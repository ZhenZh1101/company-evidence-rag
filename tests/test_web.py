import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from bs4 import BeautifulSoup

from rag.config import Settings
from rag.ingest import Segment, Source
from rag.store import Store
from rag.web import create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.settings = Settings(db_path=root / "test.sqlite3", api_key="TEST_SECRET")
        self.evidence = 'Revenue was USD 100 million. <script>alert("document")</script>'
        source = Source("doc", "NOC", "Annual report", "report", "2025-01-30", "2024",
                        "Official publication", "local", "https://example.com/report", root / "report.txt")
        with Store(self.settings.db_path) as store:
            store.put(source, root, "fingerprint", "content", [Segment(self.evidence, "page 2")])
            self.chunk_id = store.lexical("Revenue")[0]
        self.gateway = Mock(settings=self.settings)
        self.gateway.chat.side_effect = AssertionError("Unexpected model call")
        self.gateway.embed.side_effect = AssertionError("Unexpected embedding call")
        with patch("rag.web.Gateway", return_value=self.gateway):
            self.client = TestClient(create_app(self.settings), base_url="http://127.0.0.1:8000")
        self.addCleanup(self.client.close)

    def test_local_host_and_same_origin_required(self):
        for headers in ({"Host": "attacker.example"}, {"Host": "127.0.0.1:bad"},
                        {"Origin": "https://attacker.example"}, {"Origin": "null"},
                        {"Origin": "http://127.0.0.1:9000"}, {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                denied = self.client.get("/api/stats", headers=headers)
                self.assertEqual(denied.status_code, 403)
                self.assertEqual(denied.json()['error_messages'], {
                    'en': 'Only local same-origin access is allowed.', 'zh-CN': '仅允许本机同源访问。'})
                self.assertEqual(denied.json()['detail'], denied.json()['error_messages']['en'])
        response = self.client.get("/api/stats", headers={"Origin": "http://127.0.0.1:8000"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["documents"], 1)

    def test_request_bounds_dates_and_company_validation(self):
        for fields in ({"question": " "}, {"question": "x" * 4001}, {"top_k": 21},
                       {"top_k": 0}, {"mode": "unknown"}, {"companies": [""]},
                       {"language": "fr"}, {"language": ""}, {"language": None},
                       {"companies": ["UNKNOWN"]}, {"date_from": "2025-02-30"},
                       {"date_from": "2025-02-01", "date_to": "2025-01-01"}):
            with self.subTest(fields=fields):
                response = self.client.post("/api/ask", json={
                    "question": "Revenue", "mode": "lexical", "rewrite": False, **fields})
                self.assertEqual(response.status_code, 422)
        self.gateway.chat.assert_not_called()
        self.gateway.embed.assert_not_called()

    def test_validation_language_prefers_body_and_errors_include_both_languages(self):
        for fields, headers, language in (
            ({'language': 'zh-CN'}, {}, 'zh-CN'),
            ({'language': 'en'}, {'Accept-Language': 'zh-CN'}, 'en'),
            ({}, {'Accept-Language': 'zh-CN'}, 'zh-CN'),
            ({'language': 'fr'}, {'Accept-Language': 'zh-CN'}, 'zh-CN'),
        ):
            with self.subTest(fields=fields, headers=headers):
                response = self.client.post('/api/ask', json={'question': ' ', **fields}, headers=headers)
                self.assertEqual(response.status_code, 422)
                result = response.json()
                question_error = next(error for error in result['detail'] if error['loc'] == ['body', 'question'])
                expected = '请输入问题。' if language == 'zh-CN' else 'Value error, Enter a question.'
                self.assertEqual(question_error['msg'], expected)
                self.assertIn('Enter a question.', result['error_messages']['en'])
                self.assertIn('请输入问题。', result['error_messages']['zh-CN'])
                self.assertEqual(response.headers['x-content-type-options'], 'nosniff')
        self.gateway.chat.assert_not_called()
        self.gateway.embed.assert_not_called()

    def test_runtime_errors_follow_body_language_and_preserve_dynamic_details(self):
        for language in ('en', 'zh-CN'):
            with self.subTest(language=language):
                response = self.client.post('/api/ask', json={
                    'question': 'Revenue', 'companies': ['UNKNOWN'], 'mode': 'lexical',
                    'rewrite': False, 'language': language,
                }, headers={'Accept-Language': 'zh-CN' if language == 'en' else 'en'})
                self.assertEqual(response.status_code, 422)
                result = response.json()
                self.assertEqual(result['error_messages'], {
                    'en': 'Unknown company filter: UNKNOWN', 'zh-CN': '未知公司筛选条件：UNKNOWN'})
                self.assertEqual(result['detail'], result['error_messages'][language])
        self.gateway.chat.assert_not_called()
        self.gateway.embed.assert_not_called()

    def test_html_and_source_have_correct_content_and_protection(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn('id="ask-form"', page.text)
        self.assertIn("frame-ancestors 'none'", page.headers["content-security-policy"])
        self.assertEqual(page.headers["x-content-type-options"], "nosniff")
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertNotIn("TEST_SECRET", page.text)
        document = BeautifulSoup(page.text, "html.parser")
        self.assertEqual(document.html["lang"], "en")
        language_options = document.select("select#language option")
        self.assertEqual({option["value"] for option in language_options}, {"en", "zh-CN"})
        selected = next((option for option in language_options if option.has_attr("selected")), language_options[0])
        self.assertEqual(selected["value"], "en")
        response = self.client.get(f"/api/source/{self.chunk_id}")
        self.assertIn("application/json", response.headers["content-type"])
        self.assertEqual(response.json()["text"], self.evidence)
        self.assertEqual(response.json()["locator"], "page 2")
        self.assertEqual(response.json()["company"], "NOC")
        missing_source = self.client.get("/api/source/999999")
        self.assertEqual(missing_source.status_code, 404)
        self.assertEqual(missing_source.json()["detail"], "Source passage not found.")
        chinese_source = self.client.get("/api/source/999999", headers={"Accept-Language": "zh-CN"})
        self.assertEqual(chinese_source.status_code, 404)
        self.assertEqual(chinese_source.json()["detail"], "未找到该原文片段。")
        self.assertEqual(missing_source.json()['error_messages'], chinese_source.json()['error_messages'])
        self.assertEqual(missing_source.json()['error_messages'], {
            'en': 'Source passage not found.', 'zh-CN': '未找到该原文片段。'})

    def test_answer_uses_real_store_and_gateway_errors_are_sanitized(self):
        self.gateway.chat.side_effect = None
        self.gateway.chat.return_value = json.dumps({
            "answer": "Revenue was USD 100 million. [S1]",
            "insufficient_evidence": False,
            "citations": [{"label": "S1", "quote": "Revenue was USD 100 million."}],
        })
        payload = {"question": "Revenue", "companies": ["NOC"], "mode": "lexical", "rewrite": False}
        response = self.client.post("/api/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"][0]["source"]["id"], self.chunk_id)
        self.assertFalse(response.json()["insufficient_evidence"])
        self.gateway.embed.assert_not_called()
        self.gateway.chat.side_effect = RuntimeError("Bearer TEST_SECRET")
        with self.assertLogs("rag.web", level="ERROR") as logs:
            response = self.client.post("/api/ask", json=payload)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()['detail'], response.json()['error_messages']['en'])
        self.assertEqual(response.json()['error_messages']['zh-CN'], '问答失败，请检查本地模型网关是否可用、凭据配置以及索引状态。')
        self.assertNotIn("TEST_SECRET", response.text + " ".join(logs.output))

    def test_answer_language_defaults_to_english_and_can_be_chinese(self):
        self.gateway.chat.side_effect = None
        for language, question, answer, instruction in (
            (None, "Revenue 收入是多少？", "Revenue was USD 100 million. [S1]", "English (en)"),
            ("zh-CN", "What was Revenue?", "收入为一亿美元。[S1]", "Simplified Chinese (zh-CN)"),
        ):
            with self.subTest(language=language):
                self.gateway.chat.reset_mock()
                self.gateway.chat.return_value = json.dumps({
                    "answer": answer,
                    "insufficient_evidence": False,
                    "citations": [{"label": "S1", "quote": "Revenue was USD 100 million."}],
                })
                payload = {"question": question, "mode": "lexical", "rewrite": False}
                if language:
                    payload["language"] = language
                response = self.client.post("/api/ask", json=payload)
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(result["language"], language or "en")
                self.assertEqual(result["answer"], answer)
                self.assertEqual(result["citations"][0]["quote"], "Revenue was USD 100 million.")
                self.gateway.chat.assert_called_once()
                self.assertIn(instruction, self.gateway.chat.call_args.args[0][0]["content"])

    def test_no_evidence_response_uses_requested_language_without_model_calls(self):
        for language, expected in (
            ("en", "No usable evidence was found within the selected filters. Check the import status or adjust the filters."),
            ("zh-CN", "当前筛选范围内没有检索到可用证据。请检查导入状态或调整筛选条件。"),
        ):
            with self.subTest(language=language):
                response = self.client.post("/api/ask", json={
                    "question": "Unmatchedxyz", "mode": "lexical", "rewrite": False,
                    "language": language,
                })
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(result["answer"], expected)
                self.assertEqual(result["answer_code"], "no_evidence")
                self.assertTrue(result["insufficient_evidence"])
                self.assertEqual(result["citations"], [])
        self.gateway.chat.assert_not_called()
        self.gateway.embed.assert_not_called()

    def test_markdown_answer_formats_tables_lists_and_code_without_losing_raw_answer(self):
        answer = '## 结果\n\n**收入** [S1]\n\n- 第一项\n- 第二项\n\n| 公司 | 收入 |\n| --- | ---: |\n| NOC | 100 |\n\n> 原文说明\n\n```python\nprint("<value>")\n```'
        with patch('rag.web.RAG.ask', return_value={'answer':answer}):
            response = self.client.post('/api/ask', json={'question':'Revenue'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['answer'], answer)
        document = BeautifulSoup(response.json()['answer_html'], 'html.parser')
        for tag in ('h2','strong','ul','table','blockquote','pre','code'):
            self.assertIsNotNone(document.find(tag), tag)
        self.assertEqual(document.select_one('tbody td').text, 'NOC')
        self.assertEqual(document.select_one('pre code').text, 'print("<value>")\n')
        self.assertIn('[S1]', document.get_text())

    def test_markdown_escapes_html_blocks_unsafe_links_images_and_citation_references(self):
        answer = '''<script>alert(1)</script><img src=x onerror=alert(2)>

[bad](javascript:alert(3)) [encoded](jav&#x61;script:alert(4)) [file](file:///etc/passwd)
[data](data:text/html;base64,PHNjcmlwdD4=) ![pixel](https://example.com/track.png)
[safe](https://example.com/report) [S1][S2]

[S2]: https://example.com/other
'''
        with patch('rag.web.RAG.ask', return_value={'answer':answer}):
            response = self.client.post('/api/ask', json={'question':'Revenue'})
        document = BeautifulSoup(response.json()['answer_html'], 'html.parser')
        self.assertIsNone(document.find(['script','img','iframe']))
        for link in document.find_all('a'):
            self.assertTrue(link['href'].startswith('https://'))
            self.assertFalse(any(name.startswith('on') for name in link.attrs))
        self.assertIn('<script>alert(1)</script>', document.get_text())
        self.assertIn('[S1][S2]', document.get_text())
        self.assertIsNotNone(document.find('a', href='https://example.com/report'))


if __name__ == "__main__":
    unittest.main()
