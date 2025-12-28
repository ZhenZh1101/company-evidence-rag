import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

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
                self.assertEqual(self.client.get("/api/stats", headers=headers).status_code, 403)
        response = self.client.get("/api/stats", headers={"Origin": "http://127.0.0.1:8000"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["documents"], 1)

    def test_request_bounds_dates_and_company_validation(self):
        for fields in ({"question": " "}, {"question": "x" * 4001}, {"top_k": 21},
                       {"top_k": 0}, {"mode": "unknown"}, {"companies": [""]},
                       {"companies": ["UNKNOWN"]}, {"date_from": "2025-02-30"},
                       {"date_from": "2025-02-01", "date_to": "2025-01-01"}):
            with self.subTest(fields=fields):
                response = self.client.post("/api/ask", json={
                    "question": "Revenue", "mode": "lexical", "rewrite": False, **fields})
                self.assertEqual(response.status_code, 422)
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
        response = self.client.get(f"/api/source/{self.chunk_id}")
        self.assertIn("application/json", response.headers["content-type"])
        self.assertEqual(response.json()["text"], self.evidence)
        self.assertEqual(response.json()["locator"], "page 2")
        self.assertEqual(response.json()["company"], "NOC")
        self.assertEqual(self.client.get("/api/source/999999").status_code, 404)

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
        self.assertNotIn("TEST_SECRET", response.text + " ".join(logs.output))


if __name__ == "__main__":
    unittest.main()
