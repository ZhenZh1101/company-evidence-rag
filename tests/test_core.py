import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rag.client import Gateway, parse_json
from rag.config import Settings
from rag.ingest import Segment, Source, discover
from rag.pipeline import RAG, mentioned_companies
from rag.store import Store


class FakeGateway:
    def __init__(self, vectors=None, replies=()):
        self.settings = Settings(api_key="offline-test-token")
        self.vectors = vectors or {}
        self.replies = list(replies)
        self.embedded = []
        self.messages = []

    def embed(self, texts):
        self.embedded.extend(texts)
        values = np.array([self.vectors.get(t, [1.0, 0.0]) for t in texts], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    def chat(self, messages, max_tokens=2400):
        self.messages.append(messages.copy())
        if not self.replies:
            raise AssertionError("Unexpected generation call in offline test")
        return self.replies.pop(0)


class StoreAndRAGTests(unittest.TestCase):
    def test_supplied_company_aliases_match_exact_names_and_keep_issuer_scope(self):
        aliases = {'ALFA': ['Acme Labs', '艾克米', 'Shared Name'],
                   'BETA': ['B&B, Inc.', 'Shared Name'], 'PLAIN': []}
        for question, expected in (
            ('alfa的销售额和BETA的销售额', ['ALFA', 'BETA']),
            ('aCmE lAbS revenue', ['ALFA']),
            ('艾克米的销售额', ['ALFA']),
            ('(B&B, Inc.) revenue', ['BETA']),
            ('Shared Name revenue', ['ALFA', 'BETA']),
            ('ALFAX XBETA PLAIN_ Acme LabsPlus MyAcme Labs', []),
            ('Acme revenue', []),
            ('Labs revenue', []),
            ('Unknown Company revenue', []),
        ):
            with self.subTest(question=question):
                self.assertEqual(mentioned_companies(question, aliases, aliases), expected)
        self.assertEqual(mentioned_companies('Acme Labs and BETA', {'PLAIN'}, aliases), [])
        self.assertEqual(mentioned_companies('Acme Labs revenue', {'ALFA'}), [])
        self.assertEqual(mentioned_companies('艾克米 revenue', {'ALFA'}), [])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.store = Store(self.root / "test.sqlite3")
        self.addCleanup(self.store.db.close)

    def add(self, key, text, company="NOC", publication_date="2026-07-21", period="2026-Q2", company_aliases=()):
        source = Source(key, company, key, "results", publication_date, period,
                        "disclosure date" if publication_date else "quarter only", "local",
                        "https://example.test/" + key, self.root / (key + ".txt"),
                        company_aliases=list(company_aliases))
        segments = [Segment(t, f"paragraph {i + 1}") for i, t in enumerate(text if isinstance(text, list) else [text])]
        self.store.put(source, self.root, "fingerprint-" + key, "hash-" + key, segments)
        return source

    def test_hybrid_fuses_exact_and_semantic_rankings(self):
        exact = "epsilon epsilon epsilon logistics contract"
        shared = "epsilon backlog increased in the quarter"
        semantic = "Orders expanded substantially during the reporting period"
        unrelated = "Additional contextual information"
        for key, text in zip(("exact", "shared", "semantic", "unrelated"), (exact, shared, semantic, unrelated)):
            self.add(key, text)
        gateway = FakeGateway({exact: [0, 1], shared: [0.8, 0.6], semantic: [1, 0], unrelated: [0.6, 0.8], "epsilon": [1, 0]})
        self.assertEqual(self.store.embed_pending(gateway), 4)
        rag = RAG(self.store, gateway)
        results = {mode: rag.search("epsilon", mode=mode, rewrite=False, top_k=1)["sources"][0]
                   for mode in ("lexical", "dense", "hybrid")}
        self.assertEqual(results["lexical"]["document_id"], "exact")
        self.assertEqual(results["dense"]["document_id"], "semantic")
        self.assertEqual(results["hybrid"]["document_id"], "shared")
        self.assertEqual(results["hybrid"]["score"], round(2 / 62, 6))

    def test_company_specific_subqueries_do_not_search_the_other_company(self):
        self.add('cedar', 'Quarter one revenue.', company='CDR', company_aliases=['Cedar Research'])
        self.add('harbor', 'Quarter two sales.', company='HBR', company_aliases=['Harbor Labs'])
        gateway = FakeGateway(replies=[json.dumps({'queries': ['Cedar Research quarter one revenue', 'Harbor Labs quarter two sales']})])
        with patch.object(self.store, 'lexical', wraps=self.store.lexical) as search:
            RAG(self.store, gateway).search('Compare Cedar Research and Harbor Labs revenue', mode='lexical')
        calls = [(call.args[0], call.kwargs['companies']) for call in search.call_args_list]
        self.assertEqual(calls[2:], [('Cedar Research quarter one revenue', ['CDR']), ('Harbor Labs quarter two sales', ['HBR'])])
        planner_data = json.loads(gateway.messages[0][1]['content'])
        self.assertEqual(set(planner_data['company_document_titles']), {'CDR','HBR'})
        self.assertEqual(planner_data['company_aliases'], {'CDR': ['Cedar Research'], 'HBR': ['Harbor Labs']})

    def test_archive_company_aliases_reimport_without_reembedding_and_delete_with_document(self):
        from rag.cli import ingest
        archive = self.root / 'archive'
        folder = archive / 'release'
        folder.mkdir(parents=True)
        (folder / 'report.txt').write_text('Revenue was 100 million.')
        metadata = {'ticker': 'ALFA', 'company': 'Acme Holdings', 'company_name': 'Acme Labs',
                    'company_aliases': [' 艾克米 ', 'Acme   Labs'], 'files': [{'path': 'report.txt'}]}
        (folder / 'meta.json').write_text(json.dumps(metadata))
        (archive / 'index.json').write_text(json.dumps([{'folder': 'release', 'company_aliases': ['Acme Group']}]))
        sources, skipped = discover(archive)
        self.assertEqual(skipped, [])
        self.assertEqual(sources[0].company_aliases, ['Acme Group', 'Acme Holdings', 'Acme Labs', '艾克米'])
        args = SimpleNamespace(paths=[str(archive)], company=None, limit=None, prune=False,
                               report=str(self.root / 'import-report.json'))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertFalse(ingest(args, self.store))
        self.add('other', 'Revenue was 200 million.', company='BETA')
        self.assertCountEqual(self.store.company_aliases()['ALFA'], sources[0].company_aliases)
        self.assertEqual(self.store.company_aliases()['BETA'], [])
        gateway = FakeGateway()
        self.assertEqual(self.store.embed_pending(gateway), 2)
        rag = RAG(self.store, gateway)
        result = rag.search('Acme Labs revenue', mode='lexical', rewrite=False)
        self.assertEqual(result['filters']['companies'], ['ALFA'])
        self.assertEqual({s['company'] for s in result['sources']}, {'ALFA'})
        result = rag.search('Acme Labs revenue', companies=['BETA'], mode='lexical', rewrite=False)
        self.assertEqual({s['company'] for s in result['sources']}, {'BETA'})

        metadata.update(company_name='Acme Renewed', company_aliases=['艾克米'])
        (folder / 'meta.json').write_text(json.dumps(metadata))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertFalse(ingest(args, self.store))
        self.assertEqual(json.loads(Path(args.report).read_text())['imported'], 1)
        self.assertEqual(self.store.embed_pending(gateway), 0)
        self.assertCountEqual(self.store.company_aliases()['ALFA'], ['Acme Group', 'Acme Holdings', 'Acme Renewed', '艾克米'])
        self.assertIsNone(rag.search('Acme Labs revenue', mode='lexical', rewrite=False)['filters']['companies'])
        self.assertEqual(rag.search('Acme Renewed revenue', mode='lexical', rewrite=False)['filters']['companies'], ['ALFA'])
        with Store(self.root / 'test.sqlite3') as reopened:
            self.assertEqual(reopened.company_aliases(), self.store.company_aliases())
        with self.store.db:
            self.store.db.execute('DELETE FROM documents WHERE id=?', (sources[0].key,))
        self.assertEqual(self.store.company_aliases(), {'BETA': []})
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM company_aliases').fetchone()[0], 0)

    def test_alias_comparison_ask_uses_both_company_scopes(self):
        self.add('cedar', 'Revenue was 100 million.', company='CDR', company_aliases=['Cedar Research'])
        self.add('harbor', 'Revenue was 200 million.', company='HBR', company_aliases=['Harbor Labs'])
        gateway = FakeGateway(replies=[json.dumps({'labels': ['S1', 'S2']}), json.dumps({
            'answer': 'Insufficient evidence.', 'insufficient_evidence': True, 'citations': []})])
        result = RAG(self.store, gateway).ask('Compare Cedar Research and Harbor Labs revenue',
                                            mode='lexical', rewrite=False)
        self.assertEqual(result['filters']['companies'], ['CDR', 'HBR'])
        self.assertEqual({s['company'] for s in result['sources']}, {'CDR', 'HBR'})
        selection = json.loads(gateway.messages[0][1]['content'])
        self.assertEqual({s['company'] for s in selection['evidence']}, {'CDR', 'HBR'})

    def test_comparison_selector_preserves_full_evidence_and_relabels_citations(self):
        cb = 'CBRS revenue 193.4 million in quarter one.'
        noc = 'NOC revenue context. ' * 80 + 'Total sales 10,876 million in quarter two.'
        self.add('cb', cb, company='CBRS')
        self.add('noc', noc, company='NOC')
        answer = {'answer':'NOC sales 10,876 million [S1]. CBRS revenue 193.4 million [S2].',
                  'insufficient_evidence':False,'citations':[{'label':'S1','quote':'Total sales 10,876 million'}, {'label':'S2','quote':cb}]}
        gateway = FakeGateway(replies=[json.dumps({'labels':['S2','S1']}),json.dumps(answer)])
        result = RAG(self.store,gateway).ask('Compare CBRS and NOC revenue',companies=['CBRS','NOC'],mode='lexical',rewrite=False)
        self.assertFalse(result['insufficient_evidence'])
        self.assertEqual([c['source']['company'] for c in result['citations']], ['NOC','CBRS'])
        selection_input = json.loads(gateway.messages[0][1]['content'])
        self.assertEqual(selection_input['evidence'][1]['text'], noc)

    def test_company_dates_and_unknown_dates_filter_all_retrievers(self):
        self.add("current", "Revenue increased to 100 million.")
        self.add("historical", "Revenue increased to 80 million.", publication_date="2025-07-21")
        self.add("undated", "Revenue increased to 110 million.", publication_date=None)
        self.add("other-company", "Revenue increased to 120 million.", company="CBRS")
        gateway = FakeGateway()
        self.store.embed_pending(gateway)
        for mode in ("lexical", "dense", "hybrid"):
            with self.subTest(mode=mode):
                result = RAG(self.store, gateway).search("Revenue", companies=["NOC"],
                    date_from="2026-01-01", date_to="2026-12-31", rewrite=False, mode=mode)
                self.assertEqual([s["document_id"] for s in result["sources"]], ["current"])
                self.assertIn("unknown dates are excluded", result["warnings"][0])
        unfiltered = RAG(self.store, gateway).search("Revenue", companies=["NOC"], rewrite=False)
        self.assertIn("undated", [s["document_id"] for s in unfiltered["sources"]])
        unknown = next(s for s in unfiltered["sources"] if s["document_id"] == "undated")
        self.assertIsNone(unknown["publication_date"])
        self.assertEqual(unknown["publication_period"], "2026-Q2")

    def test_comparison_reserves_both_companies_and_deduplicates_copies(self):
        text = "Revenue amounted to 100 million in the quarter."
        self.add("noc", text)
        self.add("noc-copy", text)
        self.add("cbrs", text, company="CBRS")
        gateway = FakeGateway()
        self.assertEqual(self.store.embed_pending(gateway), 1)
        result = RAG(self.store, gateway).search("Compare revenue", companies=["NOC", "CBRS"], rewrite=False, top_k=10)
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual({s["company"] for s in result["sources"]}, {"NOC", "CBRS"})

    def test_update_removes_stale_fts_and_reuses_unchanged_embedding(self):
        self.add("report", ["Obsolete dividend figures", "Retained financial context"])
        gateway = FakeGateway()
        self.assertEqual(self.store.embed_pending(gateway), 2)
        self.add("report", ["Replacement earnings figures", "Retained financial context"])
        self.assertEqual(self.store.lexical("Obsolete"), [])
        self.assertEqual(len(self.store.lexical("Replacement")), 1)
        self.assertEqual(self.store.embed_pending(gateway), 1)
        self.assertEqual(self.store.embed_pending(gateway), 0)
        self.assertEqual(gateway.embedded.count("Retained financial context"), 1)
        self.assertEqual(self.store.stats()["indexed_chunks"], 2)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0], 2)
        with self.store.db:
            self.store.db.execute("DELETE FROM documents WHERE id='report'")
        self.assertEqual(self.store.lexical("Replacement"), [])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0], 0)

    def test_prune_never_drops_sources_after_incomplete_archive_discovery(self):
        from rag.cli import ingest
        archive = self.root / "archive"
        folder = archive / "release"
        folder.mkdir(parents=True)
        (folder / "report.txt").write_text("Previously indexed disclosure remains available.")
        (folder / "meta.json").write_text(json.dumps({"ticker": "NOC", "files": [{"path": "report.txt"}]}))
        (archive / "index.json").write_text(json.dumps([{"folder": "release"}]))
        report_path = self.root / "ingest-report.json"
        args = SimpleNamespace(paths=[str(archive)], company="NOC", limit=None,
                               prune=False, report=str(report_path))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertFalse(ingest(args, self.store))
        self.assertEqual(self.store.stats()["documents"], 1)
        self.assertEqual(self.store.db.execute("SELECT root FROM documents").fetchone()[0], str(archive.resolve()))
        (folder / "meta.json").write_text("{broken json")
        args.prune = True
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            ingest(args, self.store)
        self.assertEqual(self.store.stats()["documents"], 1)
        self.assertTrue(self.store.lexical("Previously"))
        report = json.loads(report_path.read_text())
        self.assertTrue(any(item.get("kind") == "error" for item in report["skipped"]))
        self.assertTrue(any("Pruning disabled" in item["message"] for item in report["warnings"]))

    def test_unembedded_index_does_not_lock_endpoint_or_model(self):
        self.add("report", "Revenue was 100 million.")
        first = FakeGateway()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            RAG(self.store, first).search("Revenue", rewrite=False)
        self.assertEqual(self.store.stats()["indexed_chunks"], 0)
        replacement = FakeGateway()
        replacement.settings = Settings(base_url="http://replacement.test/v1", embedding_model="replacement-embedding",
                                        api_key="offline-test-token")
        self.assertEqual(self.store.embed_pending(replacement), 1)
        self.assertEqual(self.store.meta("embedding_identity"), "http://replacement.test/v1|replacement-embedding")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.store.embed_pending(first)

    def test_embedding_identity_dimension_and_partial_index_fail_closed(self):
        self.add("one", "First disclosure")
        gateway = FakeGateway()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            RAG(self.store, gateway).search("disclosure", rewrite=False)
        self.assertTrue(RAG(self.store, gateway).search("disclosure", rewrite=False, mode="lexical")["sources"])
        self.store.embed_pending(gateway)
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.store.dense(np.array([1, 0, 0]))
        self.add("two", "Second disclosure")
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.store.embed_pending(FakeGateway({"Second disclosure": [1, 0, 0]}))
        changed = FakeGateway()
        changed.settings = Settings(base_url="http://different.test/v1", api_key="offline-test-token")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.store.embed_pending(changed)

    def test_unknown_company_rejected_and_empty_scope_refuses_without_generation(self):
        self.add("report", "Revenue was 100 million.")
        gateway = FakeGateway()
        rag = RAG(self.store, gateway)
        with self.assertRaisesRegex(ValueError, "Unknown company"):
            rag.ask("Revenue", companies=["AAPL"], mode="lexical", rewrite=False)
        result = rag.ask("Revenue", companies=["NOC"], date_from="2030-01-01", mode="lexical", rewrite=False)
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["citations"], [])
        self.assertEqual(gateway.messages, [])

    def test_malformed_planner_falls_back_to_original_question(self):
        self.add("report", "Revenue was 100 million.")
        for reply in ("not json", "[]", '{"queries":"bad"}', "```"):
            with self.subTest(reply=reply):
                result = RAG(self.store, FakeGateway(replies=[reply])).search("Revenue", mode="lexical")
                self.assertEqual(result["queries"], ["Revenue"])
                self.assertEqual(len(result["sources"]), 1)
                self.assertTrue(result["warnings"])

    def test_invalid_quotes_or_broken_json_twice_produce_refusal(self):
        self.add("report", "Revenue was 100 million in fiscal 2026.")
        invalid = json.dumps({"answer": "Revenue was 999 million. [S1]", "insufficient_evidence": False,
                              "citations": [{"label": "S1", "quote": "Revenue was 999 million"}]})
        for reply in (invalid, "```"):
            with self.subTest(reply=reply):
                gateway = FakeGateway(replies=[reply, reply])
                result = RAG(self.store, gateway).ask("Revenue", mode="lexical", rewrite=False)
                self.assertTrue(result["insufficient_evidence"])
                self.assertEqual(result["citations"], [])
                self.assertNotIn("999", result["answer"])
                self.assertEqual(len(gateway.messages), 2)

    def test_valid_quote_preserves_traceable_source(self):
        self.add("report", "Revenue was 100 million in fiscal 2026.")
        reply = json.dumps({"answer": "Revenue was 100 million in fiscal 2026. [S1]", "insufficient_evidence": False,
                            "citations": [{"label": "S1", "quote": "Revenue was 100 million in fiscal 2026."}]})
        result = RAG(self.store, FakeGateway(replies=[reply])).ask("Revenue", mode="lexical", rewrite=False)
        self.assertFalse(result["insufficient_evidence"])
        source = result["citations"][0]["source"]
        self.assertEqual(source["company"], "NOC")
        self.assertEqual(source["locator"], "paragraph 1")
        self.assertEqual(source["path"], str(self.root / "report.txt"))

    def test_invalid_question_dates_and_top_k_rejected(self):
        rag = RAG(self.store, FakeGateway())
        for options in ({"question": ""}, {"question": "x", "date_from": "2026-02-30"},
                        {"question": "x", "date_from": "2027-01-01", "date_to": "2026-01-01"},
                        {"question": "x", "top_k": 0}, {"question": "x", "top_k": 21}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                rag.search(**options, rewrite=False)


class ClientTests(unittest.TestCase):
    def test_embedding_batches_respect_limits_preserve_order_and_reject_dimension_changes(self):
        gateway = Gateway(Settings(api_key="offline-test-token"))
        texts = [f"{i}:small" for i in range(130)] + [f"{i}:" + "x" * 30000 for i in range(130, 133)]
        recorded = []
        later_started = Event()
        mismatched = False

        def response(route, payload):
            self.assertEqual(route, "embeddings")
            batch = payload["input"]
            recorded.append(batch.copy())
            first = int(batch[0].split(":", 1)[0])
            if first == 0:
                self.assertTrue(later_started.wait(2), "Expected another batch to execute while the first is delayed")
            else:
                later_started.set()
            entries = []
            for index, text in enumerate(batch):
                vector = [int(text.split(":", 1)[0]) + 1, 1]
                if mismatched and first == 64:
                    vector.append(1)
                entries.append({"index": index, "embedding": vector})
            return {"data": list(reversed(entries))}

        with patch.object(gateway, "_post", side_effect=response):
            vectors = gateway.embed(texts)
        self.assertGreater(len(recorded), 3)
        self.assertTrue(all(len(batch) <= 64 and sum(map(len, batch)) <= 60000 for batch in recorded))
        self.assertTrue(any(len(batch) == 64 for batch in recorded))
        self.assertCountEqual([text for batch in recorded for text in batch], texts)
        expected = np.array([[i + 1, 1] for i in range(len(texts))], dtype=np.float64)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(vectors, expected, rtol=1e-6)
        mismatched = True
        later_started.clear()
        with patch.object(gateway, "_post", side_effect=response), self.assertRaisesRegex(RuntimeError, "dimensions between batches"):
            gateway.embed(texts)

    def test_embeddings_sorted_normalized_and_invalid_vectors_rejected(self):
        gateway = Gateway(Settings(api_key="offline-test-token"))
        good = {"data": [{"index": 1, "embedding": [0, 5]}, {"index": 0, "embedding": [3, 0]}]}
        with patch.object(gateway, "_post", return_value=good):
            np.testing.assert_allclose(gateway.embed(["a", "b"]), [[1, 0], [0, 1]])
        for embeddings in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [[]], [[1, 2], [1]]):
            payload = {"data": [{"index": i, "embedding": v} for i, v in enumerate(embeddings)]}
            with self.subTest(embeddings=embeddings), patch.object(gateway, "_post", return_value=payload), self.assertRaises(RuntimeError):
                gateway.embed(["text"] * len(embeddings))
        for indexes in ([0, 0], [0], [0, 2]):
            payload = {"data": [{"index": i, "embedding": [1, 0]} for i in indexes]}
            with self.subTest(indexes=indexes), patch.object(gateway, "_post", return_value=payload), self.assertRaises(RuntimeError):
                gateway.embed(["a", "b"])

    def test_large_finite_embedding_must_not_silently_become_zero_vector(self):
        gateway = Gateway(Settings(api_key="offline-test-token"))
        payload = {"data": [{"index": 0, "embedding": [1e30, 1e30]}]}
        with patch.object(gateway, "_post", return_value=payload):
            try:
                vector = gateway.embed(["text"])[0]
            except RuntimeError:
                return  # Explicitly rejecting an unnormalizable response is also safe.
        self.assertTrue(np.isfinite(vector).all())
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_remote_endpoint_never_implicitly_receives_local_gateway_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".openclaw").mkdir()
            (root / ".openclaw/openclaw.json").write_text(json.dumps({"gateway": {"auth": {"token": "local-test-secret"}}}))
            with patch("rag.config.Path.home", return_value=root), patch.dict("os.environ", {"RAG_BASE_URL": "https://remote.example/v1"}, clear=True):
                self.assertEqual(Settings.from_env().api_key, "")
            with patch("rag.config.Path.home", return_value=root), patch.dict("os.environ", {"RAG_BASE_URL": "http://127.0.0.1:18789/v1"}, clear=True):
                self.assertEqual(Settings.from_env().api_key, "local-test-secret")

    def test_normal_json_and_fenced_json(self):
        self.assertEqual(parse_json('{"queries": []}'), {"queries": []})
        self.assertEqual(parse_json('```json\n{"queries": []}\n```'), {"queries": []})


if __name__ == "__main__":
    unittest.main()
