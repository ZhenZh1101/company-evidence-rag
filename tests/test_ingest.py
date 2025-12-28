import json
import tempfile
import unittest
from pathlib import Path

from rag.ingest import Segment, Source, chunk_segments, discover, extract


def source(path):
    return Source("test", "NOC", "Test document", "report", "2026-07-21", None, "official date", "releases", "https://example.com/report", path)


class IngestTests(unittest.TestCase):
    def test_archive_manifest_tables_provenance_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "releases" / "quarterly"
            folder.mkdir(parents=True)
            (root / "audit").mkdir()
            (root / "audit" / "bogus.txt").write_text("AUDIT IS NOT EVIDENCE")
            (folder / "article.html").write_text('<article><h1>Quarterly results</h1><p>USD millions</p><table><tr><th>Metric</th><th>2026</th><th>2025</th></tr><tr><td>Sales</td><td>10,876</td><td>10,351</td></tr></table><p>Final paragraph.</p></article>')
            (folder / "page.txt").write_text("Duplicate webpage")
            (folder / "attachment.csv").write_text("Measure,2026,2025\nEPS,7.68,8.15\n")
            (folder / "meta.json").write_text(json.dumps({"ticker": "NOC", "files": [
                {"filename": "article.html", "source_url": "https://example.com/quarterly"},
                {"path": "page.txt"}, {"path": "attachment.csv", "source_url": "https://example.com/attachment"},
                {"path": "../../../escape.txt"}]}))
            (root / "index.json").write_text(json.dumps([{"folder": "releases/quarterly", "title": "Q2 results", "category": "quarterly_results", "publication_date": None, "publication_period": "2026-Q2", "scope": "releases", "date_basis": "Quarter only", "url": "https://example.com/quarterly"}]))
            sources, report = discover(root)
            self.assertEqual({s.path.name for s in sources}, {"article.html", "attachment.csv"})
            attachment = next(s for s in sources if s.path.suffix == ".csv")
            self.assertIsNone(attachment.publication_date)
            self.assertEqual(attachment.publication_period, "2026-Q2")
            self.assertEqual(attachment.aliases[0]["date_basis"], "Quarter only")
            self.assertTrue(any("escapes" in s["reason"] for s in report))
            segments = extract(next(s for s in sources if s.path.suffix == ".html"))
            table = next(s for s in segments if s.locator == "table 1")
            self.assertIn("USD millions", table.text)
            self.assertIn("Sales | 10,876 | 10,351", table.text)
            self.assertTrue(any("Final paragraph" in s.text for s in segments))

    def test_discovery_marks_broken_metadata_and_missing_files_as_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for name, metadata in [("broken", "{invalid json"), ("missing_files", "{}"),
                                   ("missing_path", '{"files":[{}]}'),
                                   ("missing_file", '{"files":[{"path":"lost.txt"}]}'),
                                   ("unsafe_path", '{"files":[{"path":"../../outside.txt"}]}')]:
                folder = root / name
                folder.mkdir()
                (folder / "meta.json").write_text(metadata)
                records.append({"folder": name})
            (root / "index.json").write_text(json.dumps(records))
            sources, report = discover(root)
            self.assertEqual(sources, [])
            self.assertEqual(len(report), 5)
            self.assertTrue(all(r.get("kind") == "error" for r in report))

    def test_capture_wrappers_suppressed_only_when_canonical_article_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for name, canonical in [("with_article", True), ("fragment_only", False)]:
                folder = root / name
                folder.mkdir()
                filenames = ["source_fragment.html", "source_listing.html", "ir_page.txt", "unavailable_historical_calendar.txt"]
                if canonical:
                    filenames.append("article.html")
                for filename in filenames:
                    (folder / filename).write_text("Source text")
                (folder / "meta.json").write_text(json.dumps({"files": [{"path": filename} for filename in filenames]}))
                records.append({"folder": name, "category": "blogs"})
            (root / "index.json").write_text(json.dumps(records))
            sources, _ = discover(root)
            self.assertEqual({s.path.name for s in sources if s.path.parent.name == "with_article"}, {"article.html"})
            self.assertEqual({s.path.name for s in sources if s.path.parent.name == "fragment_only"}, {"source_fragment.html", "source_listing.html", "ir_page.txt"})

    def test_sec_selects_primary_html_and_exhibit_not_transformed_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "filing"
            (folder / "sec_documents").mkdir(parents=True)
            names = ["sec_documents/noc-20260630.htm", "sec_documents/exhibit99.htm", "sec_documents/R1.htm", "sec_documents/123.txt", "sec_documents/123-index.html", "123.pdf", "123.docx", "123.xls", "page.html", "filing.html"]
            for name in names:
                (folder / name).write_text("filing")
            (folder / "meta.json").write_text(json.dumps({"accession": "123", "sec_submission": {"primaryDocument": "noc-20260630.htm"}, "files": [{"path": n} for n in names]}))
            (root / "index.json").write_text(json.dumps([{"folder": "filing", "category": "sec_filings"}]))
            sources, report = discover(root, "NOC")
            self.assertEqual({s.path.name for s in sources}, {"noc-20260630.htm", "exhibit99.htm"})
            self.assertEqual(len(report), len(names) - 2)

    def test_chunk_repeats_headers_and_preserves_every_table_row(self):
        rows = [f"Metric {i} | {i * 10}.5 | {i * 9}.5" for i in range(100)]
        original = "Table context: USD millions\nMetric | 2026 | 2025\n[Rows]\n" + "\n".join(rows)
        chunks = chunk_segments([Segment(original, "table 1")], max_chars=400, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 400 for chunk in chunks))
        self.assertTrue(all("USD millions" in chunk.text and "2026 | 2025" in chunk.text for chunk in chunks))
        for row in rows:
            self.assertTrue(any(row in chunk.text for chunk in chunks), row)

    def test_long_unbroken_rows_do_not_drop_characters(self):
        row = "".join(str(i % 10) for i in range(1500))
        chunks = chunk_segments([Segment("Metric | Value\n[Rows]\n" + row, "table 1")], 300, 20)
        self.assertEqual("".join(c.text.split("[Rows]\n", 1)[1] for c in chunks), row)

    def test_nested_tables_keep_inner_cells_and_company_scopes_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.html"
            path.write_text('<table><tr><td>Report header<table><tr><th>Year</th><th>Sales</th></tr><tr><td>2026</td><td>10876</td></tr></table>Report footer</td></tr></table>')
            a, _ = discover(path, "A")
            b, _ = discover(path, "B")
            self.assertNotEqual(a[0].key, b[0].key)
            segments = extract(a[0])
            self.assertTrue(any("2026 | 10876" in x.text for x in segments))
            self.assertTrue(any("Report header" in x.text for x in segments))
            self.assertTrue(any("Report footer" in x.text for x in segments))

    def test_generic_directory_excludes_escape_and_empty_document_is_error(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as elsewhere:
            root = Path(directory)
            (root / "empty.txt").write_text("")
            external = Path(elsewhere) / "external.md"
            external.write_text("outside")
            (root / "linked.md").symlink_to(external)
            sources, report = discover(root, "TEST")
            self.assertEqual([s.path.name for s in sources], ["empty.txt"])
            self.assertTrue(any("escapes" in r["reason"] for r in report))
            with self.assertRaisesRegex(ValueError, "No extractable text"):
                extract(sources[0])

    def test_html_removes_hidden_prompt_boilerplate_but_keeps_div_prose(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filing.html"
            path.write_text('<html><body><nav>NAVIGATION</nav><script>BAD SCRIPT</script><ix:hidden>HIDDEN XBRL</ix:hidden><div>Actual filing prose: sales increased.</div><table><tr><td>Revenue</td><td>193.4</td></tr></table></body></html>')
            text = "\n".join(s.text for s in extract(source(path)))
            self.assertIn("Actual filing prose", text)
            self.assertIn("193.4", text)
            self.assertNotIn("HIDDEN XBRL", text)
            self.assertNotIn("NAVIGATION", text)
            self.assertNotIn("BAD SCRIPT", text)


if __name__ == "__main__":
    unittest.main()
