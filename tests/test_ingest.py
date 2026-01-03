import json
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

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

    def test_material_directory_snapshots_do_not_inherit_document_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for name, category, item_url, capture_url, attachment in [
                ("annual", "annual_report", "https://example.com/annual.pdf", "https://example.com/", True),
                ("quarterly", "quarterly_results", "https://example.com/quarterly#Q3-2024", "https://example.com/quarterly", True),
                ("event", "earnings_call", "https://example.com/event", "https://example.com/event", False),
                ("article", "news", "https://example.com/news", "https://example.com/news", True),
            ]:
                folder = root / name
                folder.mkdir()
                (folder / "page.txt").write_text("Future 2026 homepage content" if name in {"annual", "quarterly"} else "Actual dated body")
                files = [{"path": "page.txt", "kind": "readable_text", "source_url": capture_url}]
                if attachment:
                    (folder / "report.pdf").write_bytes(b"%PDF")
                    files.append({"path": "report.pdf", "kind": "attachment", "source_url": item_url if name == "annual" else "https://example.com/attachment.pdf"})
                (folder / "meta.json").write_text(json.dumps({"url": item_url, "files": files}))
                records.append({"folder": name, "category": category, "publication_date": "2024-10-24"})
            (root / "index.json").write_text(json.dumps(records))
            sources, report = discover(root)
            self.assertEqual({s.path.parent.name for s in sources if s.path.name == "page.txt"}, {"event", "article"})
            self.assertEqual(sum(s.path.name == "report.pdf" for s in sources), 3)
            self.assertEqual(sum("snapshot is not dated document evidence" in r["reason"] for r in report), 2)
            self.assertFalse(any(r.get("kind") == "error" for r in report))

    def test_pdf_ocr_only_reads_blank_pages_and_preserves_order(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            self.assertLessEqual(kwargs["timeout"], 120)
            output = {"pdftotext": "Existing first page\f\fExisting third page\f",
                      "pdftoppm": "", "tesseract": "Recognized second page"}[command[0]]
            return subprocess.CompletedProcess(command, 0, output, "")

        with patch("rag.ingest.shutil.which", side_effect=lambda name: "/bin/" + name), patch("rag.ingest.subprocess.run", side_effect=run):
            with warnings.catch_warnings(record=True) as notices:
                segments = extract(source(Path("/example/report.pdf")), ocr=True)
        self.assertEqual([x.text for x in segments], ["Existing first page", "Recognized second page", "Existing third page"])
        self.assertEqual([x.locator for x in segments], ["page 1", "page 2 (OCR; verify against original)", "page 3"])
        self.assertEqual([c[0] for c in calls], ["pdftotext", "pdftoppm", "tesseract"])
        self.assertEqual(calls[1][1:5], ["-f", "2", "-l", "2"])
        self.assertEqual(calls[2][-2:], ["-l", "eng"])
        self.assertTrue(any("used local English OCR" in str(w.message) for w in notices))

    def test_pdf_ocr_missing_binary_is_actionable_error(self):
        with patch("rag.ingest.shutil.which", side_effect=lambda name: None if name == "tesseract" else "/bin/" + name), patch("rag.ingest.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "\f", "")):
            with self.assertRaisesRegex(ValueError, "requires tesseract"):
                extract(source(Path("/example/scan.pdf")), ocr=True)

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

    def test_html_removes_only_globally_empty_columns_and_pads_ragged_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "columns.html"
            path.write_text("""<table>
                <tr><th>Metric</th><th></th><th>2026</th><th></th><th></th><th>2025</th><th></th></tr>
                <tr><td>Sales</td><td></td><td>$</td><td>193406</td><td></td><td>$</td><td>99512</td></tr>
                <tr><td>Only prior</td><td></td><td></td><td></td><td></td><td>$</td><td>123</td></tr>
                <tr><td>Ragged</td></tr>
            </table>""")
            table = next(s for s in extract(source(path)) if s.locator == "table 1")
            self.assertIn("Metric | 2026 |  | 2025 |", table.text)
            self.assertIn("Sales | $ | 193406 | $ | 99512", table.text)
            self.assertIn("Only prior |  |  | $ | 123", table.text)
            self.assertIn("Ragged |  |  |  |", table.text)

    def test_div_caption_units_follow_wrappers_but_not_prior_table_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.html"
            path.write_text("""<html><body>
                <div>CEREBRAS SYSTEMS INC.</div><div>RECONCILIATION</div>
                <div>(unaudited)</div><div>(in thousands)</div>
                <div><div><table><tr><th>Metric</th><th>2026</th></tr><tr><td>Revenue</td><td>193406</td></tr></table></div></div>
                <table><tr><td>Boundary target</td><td>17</td></tr></table>
                <p>Old units in millions</p><hr>
                <table><tr><td>After rule target</td><td>42</td></tr></table>
                <div>(in millions)</div><table><tr><td>Fresh caption target</td><td>63</td></tr></table>
            </body></html>""")
            tables = [s for s in extract(source(path)) if s.locator.startswith("table ")]
            first = next(s for s in tables if "193406" in s.text)
            self.assertIn("(in thousands)", first.text)
            self.assertIn("(unaudited)", first.text)
            self.assertIn("2026", first.text)
            self.assertNotIn("in thousands", next(s for s in tables if "Boundary target" in s.text).text)
            self.assertNotIn("in millions", next(s for s in tables if "After rule target" in s.text).text)
            self.assertIn("(in millions)", next(s for s in tables if "Fresh caption target" in s.text).text)

    def test_ixbrl_hidden_resources_removed_but_visible_facts_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filing.htm"
            path.write_text("""<html><body>
              <div style=" DISPLAY : none !important; "><ix:header><ix:resources>
                <xbrli:context id="c1"><xbrldi:explicitMember>noc:AllOtherGeographicRegionDomain</xbrldi:explicitMember></xbrli:context>
              </ix:resources></ix:header></div>
              <ix:header><ix:resources>UNSTYLED RESOURCE</ix:resources></ix:header>
              <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
              <div style="color:black;visibility:hidden">INVISIBLE TEXT</div>
              <p><ix:nonnumeric>Consolidated results</ix:nonnumeric> in millions</p>
              <table><tr><th>Period</th><th>Sales</th></tr><tr><td>Q2 2026</td><td><ix:nonfraction contextRef="c1">10,876</ix:nonfraction></td></tr></table>
              <ix:continuation>Visible explanatory note</ix:continuation>
            </body></html>""")
            text = "\n".join(s.text for s in extract(source(path)))
            for hidden in ["AllOtherGeographicRegionDomain", "UNSTYLED RESOURCE", "iso4217", "INVISIBLE TEXT"]:
                self.assertNotIn(hidden, text)
            for visible in ["Consolidated results", "in millions", "10,876", "Visible explanatory note"]:
                self.assertIn(visible, text)

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
