import copy
import json
import tempfile
import unittest
from pathlib import Path

from rag.archive import normalize_record, read_manifest


class ArchiveTests(unittest.TestCase):
    def test_manifest_recognition_priority_and_invalid_catalogs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(read_manifest(root))
            (root / "inventory.json").write_text(json.dumps([{"folder": "inventory"}]))
            self.assertEqual(read_manifest(root), [{"folder": "inventory"}])
            (root / "index.json").write_text(json.dumps([{"folder": "index"}]))
            self.assertEqual(read_manifest(root), [{"folder": "index"}])
            (root / "index.json").write_text('{"ordinary": "document"}')
            self.assertIsNone(read_manifest(root))
            for invalid in ('{broken', '[{"folder":"one"}, {}]', '[{"folder":null}]'):
                (root / "index.json").write_text(invalid)
                with self.assertRaises(ValueError):
                    read_manifest(root)

    def test_amkr_local_files_include_attachments_and_localized_urls(self):
        item = {"folder": "IR_press_releases/release", "date": "2026-07-27", "url": "https://example.com/release", "category": "SEC filings"}
        meta = {"scope": {"start": "2024-10-07"}, "accession": "123", "form": "8-K", "sec_submission": {"reportDate": "2026-06-30"},
                "local_files": [{"filename": "page.html"}, {"filename": "slides.pdf"}, {"filename": "content_zh.txt"}],
                "files": [{"filename": "page.html", "kind": "source_html"}],
                "attachments": [{"filename": "slides.pdf", "url": "https://example.com/slides", "publication_date": "2026-07"}],
                "language_versions": [{"text_file": "content_zh.txt", "language": "zh", "url": "https://example.com/zh"}]}
        original = copy.deepcopy((item, meta))
        row, normalized = normalize_record(item, meta)
        self.assertEqual((item, meta), original)
        self.assertEqual(row["category"], "sec_filings")
        self.assertEqual(row["publication_date"], "2026-07-27")
        self.assertIsInstance(normalized["scope"], str)
        self.assertEqual(normalized["sec_submission"], meta["sec_submission"])
        self.assertEqual([f["source_url"] for f in normalized["files"]], ["https://example.com/release", "https://example.com/slides", "https://example.com/zh"])
        self.assertEqual(normalized["files"][1]["publication_period"], "2026-07")
        self.assertIsNone(normalized["files"][1]["publication_date"])

    def test_vst_inventory_fallback_converts_only_exact_folder_prefix(self):
        item = {"folder": "releases/report", "publication_date": "2026-07", "source_url": "https://example.com/report", "files": [
            {"path": "releases/report/report.pdf", "filename": "report.pdf"},
            {"path": "../../escape.txt", "filename": "escape.txt"},
            {"path": "/outside/report.pdf", "filename": "report.pdf"},
            {"path": "releases/report/../../outside.txt"}]}
        row, meta = normalize_record(item, {})
        self.assertEqual([f["path"] for f in meta["files"]], ["report.pdf", "../../escape.txt", "/outside/report.pdf", "../../outside.txt"])
        self.assertEqual(row["publication_period"], "2026-07")
        self.assertIsNone(row["publication_date"])
        self.assertEqual((row["url"], meta["url"], row["scope"]), (item["source_url"], item["source_url"], "releases"))

    def test_nok_unknown_dates_and_month_year_precision(self):
        row, meta = normalize_record({"folder": "undated_supplement/article", "publication_date": None},
                                     {"publication_date": "2026-10-07", "event_date": "2026-10-08", "files": []})
        self.assertIsNone(row["publication_date"])
        self.assertIsNone(meta["publication_date"])
        self.assertEqual(row["scope"], "undated_supplement")
        for period in ("2025-08", "2025"):
            row, meta = normalize_record({"folder": "releases/policy", "publication_date": period}, {"files": []})
            self.assertEqual(row["publication_period"], period)
            self.assertIsNone(meta["publication_date"])
        row, _ = normalize_record({"folder": "events"}, {"event_date": "2026-10-08", "files": []})
        self.assertIsNone(row["publication_date"])


if __name__ == "__main__":
    unittest.main()
