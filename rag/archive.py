"""Normalize local archive catalogs without modifying their source metadata."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path


def read_manifest(root: Path) -> list | None:
    if not root.is_dir():
        return None
    manifest = next((root / name for name in ("index.json", "inventory.json") if (root / name).exists()), None)
    if manifest is None:
        return None
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list) or (records and not any(isinstance(row, dict) and "folder" in row for row in records)):
        return None
    if any(not isinstance(row, dict) or not isinstance(row.get("folder"), str) or not row["folder"] for row in records):
        raise ValueError(f"Archive manifest entries must have a folder: {manifest}")
    return records


def _normalize_date(record: dict) -> None:
    value = record.get("publication_date")
    if isinstance(value, str) and re.fullmatch(r"\d{4}(?:-\d{2})?", value):
        date.fromisoformat(value + ("-01-01" if len(value) == 4 else "-01"))
        record["publication_period"], record["publication_date"] = value, None


def normalize_record(item: dict, meta: dict) -> tuple[dict, dict]:
    """Return copied catalog/metadata records with folder-relative file paths."""
    item, meta = dict(item), dict(meta)
    for key in ("publication_date", "date"):
        if key in item:
            item["publication_date"] = item[key]
            break
    else:
        item["publication_date"] = meta.get("publication_date", meta.get("date", meta.get("published_date")))
    item.setdefault("publication_period", meta.get("publication_period"))
    _normalize_date(item)
    meta["publication_date"] = item["publication_date"]
    meta["publication_period"] = item.get("publication_period")
    url = item.get("url") or item.get("source_url") or meta.get("url") or meta.get("source_url") or ""
    item["url"] = meta["url"] = url
    category = item.get("category") or meta.get("category") or "document"
    item["category"] = meta["category"] = "sec_filings" if category == "SEC filings" else category
    folder = item.get("folder", "")
    scope = next((value for value in (item.get("scope"), meta.get("scope")) if isinstance(value, str) and value), None)
    if scope is None:
        scope = "supplement" if item.get("supplement") or meta.get("is_supplement") else (folder.split("/", 1)[0] or "local")
    item["scope"] = meta["scope"] = scope

    if "local_files" in meta:
        registered = meta["local_files"]
    elif "files" in meta:
        registered = meta["files"]
    else:
        registered = item.get("files")
    if not isinstance(registered, list):
        raise ValueError("Archive metadata files must be a list")

    def relative(record):
        value = record.get("path") or record.get("filename")
        # Only strip this exact catalog folder; leave traversal/absolute paths for _safe_path.
        prefix = folder.rstrip("/") + "/"
        if folder and isinstance(value, str) and value.startswith(prefix) and not Path(value).is_absolute():
            value = value[len(prefix):]
        return value

    details = {}
    for key in ("files", "attachments"):
        for record in meta.get(key, []) if isinstance(meta.get(key, []), list) else []:
            if isinstance(record, dict) and relative(record):
                name = relative(record)
                details[name] = {**details.get(name, {}), **record}
                if record.get("source_url") or record.get("url"):
                    details[name]["source_url"] = record.get("source_url") or record["url"]
    for version in meta.get("language_versions", []):
        for key in ("html_file", "text_file"):
            if version.get(key) and version.get("url"):
                name = version[key]
                details[name] = {**details.get(name, {}), "source_url": version["url"], "language": version.get("language")}
    files = []
    for record in registered:
        if not isinstance(record, dict):
            files.append(record)
            continue
        name = relative(record)
        normalized = {**details.get(name, {}), **record}
        if name is not None:
            normalized["path"] = name
        normalized["source_url"] = normalized.get("source_url") or normalized.get("url") or url
        _normalize_date(normalized)
        files.append(normalized)
    meta["files"] = files
    return item, meta
