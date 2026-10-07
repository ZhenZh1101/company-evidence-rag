"""Package the existing index and original archives for a Hugging Face Docker Space.

No documents are parsed and no model APIs are called. Archives stay compressed;
the UI reads evidence from SQLite and links to the original public source URLs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tarfile


OPENCLAW_IDENTITY = 'http://127.0.0.1:18789/v1|openclaw/llm-gpt55|openai/text-embedding-3-large'
OPENAI_IDENTITY = 'https://api.openai.com/v1|text-embedding-3-large'
EXCLUDED_DIRECTORIES = {'audit', '_audit', '_attachment_cache', '__pycache__', 'node_modules'}
EXCLUDED_SUFFIXES = {'.log', '.py', '.pyc', '.pyo', '.pem', '.key'}


def archive_files(root):
    """Include original materials/metadata, excluding collector runtime artifacts."""
    files, excluded = [], []
    for directory, directories, names in os.walk(root):
        base = Path(directory)
        for name in sorted(directories):
            path = base / name
            if name.startswith('.') or name in EXCLUDED_DIRECTORIES:
                directories.remove(name)
                excluded.append(path.relative_to(root).as_posix() + '/')
            elif path.is_symlink():
                raise ValueError(f'Source archive contains a symlink: {path}')
        directories.sort()
        for name in sorted(names):
            path = base / name
            if name.startswith('.') or path.suffix.lower() in EXCLUDED_SUFFIXES:
                excluded.append(path.relative_to(root).as_posix())
            elif path.is_symlink() or not path.is_file():
                raise ValueError(f'Source archive contains a non-regular file: {path}')
            else:
                files.append(path)
    return files, sorted(excluded)


def inspect_database(db):
    metadata = dict(db.execute('SELECT key,value FROM metadata'))
    identity = metadata.get('embedding_identity')
    if identity not in (OPENCLAW_IDENTITY, OPENAI_IDENTITY):
        raise ValueError('Only a verified OpenAI text-embedding-3-large index can be packaged.')
    if metadata.get('embedding_dimension') != '3072':
        raise ValueError('Expected 3072-dimensional text-embedding-3-large vectors.')
    if db.execute('SELECT 1 FROM embeddings WHERE dimension != 3072 OR length(vector) != 12288 LIMIT 1').fetchone():
        raise ValueError('Cached vector dimension or byte length is invalid.')
    chunks = db.execute('SELECT count(*) FROM chunks').fetchone()[0]
    embedded = db.execute('SELECT count(*) FROM chunks JOIN embeddings USING(text_hash)').fetchone()[0]
    if not chunks or embedded != chunks:
        raise ValueError('Every existing chunk must have a cached embedding before packaging.')
    if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok' or db.execute('PRAGMA foreign_key_check').fetchone():
        raise ValueError('Source database integrity check failed.')
    return {
        'documents': db.execute('SELECT count(*) FROM documents').fetchone()[0],
        'chunks': chunks,
        'embeddings': db.execute('SELECT count(*) FROM embeddings').fetchone()[0],
        'embedded_chunks': embedded,
        'embedding_dimension': 3072,
        'original_embedding_identity': identity,
        'embedding_identity': OPENAI_IDENTITY,
        'companies': dict(db.execute('SELECT company,count(*) FROM documents GROUP BY company ORDER BY company')),
    }


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_bundle(source_db, output, *, runtime_root=Path('/app/space-data'), progress=print):
    source_db, output = Path(source_db).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    with sqlite3.connect(source_db.as_uri() + '?mode=ro', uri=True) as source:
        source.execute('BEGIN')  # Keep inventory and backup on the same SQLite snapshot.
        report = inspect_database(source)
        documents = source.execute('SELECT id,root,path FROM documents').fetchall()
        roots = sorted({Path(row[1]).resolve() for row in documents})
        if len({root.name for root in roots}) != len(roots):
            raise ValueError('Archive roots must have unique directory names.')
        inventories = {}
        for root in roots:
            if not root.is_dir() or output.is_relative_to(root):
                raise ValueError(f'Missing archive root or output inside source archive: {root}')
            files, excluded = archive_files(root)
            inventories[root] = files, excluded
        included = {path for files, _ in inventories.values() for path in files}
        changes = []
        for document_id, old_root, old_path in documents:
            root, path = Path(old_root).resolve(), Path(old_path).resolve()
            relative = path.relative_to(root)
            if path not in included:
                raise ValueError(f'Indexed source is missing or excluded from the bundle: {path}')
            archive = runtime_root / 'archive' / (root.name + '.tar.gz')
            changes.append((str(archive), str(archive) + ':' + relative.as_posix(), document_id))
        output.mkdir(parents=True)
        try:
            db_path = output / 'rag.sqlite3'
            progress('Backing up the complete SQLite index...', flush=True)
            with sqlite3.connect(db_path) as target:
                source.backup(target)
                target.execute('PRAGMA journal_mode=DELETE')
                target.executemany('UPDATE documents SET root=?,path=? WHERE id=?', changes)
                target.executemany('INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)', [
                    ('deployment_original_embedding_identity', report['original_embedding_identity']),
                    ('embedding_identity', OPENAI_IDENTITY),
                ])
                # The new runtime location gets a fresh Qdrant collection from cached vectors.
                target.execute("DELETE FROM metadata WHERE key='qdrant_target'")
                target.commit()
                inspect_database(target)
            report['database'] = {'file': 'rag.sqlite3', 'bytes': db_path.stat().st_size, 'sha256': checksum(db_path)}
            report['archives'] = []
            (output / 'archive').mkdir()
            for root, (files, excluded) in inventories.items():
                archive = output / 'archive' / (root.name + '.tar.gz')
                progress(f'Packing {root.name}: {len(files):,} source files...', flush=True)
                with tarfile.open(archive, 'w:gz', compresslevel=1) as bundle:
                    for path in files:
                        bundle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
                report['archives'].append({
                    'file': archive.relative_to(output).as_posix(),
                    'source_files': len(files), 'source_bytes': sum(path.stat().st_size for path in files),
                    'bytes': archive.stat().st_size, 'sha256': checksum(archive),
                    'excluded_collector_artifacts': excluded,
                })
            report['archive_note'] = ('Original materials and metadata are preserved in compressed archives. '
                                      'Collector audits, download caches, code, logs, hidden files and credentials are excluded. '
                                      'Database source paths use archive.tar.gz:member notation; the UI reads indexed text and public source URLs.')
            report['total_bytes'] = report['database']['bytes'] + sum(item['bytes'] for item in report['archives'])
            (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            return report
        except BaseException:
            shutil.rmtree(output)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('data/expansion-20261007.sqlite3'))
    parser.add_argument('--output', type=Path, default=Path('space-data'))
    args = parser.parse_args()
    report = prepare_bundle(args.db, args.output)
    print(f"Prepared {report['documents']:,} documents / {report['chunks']:,} chunks "
          f"in {args.output} ({report['total_bytes'] / 1_000_000_000:.2f} GB).")


if __name__ == '__main__':
    main()
