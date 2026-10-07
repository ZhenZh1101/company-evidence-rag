import json
import os
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest

from scripts.prepare_space import OPENAI_IDENTITY, OPENCLAW_IDENTITY, prepare_bundle


class SpaceBundleTests(unittest.TestCase):
    def make_index(self, directory):
        root = directory / 'source-company'
        root.mkdir()
        document = root / 'report.txt'
        document.write_text('Original report.')
        os.link(document, root / 'copy.txt')  # The real archives contain hard-linked originals.
        (root / 'meta.json').write_text('{"source_url":"https://example.org/report"}')
        (root / '.env').write_text('FAKE_SECRET=do-not-copy')
        (root / 'collector.log').write_text('operational output')
        (root / '_attachment_cache').mkdir()
        (root / '_attachment_cache' / 'duplicate.txt').write_text('download cache')
        db_path = directory / 'original.sqlite3'
        with sqlite3.connect(db_path) as db:
            db.executescript('''
                CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE documents(id TEXT PRIMARY KEY,root TEXT,path TEXT,company TEXT,source_url TEXT,aliases TEXT);
                CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id TEXT REFERENCES documents(id),text_hash TEXT);
                CREATE TABLE embeddings(text_hash TEXT PRIMARY KEY,dimension INTEGER,vector BLOB);
            ''')
            db.executemany('INSERT INTO metadata VALUES (?,?)', [
                ('embedding_identity', OPENCLAW_IDENTITY), ('embedding_dimension', '3072'),
                ('qdrant_target', 'http://local|previous'),
            ])
            db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?)',
                       ('source-1', str(root), str(document), 'Company', 'https://example.org/report', '[]'))
            db.execute('INSERT INTO chunks VALUES (?,?,?)', (1, 'source-1', 'text-hash'))
            db.execute('INSERT INTO embeddings VALUES (?,?,?)', ('text-hash', 3072, b'\x00\x00\x80\x3f' * 3072))
        return db_path

    def test_bundle_preserves_originals_vectors_and_provenance_without_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            db_path = self.make_index(directory)
            output = directory / 'bundle'
            report = prepare_bundle(db_path, output, progress=lambda *args, **kwargs: None)
            self.assertEqual(report['documents'], 1)
            self.assertEqual(report['embedded_chunks'], 1)
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['database']['sha256'], report['database']['sha256'])
            archive = report['archives'][0]
            with tarfile.open(output / archive['file']) as bundle:
                self.assertEqual(set(bundle.getnames()), {'copy.txt', 'report.txt', 'meta.json'})
                self.assertEqual(bundle.extractfile('report.txt').read(), b'Original report.')
            self.assertEqual(archive['excluded_collector_artifacts'], ['.env', '_attachment_cache/', 'collector.log'])
            with sqlite3.connect(output / 'rag.sqlite3') as db:
                metadata = dict(db.execute('SELECT * FROM metadata'))
                self.assertEqual(metadata['embedding_identity'], OPENAI_IDENTITY)
                self.assertEqual(metadata['deployment_original_embedding_identity'], OPENCLAW_IDENTITY)
                self.assertNotIn('qdrant_target', metadata)
                self.assertEqual(db.execute('SELECT path,source_url,aliases FROM documents').fetchone(),
                                 ('/app/space-data/archive/source-company.tar.gz:report.txt', 'https://example.org/report', '[]'))
                self.assertEqual(db.execute('SELECT vector FROM embeddings').fetchone()[0], b'\x00\x00\x80\x3f' * 3072)
            with sqlite3.connect(db_path) as db:
                self.assertEqual(dict(db.execute('SELECT * FROM metadata'))['embedding_identity'], OPENCLAW_IDENTITY)
                self.assertEqual(db.execute('SELECT path FROM documents').fetchone()[0], str(directory / 'source-company' / 'report.txt'))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                prepare_bundle(db_path, output)

    def test_rejects_unverified_or_incomplete_vectors_and_missing_original(self):
        for change, message in [
                ("UPDATE metadata SET value='different-model' WHERE key='embedding_identity'", 'verified'),
                ("UPDATE embeddings SET dimension=1536", 'dimension'),
                ('DELETE FROM embeddings', 'Every existing chunk'),
                ("UPDATE documents SET path=root || '/missing.txt'", 'missing or excluded')]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                db_path = self.make_index(directory)
                with sqlite3.connect(db_path) as db:
                    db.execute(change)
                output = directory / 'bundle'
                with self.assertRaisesRegex(ValueError, message):
                    prepare_bundle(db_path, output, progress=lambda *args, **kwargs: None)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
