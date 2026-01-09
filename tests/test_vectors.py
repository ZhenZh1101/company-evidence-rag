import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from qdrant_client import QdrantClient, models

from rag.config import Settings
from rag.ingest import Segment, Source
from rag.pipeline import RAG
from rag.store import Store, digest


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.client = QdrantClient(':memory:')
        self.addCleanup(self.client.close)
        self.store = Store(self.root / 'index.sqlite3', vector_client=self.client)
        self.addCleanup(self.store.close)
        self.gateway = SimpleNamespace(settings=Settings(api_key='offline-test-token'),
                                       embed=Mock(side_effect=lambda texts: np.array([[1, 0]] * len(texts), dtype=np.float32)))

    def add(self, key, text, *, store=None, company='ALFA', category='results', publication_date='2026-07-21'):
        store = store or self.store
        source = Source(key, company, key, category, publication_date, None, None,
                        'local', None, self.root / (key + '.txt'))
        store.put(source, self.root, key, digest(text), [Segment(text, 'paragraph 1')])
        return store.db.execute('SELECT id FROM chunks WHERE document_id=?', (key,)).fetchone()[0]

    def test_legacy_cache_migrates_without_embedding_and_survives_reopening(self):
        path = self.root / 'legacy.sqlite3'
        text = 'Legacy revenue evidence.'
        identity = self.gateway.settings.base_url + '|' + self.gateway.settings.embedding_model + '|' + self.gateway.settings.openclaw_embedding_model
        with sqlite3.connect(path) as legacy:
            legacy.executescript('''
                CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE documents(id TEXT PRIMARY KEY,root TEXT NOT NULL,fingerprint TEXT NOT NULL,content_hash TEXT NOT NULL,
                  company TEXT NOT NULL,title TEXT NOT NULL,category TEXT NOT NULL,publication_date TEXT,
                  publication_period TEXT,date_basis TEXT,scope TEXT,source_url TEXT,path TEXT,aliases TEXT);
                CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                  ordinal INTEGER NOT NULL,text TEXT NOT NULL,locator TEXT NOT NULL,text_hash TEXT NOT NULL,UNIQUE(document_id,ordinal));
                CREATE TABLE embeddings(text_hash TEXT PRIMARY KEY,dimension INTEGER NOT NULL,vector BLOB NOT NULL);
            ''')
            legacy.executemany('INSERT INTO metadata VALUES (?,?)', [('embedding_identity', identity), ('embedding_dimension', '2')])
            legacy.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                           ('legacy', str(self.root), 'fp', 'hash', 'ALFA', 'Annual results', 'results', '2026-07-21',
                            None, None, 'local', None, str(self.root / 'legacy.txt'), '[]'))
            legacy.execute('INSERT INTO chunks VALUES (?,?,?,?,?,?)', (7, 'legacy', 0, text, 'page 2', digest(text)))
            legacy.execute('INSERT INTO embeddings VALUES (?,?,?)', (digest(text), 2, np.array([1, 0], dtype='<f4').tobytes()))
        with Store(path, vector_client=self.client) as migrated:
            with self.assertRaises((ValueError, RuntimeError)):
                migrated.dense(np.array([1, 0]))
            migrated.sync_vectors(batch_size=1)
            self.assertEqual(migrated.dense(np.array([1, 0])), [7])
            self.assertEqual(migrated.embed_pending(self.gateway), 0)
            self.assertEqual(migrated.meta('embedding_identity'), identity)
            self.assertEqual(migrated.source(7)['text'], text)
        with Store(path, vector_client=self.client) as reopened:
            self.assertEqual(reopened.dense(np.array([1, 0])), [7])
        self.gateway.embed.assert_not_called()

    def test_update_reused_chunk_id_and_sql_prune_remove_stale_vectors(self):
        old_id = self.add('report', 'Obsolete revenue.')
        self.store.embed_pending(self.gateway)
        replacement_id = self.add('report', 'Replacement revenue.', company='BETA')
        self.assertEqual(replacement_id, old_id)
        with self.assertRaises((ValueError, RuntimeError)):
            self.store.dense(np.array([1, 0]))
        self.store.embed_pending(self.gateway)
        self.assertEqual(self.store.dense(np.array([1, 0]), companies=['ALFA']), [])
        self.assertEqual(self.store.dense(np.array([1, 0]), companies=['BETA']), [replacement_id])
        self.assertEqual(self.store.source(replacement_id)['text'], 'Replacement revenue.')
        with self.store.db:
            self.store.db.execute('DELETE FROM documents WHERE id=?', ('report',))
        self.store.sync_vectors()
        self.assertEqual(self.store.dense(np.array([1, 0])), [])
        collection = self.client.get_collections().collections[0].name
        self.assertEqual(self.client.count(collection_name=collection, exact=True).count, 0)
        self.client.delete_collection(collection_name=collection)
        self.assertEqual(self.store.dense(np.array([1, 0])), [])

    def test_dense_filters_precede_limit_and_exclude_unknown_dates(self):
        wanted = self.add('wanted', 'Selected disclosure.')
        self.add('category', 'Excluded category.', category='presentation')
        self.add('company', 'Excluded company.', company='BETA')
        self.add('old', 'Excluded old disclosure.', publication_date='2025-07-21')
        undated = self.add('unknown', 'Undated disclosure.', publication_date=None)
        self.store.embed_pending(self.gateway)
        self.assertEqual(self.store.dense(np.array([1, 0]), limit=1, companies=['ALFA'], categories=['results'],
                                         date_from='2026-07-21', date_to='2026-07-21'), [wanted])
        self.assertIn(undated, self.store.dense(np.array([1, 0]), companies=['ALFA'], categories=['results']))

    def test_failed_upsert_resumes_from_cached_embeddings(self):
        chunk_id = self.add('report', 'Recoverable evidence.')
        with patch.object(self.client, 'upsert', side_effect=RuntimeError('temporary vector outage')):
            with self.assertRaises(RuntimeError):
                self.store.embed_pending(self.gateway)
        with self.assertRaises((ValueError, RuntimeError)):
            self.store.dense(np.array([1, 0]))
        self.assertEqual(self.store.embed_pending(self.gateway), 0)
        self.gateway.embed.assert_called_once()
        self.assertEqual(self.store.dense(np.array([1, 0])), [chunk_id])

    def test_sync_recovers_missing_points_when_other_updates_are_pending(self):
        first = self.add('first', 'First retained evidence.')
        second = self.add('second', 'Second retained evidence.')
        self.store.embed_pending(self.gateway)
        collection = self.client.get_collections().collections[0].name
        self.client.delete(collection_name=collection, points_selector=models.PointIdsList(points=[first]), wait=True)
        self.assertEqual(self.add('second', 'Second retained evidence.', company='BETA'), second)
        self.store.sync_vectors()
        self.assertEqual(self.store.dense(np.array([1, 0]), companies=['ALFA']), [first])
        self.assertEqual(self.store.dense(np.array([1, 0]), companies=['BETA']), [second])
        self.gateway.embed.assert_called_once()

    def test_mismatched_source_hash_refuses_stale_vector_and_queues_repair(self):
        chunk_id = self.add('report', 'Current source evidence.')
        self.store.embed_pending(self.gateway)
        collection = self.client.get_collections().collections[0].name
        self.client.set_payload(collection_name=collection, payload={'text_hash': 'stale-source-hash'}, points=[chunk_id], wait=True)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.store.dense(np.array([1, 0]))
        self.assertEqual(self.store.stats()['pending_vector_changes'], 1)
        self.store.sync_vectors()
        self.assertEqual(self.store.dense(np.array([1, 0])), [chunk_id])
        self.gateway.embed.assert_called_once()

    def test_independent_sqlite_databases_use_independent_collections(self):
        first = self.add('first', 'First database evidence.')
        self.store.embed_pending(self.gateway)
        with Store(self.root / 'second.sqlite3', vector_client=self.client) as other:
            second = self.add('second', 'Second database evidence.', store=other, company='BETA')
            other.embed_pending(self.gateway)
            self.assertEqual(first, second)
            self.assertEqual(other.dense(np.array([1, 0]), companies=['ALFA']), [])
            self.assertEqual(self.store.dense(np.array([1, 0]), companies=['BETA']), [])
            self.assertEqual(other.source(other.dense(np.array([1, 0]))[0])['text'], 'Second database evidence.')
        self.assertEqual(len(self.client.get_collections().collections), 2)
        self.assertEqual(self.store.dense(np.array([1, 0])), [first])

    def test_missing_or_partial_collection_refuses_dense_and_recovers_from_cache(self):
        chunk_id = self.add('report', 'Recoverable collection evidence.')
        self.store.embed_pending(self.gateway)
        self.gateway.embed.assert_called_once()
        collection = self.client.get_collections().collections[0].name
        for remove in (
            lambda: self.client.delete(collection_name=collection, points_selector=models.PointIdsList(points=[chunk_id]), wait=True),
            lambda: self.client.delete_collection(collection_name=collection),
        ):
            with self.subTest(remove=remove):
                remove()
                with self.assertRaises((ValueError, RuntimeError)):
                    RAG(self.store, self.gateway).search('Revenue', mode='dense', rewrite=False)
                self.assertTrue(RAG(self.store, self.gateway).search('Recoverable', mode='lexical', rewrite=False)['sources'])
                self.store.sync_vectors()
                self.assertEqual(self.store.dense(np.array([1, 0])), [chunk_id])

    def test_sync_cli_restores_collection_without_constructing_gateway(self):
        from rag.cli import main
        chunk_id = self.add('report', 'CLI migration evidence.')
        self.store.embed_pending(self.gateway)
        collection = self.client.get_collections().collections[0].name
        self.client.delete_collection(collection_name=collection)
        settings = Settings(db_path=self.root / 'index.sqlite3')
        stdout = io.StringIO()
        with patch('sys.argv', ['company-rag', 'sync-vectors', '--batch-size', '1']), \
                patch('rag.cli.Settings.from_env', return_value=settings), \
                patch('rag.cli.Store', side_effect=lambda path, url: Store(path, url, vector_client=self.client)), \
                patch('rag.cli.Gateway', side_effect=AssertionError('Migration must not instantiate a model gateway')), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            main()
        self.assertEqual(json.loads(stdout.getvalue())['indexed_chunks'], 1)
        self.assertEqual(self.store.dense(np.array([1, 0])), [chunk_id])

    def test_qdrant_url_configuration_and_invalid_endpoints(self):
        with patch.dict('os.environ', {'RAG_QDRANT_URL': 'http://127.0.0.1:7333/'}, clear=True):
            self.assertEqual(Settings.from_env().qdrant_url, 'http://127.0.0.1:7333')
        for endpoint in ('', 'localhost:6333', 'file:///tmp/qdrant', 'http://user:secret@localhost:6333',
                         'http://localhost:6333?key=secret', 'http://localhost:6333#fragment'):
            with self.subTest(endpoint=endpoint), patch.dict('os.environ', {'RAG_QDRANT_URL': endpoint}, clear=True):
                with self.assertRaisesRegex(ValueError, 'RAG_QDRANT_URL'):
                    Settings.from_env()

    def test_doctor_checks_models_when_qdrant_fails_and_exits_nonzero(self):
        from rag.cli import main
        gateway = Mock()
        gateway.embed.return_value = np.array([[1, 0], [1, 0], [0, 1]])
        gateway.chat.side_effect = RuntimeError('chat unavailable')
        stdout = io.StringIO()
        with patch('sys.argv', ['company-rag', 'doctor']), \
                patch('rag.cli.Settings.from_env', return_value=Settings()), \
                patch('qdrant_client.QdrantClient', side_effect=RuntimeError('vector service unavailable')), \
                patch('rag.cli.Gateway', return_value=gateway), redirect_stdout(stdout), \
                self.assertRaises(SystemExit) as outcome:
            main()
        self.assertEqual(outcome.exception.code, 1)
        report = json.loads(stdout.getvalue())
        self.assertEqual(set(report['errors']), {'qdrant', 'chat'})
        self.assertEqual(report['embedding_dimension'], 2)
        gateway.embed.assert_called_once()
        gateway.chat.assert_called_once()


if __name__ == '__main__':
    unittest.main()
