import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rag.client import Gateway
from rag.config import Settings
from rag.ingest import Segment, Source
from rag.pipeline import RAG
from rag.store import Store


class EmbeddingConfigurationTests(unittest.TestCase):
    def test_default_openclaw_request_matches_curl_and_does_not_change_chat(self):
        with patch.dict('os.environ', {'RAG_API_KEY': 'test-token'}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.embedding_openclaw_model, 'openai/text-embedding-3-large')
        gateway = Gateway(settings)
        texts = ['文本1', '文本2']
        replies = [
            {'data': [{'index': 1, 'embedding': [0, 5]}, {'index': 0, 'embedding': [3, 4]}]},
            {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]},
        ]
        with patch('rag.client.request.build_opener') as build:
            build.return_value.open.side_effect = [io.BytesIO(json.dumps(reply).encode()) for reply in replies]
            np.testing.assert_allclose(gateway.embed(texts), [[0.6, 0.8], [0, 1]])
            self.assertEqual(gateway.chat([{'role': 'user', 'content': 'Reply OK.'}]), 'OK')
        embedding, chat = [call.args[0] for call in build.return_value.open.call_args_list]
        self.assertEqual(embedding.full_url, 'http://127.0.0.1:18789/v1/embeddings')
        self.assertEqual(embedding.get_header('Authorization'), 'Bearer test-token')
        self.assertEqual(embedding.get_header('Content-type'), 'application/json')
        self.assertEqual(embedding.get_header('X-openclaw-model'), 'openai/text-embedding-3-large')
        self.assertEqual(json.loads(embedding.data), {'model': 'openclaw/llm-gpt55', 'input': texts})
        self.assertIsNone(chat.get_header('X-openclaw-model'))
        self.assertEqual(json.loads(chat.data)['model'], 'openclaw/llm-gpt55')
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / 'index.sqlite3') as store:
            store.check_embedding_identity(gateway)
            self.assertEqual(store.meta('embedding_identity'),
                             'http://127.0.0.1:18789/v1|openclaw/llm-gpt55|openai/text-embedding-3-large')

    def test_environment_override_and_explicit_empty_optout(self):
        for override in ('openai/text-embedding-3-small', ''):
            with self.subTest(override=override), patch.dict('os.environ', {
                    'RAG_API_KEY': 'test-token', 'RAG_EMBEDDING_OPENCLAW_MODEL': override}, clear=True):
                settings = Settings.from_env()
            self.assertEqual(settings.embedding_openclaw_model, override)
            self.assertEqual(settings.openclaw_embedding_model, override)
            with patch('rag.client.request.build_opener') as build:
                build.return_value.open.return_value = io.BytesIO(
                    b'{"data":[{"index":0,"embedding":[1,0]}]}')
                Gateway(settings).embed(['Source text'])
            req = build.return_value.open.call_args.args[0]
            self.assertEqual(req.get_header('X-openclaw-model'), override or None)
            self.assertEqual(json.loads(req.data)['model'], 'openclaw/llm-gpt55')

    def test_direct_openai_ignores_openclaw_header_and_preserves_index_identity(self):
        with patch.dict('os.environ', {
                'RAG_BASE_URL': 'https://api.openai.com/v1', 'RAG_API_KEY': 'direct-test-key',
                'RAG_EMBEDDING_MODEL': 'text-embedding-3-small'}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.openclaw_embedding_model, '')
        gateway = Gateway(settings)
        with patch('rag.client.request.build_opener') as build:
            build.return_value.open.return_value = io.BytesIO(
                b'{"data":[{"index":0,"embedding":[1,0]}]}')
            gateway.embed(['Source text'])
        req = build.return_value.open.call_args.args[0]
        self.assertEqual(req.full_url, 'https://api.openai.com/v1/embeddings')
        self.assertEqual(req.get_header('Authorization'), 'Bearer direct-test-key')
        self.assertIsNone(req.get_header('X-openclaw-model'))
        self.assertEqual(json.loads(req.data), {'model': 'text-embedding-3-small', 'input': ['Source text']})
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / 'index.sqlite3') as store:
            store.check_embedding_identity(gateway)
            with store.db:
                store.db.execute('INSERT INTO embeddings VALUES (?,?,?)',
                                 ('text-hash', 2, np.array([1, 0], dtype='<f4').tobytes()))
            store.check_embedding_identity(Gateway(replace(settings, embedding_openclaw_model='other/model')))
            self.assertEqual(store.meta('embedding_identity'),
                             'https://api.openai.com/v1|text-embedding-3-small')

    def test_legacy_or_small_index_rejected_for_large_even_with_equal_dimensions(self):
        vectors = np.array([[1, 0]], dtype=np.float32)
        for previous_model in ('', 'openai/text-embedding-3-small'):
            with self.subTest(previous_model=previous_model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                settings = Settings(api_key='test-token', embedding_openclaw_model=previous_model)
                old_gateway = Gateway(settings)
                large_gateway = Gateway(replace(settings, embedding_openclaw_model='openai/text-embedding-3-large'))
                with Store(root / 'index.sqlite3') as store:
                    source = Source('report', 'NOC', 'Annual results', 'results', '2026-07-21', '2026',
                                    'disclosure date', 'local', 'https://example.test/report', root / 'report.txt')
                    store.put(source, root, 'fingerprint', 'content-hash', [Segment('Revenue increased.', 'paragraph 1')])
                    with patch.object(old_gateway, 'embed', return_value=vectors):
                        self.assertEqual(store.embed_pending(old_gateway), 1)
                    original_identity = 'http://127.0.0.1:18789/v1|openclaw/llm-gpt55'
                    if previous_model:
                        original_identity += '|' + previous_model
                    self.assertEqual(store.meta('embedding_identity'), original_identity)
                    with patch.object(large_gateway, 'embed', return_value=vectors) as embed:
                        with self.assertRaisesRegex(ValueError, 'differs'):
                            store.embed_pending(large_gateway)
                        for mode in ('dense', 'hybrid'):
                            with self.assertRaisesRegex(ValueError, 'differs'):
                                RAG(store, large_gateway).search('Revenue', mode=mode, rewrite=False)
                        embed.assert_not_called()
                    self.assertEqual(store.meta('embedding_identity'), original_identity)
                    self.assertEqual(store.meta('embedding_dimension'), '2')
                    self.assertEqual(store.stats()['indexed_chunks'], 1)


if __name__ == '__main__':
    unittest.main()
