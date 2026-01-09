import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from qdrant_client import QdrantClient

from rag.client import Gateway
from rag.config import Settings
from rag.ingest import Segment, Source
from rag.pipeline import RAG
from rag.store import Store


OPENAI_ENV = {
    'RAG_BASE_URL': 'https://api.openai.com/v1/',
    'RAG_API_KEY': 'openai-test-key',
    'RAG_EMBEDDING_MODEL': 'text-embedding-3-small',
    'RAG_CHAT_MODEL': 'gpt-4.1-mini',
    'RAG_CHAT_TOKEN_LIMIT_FIELD': 'max_completion_tokens',
    'RAG_CHAT_TEMPERATURE': '0',
}


class OpenAITests(unittest.TestCase):
    def test_openai_document_index_and_hybrid_answer_use_shared_configuration(self):
        with patch.dict('os.environ', OPENAI_ENV, clear=True):
            settings = Settings.from_env()
        gateway = Gateway(settings)
        quote = 'Revenue was 100 million in fiscal 2026.'
        question = 'What was NOC revenue?'
        rewrite = 'NOC fiscal 2026 revenue'
        answer = {'answer': 'NOC revenue was 100 million in fiscal 2026. [S1]',
                  'insufficient_evidence': False, 'citations': [{'label': 'S1', 'quote': quote}]}
        replies = [
            {'data': [{'index': 0, 'embedding': [3, 4]}]},
            {'choices': [{'message': {'content': json.dumps({'queries': [rewrite]})}, 'finish_reason': 'stop'}]},
            {'data': [{'index': 1, 'embedding': [3, 4]}, {'index': 0, 'embedding': [3, 4]}]},
            {'choices': [{'message': {'content': json.dumps(answer)}, 'finish_reason': 'stop'}]},
        ]
        with tempfile.TemporaryDirectory() as directory, patch('rag.client.request.build_opener') as build:
            root = Path(directory)
            source = Source('report', 'NOC', 'Annual results', 'results', '2026-07-21', '2026',
                            'disclosure date', 'local', 'https://example.test/report', root / 'report.txt')
            build.return_value.open.side_effect = [io.BytesIO(json.dumps(reply).encode()) for reply in replies]
            vectors = QdrantClient(':memory:')
            self.addCleanup(vectors.close)
            with Store(root / 'openai.sqlite3', vector_client=vectors) as store:
                store.put(source, root, 'fingerprint', 'content-hash', [Segment(quote, 'paragraph 1')])
                self.assertEqual(store.embed_pending(gateway), 1)
                identity = store.meta('embedding_identity')
                self.assertEqual(identity, 'https://api.openai.com/v1|text-embedding-3-small')
                result = RAG(store, gateway).ask(question, companies=['NOC'], mode='hybrid')
                self.assertEqual(store.meta('embedding_identity'), identity)
                self.assertEqual(store.embed_pending(gateway), 0)
                changed = Gateway(replace(settings, embedding_model='text-embedding-3-large'))
                with self.assertRaisesRegex(ValueError, 'differs'):
                    store.embed_pending(changed)
                with self.assertRaisesRegex(ValueError, 'differs'):
                    RAG(store, changed).search(question, mode='dense', rewrite=False)
            requests = [call.args[0] for call in build.return_value.open.call_args_list]

        self.assertEqual(len(requests), 4)
        self.assertEqual([req.full_url for req in requests], [
            'https://api.openai.com/v1/' + route
            for route in ('embeddings', 'chat/completions', 'embeddings', 'chat/completions')])
        self.assertTrue(all(req.get_header('Authorization') == 'Bearer openai-test-key' for req in requests))
        payloads = [json.loads(req.data) for req in requests]
        self.assertEqual(payloads[0], {'model': 'text-embedding-3-small', 'input': [quote]})
        self.assertEqual(payloads[2], {'model': 'text-embedding-3-small', 'input': [question, rewrite]})
        for payload, limit in ((payloads[1], 500), (payloads[3], 4000)):
            self.assertEqual(payload['model'], 'gpt-4.1-mini')
            self.assertEqual(payload['temperature'], 0)
            self.assertEqual(payload['max_completion_tokens'], limit)
            self.assertNotIn('max_tokens', payload)
            self.assertNotIn('thinking', payload)
            self.assertNotIn('reasoning_effort', payload)
        self.assertFalse(result['insufficient_evidence'])
        self.assertEqual(result['warning_codes'], [])
        self.assertEqual(result['answer'], answer['answer'])
        self.assertEqual(result['citations'][0]['quote'], quote)
        self.assertEqual(result['citations'][0]['source']['text'], quote)
        self.assertEqual(result['citations'][0]['source']['company'], 'NOC')
        self.assertEqual(result['citations'][0]['source']['locator'], 'paragraph 1')

    def test_openai_embeddings_and_zai_chat_keep_credentials_and_parameters_separate(self):
        env = dict(OPENAI_ENV, RAG_CHAT_BASE_URL='https://api.z.ai/api/paas/v4',
                   RAG_CHAT_API_KEY='zai-test-key', RAG_CHAT_MODEL='glm-4.7',
                   RAG_CHAT_TOKEN_LIMIT_FIELD='max_tokens', RAG_CHAT_TEMPERATURE='0.1',
                   RAG_CHAT_THINKING='disabled')
        with patch.dict('os.environ', env, clear=True):
            gateway = Gateway(Settings.from_env())
        replies = [{'data': [{'index': 0, 'embedding': [1, 0]}]},
                   {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}]
        with patch('rag.client.request.build_opener') as build:
            build.return_value.open.side_effect = [io.BytesIO(json.dumps(reply).encode()) for reply in replies]
            gateway.embed(['Revenue increased.'])
            self.assertEqual(gateway.chat([{'role': 'user', 'content': 'Reply OK.'}], max_tokens=16), 'OK')
        embedding, chat = [call.args[0] for call in build.return_value.open.call_args_list]
        self.assertEqual(embedding.full_url, 'https://api.openai.com/v1/embeddings')
        self.assertEqual(embedding.get_header('Authorization'), 'Bearer openai-test-key')
        self.assertEqual(chat.full_url, 'https://api.z.ai/api/paas/v4/chat/completions')
        self.assertEqual(chat.get_header('Authorization'), 'Bearer zai-test-key')
        payload = json.loads(chat.data)
        self.assertEqual(payload['model'], 'glm-4.7')
        self.assertEqual(payload['temperature'], 0.1)
        self.assertEqual(payload['max_tokens'], 16)
        self.assertEqual(payload['thinking'], {'type': 'disabled'})
        self.assertNotIn('max_completion_tokens', payload)
        self.assertNotIn('thinking', json.loads(embedding.data))

    def test_optional_temperature_and_reasoning_effort_are_serialized_explicitly(self):
        for effort in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
            with self.subTest(effort=effort), patch.dict('os.environ', dict(
                    OPENAI_ENV, RAG_CHAT_TEMPERATURE='none', RAG_CHAT_REASONING_EFFORT=effort), clear=True):
                settings = Settings.from_env()
                self.assertIsNone(settings.chat_temperature)
                self.assertEqual(settings.chat_reasoning_effort, effort)
                with patch('rag.client.request.build_opener') as build:
                    build.return_value.open.return_value = io.BytesIO(
                        b'{"choices":[{"message":{"content":"OK"},"finish_reason":"stop"}]}')
                    self.assertEqual(Gateway(settings).chat([{'role': 'user', 'content': 'Reply OK.'}], 64), 'OK')
                payload = json.loads(build.return_value.open.call_args.args[0].data)
                self.assertNotIn('temperature', payload)
                self.assertEqual(payload['reasoning_effort'], effort)
                self.assertEqual(payload['max_completion_tokens'], 64)

    def test_invalid_token_limit_and_reasoning_configuration_rejected(self):
        for name, values in {
            'RAG_CHAT_TOKEN_LIMIT_FIELD': ('', 'max_output_tokens', 'temperature'),
            'RAG_CHAT_REASONING_EFFORT': ('disabled', 'automatic', '0'),
        }.items():
            for value in values:
                with self.subTest(name=name, value=value), patch.dict('os.environ', {
                        **OPENAI_ENV, name: value}, clear=True), self.assertRaises(ValueError):
                    Settings.from_env()


if __name__ == '__main__':
    unittest.main()
