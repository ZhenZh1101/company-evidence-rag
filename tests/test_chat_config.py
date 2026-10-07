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
from rag.store import Store


class ChatConfigurationTests(unittest.TestCase):
    def test_environment_defaults_and_separate_chat_settings(self):
        with patch.dict('os.environ', {'RAG_API_KEY': 'embedding-key'}, clear=True):
            defaults = Settings.from_env()
        self.assertEqual(defaults.chat_base_url, '')
        self.assertIsNone(defaults.chat_api_key)
        self.assertEqual(defaults.chat_temperature, 0)
        self.assertIsNone(defaults.chat_thinking)
        self.assertIsNone(defaults.chat_response_format)
        env = {
            'RAG_API_KEY': 'embedding-key',
            'RAG_CHAT_BASE_URL': 'https://api.z.ai/api/paas/v4/',
            'RAG_CHAT_API_KEY': 'chat-key',
            'RAG_CHAT_MODEL': 'glm-4.7',
            'RAG_CHAT_TEMPERATURE': '0.1',
            'RAG_CHAT_THINKING': 'disabled',
            'RAG_CHAT_RESPONSE_FORMAT': 'json_object',
        }
        with patch.dict('os.environ', env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.base_url, defaults.base_url)
        self.assertEqual(settings.api_key, 'embedding-key')
        self.assertEqual(settings.embedding_model, defaults.embedding_model)
        self.assertEqual(settings.chat_base_url, 'https://api.z.ai/api/paas/v4')
        self.assertEqual(settings.chat_api_key, 'chat-key')
        self.assertEqual(settings.chat_model, 'glm-4.7')
        self.assertEqual(settings.chat_temperature, 0.1)
        self.assertEqual(settings.chat_thinking, 'disabled')
        self.assertEqual(settings.chat_response_format, 'json_object')
        with patch.dict('os.environ', dict(env, RAG_CHAT_BASE_URL='', RAG_CHAT_API_KEY=''), clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.chat_base_url, '')
        self.assertEqual(settings.chat_api_key, '')

    def test_invalid_chat_environment_rejected(self):
        invalid = {
            'RAG_CHAT_BASE_URL': ['file:///tmp/chat', 'https:///v4', 'https://user:secret@example.test/v4',
                                  'https://:secret@example.test/v4', 'https://example.test/v4?key=secret',
                                  'https://example.test/v4#fragment'],
            'RAG_CHAT_TEMPERATURE': ['-0.1', '2.1', 'nan', 'inf', '-inf', 'bad'],
            'RAG_CHAT_THINKING': ['auto', 'true'],
            'RAG_CHAT_RESPONSE_FORMAT': ['json', 'json_schema', 'JSON_OBJECT', 'none'],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), patch.dict(
                        'os.environ', {'RAG_API_KEY': 'embedding-key', name: value}, clear=True):
                    with self.assertRaises(ValueError):
                        Settings.from_env()
        for temperature, thinking in [('0', 'enabled'), ('2', 'disabled')]:
            with self.subTest(temperature=temperature), patch.dict('os.environ', {
                    'RAG_API_KEY': 'embedding-key', 'RAG_CHAT_TEMPERATURE': temperature,
                    'RAG_CHAT_THINKING': thinking}, clear=True):
                settings = Settings.from_env()
                self.assertEqual(settings.chat_temperature, float(temperature))
                self.assertEqual(settings.chat_thinking, thinking)

    def test_chat_and_embeddings_use_independent_requests(self):
        settings = Settings(api_key='embedding-key', chat_base_url='https://api.z.ai/api/paas/v4',
                            chat_api_key='chat-key', chat_model='glm-4.7', chat_temperature=0.1,
                            chat_thinking='disabled', chat_response_format='json_object')
        gateway = Gateway(settings)
        messages = [{'role': 'user', 'content': 'Return a short answer.'}]
        replies = [
            {'choices': [{'message': {'content': 'Answer', 'reasoning_content': 'Private reasoning'},
                          'finish_reason': 'stop'}]},
            {'data': [{'index': 0, 'embedding': [3, 4]}]},
        ]
        with patch('rag.client.request.build_opener') as build:
            build.return_value.open.side_effect = [io.BytesIO(json.dumps(reply).encode()) for reply in replies]
            self.assertEqual(gateway.chat(messages, max_tokens=500), 'Answer')
            np.testing.assert_allclose(gateway.embed(['Source text']), [[0.6, 0.8]])
        calls = build.return_value.open.call_args_list
        chat, embedding = [call.args[0] for call in calls]
        self.assertEqual(chat.full_url, 'https://api.z.ai/api/paas/v4/chat/completions')
        self.assertEqual(chat.get_header('Authorization'), 'Bearer chat-key')
        self.assertEqual(json.loads(chat.data), {
            'model': 'glm-4.7', 'messages': messages, 'temperature': 0.1,
            'max_tokens': 500, 'thinking': {'type': 'disabled'},
            'response_format': {'type': 'json_object'},
        })
        self.assertEqual(embedding.full_url, settings.base_url + '/embeddings')
        self.assertEqual(embedding.get_header('Authorization'), 'Bearer embedding-key')
        self.assertEqual(json.loads(embedding.data), {'model': settings.embedding_model, 'input': ['Source text']})
        self.assertTrue(all(call.kwargs['timeout'] == settings.timeout for call in calls))

    def test_default_chat_and_same_endpoint_reuse_existing_key(self):
        base = 'http://127.0.0.1:18789/v1'
        for chat_base in ('', base, base + '/'):
            with self.subTest(chat_base=chat_base), patch('rag.client.request.build_opener') as build:
                build.return_value.open.return_value = io.BytesIO(b'{"choices":[{"message":{"content":"OK"}}]}')
                gateway = Gateway(Settings(base_url=base + '/', api_key='existing-key', chat_base_url=chat_base))
                self.assertEqual(gateway.chat([{'role': 'user', 'content': 'Hello'}]), 'OK')
                req = build.return_value.open.call_args.args[0]
                self.assertEqual(req.full_url, base + '/chat/completions')
                self.assertEqual(req.get_header('Authorization'), 'Bearer existing-key')
                payload = json.loads(req.data)
                self.assertEqual(payload['temperature'], 0)
                self.assertEqual(payload['max_tokens'], 2400)
                self.assertNotIn('thinking', payload)
                self.assertNotIn('response_format', payload)

    def test_text_response_format_and_empty_override(self):
        for value in ('text', ''):
            with self.subTest(value=value), patch.dict('os.environ', {
                    'RAG_API_KEY': 'chat-key', 'RAG_CHAT_RESPONSE_FORMAT': value}, clear=True):
                settings = Settings.from_env()
            self.assertEqual(settings.chat_response_format, value or None)
            with patch('rag.client.request.build_opener') as build:
                build.return_value.open.return_value = io.BytesIO(b'{"choices":[{"message":{"content":"OK"}}]}')
                self.assertEqual(Gateway(settings).chat([{'role': 'user', 'content': 'Reply with only OK.'}]), 'OK')
                payload = json.loads(build.return_value.open.call_args.args[0].data)
            if value:
                self.assertEqual(payload['response_format'], {'type': 'text'})
            else:
                self.assertNotIn('response_format', payload)

    def test_doctor_prompt_matches_chat_response_format(self):
        from rag.cli import doctor
        for response_format in (None, 'text', 'json_object'):
            with self.subTest(response_format=response_format), \
                    patch('qdrant_client.QdrantClient') as client, \
                    patch('rag.cli.Gateway') as gateway_class, patch('rag.cli.output') as output:
                client.return_value.get_collections.return_value.collections = []
                gateway = gateway_class.return_value
                gateway.embed.return_value = np.array([[1, 0], [1, 0], [0, 1]])
                json_mode = response_format == 'json_object'
                gateway.chat.return_value = '{"status":"OK"}' if json_mode else 'OK'
                self.assertFalse(doctor(Settings(chat_response_format=response_format)))
                prompt = 'Return JSON: {"status":"OK"}.' if json_mode else 'Reply with only OK.'
                gateway.chat.assert_called_once_with([{'role': 'user', 'content': prompt}], max_tokens=16)
                self.assertEqual(output.call_args.args[0]['chat'], gateway.chat.return_value)

    def test_missing_or_explicitly_empty_chat_key_never_sends_request(self):
        base = 'http://127.0.0.1:18789/v1'
        options = [
            {'chat_base_url': 'https://api.z.ai/api/paas/v4'},
            {'chat_base_url': base + '/other'},
            {'chat_base_url': 'http://127.0.0.1:18790/v1'},
            {'chat_api_key': ''},
            {'chat_base_url': 'https://api.z.ai/api/paas/v4', 'chat_api_key': ''},
        ]
        for option in options:
            with self.subTest(option=option), patch('rag.client.request.build_opener') as build:
                with self.assertRaises(RuntimeError) as caught:
                    Gateway(Settings(base_url=base, api_key='local-secret', **option)).chat([])
                build.return_value.open.assert_not_called()
                self.assertNotIn('local-secret', str(caught.exception))

    def test_local_config_token_does_not_escape_through_chat_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.openclaw').mkdir()
            (root / '.openclaw/openclaw.json').write_text(json.dumps({'gateway': {'auth': {'token': 'local-secret'}}}))
            with patch('rag.config.Path.home', return_value=root), patch.dict('os.environ', {
                    'RAG_CHAT_BASE_URL': 'https://api.z.ai/api/paas/v4'}, clear=True):
                settings = Settings.from_env()
            self.assertEqual(settings.api_key, 'local-secret')
            with patch('rag.client.request.build_opener') as build, self.assertRaises(RuntimeError):
                Gateway(settings).chat([])
            build.return_value.open.assert_not_called()

    def test_changing_chat_provider_preserves_populated_embedding_identity(self):
        settings = Settings(api_key='embedding-key')
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / 'index.sqlite3') as store:
            store.check_embedding_identity(Gateway(settings))
            with store.db:
                store.db.execute('INSERT INTO embeddings VALUES (?,?,?)',
                                 ('text-hash', 2, np.array([1, 0], dtype='<f4').tobytes()))
            original = store.meta('embedding_identity')
            chat_settings = replace(settings, chat_base_url='https://api.z.ai/api/paas/v4',
                                    chat_api_key='chat-key', chat_model='glm-4.7',
                                    chat_temperature=0.1, chat_thinking='disabled')
            store.check_embedding_identity(Gateway(chat_settings))
            self.assertEqual(store.meta('embedding_identity'), original)
            self.assertEqual(store.db.execute('SELECT count(*) FROM embeddings').fetchone()[0], 1)
            with self.assertRaises(ValueError):
                store.check_embedding_identity(Gateway(replace(chat_settings, embedding_model='different-model')))


if __name__ == '__main__':
    unittest.main()
