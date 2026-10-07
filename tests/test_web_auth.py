from dataclasses import replace
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from rag.auth import COOKIE_NAME, SESSION_SECONDS, hash_password
from rag.config import Settings
from rag.web import create_app


class WebAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password = 'test-login-password'
        cls.encoded = hash_password(cls.password)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.username = 'integration-friend'
        self.settings = Settings(db_path=Path(directory.name) / 'test.sqlite3',
                                 api_key='TEST_API_SECRET', chat_api_key='TEST_CHAT_SECRET',
                                 auth_users={self.username: self.encoded}, auth_required=True)
        gateway_patch = patch('rag.web.Gateway', return_value=Mock())
        self.gateway = gateway_patch.start().return_value
        self.addCleanup(gateway_patch.stop)
        store_patch = patch('rag.web.Store')
        self.store = store_patch.start()
        self.addCleanup(store_patch.stop)
        self.store.return_value.__enter__.return_value.stats.return_value = {'documents': 7}
        rag_patch = patch('rag.web.RAG')
        self.rag = rag_patch.start()
        self.addCleanup(rag_patch.stop)
        self.client = TestClient(create_app(self.settings), base_url='https://127.0.0.1')
        self.addCleanup(self.client.close)

    def login(self):
        response = self.client.post('/api/login', json={'username': self.username, 'password': self.password})
        self.assertEqual(response.status_code, 200)
        return response

    def assert_no_provider_calls(self):
        self.gateway.chat.assert_not_called()
        self.gateway.embed.assert_not_called()
        self.rag.assert_not_called()

    def test_anonymous_page_is_login_and_every_data_route_is_denied_before_handlers(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        document = BeautifulSoup(response.text, 'html.parser')
        self.assertIsNotNone(document.select_one('form#login-form input[name="username"]'))
        self.assertEqual(document.select_one('input[name="password"]')['type'], 'password')
        self.assertIsNone(document.select_one('form#ask-form'))
        for secret in (self.username, self.password, self.encoded, 'TEST_API_SECRET', 'TEST_CHAT_SECRET'):
            self.assertNotIn(secret, response.text)
        for method, path in (('GET', '/api/stats'), ('POST', '/api/ask'),
                             ('GET', '/api/source/test-chunk'), ('GET', '/api/session'),
                             ('GET', '/openapi.json'), ('GET', '/docs'), ('GET', '/unregistered')):
            with self.subTest(method=method, path=path):
                denied = self.client.request(method, path)
                self.assertEqual(denied.status_code, 401)
                self.assertEqual(denied.headers['cache-control'], 'no-store')
                self.assertEqual(denied.headers['x-content-type-options'], 'nosniff')
                self.assertIn('content-security-policy', denied.headers)
        self.store.assert_not_called()
        self.assert_no_provider_calls()

    def test_login_sets_secure_cookie_and_allows_profile_and_stats(self):
        response = self.login()
        self.assertEqual(response.json(), {'username': self.username})
        cookie = SimpleCookie(response.headers['set-cookie'])[COOKIE_NAME]
        self.assertTrue(cookie['secure'])
        self.assertTrue(cookie['httponly'])
        self.assertEqual(cookie['samesite'], 'lax')
        self.assertEqual(cookie['path'], '/')
        self.assertEqual(cookie['max-age'], str(SESSION_SECONDS))
        self.assertEqual(cookie['domain'], '')
        self.assertEqual(self.client.get('/api/session').json(), {
            'username': self.username, 'authentication_required': True})
        self.assertEqual(self.client.get('/api/stats').json(), {'documents': 7})
        self.assertIn('id="ask-form"', self.client.get('/').text)
        self.assertEqual(self.client.get('/openapi.json').status_code, 404)
        self.assert_no_provider_calls()
        hosted = replace(self.settings, public_origin='https://owner-evidence.hf.space')
        with TestClient(create_app(hosted), base_url='http://owner-evidence.hf.space') as client:
            landing = client.get('/', headers={'Sec-Fetch-Site': 'cross-site', 'Sec-Fetch-Dest': 'iframe'})
            self.assertEqual(landing.status_code, 200)
            self.assertIn('id="open-app"', landing.text)
            proxied = client.post('/api/login', json={'username': self.username, 'password': self.password})
            self.assertEqual(proxied.status_code, 200)
            self.assertTrue(SimpleCookie(proxied.headers['set-cookie'])[COOKIE_NAME]['secure'])

    def test_wrong_or_malformed_credentials_never_sign_in_or_reflect_passwords(self):
        marker = 'DO_NOT_REFLECT_CREDENTIAL'
        for body in ({'username': self.username, 'password': marker},
                     {'username': 'unknown-account', 'password': marker},
                     {'username': '', 'password': marker},
                     {'username': self.username, 'password': [marker]},
                     {'username': self.username, 'password': marker * 50}):
            with self.subTest(username=body['username']):
                denied = self.client.post('/api/login', json=body)
                self.assertEqual(denied.status_code, 401)
                self.assertNotIn(marker, denied.text)
                self.assertNotIn('set-cookie', denied.headers)
        malformed = self.client.post('/api/login', content='{"password":"' + marker,
                                     headers={'Content-Type': 'application/json'})
        self.assertEqual(malformed.status_code, 401)
        self.assertNotIn(marker, malformed.text)
        self.assertEqual(self.client.get('/api/session').status_code, 401)
        self.store.assert_not_called()
        self.assert_no_provider_calls()

    def test_cross_origin_changes_are_rejected_and_logout_revokes_copied_cookie(self):
        self.login()
        token = self.client.cookies.get(COOKIE_NAME)
        for endpoint in ('/api/login', '/api/logout'):
            for headers in ({'Origin': 'https://attacker.example'}, {'Sec-Fetch-Site': 'cross-site'}):
                with self.subTest(endpoint=endpoint, headers=headers):
                    response = self.client.post(endpoint, headers=headers,
                                                json={'username': self.username, 'password': self.password})
                    self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get('/api/session').status_code, 200)
        tampered = ('a' if token[0] != 'a' else 'b') + token[1:]
        for invalid in (tampered, 'hostile-cookie', token + 'extra'):
            response = self.client.get('/api/session', headers={'Cookie': f'{COOKIE_NAME}={invalid}'})
            self.assertEqual(response.status_code, 401)
        self.login()
        self.assertEqual(self.client.get('/api/session', headers={'Cookie': f'{COOKIE_NAME}={token}'}).status_code, 401)
        current = self.client.cookies.get(COOKIE_NAME)
        response = self.client.post('/api/logout')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(SimpleCookie(response.headers['set-cookie'])[COOKIE_NAME]['max-age'], '0')
        self.assertIsNone(self.client.cookies.get(COOKIE_NAME))
        self.assertEqual(self.client.get('/api/session').status_code, 401)
        replay = self.client.get('/api/stats', headers={'Cookie': f'{COOKIE_NAME}={current}'})
        self.assertEqual(replay.status_code, 401)
        self.store.assert_not_called()
        self.assert_no_provider_calls()

    def test_required_missing_or_malformed_user_configuration_fails_closed(self):
        baseline = {'RAG_BASE_URL': 'https://gateway.invalid/v1', 'RAG_API_KEY': 'TEST_CONFIG_SECRET',
                    'RAG_AUTH_REQUIRED': '1'}
        marker = 'INVALID_CONFIG_CREDENTIAL'
        for raw in (None, '', 'not-json-' + marker, '[]', '{}', 'null',
                    json.dumps({'friend': marker})):
            environment = dict(baseline)
            if raw is not None:
                environment['RAG_AUTH_USERS_JSON'] = raw
            with self.subTest(raw_type=type(raw).__name__), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ValueError) as caught:
                    Settings.from_env()
                self.assertNotIn(marker, str(caught.exception))
                self.assertNotIn('TEST_CONFIG_SECRET', str(caught.exception))
        with patch.dict(os.environ, {**baseline, 'RAG_AUTH_REQUIRED': 'invalid'}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()
        with patch.dict(os.environ, {**baseline, 'RAG_AUTH_USERS_JSON': json.dumps({self.username: self.encoded})}, clear=True):
            configured = Settings.from_env()
            self.assertTrue(configured.auth_required)
            self.assertEqual(configured.auth_users, {self.username: self.encoded})
        with self.assertRaises(ValueError):
            create_app(replace(self.settings, auth_users=None))
        self.store.assert_not_called()
        self.assert_no_provider_calls()


if __name__ == '__main__':
    unittest.main()
