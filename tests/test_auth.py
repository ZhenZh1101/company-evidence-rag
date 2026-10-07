from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import unittest
from unittest.mock import patch

from rag.auth import (AuthService, InvalidCredentials, LoginBusy, LoginRateLimited,
                      SESSION_SECONDS, hash_password, validate_users)


class AuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoded = hash_password('test-passphrase')

    def setUp(self):
        self.auth = AuthService({'friend': self.encoded})

    def test_hash_format_salts_and_configuration_fail_closed(self):
        other = hash_password('test-passphrase')
        self.assertNotEqual(self.encoded, other)
        self.assertRegex(self.encoded, r'^pbkdf2_sha256\$600000\$[0-9a-f]{32}\$[0-9a-f]{64}$')
        users = {'friend': self.encoded}
        checked = validate_users(users)
        users.clear()
        self.assertIn('friend', checked)
        for invalid in ({}, [], None, {'friend': ''}, {'friend': 'plaintext'},
                        {'': self.encoded}, {' friend': self.encoded},
                        {'friend': self.encoded.replace('600000', '1')},
                        {'friend': self.encoded + '\n'}):
            with self.subTest(invalid_type=type(invalid).__name__):
                with self.assertRaisesRegex(ValueError, '^Invalid authentication user configuration.$'):
                    AuthService(invalid)
        self.assertIn('friend', validate_users({'friend': hash_password('eightchr')}))
        for password in ('short', 'x' * 1025, '\ud800' * 8, None):
            with self.assertRaises(ValueError):
                hash_password(password)

    def test_sessions_authenticate_expire_logout_and_store_only_token_hashes(self):
        with patch('rag.auth.time.monotonic', return_value=100) as clock:
            token = self.auth.login('friend', 'test-passphrase', 'client')
            self.assertEqual(self.auth.current_user(token), 'friend')
            self.assertNotIn(token, self.auth._sessions)
            self.assertIn(hashlib.sha256(token.encode()).hexdigest(), self.auth._sessions)
            changed = ('a' if token[0] != 'a' else 'b') + token[1:]
            for invalid in (None, '', changed, token + 'x', 'é' * 43):
                self.assertIsNone(self.auth.current_user(invalid))
            clock.return_value = 100 + SESSION_SECONDS - 1
            self.assertEqual(self.auth.current_user(token), 'friend')
            clock.return_value += 1
            self.assertIsNone(self.auth.current_user(token))
            token = self.auth.login('friend', 'test-passphrase', 'client')
            self.auth.logout(token)
            self.auth.logout(token)
            self.auth.logout(None)
            self.assertIsNone(self.auth.current_user(token))

    def test_invalid_and_unknown_users_do_same_password_hash_work(self):
        original = hashlib.pbkdf2_hmac
        with patch('rag.auth.hashlib.pbkdf2_hmac', wraps=original) as hashing:
            for username in ('friend', 'missing'):
                with self.assertRaisesRegex(InvalidCredentials, '^Incorrect username or password.$'):
                    self.auth.login(username, 'wrong', 'client')
            self.assertEqual(hashing.call_count, 2)
            for call in hashing.call_args_list:
                self.assertEqual(call.args[0], 'sha256')
                self.assertEqual(call.args[3], 600_000)
        self.assertEqual(self.auth._sessions, {})
        with self.assertRaises(InvalidCredentials):
            self.auth.login('friend', '\ud800', 'client')

    def test_all_login_attempts_have_rolling_per_ip_limit_before_hashing(self):
        with patch('rag.auth.time.monotonic', return_value=1000) as clock:
            with patch('rag.auth._verify_password', return_value=False) as verify:
                for _ in range(10):
                    with self.assertRaises(InvalidCredentials):
                        self.auth.login('friend', 'wrong', 'a')
                clock.return_value = 1000.2
                with self.assertRaises(LoginRateLimited) as caught:
                    self.auth.login('friend', 'test-passphrase', 'a')
                self.assertEqual(caught.exception.retry_after, 60)
                self.assertEqual(verify.call_count, 10)
                with self.assertRaises(InvalidCredentials):
                    self.auth.login('friend', 'wrong', 'b')
                clock.return_value = 1060
                with self.assertRaises(InvalidCredentials):
                    self.auth.login('friend', 'wrong', 'a')
            with patch('rag.auth._verify_password', return_value=True):
                for _ in range(9):
                    self.auth.login('friend', 'test-passphrase', 'a')
                with self.assertRaises(LoginRateLimited):
                    self.auth.login('friend', 'test-passphrase', 'a')

    def test_hash_work_is_bounded_and_releases_slots_after_failures(self):
        started = threading.Barrier(3)
        release = threading.Event()
        self.addCleanup(release.set)

        def slow(*args):
            started.wait(timeout=3)
            if not release.wait(3):
                raise AssertionError('Password worker did not finish.')
            return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            with patch('rag.auth._verify_password', side_effect=slow):
                futures = [pool.submit(self.auth.login, 'friend', 'wrong', str(i)) for i in range(2)]
                try:
                    started.wait(timeout=3)
                    with self.assertRaises(LoginBusy):
                        self.auth.login('friend', 'wrong', 'third')
                finally:
                    release.set()
                for future in futures:
                    with self.assertRaises(InvalidCredentials):
                        future.result(timeout=3)
        token = self.auth.login('friend', 'test-passphrase', 'third')
        self.assertEqual(self.auth.current_user(token), 'friend')

    def test_session_and_ip_storage_are_bounded_and_restart_revokes(self):
        with patch('rag.auth.MAX_SESSIONS', 2), patch('rag.auth.MAX_LOGIN_CLIENTS', 2):
            with patch('rag.auth._verify_password', return_value=True):
                tokens = [self.auth.login('friend', 'test-passphrase', 'client') for _ in range(3)]
                self.assertIsNone(self.auth.current_user(tokens[0]))
                self.assertEqual(self.auth.current_user(tokens[-1]), 'friend')
                self.assertEqual(len(self.auth._sessions), 2)
                self.auth.login('friend', 'test-passphrase', 'second')
                with self.assertRaises(LoginBusy):
                    self.auth.login('friend', 'test-passphrase', 'third')
                self.assertEqual(len(self.auth._attempts), 2)
        restarted = AuthService({'friend': self.encoded})
        self.assertIsNone(restarted.current_user(tokens[-1]))


if __name__ == '__main__':
    unittest.main()
