"""Password login and bounded, expiring sessions for the single-worker app."""

from collections import deque
import hashlib
import math
import re
import secrets
import threading
import time


COOKIE_NAME = 'rag_session'
SESSION_SECONDS = 12 * 60 * 60
PASSWORD_ITERATIONS = 600_000
MAX_SESSIONS = 1000
MAX_LOGIN_CLIENTS = 10_000
_HASH_PATTERN = re.compile(r'pbkdf2_sha256\$600000\$[0-9a-f]{32}\$[0-9a-f]{64}')
_TOKEN_PATTERN = re.compile(r'[A-Za-z0-9_-]{43}')


class InvalidCredentials(Exception):
    def __init__(self):
        super().__init__('Incorrect username or password.')


class LoginBusy(Exception):
    retry_after = 1

    def __init__(self):
        super().__init__('Login is temporarily busy. Try again later.')


class LoginRateLimited(Exception):
    def __init__(self, retry_after):
        self.retry_after = retry_after
        super().__init__('Too many login attempts. Try again later.')


def hash_password(password):
    if not isinstance(password, str) or not 8 <= len(password) <= 1024:
        raise ValueError('Password must contain 8–1024 characters.')
    try:
        password_bytes = password.encode('utf-8')
    except UnicodeError:
        raise ValueError('Password must be valid Unicode text.') from None
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac('sha256', password_bytes, salt, PASSWORD_ITERATIONS)
    return f'pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}'


def validate_users(users):
    if not isinstance(users, dict) or not users or len(users) > 1000:
        raise ValueError('Invalid authentication user configuration.')
    for username, encoded in users.items():
        if (not isinstance(username, str) or not 1 <= len(username) <= 100
                or username != username.strip() or not isinstance(encoded, str)
                or not _HASH_PATTERN.fullmatch(encoded)):
            raise ValueError('Invalid authentication user configuration.')
    return dict(users)


def _verify_password(password, encoded):
    _, _, salt, expected = encoded.split('$')
    actual = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), PASSWORD_ITERATIONS)
    return secrets.compare_digest(actual, bytes.fromhex(expected))


def _session_key(token):
    if not isinstance(token, str) or not _TOKEN_PATTERN.fullmatch(token):
        return None
    return hashlib.sha256(token.encode('ascii')).hexdigest()


class AuthService:
    def __init__(self, users):
        self._users = validate_users(users)
        self._dummy_hash = hash_password(secrets.token_urlsafe(32))
        # ponytail: sessions and login limits live in one worker; restart logs
        # everyone out. Use shared storage before adding workers or replicas.
        self._sessions = {}
        self._attempts = {}
        self._lock = threading.Lock()
        self._hash_slots = threading.BoundedSemaphore(2)

    def _record_attempt(self, client_ip):
        with self._lock:
            now = time.monotonic()
            self._attempts = {client: times for client, times in self._attempts.items()
                              if times[-1] > now - 60}
            if client_ip not in self._attempts:
                if len(self._attempts) >= MAX_LOGIN_CLIENTS:
                    raise LoginBusy()
                self._attempts[client_ip] = deque()
            attempts = self._attempts[client_ip]
            while attempts and attempts[0] <= now - 60:
                attempts.popleft()
            if len(attempts) >= 10:
                raise LoginRateLimited(max(1, math.ceil(attempts[0] + 60 - now)))
            # All admitted attempts count, including correct passwords and busy
            # hash workers. This quota is independent of question admission.
            attempts.append(now)

    def login(self, username, password, client_ip):
        self._record_attempt(client_ip)
        if (not isinstance(username, str) or not 1 <= len(username) <= 100
                or not isinstance(password, str) or len(password) > 1024):
            raise InvalidCredentials()
        if not self._hash_slots.acquire(blocking=False):
            raise LoginBusy()
        try:
            encoded = self._users.get(username, self._dummy_hash)
            try:
                matches = _verify_password(password, encoded)
            except UnicodeError:
                raise InvalidCredentials() from None
            if not matches or username not in self._users:
                raise InvalidCredentials()
        finally:
            self._hash_slots.release()
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = time.monotonic()
            self._sessions = {key: session for key, session in self._sessions.items() if session[1] > now}
            if len(self._sessions) >= MAX_SESSIONS:
                del self._sessions[next(iter(self._sessions))]
            self._sessions[_session_key(token)] = (username, now + SESSION_SECONDS)
        return token

    def current_user(self, token):
        key = _session_key(token)
        if key is None:
            return None
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return None
            if session[1] <= time.monotonic():
                del self._sessions[key]
                return None
            return session[0]

    def logout(self, token):
        key = _session_key(token)
        with self._lock:
            self._sessions.pop(key, None)
