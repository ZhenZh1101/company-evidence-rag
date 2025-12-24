from dataclasses import dataclass
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path('data/rag.sqlite3')
    base_url: str = 'http://127.0.0.1:18789/v1'
    chat_model: str = 'openclaw/llm-gpt55'
    embedding_model: str = 'openclaw/llm-gpt55'
    api_key: str = ''
    timeout: float = 180

    @classmethod
    def from_env(cls):
        base_url = os.getenv('RAG_BASE_URL', cls.base_url).rstrip('/')
        parsed = urlsplit(base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username:
            raise ValueError('RAG_BASE_URL must be an HTTP(S) endpoint without credentials.')
        token = os.getenv('RAG_API_KEY', '')
        # Never forward the local gateway credential to a differently configured host.
        if not token and parsed.hostname in ('localhost', '127.0.0.1', '::1'):
            config = Path.home() / '.openclaw/openclaw.json'
            if config.exists():
                token = json.loads(config.read_text()).get('gateway', {}).get('auth', {}).get('token', '')
        return cls(Path(os.getenv('RAG_DB_PATH', 'data/rag.sqlite3')), base_url,
                   os.getenv('RAG_CHAT_MODEL', cls.chat_model),
                   os.getenv('RAG_EMBEDDING_MODEL', cls.embedding_model), token,
                   float(os.getenv('RAG_TIMEOUT', '180')))
