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
    embedding_openclaw_model: str = 'openai/text-embedding-3-large'
    api_key: str = ''
    timeout: float = 180
    chat_base_url: str = ''
    chat_api_key: str | None = None
    chat_temperature: float | None = 0
    chat_thinking: str | None = None
    chat_token_limit_field: str = 'max_tokens'
    chat_reasoning_effort: str | None = None

    @property
    def openclaw_embedding_model(self):
        return self.embedding_openclaw_model if self.embedding_model.startswith('openclaw/') else ''

    @classmethod
    def from_env(cls):
        base_url = os.getenv('RAG_BASE_URL', cls.base_url).rstrip('/')
        chat_base_url = os.getenv('RAG_CHAT_BASE_URL', '').rstrip('/')
        for name, endpoint in (('RAG_BASE_URL', base_url), ('RAG_CHAT_BASE_URL', chat_base_url or base_url)):
            parsed = urlsplit(endpoint)
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                    or parsed.username is not None or parsed.query or parsed.fragment):
                raise ValueError(f'{name} must be an HTTP(S) endpoint without credentials, query or fragment.')
        raw_temperature = os.getenv('RAG_CHAT_TEMPERATURE', '0')
        temperature = None if raw_temperature == 'none' else float(raw_temperature)
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError('RAG_CHAT_TEMPERATURE must be between 0 and 2, or none to omit it.')
        thinking = os.getenv('RAG_CHAT_THINKING') or None
        if thinking not in (None, 'enabled', 'disabled'):
            raise ValueError('RAG_CHAT_THINKING must be enabled or disabled, or unset.')
        token_limit_field = os.getenv('RAG_CHAT_TOKEN_LIMIT_FIELD', cls.chat_token_limit_field)
        if token_limit_field not in ('max_tokens', 'max_completion_tokens'):
            raise ValueError('RAG_CHAT_TOKEN_LIMIT_FIELD must be max_tokens or max_completion_tokens.')
        reasoning_effort = os.getenv('RAG_CHAT_REASONING_EFFORT') or None
        if reasoning_effort not in (None, 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
            raise ValueError('Invalid RAG_CHAT_REASONING_EFFORT.')
        parsed = urlsplit(base_url)
        token = os.getenv('RAG_API_KEY', '')
        # Never forward the local gateway credential to a differently configured host.
        if not token and parsed.hostname in ('localhost', '127.0.0.1', '::1'):
            config = Path.home() / '.openclaw/openclaw.json'
            if config.exists():
                token = json.loads(config.read_text()).get('gateway', {}).get('auth', {}).get('token', '')
        return cls(db_path=Path(os.getenv('RAG_DB_PATH', 'data/rag.sqlite3')), base_url=base_url,
                   chat_model=os.getenv('RAG_CHAT_MODEL', cls.chat_model),
                   embedding_model=os.getenv('RAG_EMBEDDING_MODEL', cls.embedding_model), api_key=token,
                   embedding_openclaw_model=os.getenv('RAG_EMBEDDING_OPENCLAW_MODEL', cls.embedding_openclaw_model),
                   timeout=float(os.getenv('RAG_TIMEOUT', '180')), chat_base_url=chat_base_url,
                   chat_api_key=os.getenv('RAG_CHAT_API_KEY'), chat_temperature=temperature,
                   chat_thinking=thinking, chat_token_limit_field=token_limit_field,
                   chat_reasoning_effort=reasoning_effort)
