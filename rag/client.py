"""Small OpenAI-compatible client; errors never include credentials or response bodies."""
import json
import time
from urllib import error, request
import numpy as np
from .config import Settings


class Gateway:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _post(self, route, payload):
        if not self.settings.api_key:
            raise RuntimeError('Set RAG_API_KEY or configure gateway.auth.token in ~/.openclaw/openclaw.json.')
        encoded = json.dumps(payload).encode()
        # Prevent HTTP redirects from forwarding Authorization to another destination.
        class NoRedirect(request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        opener = request.build_opener(NoRedirect)
        for attempt in range(3):
            req = request.Request(self.settings.base_url + '/' + route, encoded,
                                  {'Authorization': 'Bearer ' + self.settings.api_key,
                                   'Content-Type': 'application/json'})
            try:
                with opener.open(req, timeout=self.settings.timeout) as response:
                    return json.load(response)
            except error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f'Gateway {route} returned HTTP {exc.code}.') from None
            except (error.URLError, TimeoutError, OSError):
                raise RuntimeError(f'Gateway {route} is unavailable or timed out.') from None
            except (ValueError, KeyError, TypeError):
                raise RuntimeError(f'Gateway {route} returned an invalid response.') from None

    def embed(self, texts):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        data = self._post('embeddings', {'model': self.settings.embedding_model, 'input': texts})
        try:
            entries = sorted(data['data'], key=lambda item: item['index'])
            if [x['index'] for x in entries] != list(range(len(texts))):
                raise ValueError('missing or repeated indexes')
            vectors = np.array([item['embedding'] for item in entries], dtype=np.float64)
            if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
                raise ValueError('invalid vectors')
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if (norms <= 0).any() or not np.isfinite(norms).all():
                raise ValueError('zero vector')
            return (vectors / norms).astype(np.float32)
        except (KeyError, ValueError, TypeError):
            raise RuntimeError('Gateway embeddings must be finite, nonzero, consistent vectors with valid indexes.') from None

    def chat(self, messages, max_tokens=2400):
        data = self._post('chat/completions', {'model': self.settings.chat_model,
                         'messages': messages, 'temperature': 0, 'max_tokens': max_tokens})
        try:
            choice = data['choices'][0]
            content = choice['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise ValueError()
            if choice.get('finish_reason') == 'length':
                raise RuntimeError('Gateway answer exceeded the output limit; shorten the question.')
            return content
        except (KeyError, IndexError, ValueError, TypeError):
            raise RuntimeError('Gateway chat returned no usable text.') from None


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        if '\n' not in text or not text.endswith('```'):
            raise ValueError('Malformed JSON fence.')
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(text)
