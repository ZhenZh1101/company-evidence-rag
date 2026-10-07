"""Small OpenAI-compatible client; errors never include credentials or response bodies."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from urllib import error, request
import numpy as np
from .config import Settings


class Gateway:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _post(self, route, payload):
        base_url, api_key = self.settings.base_url.rstrip('/'), self.settings.api_key
        if route == 'chat/completions':
            base_url = (self.settings.chat_base_url or base_url).rstrip('/')
            api_key = self.settings.chat_api_key
            # Only the same endpoint may inherit the shared gateway credential.
            if api_key is None and base_url == self.settings.base_url.rstrip('/'):
                api_key = self.settings.api_key
            if not api_key:
                raise RuntimeError('Set RAG_CHAT_API_KEY; a separate chat endpoint requires its own key. '
                                   'For the shared gateway, set RAG_API_KEY or configure gateway.auth.token.')
        elif not api_key:
            raise RuntimeError('Set RAG_API_KEY or configure gateway.auth.token in ~/.openclaw/openclaw.json.')
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        # Prevent HTTP redirects from forwarding Authorization to another destination.
        class NoRedirect(request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        opener = request.build_opener(NoRedirect)
        for attempt in range(5):
            req = request.Request(base_url + '/' + route, encoded,
                                  {'Authorization': 'Bearer ' + api_key,
                                   'Content-Type': 'application/json'})
            if route == 'embeddings' and self.settings.openclaw_embedding_model:
                req.add_header('x-openclaw-model', self.settings.openclaw_embedding_model)
            try:
                with opener.open(req, timeout=self.settings.timeout) as response:
                    return json.load(response)
            except error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
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
        batches, batch, characters = [], [], 0
        for text in texts:
            if not isinstance(text, str) or not text.strip() or len(text) > 60000:
                raise ValueError('Embedding inputs must be nonempty strings of at most 60000 characters.')
            # The configured gateway has a 65,536-character total request limit.
            if batch and (characters + len(text) > 60000 or len(batch) >= 64):
                batches.append(batch)
                batch, characters = [], 0
            batch.append(text)
            characters += len(text)
        batches.append(batch)
        if len(batches) == 1:
            return self._embed_batch(batches[0])
        with ThreadPoolExecutor(max_workers=2) as pool:
            parts = list(pool.map(self._embed_batch, batches))
        if len({part.shape[1] for part in parts}) != 1:
            raise RuntimeError('Gateway changed embedding dimensions between batches.')
        return np.concatenate(parts)

    def _embed_batch(self, texts):
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
        payload = {'model': self.settings.chat_model, 'messages': messages,
                   self.settings.chat_token_limit_field: max(max_tokens, self.settings.chat_min_tokens)}
        if self.settings.chat_temperature is not None:
            payload['temperature'] = self.settings.chat_temperature
        if self.settings.chat_thinking is not None:
            payload['thinking'] = {'type': self.settings.chat_thinking}
        if self.settings.chat_reasoning_effort is not None:
            payload['reasoning_effort'] = self.settings.chat_reasoning_effort
        if self.settings.chat_response_format is not None:
            payload['response_format'] = {'type': self.settings.chat_response_format}
        data = self._post('chat/completions', payload)
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
