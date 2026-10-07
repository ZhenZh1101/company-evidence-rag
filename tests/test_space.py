from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from rag.client import Gateway
from rag.config import Settings
from rag.space import prepare_database


class SpaceTests(unittest.TestCase):
    def test_public_origin_and_reasoning_budget_configuration(self):
        with patch.dict('os.environ', {
                'SPACE_HOST': 'owner-research.hf.space',
                'RAG_BASE_URL': 'https://api.openai.com/v1',
                'RAG_CHAT_BASE_URL': 'https://api.z.ai/api/paas/v4',
                'RAG_CHAT_MODEL': 'glm-5.3-flash',
                'RAG_CHAT_MIN_TOKENS': '8192', 'RAG_CHAT_THINKING': 'enabled',
                'RAG_CHAT_REASONING_EFFORT': 'low', 'RAG_CHAT_TEMPERATURE': '1',
                'RAG_RATE_LIMIT_DB_PATH': '/tmp/rates.sqlite3'}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.public_origin, 'https://owner-research.hf.space')
        self.assertEqual(settings.rate_limit_db_path, Path('/tmp/rates.sqlite3'))
        with patch.object(Gateway, '_post', return_value={
                'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}) as post:
            for requested, expected in ((16, 8192), (500, 8192), (4000, 8192), (12000, 12000)):
                Gateway(settings).chat([], max_tokens=requested)
                payload = post.call_args.args[1]
                self.assertEqual(payload['max_tokens'], expected)
                self.assertEqual(payload['thinking'], {'type': 'enabled'})
                self.assertEqual(payload['reasoning_effort'], 'low')
            Gateway(replace(settings, chat_min_tokens=0)).chat([], max_tokens=16)
            self.assertEqual(post.call_args.args[1]['max_tokens'], 16)
        for name, value in [('RAG_PUBLIC_ORIGIN', 'https://user:pass@host'),
                            ('RAG_PUBLIC_ORIGIN', 'https://host/path'),
                            ('RAG_PUBLIC_ORIGIN', 'https://host:bad'),
                            ('RAG_CHAT_MIN_TOKENS', '-1'), ('RAG_CHAT_MIN_TOKENS', '128001')]:
            with self.subTest(name=name, value=value), patch.dict('os.environ', {name: value}, clear=True):
                with self.assertRaises(ValueError):
                    Settings.from_env()

    def test_seed_copy_is_complete_and_never_overwrites_existing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed, live = root / 'seed.sqlite3', root / 'runtime/index.sqlite3'
            with self.assertRaisesRegex(RuntimeError, 'Missing'):
                prepare_database(seed, live)
            with sqlite3.connect(seed) as db:
                db.executescript('''CREATE TABLE chunks(text_hash TEXT);
                    CREATE TABLE embeddings(text_hash TEXT);
                    CREATE TABLE metadata(key TEXT, value TEXT);
                    INSERT INTO chunks VALUES ('hash');
                    INSERT INTO embeddings VALUES ('hash');
                    INSERT INTO metadata VALUES
                    ('embedding_identity','https://api.openai.com/v1|text-embedding-3-large');''')
            original = seed.read_bytes()
            prepare_database(seed, live)
            self.assertEqual(original, live.read_bytes())
            with sqlite3.connect(live) as db:
                db.execute("INSERT INTO metadata VALUES ('runtime-only','preserved')")
            prepare_database(seed, live)
            with sqlite3.connect(live) as db:
                self.assertEqual(db.execute("SELECT value FROM metadata WHERE key='runtime-only'").fetchone(),
                                 ('preserved',))
                db.execute('DELETE FROM embeddings')
            with self.assertRaisesRegex(RuntimeError, 'complete embedding'):
                prepare_database(seed, live)
            self.assertEqual(seed.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
