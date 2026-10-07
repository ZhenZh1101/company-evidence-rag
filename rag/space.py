"""Start the frozen-data Docker Space; never ingest or generate embeddings."""

from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import URLError

from .config import Settings


def prepare_database(seed: Path, destination: Path):
    if not seed.is_file():
        raise RuntimeError('Missing space-data/rag.sqlite3. Prepare the snapshot and mount its dataset at /app/space-data.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix('.partial')
        shutil.copyfile(seed, temporary)
        temporary.replace(destination)
    # Refuse empty/partial data instead of silently starting a blank application.
    db = sqlite3.connect(f'{destination.resolve().as_uri()}?mode=ro', uri=True)
    try:
        chunks, embedded = db.execute('SELECT count(*),count(e.text_hash) FROM chunks c '
                                     'LEFT JOIN embeddings e USING(text_hash)').fetchone()
        identity = db.execute("SELECT value FROM metadata WHERE key='embedding_identity'").fetchone()
        if not chunks or chunks != embedded:
            raise RuntimeError('The deployment snapshot must contain a complete embedding cache.')
        if identity != ('https://api.openai.com/v1|text-embedding-3-large',):
            raise RuntimeError('The deployment snapshot must use OpenAI text-embedding-3-large.')
    finally:
        db.close()


def main():
    settings = Settings.from_env()
    prepare_database(Path('/app/space-data/rag.sqlite3'), settings.db_path)
    children = []

    def stop(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        qdrant = subprocess.Popen(['qdrant', '--config-path', '/app/config/config.yaml'])
        children.append(qdrant)
        deadline = time.monotonic() + 60
        while True:
            if qdrant.poll() is not None:
                raise RuntimeError('Qdrant exited during startup.')
            try:
                with urlopen('http://127.0.0.1:6333/readyz', timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError('Qdrant was not ready within 60 seconds.')
            time.sleep(0.5)

        print('Restoring searchable vectors from the existing cache; no model API calls.', flush=True)
        sync = subprocess.Popen([sys.executable, '-m', 'rag.cli', 'sync-vectors', '--batch-size', '128'])
        children.append(sync)
        while sync.poll() is None:
            if qdrant.poll() is not None:
                raise RuntimeError('Qdrant exited while restoring cached vectors.')
            time.sleep(0.5)
        if sync.returncode:
            raise RuntimeError('Cached vector restoration failed; refusing to serve a partial index.')

        server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'rag.web:create_app', '--factory',
                                   '--host', '0.0.0.0', '--port', '7860', '--workers', '1',
                                   '--no-proxy-headers', '--timeout-graceful-shutdown', '30'])
        children.append(server)
        while server.poll() is None and qdrant.poll() is None:
            time.sleep(0.5)
        raise RuntimeError('A Space service exited; restarting the container is required.')
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            try:
                child.wait(timeout=35)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
