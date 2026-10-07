"""Single-worker question admission with a durable, rolling IP quota."""

import asyncio
from collections import deque
from contextlib import closing
import math
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from starlette.concurrency import run_in_threadpool


class RateLimitExceeded(Exception):
    def __init__(self, retry_after):
        self.retry_after = retry_after


class QueueFull(Exception):
    pass


class AdmissionController:
    """One running question and at most five FIFO waiters, in one app process.

    Accepted questions reserve quota immediately. Queued cancellations refund it;
    started questions consume quota even if the upstream service fails.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS questions '
                       '(id TEXT PRIMARY KEY, client TEXT NOT NULL, asked_at REAL NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS questions_client_time ON questions(client, asked_at)')
            db.execute('CREATE INDEX IF NOT EXISTS questions_time ON questions(asked_at)')
        self.waiting = deque()
        self.running = False
        self.tasks = set()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=0.1)

    def _reserve(self, client):
        now = time.time()
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM questions WHERE asked_at <= ?', (now - 86400,))
            times = [row[0] for row in db.execute(
                'SELECT asked_at FROM questions WHERE client = ? ORDER BY asked_at', (client,))]
            waits = [times[-limit] + seconds - now for limit, seconds in ((5, 60), (50, 86400))
                     if len(times) >= limit and times[-limit] > now - seconds]
            if waits:
                raise RateLimitExceeded(max(1, math.ceil(max(waits))))
            reservation = uuid4().hex
            db.execute('INSERT INTO questions VALUES (?, ?, ?)', (reservation, client, now))
            return reservation

    def _refund(self, reservation):
        with closing(self._connect()) as db, db:
            db.execute('DELETE FROM questions WHERE id = ?', (reservation,))

    def _release(self):
        while self.waiting:
            ready = self.waiting.popleft()
            if not ready.cancelled():
                ready.set_result(None)
                return
        self.running = False

    async def _execute(self, operation):
        try:
            return await run_in_threadpool(operation)
        finally:
            self._release()

    async def run(self, client, operation):
        # ponytail: in-process FIFO; run exactly one worker/replica. A shared queue
        # is required before scaling out. No await before reserving a queue slot.
        if self.running and len(self.waiting) >= 5:
            raise QueueFull()
        reservation = self._reserve(client)
        ready = asyncio.get_running_loop().create_future()
        if self.running:
            self.waiting.append(ready)
        else:
            self.running = True
            ready.set_result(None)
        try:
            await ready
        except asyncio.CancelledError:
            if ready in self.waiting:
                self.waiting.remove(ready)
            elif not ready.cancelled():
                # A slot may have been handed over immediately before cancel.
                # A cancelled future was skipped by _release and owns no slot.
                self._release()
            self._refund(reservation)
            raise

        task = asyncio.create_task(self._execute(operation))
        self.tasks.add(task)

        def completed(done):
            self.tasks.discard(done)
            if not done.cancelled():
                done.exception()  # Retrieve failures when the caller disconnected.

        task.add_done_callback(completed)
        # Disconnecting a running caller must not admit another question while
        # its synchronous gateway work is still executing in the thread pool.
        return await asyncio.shield(task)
