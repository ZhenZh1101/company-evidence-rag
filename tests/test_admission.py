import asyncio
from contextlib import closing
from ipaddress import ip_network
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from starlette.requests import Request

from rag.admission import AdmissionController, QueueFull, RateLimitExceeded
from rag.config import Settings
from rag.web import client_ip, create_app


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'limits.sqlite3'
        self.admission = AdmissionController(self.path)

    def reservations(self):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute('SELECT count(*) FROM questions').fetchone()[0]

    async def test_rolling_windows_ip_isolation_and_restart(self):
        with patch('rag.admission.time.time', return_value=100000) as clock:
            for _ in range(5):
                await self.admission.run('a', lambda: None)
            with self.assertRaises(RateLimitExceeded) as caught:
                await self.admission.run('a', lambda: self.fail('Over-quota work started'))
            self.assertEqual(caught.exception.retry_after, 60)
            self.assertEqual(self.reservations(), 5)
            await self.admission.run('b', lambda: None)
            restarted = AdmissionController(self.path)
            with self.assertRaises(RateLimitExceeded):
                await restarted.run('a', lambda: None)
            for minute in range(1, 10):
                clock.return_value = 100000 + minute * 60
                for _ in range(5):
                    await restarted.run('a', lambda: None)
            clock.return_value = 100600
            with self.assertRaises(RateLimitExceeded) as caught:
                await restarted.run('a', lambda: None)
            self.assertEqual(caught.exception.retry_after, 85800)
            self.assertEqual(self.reservations(), 51)
            clock.return_value = 186400
            for _ in range(5):
                await restarted.run('a', lambda: None)
            with self.assertRaises(RateLimitExceeded):
                await restarted.run('a', lambda: None)

    async def wait_for_queue(self, size):
        async with asyncio.timeout(2):
            while len(self.admission.waiting) != size:
                await asyncio.sleep(0.001)

    async def test_five_waiters_fifo_rejection_and_queued_cancellation_refund(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        order = []

        def work(number):
            if number == 0:
                started.set()
                self.assertTrue(release.wait(5))
            order.append(number)
            return number

        first = asyncio.create_task(self.admission.run('0', lambda: work(0)))
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        pending = [asyncio.create_task(self.admission.run(str(i), lambda i=i: work(i))) for i in range(1, 6)]
        await self.wait_for_queue(5)
        with self.assertRaises(QueueFull):
            await self.admission.run('rejected', lambda: self.fail('Full queue started work'))
        self.assertEqual(self.reservations(), 6)
        pending[1].cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending[1]
        self.assertEqual(len(self.admission.waiting), 4)
        self.assertEqual(self.reservations(), 5)
        replacement = asyncio.create_task(self.admission.run('6', lambda: work(6)))
        await self.wait_for_queue(5)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, *pending, replacement, return_exceptions=True), 3)
        self.assertEqual(order, [0, 1, 3, 4, 5, 6])
        self.assertFalse(self.admission.running)
        self.assertEqual(self.reservations(), 6)

    async def test_running_cancellation_keeps_slot_until_work_finishes_and_failures_release(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def slow():
            started.set()
            self.assertTrue(release.wait(5))
            raise RuntimeError('upstream failed after disconnect')

        first = asyncio.create_task(self.admission.run('a', slow))
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertTrue(self.admission.running)
        second = asyncio.create_task(self.admission.run('b', lambda: 2))
        await self.wait_for_queue(1)
        self.assertFalse(second.done())
        release.set()
        self.assertEqual(await asyncio.wait_for(second, 3), 2)
        self.assertFalse(self.admission.running)
        self.assertEqual(self.reservations(), 2)
        with self.assertRaisesRegex(ValueError, 'failed'):
            await self.admission.run('c', lambda: (_ for _ in ()).throw(ValueError('failed')))
        self.assertEqual(await self.admission.run('d', lambda: 4), 4)

    async def test_cancelled_waiter_skipped_during_handover_cannot_release_next_slot(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.admission.running = True  # A currently executing request owns the slot.
        cancelled = asyncio.create_task(self.admission.run('a', lambda: self.fail('Cancelled work ran')))

        def slow():
            started.set()
            self.assertTrue(release.wait(5))

        next_request = asyncio.create_task(self.admission.run('b', slow))
        await self.wait_for_queue(2)
        cancelled.cancel()
        # The active request ends before the cancelled coroutine gets CPU time.
        self.admission._release()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        self.assertTrue(self.admission.running)
        release.set()
        await asyncio.wait_for(next_request, 3)
        self.assertFalse(self.admission.running)
        self.assertEqual(self.reservations(), 1)

    async def test_http_disconnect_removes_queued_request_and_refunds_it(self):
        app = create_app(Settings(db_path=self.path.with_name('evidence.sqlite3'), rate_limit_db_path=self.path))
        app.state.admission.running = True  # Simulate a busy worker.
        incoming = asyncio.Queue()
        await incoming.put({'type': 'http.request', 'body': json.dumps({'question': 'Revenue'}).encode(),
                            'more_body': False})
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST',
                 'scheme': 'http', 'path': '/api/ask', 'raw_path': b'/api/ask', 'query_string': b'',
                 'server': ('127.0.0.1', 8000), 'client': ('198.51.100.9', 1234),
                 'headers': [(b'host', b'127.0.0.1:8000'), (b'content-type', b'application/json')]}
        messages = []

        async def send(message):
            messages.append(message)

        with patch('rag.web.RAG.ask') as answer:
            request = asyncio.create_task(app(scope, incoming.get, send))
            async with asyncio.timeout(2):
                while not app.state.admission.waiting:
                    await asyncio.sleep(0.001)
            self.assertEqual(self.reservations(), 1)
            await incoming.put({'type': 'http.disconnect'})
            await asyncio.wait_for(request, 2)
            self.assertEqual(len(app.state.admission.waiting), 0)
            self.assertEqual(self.reservations(), 0)
            self.assertEqual(messages[0]['status'], 499)
            answer.assert_not_called()


class ClientIPTests(unittest.TestCase):
    def test_only_configured_proxies_and_rightmost_untrusted_address_are_used(self):
        trusted = (ip_network('10.0.0.0/8'),)

        def resolve(peer, headers):
            request = Request({'type': 'http', 'client': (peer, 1234), 'headers': headers})
            return client_ip(request, trusted)

        self.assertEqual(resolve('198.51.100.9', [(b'x-forwarded-for', b'203.0.113.99')]), '198.51.100.9')
        self.assertEqual(resolve('10.0.0.2', [(b'x-forwarded-for', b'203.0.113.99, 198.51.100.9, 10.0.0.3')]), '198.51.100.9')
        self.assertEqual(resolve('10.0.0.2', [(b'x-forwarded-for', b'203.0.113.99'),
                                              (b'x-forwarded-for', b'198.51.100.9, 10.0.0.3')]), '198.51.100.9')
        self.assertEqual(resolve('10.0.0.2', [(b'x-forwarded-for', b'203.0.113.99, invalid')]), '10.0.0.2')
        self.assertEqual(resolve('10.0.0.2', [(b'x-real-ip', b'203.0.113.99')]), '10.0.0.2')
        self.assertEqual(resolve('::ffff:10.0.0.2', [(b'x-forwarded-for', b'::ffff:198.51.100.9')]), '198.51.100.9')
        self.assertEqual(resolve('2001:0db8:0:0:0:0:0:1', []), '2001:db8::1')


if __name__ == '__main__':
    unittest.main()
