import io
import json
import unittest

from serving_engine.backends import NumpyBackend
from serving_engine.engine import Config
from serving_engine.server import Stream, Worker, handler_for


class WorkerTests(unittest.TestCase):
    def test_stream_completes_and_releases_request(self):
        worker = Worker(NumpyBackend(), Config())
        worker.thread.start()
        try:
            stream = Stream("a", "A small prompt", 5)
            worker.submit(stream)
            self.assertEqual(stream.ready.get(timeout=5)[0], 200)
            events = [stream.events.get(timeout=5) for _ in range(6)]
            self.assertEqual([e["type"] for e in events], ["token"] * 5 + ["done"])
            self.assertEqual(events[-1]["reason"], "length")
        finally:
            worker.close()
        self.assertEqual(worker.engine.reserved, 0)
        self.assertFalse(worker.thread.is_alive())

    def test_cancel_before_admission(self):
        worker = Worker(NumpyBackend(), Config())
        stream = Stream("a", "prompt", 5)
        stream.cancelled.set()
        worker.submit(stream)
        worker.thread.start()
        try:
            self.assertEqual(stream.ready.get(timeout=5)[0], 408)
        finally:
            worker.close()
        self.assertFalse(worker.streams)

    def test_oversized_prompt_is_rejected(self):
        worker = Worker(NumpyBackend(), Config(max_context=20))
        worker.thread.start()
        try:
            stream = Stream("a", "x" * 21, 5)
            worker.submit(stream)
            self.assertEqual(stream.ready.get(timeout=5)[0], 400)
        finally:
            worker.close()


class Connection:
    """In-memory transport exercises the real HTTP handler without opening a port."""
    def __init__(self, raw):
        self.input = io.BytesIO(raw)
        self.output = bytearray()

    def makefile(self, *_args):
        return self.input

    def sendall(self, data):
        self.output.extend(data)

    def settimeout(self, _value):
        pass


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.worker = Worker(NumpyBackend(), Config())
        self.worker.thread.start()
        self.handler = handler_for(self.worker)

    def tearDown(self):
        self.worker.close()

    def send(self, body, path="/generate"):
        raw = (f"POST {path} HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
        conn = Connection(raw)
        self.handler(conn, ("127.0.0.1", 1234), object())
        return bytes(conn.output)

    def test_sse_protocol(self):
        response = self.send(json.dumps(dict(prompt="hello", max_tokens=3)).encode())
        self.assertIn(b"200 OK", response)
        self.assertIn(b"text/event-stream", response)
        events = [json.loads(line[6:]) for line in response.splitlines() if line.startswith(b"data: ")]
        self.assertEqual([e["type"] for e in events], ["token"] * 3 + ["done"])

    def test_invalid_requests_release_slots(self):
        for body in [b"broken", b"[]", b'{"prompt":""}',
                     b'{"prompt":"hi","max_tokens":true}',
                     b'{"prompt":"hi","max_tokens":257}'] * 8:
            self.assertIn(b"400", self.send(body).splitlines()[0])
        self.assertIn(b"200", self.send(b'{"prompt":"ok","max_tokens":1}').splitlines()[0])

    def test_missing_route(self):
        self.assertIn(b"404", self.send(b"{}", "/missing").splitlines()[0])

    def test_health(self):
        conn = Connection(b"GET /health HTTP/1.0\r\n\r\n")
        self.handler(conn, ("127.0.0.1", 1234), object())
        self.assertIn(b'"status": "ok"', conn.output)


if __name__ == "__main__":
    unittest.main()
