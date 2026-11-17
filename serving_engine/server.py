"""Local HTTP/SSE demo, not a production or OpenAI-compatible server.

The worker alone owns all inference state. Handler threads only exchange messages.
Streams carry token IDs; clients may decode the accumulated IDs with the model tokenizer.
"""
import argparse
import json
import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import perf_counter

from .backends import make_backend
from .engine import Config, Engine, Overloaded, Request


@dataclass
class Stream:
    id: str
    prompt: str
    max_tokens: int
    arrival: float = field(default_factory=perf_counter)
    cancelled: threading.Event = field(default_factory=threading.Event)
    ready: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=1))
    events: queue.Queue = field(init=False)

    def __post_init__(self):
        self.events = queue.Queue(maxsize=self.max_tokens + 2)


class Worker:
    def __init__(self, backend, config):
        self.backend = backend
        self.commands = queue.Queue(maxsize=config.max_pending)
        self.streams = {}
        self.stop = threading.Event()
        self.engine = Engine(backend, config, on_token=self._token, on_done=self._done)
        self.thread = threading.Thread(target=self._run, name="inference-owner", daemon=True)

    def _token(self, request, token):
        self.streams[request.id].events.put_nowait(
            dict(type="token", id=request.id, token_id=token))

    def _done(self, request):
        stream = self.streams.pop(request.id)
        stream.events.put_nowait(dict(type="done", id=request.id, reason=request.reason))

    def submit(self, stream):
        try:
            self.commands.put_nowait(stream)
        except queue.Full as exc:
            raise Overloaded("command queue full") from exc

    def _accept(self, stream):
        if stream.cancelled.is_set():
            stream.ready.put_nowait((408, "cancelled before admission"))
            return
        try:
            tokens = self.backend.encode(stream.prompt)
            request = Request(stream.id, tokens, stream.max_tokens, stream.arrival)
            self.engine.submit(request)
        except Overloaded as exc:
            stream.ready.put_nowait((429, str(exc)))
        except (ValueError, TypeError) as exc:
            stream.ready.put_nowait((400, str(exc)))
        except Exception:
            logging.exception("tokenization failed")
            stream.ready.put_nowait((500, "tokenization failed"))
        else:
            self.streams[stream.id] = stream
            stream.ready.put_nowait((200, "accepted"))

    def _run(self):
        while not self.stop.is_set():
            # A bounded drain ensures continuous arrivals cannot starve model steps.
            for _ in range(self.commands.maxsize):
                try:
                    self._accept(self.commands.get_nowait())
                except queue.Empty:
                    break
            for stream in list(self.streams.values()):
                if stream.cancelled.is_set():
                    self.engine.cancel(stream.id)
            if self.engine.busy:
                try:
                    self.engine.step()
                except Exception:
                    logging.exception("model step failed; request terminated")
            else:
                self.stop.wait(.001)
        for stream in list(self.streams.values()):
            self.engine.cancel(stream.id)
        while True:
            try:
                self.commands.get_nowait().ready.put_nowait((503, "server stopping"))
            except queue.Empty:
                break

    def close(self):
        self.stop.set()
        self.thread.join(timeout=30)


def handler_for(worker):
    class Handler(BaseHTTPRequestHandler):
        # Bound concurrent handlers that can submit model work. This is a local demo.
        slots = threading.BoundedSemaphore(worker.engine.config.max_pending)

        def log_message(self, *_args):
            pass  # Do not log user prompts or token output.

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            body = json.dumps(dict(status="ok", backend=worker.backend.metadata["backend"])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            if not self.slots.acquire(blocking=False):
                self.send_error(429, "too many streams")
                return
            stream = None
            try:
                self.connection.settimeout(30)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    self.send_error(413, "body must be 1 to 65536 bytes")
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
                prompt, max_tokens = payload.get("prompt"), payload.get("max_tokens", 64)
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("nonempty string prompt required")
                if type(max_tokens) is not int or not 1 <= max_tokens <= 256:
                    raise ValueError("max_tokens must be an integer between 1 and 256")
                stream = Stream(uuid.uuid4().hex, prompt, max_tokens)
                worker.submit(stream)
                try:
                    status, message = stream.ready.get(timeout=30)
                except queue.Empty:
                    self.send_error(503, "admission timed out")
                    return
                if status != 200:
                    self.send_error(status, message)
                    return
            except Overloaded as exc:
                self.send_error(429, str(exc))
                return
            except (ValueError, TypeError, UnicodeDecodeError):
                self.send_error(400, "invalid JSON or generation parameters")
                return
            except OSError:
                return
            finally:
                # Successful streams retain their slot until the SSE loop exits.
                if stream is None or 'status' not in locals() or status != 200:
                    if stream is not None:
                        stream.cancelled.set()
                    self.slots.release()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                while True:
                    try:
                        event = stream.events.get(timeout=5)
                        self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    except queue.Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        event = {}
                    self.wfile.flush()
                    if event.get("type") == "done":
                        break
            except OSError:
                pass
            finally:
                stream.cancelled.set()
                self.slots.release()

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["numpy", "mlx"], default="numpy")
    p.add_argument("--model")
    p.add_argument("--policy", choices=["blocking", "chunked"], default="chunked")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()
    backend = make_backend(args.backend, args.model)
    worker = Worker(backend, Config(policy=args.policy, chunk_size=args.chunk_size,
                                    token_budget=args.chunk_size + 4))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(worker))
    worker.thread.start()
    print(f"Listening on http://127.0.0.1:{args.port}; backend={args.backend}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.close()


if __name__ == "__main__":
    main()
