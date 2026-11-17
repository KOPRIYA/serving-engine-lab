"""Single-owner scheduler. Model calls are serialized; there is no fused batching.

Both policies use the same backend, request limits, and greedy decoding.
The only treatment is scheduling: blocking prefill versus decode-first chunks.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Protocol


class Backend(Protocol):
    eos_ids: set[int]

    def new_cache(self) -> Any: ...
    def forward(self, tokens: list[int], cache: Any) -> int: ...


class Overloaded(RuntimeError):
    """Admission rejected; caller may retry after capacity is released."""


@dataclass(frozen=True)
class Config:
    policy: str = "chunked"
    chunk_size: int = 64
    token_budget: int = 128
    max_active: int = 4
    max_pending: int = 32
    max_context: int = 4096
    kv_token_capacity: int = 16384

    def __post_init__(self):
        if self.policy not in {"blocking", "chunked"}:
            raise ValueError("policy must be blocking or chunked")
        for name in ("chunk_size", "token_budget", "max_active", "max_pending",
                     "max_context", "kv_token_capacity"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.token_budget <= self.max_active:
            raise ValueError("token_budget must exceed max_active to prevent prefill starvation")


@dataclass
class Request:
    id: str
    prompt: list[int]
    max_tokens: int
    arrival: float
    ignore_eos: bool = False
    cursor: int = 0
    tokens: list[int] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)
    admitted: float | None = None
    finished: float | None = None
    reason: str | None = None
    cache: Any = None

    @property
    def reservation(self):
        # Conservative upper bound; final generated token need not enter the cache.
        return len(self.prompt) + self.max_tokens


class Engine:
    def __init__(self, backend: Backend, config: Config = Config(), *, clock=perf_counter,
                 on_token: Callable[[Request, int], None] | None = None,
                 on_done: Callable[[Request], None] | None = None, trace=False):
        self.backend, self.config, self.clock = backend, config, clock
        self.on_token, self.on_done = on_token, on_done
        self.pending = deque()
        self.active = OrderedDict()
        self.reserved = 0
        self.trace = [] if trace else None
        self._prefill_turns = deque()

    @property
    def busy(self):
        return bool(self.pending or self.active)

    def submit(self, request: Request):
        if not request.id or not request.prompt or request.max_tokens < 1:
            raise ValueError("nonempty id/prompt and positive max_tokens required")
        if request.admitted is not None or request.finished is not None or request.tokens:
            raise ValueError("request objects cannot be reused")
        if request.reservation > min(self.config.max_context, self.config.kv_token_capacity):
            raise ValueError("request exceeds context or KV token capacity")
        if request.id in self.active or any(r.id == request.id for r in self.pending):
            raise ValueError("duplicate live request id")
        if len(self.pending) >= self.config.max_pending:
            raise Overloaded("pending queue full")
        self.pending.append(request)

    def _finish(self, request, reason):
        if request.id in self.active:
            del self.active[request.id]
            self.reserved -= request.reservation
        self._prefill_turns = deque(x for x in self._prefill_turns if x != request.id)
        request.cache = None  # Release all sequence-owned KV arrays.
        request.reason, request.finished = reason, self.clock()
        if self.on_done:
            self.on_done(request)

    def cancel(self, request_id):
        if request_id in self.active:
            self._finish(self.active[request_id], "cancelled")
            return True
        for request in self.pending:
            if request.id == request_id:
                self.pending.remove(request)
                self._finish(request, "cancelled")
                return True
        return False

    def _admit(self):
        while self.pending and len(self.active) < self.config.max_active:
            request = self.pending[0]
            if self.reserved + request.reservation > self.config.kv_token_capacity:
                break  # FIFO admission; no small-request bypass.
            self.pending.popleft()
            try:
                cache = self.backend.new_cache()
            except Exception:
                # Fail this request once; do not leave it blocking FIFO admission.
                self._finish(request, "error")
                raise
            request.cache, request.admitted = cache, self.clock()
            self.reserved += request.reservation
            self.active[request.id] = request
            self._prefill_turns.append(request.id)

    def _forward(self, request, tokens, phase):
        start = self.clock()
        try:
            token = self.backend.forward(tokens, request.cache)
        except Exception:
            self._finish(request, "error")
            raise
        if self.trace is not None:
            self.trace.append(dict(request_id=request.id, phase=phase, input_tokens=len(tokens),
                                   start=start, end=self.clock()))
        return token

    def _emit(self, request, token):
        request.tokens.append(token)
        request.token_times.append(self.clock())
        if self.on_token:
            self.on_token(request, token)
        if not request.ignore_eos and token in self.backend.eos_ids:
            self._finish(request, "eos")
        elif len(request.tokens) >= request.max_tokens:
            self._finish(request, "length")

    def _prefill(self, request, count):
        end = min(request.cursor + count, len(request.prompt))
        token = self._forward(request, request.prompt[request.cursor:end], "prefill")
        consumed = end - request.cursor
        request.cursor = end
        if end == len(request.prompt):
            self._prefill_turns = deque(x for x in self._prefill_turns if x != request.id)
            self._emit(request, token)
        return consumed

    def step(self):
        """One scheduling iteration; returns the number of model input tokens processed."""
        self._admit()
        if not self.active:
            return 0
        if self.config.policy == "blocking" and self._prefill_turns:
            r = self.active[self._prefill_turns[0]]
            return self._prefill(r, len(r.prompt) - r.cursor)

        used = 0
        # Snapshot prevents a newly-prefilled request from also decoding in this tick.
        for r in list(self.active.values()):
            if r.cursor == len(r.prompt):
                token = self._forward(r, [r.tokens[-1]], "decode")
                used += 1
                self._emit(r, token)

        if self.config.policy == "chunked" and self._prefill_turns:
            r = self.active[self._prefill_turns[0]]
            count = min(self.config.chunk_size, self.config.token_budget - used)
            used += self._prefill(r, count)
            if r.id in self._prefill_turns:
                self._prefill_turns.rotate(-1)
        return used

