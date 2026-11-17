"""Real CPU reference transformer and optional pretrained Apple GPU backend."""
import importlib.metadata
import json
from pathlib import Path


class NumpyBackend:
    """Random-weight, two-layer causal transformer; a correctness/CPU fixture only.

    Byte vocabulary, sinusoidal positions, pre-norm attention and GELU MLP.
    Each layer keeps its own actual K/V arrays. No sleeps or simulated timings.
    This model has NOT been trained and generates meaningless text.
    """
    eos_ids = set()

    def __init__(self, width=64, layers=2, seed=17):
        import numpy as np
        self.np, self.width, self.layers = np, width, layers
        rng = np.random.default_rng(seed)

        def weight(a, b):
            return (rng.standard_normal((a, b)) / a**0.5).astype(np.float32)

        self.embedding = weight(256, width)
        self.weights = [(weight(width, 3 * width), weight(width, width),
                         weight(width, 4 * width), weight(4 * width, width))
                        for _ in range(layers)]
        self.head = weight(width, 256)
        self.metadata = dict(backend="numpy", device="CPU", model="random-transformer",
                             trained=False, width=width, layers=layers, seed=seed,
                             numpy=np.__version__, precision="float32")

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")

    def new_cache(self):
        return [None] * self.layers

    def logits(self, tokens, cache):
        np, d = self.np, self.width
        offset = 0 if cache[0] is None else len(cache[0][0])
        positions = np.arange(offset, offset + len(tokens), dtype=np.float32)[:, None]
        rates = np.exp(-np.arange(0, d, 2, dtype=np.float32) * np.float32(np.log(10000)) / d)
        pos = np.zeros((len(tokens), d), dtype=np.float32)
        pos[:, 0::2], pos[:, 1::2] = np.sin(positions * rates), np.cos(positions * rates)
        x = self.embedding[tokens] + pos

        def norm(a):
            return (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)

        for i, (qkv, out, up, down) in enumerate(self.weights):
            q, k, v = np.split(norm(x) @ qkv, 3, axis=-1)
            if cache[i] is not None:
                k = np.concatenate((cache[i][0], k))
                v = np.concatenate((cache[i][1], v))
            cache[i] = (k, v)
            scores = (q @ k.T) * np.float32(d**-0.5)
            future = np.arange(len(k))[None, :] > np.arange(offset, offset + len(tokens))[:, None]
            scores = np.where(future, -np.inf, scores)
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            x = x + (probs @ v) @ out
            h = norm(x) @ up
            h = 0.5 * h * (1 + np.tanh(np.float32((2 / np.pi)**0.5) * (h + 0.044715 * h**3)))
            x = x + h @ down
        return norm(x[-1:]) @ self.head

    def forward(self, tokens, cache):
        return int(self.logits(tokens, cache).argmax())


class MLXBackend:
    """Use an existing local MLX model directory; never download model weights implicitly."""
    def __init__(self, model_path):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        path = Path(model_path)
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("--model must be a local MLX model directory with config.json")
        self.mx, self.make_prompt_cache = mx, make_prompt_cache
        self.model, self.tokenizer = load(str(path))
        mx.eval(self.model.parameters())
        self.eos_ids = set(self.tokenizer.eos_token_ids)
        self.metadata = dict(backend="mlx", model=path.name, device=mx.metal.device_info(),
                             mlx=importlib.metadata.version("mlx"),
                             mlx_lm=importlib.metadata.version("mlx-lm"),
                             config=json.loads((path / "config.json").read_text()))

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def new_cache(self):
        return self.make_prompt_cache(self.model)

    def forward(self, tokens, cache):
        mx = self.mx
        logits = self.model(mx.array(tokens)[None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        # Lazy graph construction is not inference latency. Synchronize every event.
        mx.eval(token, [c.state for c in cache])
        return int(token.item())


def make_backend(name, model=None):
    if name == "numpy":
        return NumpyBackend()
    if name == "mlx":
        if not model:
            raise ValueError("--model required for MLX")
        return MLXBackend(model)
    raise ValueError(f"unknown backend: {name}")
