"""Optional real-model check: never silently substitute a CPU fixture for MLX."""
import os
import unittest

from serving_engine.backends import MLXBackend
from serving_engine.engine import Config, Engine, Request


@unittest.skipUnless(os.environ.get("SERVING_ENGINE_MLX_MODEL"), "Metal/model integration is opt-in")
class MLXTests(unittest.TestCase):
    def test_interleaved_outputs_equal_isolated_generation(self):
        backend = MLXBackend(os.environ["SERVING_ENGINE_MLX_MODEL"])
        prompts = ["Explain why KV caching speeds up language model generation.",
                   "Write a Python function to add two integers and explain its arguments.",
                   "Summarize this text: " + "A long prompt competes with active decoding. " * 40]
        expected = []
        for i, prompt in enumerate(prompts):
            request = Request(str(i), backend.encode(prompt), 16, 0, ignore_eos=True)
            engine = Engine(backend, Config(policy="blocking"))
            engine.submit(request)
            while engine.busy:
                engine.step()
            expected.append(request.tokens)
        requests = [Request(str(i), backend.encode(prompt), 16, 0, ignore_eos=True)
                    for i, prompt in enumerate(prompts)]
        engine = Engine(backend, Config(chunk_size=16))
        for request in requests:
            engine.submit(request)
        while engine.busy:
            engine.step()
        self.assertEqual([r.tokens for r in requests], expected)


if __name__ == "__main__":
    unittest.main()

