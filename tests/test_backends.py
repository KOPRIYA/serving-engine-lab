import unittest

import numpy as np

from serving_engine.backends import NumpyBackend
from serving_engine.engine import Config, Engine, Request
from serving_engine.benchmark import percentile
from test_engine import drain


class BackendTests(unittest.TestCase):
    def test_chunked_logits_match_full_prefill(self):
        b = NumpyBackend()
        tokens = list(range(67))
        expected = b.logits(tokens, b.new_cache())
        self.assertEqual(expected.dtype, np.float32)
        for size in [1, 7, 16, 64, 100]:
            cache = b.new_cache()
            for start in range(0, len(tokens), size):
                actual = b.logits(tokens[start:start + size], cache)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            self.assertEqual(len(cache[0][0]), len(tokens))

    def test_decode_cache_matches_full_recomputation(self):
        b = NumpyBackend()
        tokens = list(range(32))
        cache = b.new_cache()
        b.logits(tokens, cache)
        for token in [3, 91, 7]:
            tokens.append(token)
            actual = b.logits([token], cache)
            expected = b.logits(tokens, b.new_cache())
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_interleaved_requests_match_isolated_output(self):
        b = NumpyBackend()
        expected = []
        for i in range(3):
            e = Engine(b, Config(policy="blocking"))
            r = Request(str(i), list(range(20 + i * 7)), 12, 0, ignore_eos=True)
            e.submit(r)
            drain(e)
            expected.append(r.tokens)
        e = Engine(b, Config(chunk_size=7))
        requests = [Request(str(i), list(range(20 + i * 7)), 12, 0, ignore_eos=True)
                    for i in range(3)]
        for r in requests:
            e.submit(r)
        drain(e)
        self.assertEqual([r.tokens for r in requests], expected)

    def test_percentile(self):
        self.assertIsNone(percentile([], 95))
        self.assertEqual(percentile([4], 95), 4)
        self.assertEqual(percentile([0, 10], 50), 5)


if __name__ == "__main__":
    unittest.main()
