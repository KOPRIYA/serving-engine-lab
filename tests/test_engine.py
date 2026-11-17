import unittest

from serving_engine.engine import Config, Engine, Overloaded, Request


class Fake:
    eos_ids = {99}

    def __init__(self):
        self.calls = []

    def new_cache(self):
        return []

    def forward(self, tokens, cache):
        cache.extend(tokens)
        self.calls.append(list(tokens))
        return sum(cache) % 101


def request(name="r", n=10, out=3):
    return Request(name, [1] * n, out, 0, ignore_eos=True)


def drain(engine):
    for _ in range(10000):
        if not engine.busy:
            return
        engine.step()
    raise AssertionError("scheduler failed to make progress")


class SchedulerTests(unittest.TestCase):
    def test_decode_precedes_new_prefill(self):
        b = Fake()
        e = Engine(b, Config(chunk_size=4))
        e.submit(request("short", n=1))
        e.step()
        e.submit(request("long", n=20))
        b.calls.clear()
        e.step()
        self.assertEqual([len(x) for x in b.calls], [1, 4])

    def test_blocking_prefill_precedes_decode(self):
        b = Fake()
        e = Engine(b, Config(policy="blocking"))
        e.submit(request("short", n=1))
        e.step()
        e.submit(request("long", n=20))
        b.calls.clear()
        e.step()
        self.assertEqual([len(x) for x in b.calls], [20])

    def test_budget_with_competing_decodes(self):
        e = Engine(Fake(), Config(chunk_size=100, token_budget=5, max_active=4))
        for i in range(4):
            e.submit(request(str(i), n=30, out=10))
        while e.busy:
            self.assertLessEqual(e.step(), 5)

    def test_round_robin_prefill(self):
        e = Engine(Fake(), Config(chunk_size=2))
        a, b = request("a"), request("b")
        e.submit(a)
        e.submit(b)
        e.step()
        e.step()
        self.assertEqual((a.cursor, b.cursor), (2, 2))

    def test_memory_reservation_and_release(self):
        e = Engine(Fake(), Config(kv_token_capacity=13))
        a, b = request("a"), request("b")
        e.submit(a)
        e.submit(b)
        e.step()
        self.assertEqual(e.reserved, 13)
        self.assertIsNone(b.admitted)
        drain(e)
        self.assertEqual(e.reserved, 0)
        self.assertIsNone(a.cache)
        self.assertIsNone(b.cache)

    def test_cancel_pending_and_active(self):
        e = Engine(Fake(), Config(max_active=1))
        a, b = request("a"), request("b")
        e.submit(a)
        e.submit(b)
        e.step()
        self.assertTrue(e.cancel("b"))
        self.assertTrue(e.cancel("a"))
        self.assertFalse(e.cancel("unknown"))
        self.assertEqual((a.reason, b.reason), ("cancelled", "cancelled"))
        self.assertEqual(e.reserved, 0)
        self.assertFalse(e.busy)

    def test_empty_and_oversized(self):
        e = Engine(Fake(), Config(max_context=12))
        for r in [request(n=0), request(n=10), request(out=0)]:
            with self.assertRaises(ValueError):
                e.submit(r)

    def test_duplicate_and_queue_bound(self):
        e = Engine(Fake(), Config(max_pending=1))
        e.submit(request())
        with self.assertRaises(ValueError):
            e.submit(request())
        with self.assertRaises(Overloaded):
            e.submit(request("other"))

    def test_eos_and_length(self):
        for ignore, count, reason in [(False, 1, "eos"), (True, 2, "length")]:
            b = Fake()
            b.forward = lambda tokens, cache: 99
            e = Engine(b)
            r = Request("a", [1], 2, 0, ignore_eos=ignore)
            e.submit(r)
            drain(e)
            self.assertEqual((len(r.tokens), r.reason), (count, reason))

    def test_error_releases_cache(self):
        b = Fake()

        def broken(tokens, cache):
            raise RuntimeError("inference failure")

        b.forward = broken
        e = Engine(b)
        r = request()
        e.submit(r)
        with self.assertRaises(RuntimeError):
            e.step()
        self.assertEqual(r.reason, "error")
        self.assertEqual(e.reserved, 0)
        self.assertIsNone(r.cache)

    def test_cache_allocation_failure_does_not_block_following_requests(self):
        b = Fake()
        calls = 0

        def allocate():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise MemoryError("cache allocation failed")
            return []

        b.new_cache = allocate
        finished = []
        e = Engine(b, on_done=lambda r: finished.append(r.id))
        active, failed, following = request("active"), request("failed"), request("following")
        for r in (active, failed, following):
            e.submit(r)
        with self.assertRaises(MemoryError):
            e.step()
        self.assertEqual(failed.reason, "error")
        self.assertIsNotNone(failed.finished)
        self.assertIsNone(failed.cache)
        self.assertEqual(e.reserved, active.reservation)
        self.assertEqual([r.id for r in e.pending], ["following"])
        drain(e)
        self.assertEqual(finished.count("failed"), 1)
        self.assertEqual((active.reason, following.reason), ("length", "length"))
        self.assertEqual(e.reserved, 0)

    def test_one_token_and_chunk_boundaries(self):
        for n in [1, 3, 4, 5, 8, 9]:
            outputs = []
            for policy in ["blocking", "chunked"]:
                e = Engine(Fake(), Config(policy=policy, chunk_size=4))
                r = request(n=n, out=1)
                e.submit(r)
                drain(e)
                outputs.append(r.tokens)
                self.assertEqual(len(r.tokens), 1)
            self.assertEqual(*outputs)

    def test_no_starvation_under_decode_load(self):
        e = Engine(Fake(), Config(chunk_size=2, max_active=4, token_budget=5))
        for i in range(3):
            e.submit(request(str(i), n=1, out=100))
            e.step()
        r = request("long", n=20)
        e.submit(r)
        for _ in range(12):
            e.step()
        self.assertEqual(r.cursor, 20)

    def test_long_prefill_stall_with_deterministic_clock(self):
        # Logical time demonstrates causality; these are NOT hardware measurements.
        gaps = {}
        for policy in ("blocking", "chunked"):
            now = [0]
            b = Fake()
            original_forward = b.forward

            def forward(tokens, cache):
                now[0] += len(tokens)
                return original_forward(tokens, cache)

            b.forward = forward
            e = Engine(b, Config(policy=policy, chunk_size=4), clock=lambda: now[0])
            short = request("short", n=1, out=3)
            e.submit(short)
            e.step()
            e.submit(request("long", n=20))
            while len(short.tokens) < 3:
                e.step()
            gaps[policy] = [b - a for a, b in zip(short.token_times, short.token_times[1:])]
            drain(e)
        self.assertEqual(gaps["blocking"], [21, 2])
        self.assertEqual(gaps["chunked"], [1, 5])

    def test_config_validation(self):
        for kwargs in [dict(policy="bad"), dict(chunk_size=0), dict(token_budget=4),
                       dict(max_pending=0), dict(max_active=-1)]:
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_completed_request_cannot_be_reused(self):
        e = Engine(Fake())
        r = request()
        e.submit(r)
        drain(e)
        with self.assertRaises(ValueError):
            e.submit(r)


if __name__ == "__main__":
    unittest.main()
