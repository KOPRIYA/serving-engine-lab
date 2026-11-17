import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from serving_engine.benchmark import main, metrics
from serving_engine.verify import verify

ROOT = Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def test_metrics_include_waiting_from_scheduled_arrival(self):
        rows = [dict(arrival=0, tokens=[1, 2, 3], token_times=[1, 1.1, 1.4]),
                dict(arrival=.5, tokens=[4, 5], token_times=[1.5, 1.7])]
        result = metrics(rows, 2)
        self.assertAlmostEqual(result['ttft_p95_ms'], 1000)
        self.assertAlmostEqual(result['itl_p50_ms'], 200)
        self.assertAlmostEqual(result['itl_p95_ms'], 290)
        self.assertAlmostEqual(result['itl_max_ms'], 300)
        self.assertEqual(result['output_tokens_per_second'], 2.5)

    def test_invalid_cli_fails_before_loading_model(self):
        for args in [['--output-tokens', '1'], ['--chunks', '64', '64'],
                     ['--long', '4090', '--output-tokens', '2']]:
            with self.subTest(args=args), patch('serving_engine.benchmark.make_backend') as load:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    main(args)
                self.assertEqual(exc.exception.code, 2)
                load.assert_not_called()


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads((ROOT / 'results/cpu-reference/raw.json').read_text())

    def test_reference_evidence(self):
        result = verify(self.reference)
        self.assertEqual(result['matches'], result['comparisons'])
        self.assertEqual(result['runs'], 27)

    def test_incomplete_or_modified_evidence_is_rejected(self):
        def remove_run(p):
            p['runs'].pop()

        def remove_request(p):
            p['runs'][0]['requests'].pop()

        def duplicate_run(p):
            p['runs'].append(copy.deepcopy(p['runs'][0]))

        def change_metric(p):
            p['runs'][0]['metrics']['ttft_p95_ms'] += 1

        def change_token(p):
            p['runs'][0]['requests'][0]['tokens'][0] += 1

        def change_time(p):
            p['runs'][0]['requests'][0]['token_times'][0] = -1

        for mutate in [remove_run, remove_request, duplicate_run, change_metric,
                       change_token, change_time]:
            with self.subTest(case=mutate.__name__):
                payload = copy.deepcopy(self.reference)
                mutate(payload)
                with self.assertRaises(ValueError):
                    verify(payload)


if __name__ == '__main__':
    unittest.main()
