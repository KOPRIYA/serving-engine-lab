import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "workflow", Path(__file__).resolve().parents[1] / "scripts/workflow.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def test_missing_model_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                workflow.model_files(Path(folder))

    def test_missing_shard_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text("{}")
            (root / "part1.safetensors").write_bytes(b"fixture")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
            with self.assertRaises(ValueError):
                workflow.model_files(root)

    def test_hashes_include_actual_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text("{}")
            weights = root / "model.safetensors"
            weights.write_bytes(b"first")
            self.assertIn(weights, workflow.model_files(root))
            before = workflow.digest(weights)
            weights.write_bytes(b"second")
            self.assertNotEqual(before, workflow.digest(weights))

    def test_failure_records_exit_and_retains_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = workflow.Run(root)
            with self.assertRaises(RuntimeError):
                run.stage("failure", [sys.executable, "-c", "print('retained'); raise SystemExit(7)"],
                          os.environ.copy())
            run.finish("failed", "expected test failure")
            state = json.loads((root / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stages"][0]["exit_code"], 7)
            self.assertIn("retained", (root / "failure.log").read_text())
            hashes = json.loads((root / "checksums.json").read_text())
            self.assertEqual(hashes["failure.log"], workflow.digest(root / "failure.log"))

    def test_source_hashes_exclude_runtime_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "serving_engine/__pycache__").mkdir(parents=True)
            (root / "serving_engine/engine.py").write_text("# source")
            (root / "serving_engine/__pycache__/engine.pyc").write_bytes(b"cache")
            (root / "results").mkdir()
            (root / "results/raw.json").write_text("{}")
            (root / "local-notes").mkdir()
            (root / "local-notes/private.md").write_text("private notes")
            self.assertEqual(list(workflow.source_hashes(root)), ["serving_engine/engine.py"])

    def test_summary_keeps_ttft_regression_visible(self):
        def run(label, itl, ttft, throughput):
            return {"workload": "mixed", "label": label,
                    "metrics": {"itl_max_ms": itl, "ttft_p95_ms": ttft,
                                "output_tokens_per_second": throughput}}
        payload = {"backend": {"backend": "numpy"}, "variants": ["blocking", "chunked-64"],
                   "runs": [run("blocking", 10, 20, 100), run("chunked-64", 1, 24, 90)],
                   "consistency": {"matches": 2, "comparisons": 2}}
        result = workflow.experiment_summary(payload)
        self.assertIn("-90.0%", result)
        self.assertIn("+20.0%", result)
        self.assertIn("-10.0%", result)
        self.assertIn("untrained NumPy transformer, CPU only", result)


if __name__ == "__main__":
    unittest.main()
