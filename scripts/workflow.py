#!/usr/bin/env python3
"""Run correctness gates and capture reproducible local benchmark evidence.

No implicit downloads, CPU substitution for MLX, result overwrites, or publishing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_files(root):
    files = []
    for name in ("serving_engine", "scripts", "tests", "docs", ".github"):
        files.extend(p for p in (root / name).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files.extend(root / name for name in
                 ("README.md", "pyproject.toml", "LICENSE", "CONTRIBUTING.md", ".gitignore")
                 if (root / name).is_file())
    return sorted(files)


def source_hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in source_files(root)}


def model_files(model):
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError(f"Local model directory with config.json required: {model}")
    weights = sorted(model.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"No safetensors weights found in {model}")
    index = model / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text())["weight_map"].values())
        for name in names:
            target = (model / name).resolve()
            if not target.is_relative_to(model.resolve()) or not target.is_file():
                raise ValueError(f"Missing or invalid indexed weight shard: {name}")
    return sorted(set(weights + list(model.glob("*.json")) + list(model.glob("*.jinja"))))


def git_state(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve():
            return {"available": False, "reason": "project has no own repository"}
        return {"available": True, "commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "reason": "no committed Git repository available"}


class Run:
    def __init__(self, folder):
        self.folder = folder
        self.status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                       "stages": []}
        self.save()

    def save(self):
        write_json(self.folder / "status.json", self.status)

    def stage(self, name, command, env):
        stage = {"name": name, "status": "running", "log": f"{name}.log"}
        self.status["stages"].append(stage)
        self.save()
        print(f"\n[{name}]", flush=True)
        with (self.folder / stage["log"]).open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
                try:
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
        stage.update(status="passed" if code == 0 else "failed", exit_code=code)
        self.save()
        if code:
            raise RuntimeError(f"{name} failed (exit {code}); see {stage['log']}")

    def finish(self, status, error=None):
        self.status.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat())
        if error:
            self.status["error"] = error
        if status != "passed":
            for stage in self.status["stages"]:
                if stage["status"] == "running":
                    stage["status"] = "interrupted" if status == "interrupted" else "failed"
        self.save()
        hashes = {str(p.relative_to(self.folder)): digest(p)
                  for p in sorted(self.folder.rglob("*"))
                  if p.is_file() and p.name != "checksums.json"}
        write_json(self.folder / "checksums.json", hashes)


def experiment_summary(payload):
    """Describe all measured variants without selecting only the largest speedup."""
    cpu = payload["backend"]["backend"] == "numpy"
    lines = ["# Experiment summary", "",
             "Evidence scope: " + ("untrained NumPy transformer, CPU only." if cpu else
                                    "local pretrained MLX model; see raw.json for device/config."),
             "", "Bottleneck: long prompt prefill blocks token emission for active decodes.",
             "Change: decode-first scheduling with token-budgeted chunks, round-robin prefill, "
             "bounded admission and per-request KV cleanup.", "",
             "The model and precision were held constant across scheduler variants. "
             "The engine interleaves serialized calls; it does not fuse request batches.", "",
             "## Mixed workload results", "",
             "Values are medians of per-round metrics; negative change means a reduction.", "",
             "| Variant | Max ITL ms | Change | TTFT p95 ms | Change | Output tok/s | Change |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def med(label, key):
        return statistics.median(r["metrics"][key] for r in payload["runs"]
                                 if r["workload"] == "mixed" and r["label"] == label)
    keys = ("itl_max_ms", "ttft_p95_ms", "output_tokens_per_second")
    base = [med("blocking", k) for k in keys]
    for label in payload["variants"]:
        cells = [label]
        for key, baseline in zip(keys, base):
            value = med(label, key)
            cells.extend([f"{value:.3f}", f"{(value / baseline - 1) * 100:+.1f}%"])
        lines.append("| " + " | ".join(cells) + " |")
    c = payload["consistency"]
    lines += ["", f"Exact output sequences matched: {c['matches']}/{c['comparisons']}.",
              "Output consistency is not a task-quality evaluation. See benchmark/REPORT.md "
              "for all three workloads and benchmark/raw.json for every run.", "",
              "Describe this as a local custom scheduler prototype. No production rollout, "
              "upstream vLLM/SGLang/TensorRT-LLM patch, custom CUDA kernel or multi-GPU result "
              "is established by this experiment.", "",
              "Reproduce all variants with the same weights and runtime. Publish reviewed metrics "
              "alongside source and model fingerprints; retain regressions as well as improvements.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "benchmark"])
    parser.add_argument("--backend", choices=["mlx", "numpy"], default="mlx")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--short", type=int, default=48)
    parser.add_argument("--long", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--open", action="store_true", help="open successful benchmark report on macOS")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or later required")
    if min(args.rounds, args.short, args.long, *args.chunks) < 1 or args.output_tokens < 2:
        parser.error("positive rounds/prompt/chunk sizes and at least two output tokens required")
    if max(args.short, args.long) + max(args.output_tokens, 8) > 4096:
        parser.error("prompt + output (including 8-token warmup) must fit the 4096 context limit")
    if len(set(args.chunks)) != len(args.chunks):
        parser.error("chunk sizes must be distinct")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(tempfile.mkdtemp(prefix=f"local-{args.action}-{args.backend}-{stamp}-", dir=results))
    run = Run(folder)
    print(f"Evidence directory: {folder}", flush=True)
    env = dict(os.environ)
    env.pop("SERVING_ENGINE_MLX_MODEL", None)
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(key, "1")
    try:
        initial = source_hashes(ROOT)
        for name in initial:
            dest = folder / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        provenance = {"schema_version": 1, "backend": args.backend, "action": args.action,
                      "python": platform.python_version(), "git": git_state(ROOT),
                      "source_sha256": initial,
                      "parameters": {k: v for k, v in vars(args).items() if k not in {"model", "open"}},
                      "thread_environment": {k: env[k] for k in
                                             ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                              "VECLIB_MAXIMUM_THREADS")}}
        write_json(folder / "provenance.json", provenance)
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--exclude-editable"], cwd=ROOT,
                                         env=env, text=True)
        (folder / "environment-lock.txt").write_text(freeze, encoding="utf-8")
        run.stage("unit-tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], env)
        model = None
        before = {}
        if args.backend == "mlx":
            model = (args.model or Path(os.environ.get("SERVING_ENGINE_MLX_MODEL", "models/qwen/int4")))
            model = (model if model.is_absolute() else ROOT / model).resolve()
            files = model_files(model)
            print("Hashing local model weights and configuration...", flush=True)
            before = {str(p.relative_to(model)): digest(p) for p in files}
            write_json(folder / "model-sha256.json", before)
            run.stage("metal-check", [sys.executable, "-c",
                      "import mlx.core as mx; print(mx.metal.device_info())"], env)
            gpu_env = {**env, "SERVING_ENGINE_MLX_MODEL": str(model)}
            run.stage("mlx-consistency", [sys.executable, "-m", "unittest", "discover", "-s", "tests",
                                          "-p", "test_mlx.py", "-v"], gpu_env)
        if args.action == "benchmark":
            command = [sys.executable, "-m", "serving_engine.benchmark", "--backend", args.backend,
                       "--rounds", str(args.rounds), "--chunks", *map(str, args.chunks),
                       "--short", str(args.short), "--long", str(args.long),
                       "--output-tokens", str(args.output_tokens), "--out", str(folder / "benchmark")]
            if model:
                command += ["--model", str(model)]
            run.stage("benchmark", command, env)
            run.stage("verify-evidence", [sys.executable, "-m", "serving_engine.verify",
                                         str(folder / "benchmark/raw.json")], env)
            payload = json.loads((folder / "benchmark/raw.json").read_text())
            if payload["consistency"]["matches"] != payload["consistency"]["comparisons"]:
                raise RuntimeError("Output consistency failed")
            if payload["consistency"]["comparisons"] == 0:
                raise RuntimeError("No output comparisons were performed")
            expected_runs = args.rounds * (len(args.chunks) + 1) * 3
            if len(payload["runs"]) != expected_runs:
                raise RuntimeError("Incomplete benchmark")
        if source_hashes(ROOT) != initial:
            raise RuntimeError("Project source changed during the run; retain evidence and rerun")
        if model and {str(p.relative_to(model)): digest(p) for p in model_files(model)} != before:
            raise RuntimeError("Model files changed during the run; retain evidence and rerun")
        if args.action == "benchmark":
            (folder / "EXPERIMENT_SUMMARY.md").write_text(experiment_summary(payload), encoding="utf-8")
        run.finish("passed")
    except KeyboardInterrupt:
        run.finish("interrupted", "Interrupted by user")
        return 130
    except Exception as exc:
        run.finish("failed", str(exc))
        print(f"FAILED: {exc}\nPartial evidence retained: {folder}", file=sys.stderr)
        return 1
    print(f"\nPASSED. Evidence: {folder}")
    if args.action == "benchmark":
        print(f"Experiment summary: {folder / 'EXPERIMENT_SUMMARY.md'}")
        if args.open:
            if sys.platform == "darwin":
                subprocess.run(["open", str(folder / "benchmark/REPORT.md")], check=False)
            else:
                print("--open is macOS-only; open benchmark/REPORT.md manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
