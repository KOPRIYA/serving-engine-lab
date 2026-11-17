"""Counterbalanced in-process replay; exports every timestamp, token, and model call."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep

from .backends import make_backend
from .engine import Config, Engine, Request


def percentile(values, p):
    """Linear interpolation, including single-element samples."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def metrics(rows, duration):
    ttft = [(r["token_times"][0] - r["arrival"]) * 1000 for r in rows]
    itl = [(b - a) * 1000 for r in rows
           for a, b in zip(r["token_times"], r["token_times"][1:])]
    worst = [max((b - a) * 1000 for a, b in zip(r["token_times"], r["token_times"][1:]))
             for r in rows if len(r["token_times"]) > 1]
    return dict(ttft_p50_ms=percentile(ttft, 50), ttft_p95_ms=percentile(ttft, 95),
                itl_p50_ms=percentile(itl, 50), itl_p95_ms=percentile(itl, 95),
                itl_p99_ms=percentile(itl, 99), itl_max_ms=max(itl) if itl else None,
                request_worst_itl_p95_ms=percentile(worst, 95),
                output_tokens_per_second=sum(len(r["tokens"]) for r in rows) / duration,
                duration_seconds=duration, requests=len(rows), itl_samples=len(itl))


def workloads(backend, short, long, output):
    # Repeat a fixed textual fixture then slice token IDs for exact input lengths.
    seed = backend.encode("Explain how a cache helps a computer reuse previous work. ")

    def prompt(n):
        return (seed * (n // len(seed) + 1))[:n]

    specs = {
        "mixed": [(0, short), (0, short), (.025, long), (.025, long),
                  (.04, short), (.06, short)],
        "short_only": [(i * .01, short) for i in range(6)],
        "long_only": [(0, long), (0, long)],
    }
    return {name: [dict(id=f"{name}-{i}", arrival=a, prompt=prompt(n), max_tokens=output)
                   for i, (a, n) in enumerate(spec)] for name, spec in specs.items()}


def run_once(backend, config, workload):
    engine = Engine(backend, config, trace=True)
    start = perf_counter()
    requests = [Request(x["id"], list(x["prompt"]), x["max_tokens"], start + x["arrival"],
                        ignore_eos=True) for x in workload]
    cursor = 0
    while cursor < len(requests) or engine.busy:
        now = perf_counter()
        while cursor < len(requests) and requests[cursor].arrival <= now:
            engine.submit(requests[cursor])
            cursor += 1
        if engine.busy:
            engine.step()
        elif cursor < len(requests):
            sleep(max(0, requests[cursor].arrival - perf_counter()))
    duration = perf_counter() - start
    rows = []
    for r in requests:
        assert r.reason == "length" and len(r.tokens) == r.max_tokens
        rows.append(dict(id=r.id, arrival=r.arrival - start, admitted=r.admitted - start,
                         finished=r.finished - start, prompt_tokens=len(r.prompt),
                         tokens=r.tokens, token_times=[t - start for t in r.token_times],
                         finish_reason=r.reason))
    trace = [{**s, "start": s["start"] - start, "end": s["end"] - start}
             for s in engine.trace]
    return dict(config=asdict(config), metrics=metrics(rows, duration), requests=rows, trace=trace)


def fingerprint(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "serving_engine").glob("*.py"))}


def report(payload):
    runs = payload["runs"]
    lines = ["# Measured benchmark", "", f"Backend: `{payload['backend']['backend']}`.",
             "Each table entry is the median of per-round metrics; raw rounds are in raw.json.",
             "These are in-process wall-clock measurements, not HTTP client measurements.", "",
             "| Workload | Policy | TTFT p95 ms | ITL p95 ms | ITL max ms | Output tok/s |",
             "|---|---|---:|---:|---:|---:|"]
    for workload in payload["workloads"]:
        for label in payload["variants"]:
            selected = [r["metrics"] for r in runs if r["workload"] == workload and r["label"] == label]
            vals = [statistics.median(r[k] for r in selected) for k in
                    ("ttft_p95_ms", "itl_p95_ms", "itl_max_ms", "output_tokens_per_second")]
            lines.append(f"| {workload} | {label} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += ["", "## Output consistency", "",
              f"Exact sequence matches against blocking in the same round: "
              f"{payload['consistency']['matches']}/{payload['consistency']['comparisons']}.",
              "Matching greedy tokens checks scheduling consistency on this fixture; "
              "it is not a task-quality evaluation.", "",
              "## Interpretation", "",
              "- NumPy runs use an untrained random-weight model on CPU. Do not claim Qwen/GPU results from them.",
              "- MLX results, if present, use the local model described in raw.json.",
              "- Throughput includes arrival gaps; model loading, tokenization and warmup are excluded.",
              "- EOS is ignored to give every request the same fixed output length.",
              "- A pooled ITL percentile can hide rare stalls; inspect maximum ITL and per-request traces.",
              "- Small samples and one machine do not establish production or multi-GPU performance.",
              "- Compare TTFT and throughput as well as ITL: chunking can trade one for another.", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["numpy", "mlx"], default="numpy")
    p.add_argument("--model")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--chunks", type=int, nargs="+", default=[64, 256])
    p.add_argument("--short", type=int, default=48)
    p.add_argument("--long", type=int, default=1024)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--out", type=Path, default=Path("results/local"))
    args = p.parse_args(argv)
    if min(args.rounds, args.short, args.long, args.output_tokens, *args.chunks) < 1:
        p.error("counts and lengths must be positive")
    if args.output_tokens < 2:
        p.error("at least two output tokens required to measure ITL")
    if len(set(args.chunks)) != len(args.chunks):
        p.error("chunk sizes must be distinct")
    if max(args.short, args.long) + max(args.output_tokens, 8) > 4096:
        p.error("prompt + output (including warmup) must fit the 4096 context limit")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        p.error("output directory must be empty; preserve prior evidence")
    backend = make_backend(args.backend, args.model)
    fixtures = workloads(backend, args.short, args.long, args.output_tokens)
    variants = {"blocking": Config(policy="blocking")}
    variants.update({f"chunked-{c}": Config(chunk_size=c, token_budget=c + 4) for c in args.chunks})
    # Prime both long prefill and decode paths with each policy, outside measurements.
    for config in variants.values():
        run_once(backend, config, [dict(id="warmup", arrival=0,
                                      prompt=fixtures["long_only"][0]["prompt"], max_tokens=8)])
    root = Path(__file__).resolve().parent.parent
    payload = dict(schema_version=1, timestamp=datetime.now(timezone.utc).isoformat(),
                   backend=backend.metadata, python=platform.python_version(),
                   platform=dict(system=platform.system(), release=platform.release(),
                                 machine=platform.machine(), cpu_count=os.cpu_count()),
                   packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                   source_sha256=fingerprint(root), rounds=args.rounds,
                   workloads=fixtures, variants=list(variants), runs=[])
    payload["thread_environment"] = {k: os.environ.get(k) for k in
                                     ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                      "VECLIB_MAXIMUM_THREADS")}
    if platform.system() == "Darwin":
        for key, syskey in [("cpu", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")]:
            try:
                payload["platform"][key] = subprocess.check_output(
                    ["sysctl", "-n", syskey], text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                payload["platform"][key] = "unavailable in execution environment"
    if args.model:
        # Hash configs and tokenizer only; large weight hashes are supplied by the user separately.
        payload["model_files_sha256"] = {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(args.model).glob("*.json")}
    args.out.mkdir(parents=True, exist_ok=True)
    for round_id in range(args.rounds):
        labels = list(variants)
        # Latin rotation counterbalances position across three default variants.
        labels = labels[round_id % len(labels):] + labels[:round_id % len(labels)]
        for workload, fixture in fixtures.items():
            for label in labels:
                result = run_once(backend, variants[label], fixture)
                result.update(round=round_id, workload=workload, label=label)
                payload["runs"].append(result)
                (args.out / "raw.json").write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps({k: result[k] for k in ("round", "workload", "label", "metrics")}),
                      flush=True)
    matches = comparisons = 0
    failures = []
    for run in payload["runs"]:
        if run["label"] == "blocking":
            continue
        baseline = next(r for r in payload["runs"] if r["label"] == "blocking" and
                        r["round"] == run["round"] and r["workload"] == run["workload"])
        for ref, actual in zip(baseline["requests"], run["requests"]):
            equal = ref["tokens"] == actual["tokens"]
            matches += int(equal)
            comparisons += 1
            if not equal:
                failures.append(dict(round=run["round"], variant=run["label"], request=actual["id"]))
    payload["consistency"] = dict(matches=matches, comparisons=comparisons, failures=failures)
    (args.out / "raw.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out / "REPORT.md").write_text(report(payload))
    if failures:
        raise SystemExit("Output mismatch: evidence saved; investigate before claiming consistency")


if __name__ == "__main__":
    main()
