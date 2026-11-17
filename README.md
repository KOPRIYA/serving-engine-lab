# Serving Engine Lab

An inference systems experiment studying **prefill/decode interference**.
It compares a blocking scheduler with decode-first, token-budgeted, round-robin chunked prefill.
Both policies use the same model, precision, per-request KV cache and greedy decoding.

Long prompt processing can delay tokens for requests already decoding. This project makes that
interference observable and measures the latency/throughput trade-offs of bounded prefill chunks.

## Implementation

- Decode-first scheduling with round-robin progress for unfinished prompts.
- FIFO admission, active/pending limits and conservative KV token reservations.
- Per-request cache ownership, EOS/length stopping, cancellation and error cleanup.
- NumPy reference transformer and optional pretrained MLX backend on Apple Silicon.
- Local HTTP/SSE token-ID streaming with a single inference-owner thread.
- Reproducible benchmark runner with correctness gates, source/model hashes and raw traces.

Model calls are serialized; this is not fused continuous batching, paged KV allocation or a
production engine. Chunked prefill is an established technique, not a novel algorithm claimed
by this project. See [design and invariants](docs/DESIGN.md) and [primary references](docs/SOURCES.md).

## Code map

| Area | Responsibility |
|---|---|
| `serving_engine/engine.py` | Scheduling, admission, token reservations and request lifecycle |
| `serving_engine/backends.py` | NumPy reference and pretrained MLX execution |
| `serving_engine/server.py` | Single inference owner and HTTP/SSE streaming |
| `serving_engine/benchmark.py` | Workload replay, metrics, raw traces and output consistency |
| `scripts/workflow.py` | Validation gates, provenance and immutable local run bundles |
| `tests/` | Scheduler, cache, server and workflow regression coverage |

## Review the contribution

Start with `Engine.step()` in [engine.py](serving_engine/engine.py): the blocking branch
finishes a prompt before decoding; the chunked branch services existing decodes first and
spends the remaining token budget on one round-robin prefill chunk.
[Scheduler tests](tests/test_engine.py) exercise ordering, fairness, capacity and failure cleanup.

Verify the recorded evidence without a GPU or model download:

```bash
python -m serving_engine.verify results/apple-m5-reference/raw.json
python -m serving_engine.verify results/cpu-reference/raw.json
```

The verifier recomputes metrics from timestamps, checks the complete run/request matrix and
compares output sequences by request ID. This establishes internal consistency, not independent
measurement or authenticity. Historical source hashes may differ from the current source.
See [experimental design](docs/DESIGN.md) for controls and limitations.

## Setup

Python 3.11+ is required. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For Apple Silicon, install the MLX extras and place a converted model at `models/qwen/int4`:

```bash
python -m pip install -e '.[mlx]'
```

Models are local inputs and are not downloaded by the runner. See [model setup](docs/APPLE_GPU.md).

## Validate and benchmark

```bash
# Portable CPU reference validation
bash scripts/validate.sh --backend numpy

# Metal availability and real-model correctness
bash scripts/validate.sh

# Full GPU workflow: validation, three-round benchmark and report
bash scripts/benchmark.sh --open

# Short CPU workflow smoke test
bash scripts/benchmark.sh --backend numpy --rounds 1 --short 16 --long 64 --output-tokens 4
```

The scripts locate the repository root and use `.venv/bin/python`. Override the interpreter
with `PYTHON_BIN` or the model with `--model`. Every run creates a fresh ignored `results/local-*`
directory. Failures preserve logs and return a nonzero exit status. There is no automatic
fallback from GPU to CPU. See [workflow options and output files](docs/RUN_WORKFLOW.md).

Successful benchmarks produce `benchmark/REPORT.md`, `benchmark/raw.json`,
`EXPERIMENT_SUMMARY.md`, status, logs, checksums and source/model provenance.

## Measured Apple M5 reference

Qwen2.5-1.5B-Instruct, 4-bit affine weights, MLX 0.32.4 / MLX-LM 0.32.2, Apple M5 with
16 GiB unified memory. Three rounds, fixed 64-token outputs, mixed short/long arrivals and
up to four active requests. Values below are medians of per-round metrics.

| Mixed-workload metric | Blocking | Chunk 64 | Chunk 256 |
|---|---:|---:|---:|
| Maximum ITL per run (ms) | 685.735 | 78.629 | 119.875 |
| ITL p95 (ms) | 37.013 | 63.607 | 66.977 |
| TTFT p95 (ms) | 2954.821 | 3373.167 | 3140.737 |
| Aggregate output tokens/s | 93.287 | 73.214 | 87.246 |

Chunk 64 reduced the longest stalls but worsened p95 ITL, p95 TTFT and throughput in this
workload. This is a trade-off, not a universal speedup. All 84 sequence comparisons matched
blocking. Token equality on these fixtures is not a task-quality evaluation.

The 64-token setting targets severe stalls: maximum ITL fell 88.5%, with a 21.5% throughput
loss. The 256-token setting is a different trade-off: maximum ITL fell 82.5%, with a 6.5%
throughput loss. Both worsened mixed-workload p95 TTFT and p95 ITL. Neither result alone
justifies a production default; policy selection needs workload-specific latency requirements
and sustained-load measurements.

[Full GPU report](results/apple-m5-reference/REPORT.md) ·
[Raw GPU evidence](results/apple-m5-reference/raw.json) ·
[Provenance and limitations](results/apple-m5-reference/PROVENANCE.md)

The original [CPU reference report](results/cpu-reference/REPORT.md) and
[raw CPU evidence](results/cpu-reference/raw.json) use an untrained two-layer NumPy transformer.
CPU and GPU measurements must not be combined into one performance claim. The GPU reference
predates subsequent failure-handling and benchmark-validation changes; its recorded source hashes
identify the measured version. No new GPU performance result is claimed for the current source.
Measurements are in-process, exclude model loading/tokenization, and do not establish production
or HTTP SLOs.

## Serve locally

```bash
bash scripts/serve.sh
```

In another Terminal:

```bash
curl http://127.0.0.1:8001/health
curl -N http://127.0.0.1:8001/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain KV caching.","max_tokens":32}'
```

This local demo binds to 127.0.0.1:8001, leaving port 8000 available for another server.
Override it with `PORT=8002 bash scripts/serve.sh` or `bash scripts/serve.sh --port 8002`.
The benchmark runner calls the model directly and does not use either HTTP server.
This server loads its own model; it does not reuse the model hosted on port 8000. Events carry token IDs followed by a terminal `done` event;
decode accumulated IDs with the matching tokenizer. This is not an OpenAI-compatible API.
For the untrained CPU backend, use `bash scripts/serve.sh --backend numpy`.

## Repository hygiene

```bash
python scripts/audit_public.py
```

The audit checks the candidate public file set using Git ignore rules, including files already
tracked despite those rules. It does not change the index or publish anything. Review findings
before staging. See [contribution guidelines](CONTRIBUTING.md).

Local models, virtual environments, generated caches, local runs, machine configuration,
private notes and installer backups are ignored. Only reviewed reference results are public.
Private notes belong in `local-notes/` and are excluded from new source snapshots.

MIT license. Model weights retain their own licenses. Developed with coding-agent assistance;
implementation and measured evidence are available for review. No upstream engine patch,
custom CUDA kernel, multi-GPU result or production rollout is claimed.
