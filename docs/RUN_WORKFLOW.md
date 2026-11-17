# Reproducible validation and benchmark scripts

Run from a clone of this repository using Python 3.11+. The scripts locate the project root
automatically and do not contain personal filesystem paths. The default model is
`models/qwen/int4`, resolved relative to the project root, and the default interpreter is
`.venv/bin/python`. The scripts do not download or convert models.

## Setup (once)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mlx]'
```

Place a converted MLX model, tokenizer, config and safetensors weights in `models/qwen/int4`.
An optional safetensors index must reference existing shards. Model weights are ignored by Git.

## 1. Validate

```bash
bash scripts/validate.sh
```

Runs the ordinary test suite with the opt-in GPU test disabled, verifies local model files and
Metal availability, then explicitly enables the real-model output-consistency test. A skipped
GPU test in the ordinary suite is expected; the explicit second test must pass. A missing model,
failed test or unavailable Metal produces a nonzero exit and a retained failure record.

## 2. Benchmark and open the report

```bash
bash scripts/benchmark.sh --open
```

This command repeats validation so every benchmark has its own correctness evidence. It runs
three rounds of blocking, chunk-64 and chunk-256 across mixed/short-only/long-only workloads,
with 48/1,024 prompt tokens and fixed 64-token outputs. It never silently falls back to CPU.
`--open` opens the report only after successful validation, timing and integrity checks.

Both commands create unique directories below `results/local-*`; prior evidence is never
overwritten. The directory is printed at startup. If the process fails or is interrupted,
inspect `status.json` and the stage logs. A forcibly killed process may leave `status: running`;
that is incomplete evidence, not a pass.

## Output files

| File | Purpose |
|---|---|
| `status.json` | Passed, failed or interrupted status, with stage exit codes |
| `unit-tests.log` | Ordinary test output |
| `metal-check.log` | Metal hardware availability output |
| `mlx-consistency.log` | Pretrained-model output-consistency test output |
| `model-sha256.json` | Weight, config and tokenizer hashes |
| `environment-lock.txt` | pip freeze excluding editable installs; own source is captured separately |
| `provenance.json` | Parameters, source hashes, Python version and Git commit/dirty state when available |
| `source/` | Copy of the source, tests, scripts and documentation used for the run |
| `benchmark.log` | Benchmark process output |
| `benchmark/REPORT.md` | Readable metrics for all workloads and variants |
| `benchmark/raw.json` | Measured requests, token timestamps and model-call traces |
| `EXPERIMENT_SUMMARY.md` | Technical mixed-workload summary; created only after success |
| `checksums.json` | Hashes of the retained evidence files |

Validation-only runs omit benchmark outputs. CPU-only runs omit Metal/model files. The runner
hashes model files before and after execution and checks project source for changes. These checks
and weight hashing are outside timed inference. Environment logs can contain local paths or
package-origin information; review them before publishing. Checksums establish integrity against
accidental modification, not independent attestation of a performance claim.

## Overrides

```bash
# Another local MLX model; relative paths resolve from the repository root.
bash scripts/benchmark.sh --model models/another-model --rounds 6 --open

# Larger prompts, preserving the same model and precision across policies.
bash scripts/benchmark.sh --long 2048 --output-tokens 128

# Explicit CPU reference validation / workflow smoke test; never claim GPU results from it.
bash scripts/validate.sh --backend numpy
bash scripts/benchmark.sh --backend numpy --rounds 1 --short 16 --long 64 --output-tokens 4

# Interpreter override, useful in CI or when using an already activated environment.
PYTHON_BIN="$(command -v python)" bash scripts/validate.sh --backend numpy
```

`--model` takes precedence over `SERVING_ENGINE_MLX_MODEL`, which takes precedence over the
default folder. Benchmark inputs must fit the engine's 4,096-token context limit. Chunk sizes
must be distinct and output length must be at least two for inter-token statistics.

## Public evidence

Keep reference measurements unchanged. New runs remain ignored by Git until reviewed.
Only `results/cpu-reference/` and `results/apple-m5-reference/` are public result directories.
Publish actual metrics, regressions and fingerprints together. Do not commit complete local
run bundles without review: historical snapshots and logs may contain personal paths or notes.

`EXPERIMENT_SUMMARY.md` is a technical report. It does not assert personal ownership, a public
PR or production use. Output consistency does not establish task quality. This project is a
custom serialized scheduler, not a vLLM patch, CUDA kernel or multi-GPU implementation.

Run `python scripts/audit_public.py` before staging files. It uses Git ignore rules, flags
ignored files that are already tracked, and checks candidate public files for private paths,
credential-like strings and unexpected large binaries. It is a best-effort audit, not proof
that no sensitive content exists. Private material belongs in ignored `local-notes/`.
