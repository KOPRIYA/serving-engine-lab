# Apple GPU validation

For the automated path, use `bash scripts/validate.sh` and
`bash scripts/benchmark.sh --open`. See [the reproducible workflow](RUN_WORKFLOW.md).
The default model is now `models/qwen/int4` relative to this repository; the manual
commands below remain available for debugging.

A completed local Apple M5 run is included in `results/apple-m5-reference/`.
To reproduce it, use a normal Terminal with Metal access. Some restricted execution
environments cannot access Metal. All commands below run from the repository root.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mlx]'
python -c 'import mlx.core as mx; print(mx.metal.device_info())'
```

The tested API target is MLX 0.32.4 / MLX-LM 0.32.2. These versions were available locally;
the adapter also passed the real-model consistency check in the recorded reference run.
Your environment still needs its own validation.

Use an existing local MLX model, preferably Qwen2.5-1.5B-Instruct 4-bit to fit comfortably in
16 GB unified memory. If you have only original Hugging Face weights, convert them first:

```bash
python -m mlx_lm.convert --hf-path /absolute/path/to/Qwen2.5-1.5B-Instruct \
  --mlx-path models/qwen-int4 -q --q-bits 4
```

The local model must contain config, tokenizer and weight files. The backend does not silently
download weights. Keep model licenses and source revision alongside your local model manifest.
Do not commit model weights to this repository.

## Correctness before timing

```bash
SERVING_ENGINE_MLX_MODEL=/absolute/path/to/mlx-model \
  python -m unittest discover -s tests -p test_mlx.py -v
```

This opt-in test compares isolated blocking generation against interleaved chunked generation
on three different prompts. If it fails, inspect logits/numerical precision and cache state;
do not describe the scheduler as output-equivalent. Run a broader task-quality suite before
making application-quality claims.

## Benchmark

```bash
python -m serving_engine.benchmark --backend mlx --model /absolute/path/to/mlx-model \
  --rounds 3 --chunks 64 256 --short 48 --long 1024 --output-tokens 64 \
  --out results/local-qwen
python -m pip freeze > results/local-qwen/environment-lock.txt
```

Close competing GPU workloads and keep power settings stable. Inspect all three workloads,
all rounds, output consistency, TTFT and throughput; avoid reporting only the best metric.
Run longer trials and more rounds for a reliable performance claim. This default is a small
controlled diagnostic, not a saturation/load benchmark. The reference run shows a maximum-stall improvement alongside throughput and latency
regressions; inspect the trade-offs rather than assuming a universal speedup.

Fingerprint all weight files separately before publishing a result:

```bash
python - /absolute/path/to/mlx-model > results/local-qwen/weights-sha256.json <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
result = {}
for path in sorted(root.glob('*.safetensors')):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    result[path.name] = h.hexdigest()
if not result:
    raise SystemExit('No safetensors files found')
print(json.dumps(result, indent=2))
PY
```

## Live HTTP check

```bash
python -m serving_engine.server --backend mlx --model /absolute/path/to/mlx-model
```

In another Terminal:

```bash
curl http://127.0.0.1:8001/health
curl -N http://127.0.0.1:8001/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain how KV caching works.","max_tokens":32}'
```

Launch overlapping requests with long prompts to exercise interference. In-process benchmark
numbers do not include HTTP/network transport costs. The development session prohibited binding
a local socket, so actual socket integration was not verified there.

