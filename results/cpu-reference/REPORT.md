# Measured benchmark

Backend: `numpy`.
Each table entry is the median of per-round metrics; raw rounds are in raw.json.
These are in-process wall-clock measurements, not HTTP client measurements.

| Workload | Policy | TTFT p95 ms | ITL p95 ms | ITL max ms | Output tok/s |
|---|---|---:|---:|---:|---:|
| mixed | blocking | 18.333 | 0.456 | 9.033 | 5205.600 |
| mixed | chunked-64 | 18.561 | 0.452 | 1.019 | 5383.740 |
| mixed | chunked-256 | 17.009 | 0.449 | 2.312 | 5433.857 |
| short_only | blocking | 1.202 | 0.101 | 0.141 | 6705.184 |
| short_only | chunked-64 | 1.226 | 0.101 | 0.129 | 6715.385 |
| short_only | chunked-256 | 1.275 | 0.102 | 0.140 | 6736.143 |
| long_only | blocking | 16.349 | 0.277 | 8.417 | 3849.503 |
| long_only | chunked-64 | 16.413 | 0.287 | 0.820 | 3890.006 |
| long_only | chunked-256 | 14.382 | 0.291 | 2.345 | 4146.364 |

## Output consistency

Exact sequence matches against blocking in the same round: 84/84.
Matching greedy tokens checks scheduling consistency on this fixture; it is not a task-quality evaluation.

## Interpretation

- NumPy runs use an untrained random-weight model on CPU. Do not claim Qwen/GPU results from them.
- MLX results, if present, use the local model described in raw.json.
- Throughput includes arrival gaps; model loading, tokenization and warmup are excluded.
- EOS is ignored to give every request the same fixed output length.
- A pooled ITL percentile can hide rare stalls; inspect maximum ITL and per-request traces.
- Small samples and one machine do not establish production or multi-GPU performance.
- Compare TTFT and throughput as well as ITL: chunking can trade one for another.
