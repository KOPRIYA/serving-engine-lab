# Measured benchmark

Backend: `mlx`.
Each table entry is the median of per-round metrics; raw rounds are in raw.json.
These are in-process wall-clock measurements, not HTTP client measurements.

| Workload | Policy | TTFT p95 ms | ITL p95 ms | ITL max ms | Output tok/s |
|---|---|---:|---:|---:|---:|
| mixed | blocking | 2954.821 | 37.013 | 685.735 | 93.287 |
| mixed | chunked-64 | 3373.167 | 63.607 | 78.629 | 73.214 |
| mixed | chunked-256 | 3140.737 | 66.977 | 119.875 | 87.246 |
| short_only | blocking | 2459.428 | 39.392 | 130.148 | 100.268 |
| short_only | chunked-64 | 2482.916 | 38.257 | 69.452 | 103.670 |
| short_only | chunked-256 | 2494.351 | 40.058 | 67.700 | 103.656 |
| long_only | blocking | 640.328 | 20.951 | 337.245 | 68.017 |
| long_only | chunked-64 | 1385.641 | 21.956 | 53.918 | 49.191 |
| long_only | chunked-256 | 700.649 | 19.915 | 101.497 | 67.254 |

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
