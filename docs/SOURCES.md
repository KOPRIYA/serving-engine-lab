# Primary references

- [vLLM optimization documentation](https://docs.vllm.ai/en/latest/configuration/optimization/):
  chunked prefill and decode-priority scheduling. This project independently implements the
  established idea in a small engine. It does not modify vLLM or claim novelty for chunking.
- [MLX-LM cache implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py):
  per-layer cache creation used by the optional MLX backend.
- [MLX-LM generation implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py):
  model calls, cache advancement, greedy generation and evaluation boundaries.
- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html):
  why graph construction must not be mistaken for completed GPU computation.

Consulted 2026-11-17. Dependency versions are pinned in pyproject.toml. Main-branch documentation
links may evolve; local MLX 0.32.4 / MLX-LM 0.32.2 APIs were also inspected during implementation.

