# Validation record

## Evidence hardening, 2026-11-17

- Moved the extension roadmap into ignored local notes; removed its public README link.
- Added a model-free evidence verifier and CI checks for both reviewed reference datasets.
  Both contain 27 complete runs and 84/84 matching output comparisons; metrics recompute
  from timestamps. Missing runs/requests and altered metrics/tokens are regression-tested.
- Added a logical-clock regression for the prefill/decode stall and hand-calculated metric tests.
- Direct benchmark CLI now rejects single-token ITL runs, duplicate chunks and oversized
  contexts before loading a model. The benchmark workflow verifies fresh evidence automatically.
- Ordinary suite: **41 passed, 1 opt-in Metal test skipped**.
- Fresh CPU workflow: **27 runs, 84/84 comparisons**, including the new evidence gate.
- Public-file audit: **36 candidates, zero findings**; shell and Python syntax checks passed.
- Validation used the existing local Python/NumPy environment. Ruff, package build, remote CI,
  live socket serving and a fresh Metal run remain unverified in this session.
- No new GPU speedup is claimed. Historical raw evidence and checksums remain unchanged.


## Repository review, 2026-11-17

- Added Git ignore rules and CPU CI; local audit: 35 public candidate files, zero findings.
- Fixed cache-allocation failure handling: the failed request terminates once, releases its
  lifecycle state and cannot indefinitely block following requests. A regression test covers
  existing active work, the failed request, subsequent admission and reservation cleanup.
- Current ordinary suite: **36 passed, 1 opt-in Metal test skipped**.
- CPU benchmark: **27 runs, 84/84 exact sequence comparisons matched**, using the standard
  48/1024-token prompts, 64-token outputs, three rounds and chunks 64/256.
- Shell syntax passed; historical GPU evidence checksums verified without changing those files.
- Used an existing local Python 3.11 environment with NumPy 2.4.6. Installing dependencies
  into the new project virtual environment failed because PyPI DNS was unavailable.
- Ruff and build were unavailable locally; those CI steps and remote CI execution remain
  unverified. No new Metal benchmark or live HTTP socket validation was performed.

The GPU reference predates the engine failure-path fix. Its hashes describe the historical
source, not the current source; rerun the GPU workflow before reporting current GPU timings.
The earlier cleanup record below describes an earlier validation session.

## Cleanup validation, 2026-11-17

- Ordinary suite: **35 passed, 1 opt-in Metal test skipped**.
- Complete CPU workflow rerun: three workloads x three policies x three rounds = 27 runs;
  all 84 exact output-sequence comparisons passed.
- Public-file audit: candidate source, docs and reviewed reference data passed the audit.
- Shell syntax checks passed for validation, benchmark and serving entry points.
- Original CPU and GPU reference data retained; measured engine code was not changed.

Tests cover scheduler ordering, budgets, fairness, cache correctness, request lifecycle,
streaming handlers, workflow failure handling and publication hygiene. The audit tests verify
Git ignore behavior before repository initialization and detect ignored files already tracked.

## Recorded GPU reference

The existing completed Apple M5 run passed ordinary tests, Metal availability, pretrained-model
output consistency and the 27-run benchmark. All 37 original evidence checksums were verified
before exporting the reviewed public subset to `results/apple-m5-reference/`. The five engine
source hashes in raw.json match the unchanged serving_engine files at export.

GPU results are historical measurements from the user's normal Terminal. The cleanup execution
session cannot access Metal, so no new GPU performance run is claimed. The optional MLX test
is skipped in ordinary CPU validation and is explicitly enabled by the GPU workflow.

## Scope

HTTP handler tests use an in-memory transport. A live socket check is still required in a normal
Terminal because the restricted execution session cannot bind localhost. The server is a local
SSE demo, not an OpenAI-compatible endpoint. No production, custom CUDA or multi-GPU claim is made.
GitHub CI configuration is included; its remote execution is not established by these local checks.
