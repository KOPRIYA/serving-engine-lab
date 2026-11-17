# Apple M5 reference evidence

Recorded 2026-11-17 with the local Qwen2.5-1.5B-Instruct 4-bit MLX model,
MLX 0.32.4 / MLX-LM 0.32.2, on Apple M5 with 16 GiB unified memory.
Three rounds, three scheduling policies, three synthetic workloads, fixed 64-token outputs.
The recorded pipeline passed unit tests, Metal availability, real-model consistency and benchmark.
All 37 checksums of the original completed local bundle were verified before this export.

This directory contains byte-for-byte copies of the original raw measurements, report and
model-file hashes. Personal logs, environment paths, career material and full local snapshots
are excluded. Package versions and measured engine-source hashes remain in raw.json. The five
source hashes were verified against the repository's unchanged serving_engine package at export.
No new GPU run was performed during this cleanup; Metal was unavailable to the cleanup session.

No Git commit was available when the original run was recorded. Source hashes identify the
measured files, but this is self-recorded evidence rather than independent verification.
See raw.json for all rounds, device metadata, exact input/output tokens and consistency results.

Smaller chunks reduced maximum token stalls but also increased mixed-workload TTFT/p95 ITL
and reduced throughput. This is a scheduling trade-off, not a uniform performance improvement.
The runtime is a serialized prototype; these are not production, HTTP or multi-GPU measurements.
