# Scheduler design

## Bottleneck and treatment

The baseline admits requests FIFO and processes an entire pending prompt before returning to
active decodes. When a long prompt arrives during generation, its forward pass blocks token
emission for other requests. The chunked policy changes the scheduling loop:

1. Admit FIFO requests while both active-sequence and reserved-token limits permit.
2. Decode one token for each already-prefilled active request.
3. Spend the remaining iteration token budget on one prefill chunk.
4. Rotate that prefill request behind other unfinished prompts.
5. Release request-owned KV arrays and reservation on completion, cancellation or model error.

Both policies use identical weights, cache implementation and greedy sampling. Prefix caching,
quantization changes and cross-request fused batches are deliberately absent from the treatment.
The baseline is an explicit blocking implementation, not a claim about current vLLM defaults.

## Invariants

- `reserved` is the sum of prompt length + maximum output length for active requests.
- Every admitted request fits its individual context limit and the total KV token capacity.
- `token_budget > max_active` leaves prefill capacity even when all active requests decode.
- Each unfinished prompt gets one chunk per round-robin cycle, assuming model calls terminate.
- New requests remain FIFO when KV capacity prevents admission. This may cause head-of-line
  waiting for smaller requests; bypass and aging would be a separate policy experiment.
- Each request has independent per-layer K/V arrays; no prefix cache is shared across tenants.
- MLX forwards explicitly evaluate token and cache state before timestamping completion.
- The worker thread alone mutates engine and model state. HTTP handlers send bounded messages.
- The benchmark logs all emissions, including the first token produced by the final prefill.

The token capacity is an accounting bound, **not** a byte-accurate GPU allocator or OOM guarantee.
Attention workspaces, vocabulary logits, allocator caching and model weights are additional.
NumPy reallocates contiguous K/V arrays when appending; a production paged allocator would
require a different backend and separate benchmarks.

## Fairness and trade-offs

Smaller chunks reduce the longest uninterrupted prefill work but incur more calls and may
delay first-token completion. Round-robin prefill can also delay one long request in order to
advance another. No latency SLO is guaranteed: a token budget is not a wall-clock deadline.
Fairness is over admitted requests only. The model calls are sequential, so decode latency
also grows with active request count. Cancellation is checked between scheduling iterations;
it cannot preempt a running model call.

## Evidence methodology

The benchmark uses a predetermined arrival schedule and charges latency from **scheduled**
arrival, including time spent waiting while another model call blocks the loop. It does not
reset arrival time when work is eventually submitted. There is no concurrency-one speedup claim.
Each run uses fresh request caches, fixed output length, and pre-tokenized prompts. Model loading,
tokenization, warmup and JSON writing are excluded. Trace append overhead is included for both.
Throughput is total emitted tokens divided by run wall time, including arrival gaps.

Per-request token timestamps produce TTFT and inter-token gaps. Percentiles use linear
interpolation. The report takes the median of per-round metrics. Variant order is rotated each
round; three rounds balance the three default variants. The same workload order is retained.
Raw JSON preserves individual runs, per-request worst gaps, model calls and source fingerprints.

Output consistency is checked against blocking within the same round. Exact greedy token
matches are a useful regression gate, not proof of task quality or equivalence on all prompts.
NumPy tests also compare full/chunked logits within tolerance and incremental decode against
full recomputation. Real-model integration checks remain opt-in on Metal hardware.

## Extension candidates

Add fused batches only after tracking positions, masks and KV ownership per sequence. Add a
byte-accounted allocator before increasing concurrency substantially. Measure overload and
sustained traffic with an external load generator before making service-level latency claims.
A future vLLM port would require its own actual code change and NVIDIA benchmark; changing
vLLM configuration alone is not the same contribution as changing its engine.


## Evidence verification

`python -m serving_engine.verify PATH/TO/raw.json` checks the expected round/workload/policy
matrix, unique request IDs, complete output lengths, scheduled arrivals and timestamp ordering.
It recalculates each recorded metric and compares sequences by ID against the blocking run.
Missing runs, missing requests, inconsistent metrics or output mismatches fail verification.
The workflow runs this gate before marking a new benchmark successful; CI also verifies both
reviewed reference datasets. It does not establish measurement authenticity or source identity.

A deterministic logical-clock test isolates the scheduling mechanism: a 20-token prefill delays
an existing decoder under blocking, while four-token chunks bound individual prefill work.
This is a functional regression test, not simulated hardware evidence. Hardware claims use only
the recorded MLX timestamps. Metric tests separately check scheduled-arrival accounting and
percentile calculations against hand-calculated examples.
