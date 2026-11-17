# Contributing

Use Python 3.11+, install with `python -m pip install -e .`, and run
`python -m unittest discover -s tests -v` before proposing a change.

Keep the blocking baseline available. A scheduler change needs a test of the behavior it changes
and a rerun on the same fixtures with raw evidence. Separate scheduling, precision, model and
cache changes so performance differences have an interpretable cause. Do not use timing
thresholds in shared CI; functional assertions are the correctness gate.

For GPU changes, run the opt-in real-model test and save model/runtime fingerprints. Document
failed comparisons and latency regressions. Do not replace old raw results or describe fixture
timings as trained-model or production measurements. Never commit weights or credentials.

## Public files

Run `python scripts/audit_public.py` before staging. Git ignore rules do not remove previously
tracked files: review any tracked/ignored findings and deliberately untrack them as needed.
Do not blanket-remove tracked files or rewrite history as part of ordinary cleanup.

Store private notes in `local-notes/`. Keep local run bundles ignored and immutable; they may
contain historical source snapshots, private notes or local machine paths. Publish only reviewed
reference files with provenance and include regressions. Keep model weights and virtual
environments local. Installed package metadata may be required by an editable installation;
it is ignored rather than deleted. No script automatically commits, pushes or publishes.

## Automated checks

CPU CI runs on Python 3.11 and 3.12: lint, publication audit, shell syntax, unit tests, a short
benchmark consistency run and package build. Install `.[dev]` to run lint and build locally.
Metal tests remain opt-in. CI has no timing thresholds and does not establish GPU performance.
