#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN. Create .venv or set PYTHON_BIN to a Python 3.11+ executable." >&2
  exit 1
fi
exec "$PYTHON_BIN" "$ROOT/scripts/workflow.py" validate "$@"
