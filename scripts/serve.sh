#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
MODEL="${SERVING_ENGINE_MLX_MODEL:-$ROOT/models/qwen/int4}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN. Set up .venv or supply PYTHON_BIN." >&2
  exit 1
fi
cd "$ROOT"
exec "$PYTHON_BIN" -m serving_engine.server --backend mlx --model "$MODEL" --port "${PORT:-8001}" "$@"
