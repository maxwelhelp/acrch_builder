#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

cleanup_pycache() {
  find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
  find . -type f -name "*.pyc" -delete || true
  find . -type f -name "*.pyo" -delete || true
}

echo "[validate] cleanup stale pycache"
cleanup_pycache

echo "[validate] py_compile"
python -m py_compile arch_builder/*.py

echo "[validate] import smoke"
python - <<'PY'
from arch_builder import PrimitiveMatrix5x5, HybridScanner, LowRankSimulator, ActionExecutor, ActionMatrixModel
m = ActionMatrixModel(dim=16, slots=4, layers=1, classes=2, top_k=8, sim_rank=8)
print(type(m).__name__, m.pm.names[:5])
PY

echo "[validate] cli help"
python -m arch_builder.train_vertical_slice --help >/dev/null

echo "[validate] cleanup generated pycache"
cleanup_pycache

echo "[validate] forbidden files"
if git ls-files | grep -E '\.(pt|pth|ckpt|safetensors|pyc|pyo)$' | grep .; then
  echo "forbidden tracked weight/cache file found" >&2
  exit 4
fi

echo "[validate] OK"
