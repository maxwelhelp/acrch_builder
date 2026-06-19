#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "[validate] py_compile"
python -m py_compile arch_builder/*.py

echo "[validate] import smoke"
python - <<'PY'
from arch_builder import PrimitiveMatrix5x5, HybridScanner, LowRankSimulator, ActionExecutor, ActionMatrixModel
m = ActionMatrixModel(dim=16, slots=4, layers=1, classes=2, top_k=4, sim_rank=8)
print(type(m).__name__, m.pm.names[:5])
PY

echo "[validate] cli help"
python -m arch_builder.train_vertical_slice --help >/dev/null

echo "[validate] forbidden files"
if find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.pyc' \) | grep .; then
  echo "forbidden weight/cache file found" >&2
  exit 4
fi

echo "[validate] OK"
