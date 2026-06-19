#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-$(pwd)}"
cd "$ROOT_DIR"
mkdir -p reports/agent_inspector
python tools/project_probe/probe_learning_loop.py \
  --out reports/agent_inspector/runtime_probe.json \
  --task diff \
  --device cuda \
  --dim 64 \
  --slots 4 \
  --layers 1 \
  --top-k 25 \
  --sim-rank 16 \
  --batch-size 64 \
  --tau 1.0 \
  --input-norm none
