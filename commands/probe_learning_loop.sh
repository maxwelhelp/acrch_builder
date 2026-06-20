#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python tools/project_probe/probe_learning_loop.py \
  --device "${PROBE_DEVICE:-cpu}" \
  --task "${PROBE_TASK:-diff}" \
  --dim "${PROBE_DIM:-32}" \
  --layers "${PROBE_LAYERS:-2}" \
  --batch-size "${PROBE_BATCH_SIZE:-64}" \
  --sim-rank "${PROBE_SIM_RANK:-8}" \
  --out reports/agent_inspector/runtime_probe.json
