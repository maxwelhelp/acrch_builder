#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python tools/project_probe/forced_program_oracle.py \
  --device "${PROBE_DEVICE:-cpu}" \
  --dim "${PROBE_DIM:-32}" \
  --batch-size "${PROBE_BATCH_SIZE:-512}" \
  --out reports/agent_inspector/forced_program_oracle.json \
  --markdown reports/agent_inspector/FORCED_PROGRAM_ORACLE.md
