#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"

python -m py_compile tools/project_probe/probe_self_delta_candidate_field.py

python tools/project_probe/probe_self_delta_candidate_field.py \
  --device "$DEVICE" \
  --seed "${SEED:-1}" \
  --batch-size "${BATCH_SIZE:-4096}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-4096}" \
  --dim "${DIM:-64}" \
  --sim-rank "${SIM_RANK:-8}" \
  --proj-dim "${PROJ_DIM:-32}" \
  --top-m "${TOP_M:-32}" \
  --steps "${STEPS:-120}"
