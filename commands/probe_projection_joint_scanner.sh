#!/usr/bin/env bash
set -euo pipefail

python tools/project_probe/probe_projection_joint_scanner.py \
  --device "${DEVICE:-cuda}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  --dim "${DIM:-32}" \
  --single-proj-dim "${SINGLE_PROJ_DIM:-32}" \
  --joint-proj-dims "${JOINT_PROJ_DIMS:-4,8,16,32}" \
  --row-chunk "${ROW_CHUNK:-2}" \
  --steps "${STEPS:-120}" \
  --seed "${SEED:-1}" \
  --out "${OUT:-reports/agent_inspector/projection_joint_scanner_probe.json}"
