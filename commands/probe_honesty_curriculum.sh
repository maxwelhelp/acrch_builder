#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python tools/project_probe/honesty_curriculum_proof.py \
  --task-config "${TASK_CONFIG:-configs/tasks/chain_diff_product.yml}" \
  --transfer-task-config "${TRANSFER_TASK_CONFIG:-configs/tasks/chain_diff_merge.yml}" \
  --epochs "${EPOCHS:-1}" \
  --max-steps "${MAX_STEPS:-4}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --eval-steps "${EVAL_STEPS:-1}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-32}" \
  --dim "${DIM:-16}" \
  --top-k "${TOP_K:-8}" \
  --sim-rank "${SIM_RANK:-8}" \
  --device "${DEVICE:-cpu}" \
  --amp "${AMP:-none}" \
  --out-json reports/agent_inspector/honesty_curriculum_proof.json \
  --out-md reports/agent_inspector/HONESTY_CURRICULUM_PROOF.md
