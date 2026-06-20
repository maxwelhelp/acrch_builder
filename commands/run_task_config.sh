#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

if [[ $# -ne 1 ]]; then
  echo "usage: bash commands/run_task_config.sh configs/tasks/<task>.yml" >&2
  exit 2
fi

TASK_CONFIG="$1"
TASK_STEM="$(basename "$TASK_CONFIG" .yml)"
OUT_DIR="${OUT_DIR:-agent_reports/task_config_${TASK_STEM}_smoke}"

python -m arch_builder.train_vertical_slice \
  --task-config "$TASK_CONFIG" \
  --epochs "${EPOCHS:-1}" \
  --max-steps "${MAX_STEPS:-2}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --eval-steps "${EVAL_STEPS:-1}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --dim "${DIM:-16}" \
  --top-k "${TOP_K:-8}" \
  --sim-rank "${SIM_RANK:-8}" \
  --device "${DEVICE:-cpu}" \
  --amp "${AMP:-none}" \
  --out-dir "$OUT_DIR" \
  --latest-report "$OUT_DIR/LATEST_RUN_REPORT.md"
