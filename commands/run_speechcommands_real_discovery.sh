#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/maxwelhelp/test/sience/experiments/math_search/structured_matrix_program_export_lab/data/speechcommands/SpeechCommands}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-agent_reports/speechcommands_real_discovery_seed${SEED:-1}_${STAMP}}"

python -m arch_builder.train_audio_frontend \
  --dataset speechcommands \
  --variant structured \
  --data-root "$DATA_ROOT" \
  --classes "${CLASSES:-yes,no,up,down,left,right,on,off,stop,go}" \
  --discovery \
  --device "${DEVICE:-cuda}" \
  --amp "${AMP:-fp16}" \
  --seed "${SEED:-1}" \
  --slots "${SLOTS:-8}" \
  --dim "${DIM:-64}" \
  --layers "${LAYERS:-3}" \
  --top-k "${TOP_K:-8}" \
  --sim-rank "${SIM_RANK:-16}" \
  --final-read last \
  --epochs "${EPOCHS:-3}" \
  --steps-per-epoch "${STEPS_PER_EPOCH:-200}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-128}" \
  --eval-steps "${EVAL_STEPS:-10}" \
  --workers "${WORKERS:-6}" \
  --lr "${LR:-1e-3}" \
  --tau-start "${TAU_START:-1.5}" \
  --tau-min "${TAU_MIN:-0.8}" \
  --tau-decay "${TAU_DECAY:-0.95}" \
  --credit-budget "${CREDIT_BUDGET:-8}" \
  --credit-interval "${CREDIT_INTERVAL:-8}" \
  --credit-batch-size "${CREDIT_BATCH_SIZE:-16}" \
  --target-active-cells "${TARGET_ACTIVE_CELLS:-6}" \
  --train-limit "${TRAIN_LIMIT:-0}" \
  --val-limit "${VAL_LIMIT:-0}" \
  --test-limit "${TEST_LIMIT:-2000}" \
  --log-every "${LOG_EVERY:-20}" \
  --out-dir "$OUT" \
  --latest-report "$OUT/LATEST_RUN_REPORT.md"

echo "report: $OUT/final_report.json"
