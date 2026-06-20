#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-agent_reports/config_chain_diff_product_strong_sparse_seed${SEED:-1}}"

python -m arch_builder.train_vertical_slice \
  --task-config configs/tasks/chain_diff_product.yml \
  --device "${DEVICE:-cuda}" \
  --dim "${DIM:-64}" \
  --slots 4 \
  --layers 2 \
  --top-k 25 \
  --sim-rank 16 \
  --epochs "${EPOCHS:-10}" \
  --steps-per-epoch "${STEPS_PER_EPOCH:-120}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --eval-steps 12 \
  --eval-batch-size 256 \
  --input-norm none \
  --state-norm none \
  --final-read last \
  --amp "${AMP:-fp16}" \
  --seed "${SEED:-1}" \
  --lambda-non-expected-primitive 0.90 \
  --lambda-non-expected-active 0.35 \
  --lambda-non-expected-tape 0.35 \
  --lambda-non-expected-transform 0.12 \
  --lambda-active-budget 0.55 \
  --lambda-tape-budget 0.25 \
  --lambda-cell-choice-diversity 0.18 \
  --lambda-primitive-usage-diversity 0.16 \
  --lambda-layer-action-diversity 0.08 \
  --target-active-cells 2.0 \
  --target-active-fraction 0.055 \
  --target-tape-fraction 0.003 \
  --adapt-top-share-floor 0.22 \
  --adapt-cell-sharpness 4.0 \
  --out-dir "$OUT" \
  --latest-report LATEST_RUN_REPORT.md

python - <<PY
import json, pathlib
p=pathlib.Path("$OUT/final_report.json")
d=json.load(open(p))
keys=[
 "best_acc","last_acc",
 "expected_edge_recovery","expected_edge_choice_mass","expected_candidate_present",
 "program_expected_top_cells","program_active_cells","primitive_top_share",
 "program_verdicts","sim_disabled_delta","choice_without_sim_delta",
 "loss_accounting_error"
]
print("\\n=== STRONG SPARSE SUMMARY ===")
print("report =", p)
for k in keys:
    print(k, "=", d.get(k))
PY
