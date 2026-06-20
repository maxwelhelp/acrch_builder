#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-agent_reports/discovery_chain_diff_product_seed${SEED:-1}_$(date +%Y%m%d_%H%M%S)}"

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
  --lr "${LR:-1e-3}" \
  --seed "${SEED:-1}" \
  --supervision-mode discovery \
  --choice-sampling softmax \
  --tau-start "${TAU_START:-1.5}" \
  --tau-min "${TAU_MIN:-1.5}" \
  --tau-decay 1.0 \
  \
  --lambda-sim 0.0 \
  --lambda-choice 0.0 \
  --lambda-expected-active 0.0 \
  --lambda-non-expected-primitive 0.0 \
  --lambda-non-expected-active 0.0 \
  --lambda-non-expected-tape 0.0 \
  --lambda-non-expected-transform 0.0 \
  --lambda-branch 0.0 \
  \
  --lambda-active-budget 0.0 \
  --lambda-tape-budget 0.0 \
  --lambda-cell-choice-diversity 0.0 \
  --lambda-primitive-usage-diversity 0.0 \
  --lambda-layer-action-diversity 0.0 \
  \
  --lambda-collapse 0.02 \
  --min-transform-mass 0.10 \
  \
  --out-dir "$OUT" \
  --latest-report LATEST_RUN_REPORT.md

python - <<PY
import json, pathlib
p=pathlib.Path("$OUT/final_report.json")
d=json.load(open(p))
print("\\n=== DISCOVERY SUMMARY ===")
print("out =", "$OUT")
for k in [
 "best_acc","last_acc",
 "expected_candidate_present",
 "expected_edge_choice_mass",
 "expected_edge_recovery",
 "expected_any_recovery",
 "program_expected_top_cells",
 "program_active_cells",
 "primitive_top_share",
 "program_verdicts",
 "sim_disabled_delta",
 "choice_without_sim_delta",
 "loss_accounting_error",
]:
    print(k, "=", d.get(k))

checks = {
    "learned_accuracy_above_chance": float(d.get("last_acc", 0.0)) >= 0.60,
    "full_scan_candidates_present": float(d.get("expected_candidate_present", 0.0)) >= 0.99,
    "no_primitive_collapse": all(v != "primitive_collapse" for v in d.get("program_verdicts", [])),
}
print("checks =", checks)
if not all(checks.values()):
    raise SystemExit("DISCOVERY FAIL: learned program did not meet acceptance")
PY
