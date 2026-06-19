#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

BRANCH="${BRANCH:-main}"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${OUT_DIR:-agent_reports/vertical_slice_${TS}}"
LATEST_REPORT="LATEST_RUN_REPORT.md"

TASK="${TASK:-diff}"
EPOCHS="${EPOCHS:-5}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
MAX_STEPS="${MAX_STEPS:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_STEPS="${EVAL_STEPS:-10}"
DIM="${DIM:-64}"
SLOTS="${SLOTS:-4}"
LAYERS="${LAYERS:-1}"
TOP_K="${TOP_K:-25}"
SIM_RANK="${SIM_RANK:-16}"
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-fp16}"
LR="${LR:-3e-4}"

mkdir -p "$REPORT_DIR"

echo "[sync] branch=$BRANCH report_dir=$REPORT_DIR task=$TASK"

git checkout "$BRANCH"
git pull --rebase --autostash origin "$BRANCH"

bash commands/validate.sh | tee "$REPORT_DIR/validate.log"

set +e
python -m arch_builder.train_vertical_slice \
  --task "$TASK" \
  --epochs "$EPOCHS" \
  --steps-per-epoch "$STEPS_PER_EPOCH" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-steps "$EVAL_STEPS" \
  --dim "$DIM" \
  --slots "$SLOTS" \
  --layers "$LAYERS" \
  --top-k "$TOP_K" \
  --sim-rank "$SIM_RANK" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --lr "$LR" \
  --out-dir "$REPORT_DIR" \
  --latest-report "$LATEST_REPORT" \
  2>&1 | tee "$REPORT_DIR/train.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo "$RUN_STATUS" > "$REPORT_DIR/run_status.txt"

find arch_builder commands docs "$REPORT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + || true
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete || true

if find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.pyc' -o -name '*.pyo' \) | grep .; then
  echo "[sync] forbidden weight/cache file found; refusing commit" >&2
  exit 5
fi

git add README.md requirements.txt .gitignore arch_builder commands docs "$REPORT_DIR" "$LATEST_REPORT" 2>/dev/null || true
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors|pyc|pyo)$'; then
  echo "[sync] forbidden file staged; refusing commit" >&2
  git reset --cached . >/dev/null || true
  exit 6
fi

if git diff --cached --quiet; then
  echo "[sync] nothing to commit"
else
  git commit -m "Run vertical slice and update reports"
  git push origin "$BRANCH"
fi

echo "[sync] latest report: $LATEST_REPORT"
echo "[sync] done status=$RUN_STATUS"
exit "$RUN_STATUS"
