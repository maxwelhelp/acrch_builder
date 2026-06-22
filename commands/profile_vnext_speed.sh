#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/maxwelhelp/test/sience/experiments/math_search/structured_matrix_program_export_lab/data/speechcommands/SpeechCommands}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-agent_reports/profile_vnext_speed}"

SCANNER_ARGS=()
if [[ "${ENABLE_SINGLE_SIGNED_PROJECTION:-0}" == "1" ]]; then
  SCANNER_ARGS+=(--enable-single-signed-projection)
fi
if [[ "${ENABLE_PAIR_JL_BILINEAR:-0}" == "1" ]]; then
  SCANNER_ARGS+=(--enable-pair-jl-bilinear)
fi
if [[ "${ENABLE_SELF_DELTA_PROBE:-0}" == "1" ]]; then
  SCANNER_ARGS+=(--enable-self-delta-probe)
fi
if [[ "${ENABLE_SELF_DELTA_CHOICE:-0}" == "1" ]]; then
  SCANNER_ARGS+=(--enable-self-delta-choice)
fi

VNEXT_ARGS=()
if [[ "${ENABLE_VNEXT:-0}" == "1" ]]; then
  VNEXT_ARGS+=(--enable-vnext)
fi
if [[ "${ENABLE_UTILITY_CRITIC_PROBE:-0}" == "1" ]]; then
  VNEXT_ARGS+=(--enable-utility-critic-probe)
fi
if [[ "${ENABLE_UTILITY_CRITIC_CHOICE:-0}" == "1" ]]; then
  VNEXT_ARGS+=(--enable-utility-critic-choice)
fi
if [[ "${ENABLE_SCANNER_FEEDBACK_MEMORY:-0}" == "1" ]]; then
  VNEXT_ARGS+=(--enable-scanner-feedback-memory)
fi
if [[ "${ENABLE_MMR_CONTROLLER:-0}" == "1" ]]; then
  VNEXT_ARGS+=(--enable-mmr-controller)
fi
if [[ "${ENABLE_LAZY_EXECUTOR:-0}" == "1" ]]; then
  VNEXT_ARGS+=(--enable-lazy-executor)
fi
if [[ -n "${UTILITY_POOL_SIZE:-}" ]]; then
  VNEXT_ARGS+=(--utility-pool-size "${UTILITY_POOL_SIZE}")
fi
if [[ -n "${UTILITY_BUDGET:-}" ]]; then
  VNEXT_ARGS+=(--utility-budget "${UTILITY_BUDGET}")
fi
if [[ -n "${UTILITY_MMR_BETA:-}" ]]; then
  VNEXT_ARGS+=(--utility-mmr-beta "${UTILITY_MMR_BETA}")
fi
if [[ -n "${UTILITY_MMR_MODE:-}" ]]; then
  VNEXT_ARGS+=(--utility-mmr-mode "${UTILITY_MMR_MODE}")
fi
if [[ -n "${UTILITY_CHOICE_WARMUP_STEPS:-}" ]]; then
  VNEXT_ARGS+=(--utility-choice-warmup-steps "${UTILITY_CHOICE_WARMUP_STEPS}")
fi
if [[ -n "${MMR_CONTROLLER_WARMUP_STEPS:-}" ]]; then
  VNEXT_ARGS+=(--mmr-controller-warmup-steps "${MMR_CONTROLLER_WARMUP_STEPS}")
fi
if [[ -n "${UTILITY_CHOICE_SCALE:-}" ]]; then
  VNEXT_ARGS+=(--utility-choice-scale "${UTILITY_CHOICE_SCALE}")
fi
if [[ -n "${UTILITY_CHOICE_SCALE_MAX:-}" ]]; then
  VNEXT_ARGS+=(--utility-choice-scale-max "${UTILITY_CHOICE_SCALE_MAX}")
fi
if [[ -n "${UTILITY_MMR_IDENTITY_WEIGHT:-}" ]]; then
  VNEXT_ARGS+=(--utility-mmr-identity-weight "${UTILITY_MMR_IDENTITY_WEIGHT}")
fi
if [[ "${ENABLE_CATEGORY_SCANNER:-0}" == "1" ]]; then
  VNEXT_ARGS+=(--enable-category-scanner)
fi
if [[ "${ENABLE_AUTO_MINED_ATOMS:-0}" == "1" ]]; then
  VNEXT_ARGS+=(--enable-auto-mined-atoms)
fi

PYTHONPATH=. python tools/project_probe/profile_vnext_speed.py \
  --dataset synthetic \
  --variant structured \
  --discovery \
  --controller-baseline "${CONTROLLER_BASELINE:-learned}" \
  --device "${DEVICE:-cuda}" \
  --amp "${AMP:-fp16}" \
  --seed "${SEED:-1}" \
  --slots "${SLOTS:-8}" \
  --dim "${DIM:-64}" \
  --layers "${LAYERS:-3}" \
  --top-k "${TOP_K:-8}" \
  --sim-rank "${SIM_RANK:-16}" \
  --final-read last \
  --epochs "${EPOCHS:-1}" \
  --steps-per-epoch "${STEPS_PER_EPOCH:-50}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --workers "${WORKERS:-4}" \
  --lr "${LR:-1e-3}" \
  --tau-start "${TAU_START:-1.5}" \
  --tau-min "${TAU_MIN:-0.8}" \
  --tau-decay "${TAU_DECAY:-0.95}" \
  --credit-budget "${CREDIT_BUDGET:-2}" \
  --credit-alternative-budget "${CREDIT_ALTERNATIVE_BUDGET:-1}" \
  --credit-interval "${CREDIT_INTERVAL:-25}" \
  --credit-batch-size "${CREDIT_BATCH_SIZE:-8}" \
  --out-dir "$OUT" \
  --utility-category-k "${UTILITY_CATEGORY_K:-2}" \
  --profile-steps "${PROFILE_STEPS:-20}" \
  --profile-light \
  "${SCANNER_ARGS[@]}" \
  "${VNEXT_ARGS[@]}"
