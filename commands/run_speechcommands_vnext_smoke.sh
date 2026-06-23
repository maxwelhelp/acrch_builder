#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

OUT=agent_reports/vnext_smoke_seed1_1ep \
SEED=1 \
EPOCHS=1 \
STEPS_PER_EPOCH=200 \
BATCH_SIZE=64 \
EVAL_BATCH_SIZE=128 \
WORKERS=6 \
AMP=fp16 \
ENABLE_SINGLE_SIGNED_PROJECTION=1 \
SINGLE_PROJ_DIM=32 \
ENABLE_PAIR_JL_BILINEAR=0 \
PROJECTION_LOGIT_CAP=0 \
PRIMITIVE_TOP_SHARE_TARGET=0.65 \
PRIMITIVE_ENTROPY_FLOOR=0.55 \
ENABLE_VNEXT=1 \
ENABLE_UTILITY_CRITIC_PROBE=1 \
bash commands/run_speechcommands_real_discovery.sh "$@" || true
