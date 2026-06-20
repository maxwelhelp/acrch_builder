#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python tools/project_probe/audio_frontend_proof.py \
  --out-json reports/agent_inspector/audio_frontend_proof.json \
  --out-md reports/agent_inspector/AUDIO_FRONTEND_PROOF.md
