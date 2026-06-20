#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python tools/project_probe/token_slot_replacement_proof.py \
  --out-json reports/agent_inspector/token_slot_replacement_proof.json \
  --out-md reports/agent_inspector/TOKEN_SLOT_REPLACEMENT_PROOF.md
