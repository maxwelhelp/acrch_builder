#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python tools/project_probe/transformer_plugin_proof.py \
  --out-json reports/agent_inspector/transformer_plugin_proof.json \
  --out-md reports/agent_inspector/TRANSFORMER_PLUGIN_PROOF.md
