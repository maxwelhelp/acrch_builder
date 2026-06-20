#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python tools/project_probe/speechcommands_acceptance_proof.py \
  --out-json reports/agent_inspector/speechcommands_acceptance_proof.json \
  --out-md reports/agent_inspector/SPEECHCOMMANDS_ACCEPTANCE_PROOF.md
