#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python -m tools.project_probe.layer_specialization_proof
echo "[layer-specialization-proof] reports/agent_inspector/LAYER_SPECIALIZATION_PROOF.md"
