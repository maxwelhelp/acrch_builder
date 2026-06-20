#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python -m tools.project_probe.branching_output_proof
echo "[branching-output-proof] reports/agent_inspector/BRANCHING_OUTPUT_PROOF.md"
