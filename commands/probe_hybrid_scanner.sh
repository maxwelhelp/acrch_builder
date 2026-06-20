#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python -m tools.project_probe.hybrid_scanner_proof
echo "[hybrid-scanner-proof] reports/agent_inspector/HYBRID_SCANNER_PROOF.md"
