#!/usr/bin/env bash
set -euo pipefail

# Make sure we are at project root
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

mkdir -p reports/agent_inspector

echo "Running Deliberation Probe v2..."
python -m tools.project_probe.probe_simulator_deliberation \
    --device cuda \
    --seed 1 \
    --dim 64 \
    --rank 16 \
    --proj-dim 32 \
    --hidden 128 \
    --batch 1024 \
    --steps 400 \
    --lr 2e-3 \
    --mmr-beta 0.35 \
    --num-cells 4 \
    --num-heads 8 \
    --K 64 \
    > reports/agent_inspector/simulator_deliberation_probe_v2_results.json

echo "Deliberation Probe v2 finished. Output saved to reports/agent_inspector/simulator_deliberation_probe_v2_results.json"
