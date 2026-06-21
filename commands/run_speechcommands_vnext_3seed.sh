#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="${ROOT_OUT:-agent_reports/vnext_3seed_${STAMP}}"
mkdir -p "$ROOT_OUT"

reports=()

for seed in 1 2 3; do
  out="$ROOT_OUT/vnext_seed${seed}"
  echo "[vnext-3seed] seed=$seed out=$out"
  
  # Allow passing customization via environment variables
  SEED="$seed" OUT="$out" \
  ENABLE_VNEXT="${ENABLE_VNEXT:-1}" \
  ENABLE_UTILITY_CRITIC_PROBE="${ENABLE_UTILITY_CRITIC_PROBE:-1}" \
  ENABLE_UTILITY_CRITIC_CHOICE="${ENABLE_UTILITY_CRITIC_CHOICE:-0}" \
  bash commands/run_speechcommands_real_discovery.sh || true

  if [[ -f "$out/final_report.json" ]]; then
    reports+=("$out/final_report.json")
  else
    echo "Warning: final_report.json not found for seed $seed"
  fi
done

if [[ ${#reports[@]} -gt 0 ]]; then
  echo "Comparing vNext run with baseline..."
  python tools/project_probe/compare_vnext_to_baseline.py "${reports[@]}"
else
  echo "No reports generated."
fi
