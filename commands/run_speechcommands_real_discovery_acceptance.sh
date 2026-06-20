#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="${ROOT_OUT:-agent_reports/speech_real_discovery_acceptance_${STAMP}}"
mkdir -p "$ROOT_OUT"

learned=()
frozen=()
random=()

for mode in learned frozen random; do
  for seed in 1 2 3; do
    out="$ROOT_OUT/${mode}_seed${seed}"
    echo "[acceptance] mode=$mode seed=$seed out=$out"
    set +e
    CONTROLLER_BASELINE="$mode" SEED="$seed" OUT="$out" \
      bash commands/run_speechcommands_real_discovery.sh 2>&1 | tee "$out.run.log"
    run_status=${PIPESTATUS[0]}
    set -e
    if [[ ! -f "$out/final_report.json" ]]; then
      echo "missing report for mode=$mode seed=$seed (exit=$run_status)" >&2
      exit 1
    fi
    case "$mode" in
      learned) learned+=("$out/final_report.json") ;;
      frozen) frozen+=("$out/final_report.json") ;;
      random) random+=("$out/final_report.json") ;;
    esac
  done
done

python -m tools.project_probe.speech_real_discovery_acceptance \
  --learned "${learned[@]}" \
  --frozen "${frozen[@]}" \
  --random "${random[@]}" \
  --out "$ROOT_OUT/acceptance.json"
