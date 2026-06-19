#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-$(pwd)}"
cd "$ROOT_DIR"
bash commands/runtime_probe.sh .
bash commands/inspect_all.sh .
printf '\n[inspector] Runtime graphs:\n  reports/agent_inspector/runtime_gradient_graph.json\n  reports/agent_inspector/runtime_credit_graph.json\n  reports/agent_inspector/runtime_graph_summary.md\n\n'
