#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-$(pwd)}"
cd "$ROOT_DIR"
python tools/agent_inspector/inspect_project.py --root .
printf '\n[inspect] Open:\n  xdg-open reports/agent_inspector/static_dashboard.html\n\n'
printf '[inspect] Agent files:\n  reports/agent_inspector/AGENT_CONTEXT.md\n  reports/agent_inspector/SUMMARY.md\n  reports/agent_inspector/runtime_gradient_graph.json\n  reports/agent_inspector/runtime_credit_graph.json\n  reports/agent_inspector/runtime_probe.json\n  reports/agent_inspector/static_graph.json\n  .inspector.json\n\n'
