#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-$(pwd)}"
cd "$ROOT_DIR"
python tools/agent_inspector/inspect_project.py --root .
printf '\n[inspector] Open:\n  xdg-open reports/agent_inspector/static_dashboard.html\n\n'
printf '[inspector] Agent files:\n  reports/agent_inspector/AGENT_CONTEXT.md\n  reports/agent_inspector/static_summary.md\n  reports/agent_inspector/holes_report.md\n  reports/agent_inspector/runtime_summary.md\n  reports/agent_inspector/static_graph.json\n  .inspector.yml\n\n'
