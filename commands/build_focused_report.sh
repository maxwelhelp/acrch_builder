#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-$(pwd)}"
cd "$ROOT_DIR"
python tools/agent_inspector/focus_report.py --root . --request inspector_request.yml --graph reports/agent_inspector/static_graph.json --out reports/agent_inspector/focused_report.md
printf '\n[inspector] Focused report:\n  reports/agent_inspector/focused_report.md\n\n'
