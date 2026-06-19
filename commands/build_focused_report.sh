#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

if [[ ! -f reports/agent_inspector/static_graph.json ]]; then
  python tools/agent_inspector/inspect_project.py --root .
fi
python tools/agent_inspector/focus_report.py --root .

echo "[focused-report] report: reports/agent_inspector/focused_report.md"
