#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-$(pwd)}"
cd "$ROOT_DIR"
python tools/agent_inspector/focus_report.py --root .
printf '\n[focus] Agent file:\n  reports/agent_inspector/focused_report.md\n\n'
