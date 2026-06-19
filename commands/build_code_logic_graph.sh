#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p reports
python tools/build_code_logic_graph.py --root . --out-dir reports
printf '\nopen:\n  xdg-open reports/code_logic_graph.html\n\nagent files:\n  reports/CODE_MAP.md\n  reports/loop_closure_report.md\n  reports/code_logic_graph.json\n\n'
