#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p reports
python tools/build_code_logic_graph.py \
  --root . \
  --include-docs \
  --include-commands \
  --out-json reports/code_logic_graph.json \
  --out-html reports/code_logic_graph.html \
  --out-md reports/CODE_MAP.md \
  --out-loop reports/loop_closure_report.md
printf '\n[code-graph-v3] open:\n  xdg-open reports/code_logic_graph.html\n\n'
