#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

bash commands/probe_learning_loop.sh
python tools/agent_inspector/inspect_project.py --root .

echo "[inspect-with-probe] summary: reports/agent_inspector/SUMMARY.md"
