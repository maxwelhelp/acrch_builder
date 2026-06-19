#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python tools/agent_inspector/inspect_project.py --root .

echo "[inspect-all] summary: reports/agent_inspector/SUMMARY.md"
