#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-$(pwd)}"
cd "$ROOT_DIR"
python tools/agent_inspector/runtime_probe.py --root . --out reports/agent_inspector/runtime_probe.json
