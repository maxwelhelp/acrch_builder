#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-$(pwd)}"
cd "$ROOT_DIR"
bash commands/probe_learning_loop.sh .
bash commands/inspect_all.sh .
