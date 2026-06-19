#!/usr/bin/env python3
"""Compatibility entrypoint for the canonical project-side runtime probe."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="reports/agent_inspector/runtime_probe.json")
    args, extra = ap.parse_known_args()

    root = Path(args.root).resolve()
    probe = root / "tools" / "project_probe" / "probe_learning_loop.py"
    if not probe.exists():
        raise FileNotFoundError(f"canonical runtime probe is missing: {probe}")
    command = [sys.executable, str(probe), "--out", args.out, *extra]
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
