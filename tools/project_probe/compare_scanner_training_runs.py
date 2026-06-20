#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = (
    "acc",
    "samples_per_second",
    "credit_closed",
    "sim_disabled_delta",
    "choice_without_sim_delta",
    "single_signed_projection_usage",
    "pair_jl16_usage",
    "primitive_top_share",
    "active_cells",
    "projection_logit_cap",
    "projection_logit_clipped_fraction",
)


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["comparison_metrics"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--out", default="reports/agent_inspector/speech_scanner_1ep_comparison.json")
    args = parser.parse_args()
    old, new = load(args.old), load(args.new)
    report = {
        "old_report": args.old,
        "new_report": args.new,
        "old": {key: old.get(key, 0.0) for key in FIELDS},
        "new": {key: new.get(key, 0.0) for key in FIELDS},
        "delta_new_minus_old": {
            key: float(new.get(key, 0.0)) - float(old.get(key, 0.0)) for key in FIELDS
        },
        "scope": "single_seed_single_epoch_comparison_not_acceptance",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# SpeechCommands scanner 1ep comparison", "", "| metric | old | new | delta |", "|---|---:|---:|---:|"]
    for key in FIELDS:
        lines.append(
            f"| {key} | {report['old'][key]:.6g} | {report['new'][key]:.6g} | {report['delta_new_minus_old'][key]:+.6g} |"
        )
    lines += ["", "Single seed/epoch smoke; this is not acceptance.", ""]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
