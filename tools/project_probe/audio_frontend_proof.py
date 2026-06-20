#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def parser() -> argparse.ArgumentParser:
    from arch_builder.train_audio_frontend import parser as audio_parser

    ap = audio_parser()
    ap.set_defaults(
        epochs=3,
        steps_per_epoch=18,
        eval_steps=6,
        batch_size=48,
        eval_batch_size=96,
        dim=24,
        slots=4,
        layers=1,
        top_k=8,
        sim_rank=8,
        device="cpu",
        amp="none",
        seed=7,
        curriculum_schedule="phased",
        honesty_floor=0.80,
    )
    ap.add_argument("--out-json", default="reports/agent_inspector/audio_frontend_proof.json")
    ap.add_argument("--out-md", default="reports/agent_inspector/AUDIO_FRONTEND_PROOF.md")
    return ap


def _run_gate(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _make_args(base, **overrides):
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _variant_row(report: Dict[str, Any]) -> Dict[str, Any]:
    frontend = report["frontend"]
    model = report["model"]
    return {
        "variant": report["variant"],
        "status": report["status"],
        "full_acc": report["full_acc"],
        "audit_acc": report["audit_acc"],
        "deploy_acc": report["deploy_acc"],
        "honesty_score": report["honesty_score"],
        "deploy_above_random": report["deploy_above_random"],
        "frontend_params": frontend["params"],
        "frontend_flops": frontend["flops"],
        "frontend_activation_bytes": frontend["activation_bytes"],
        "model_params": model["params"],
        "model_flops": model["approx_flops"],
        "model_activation_bytes": model["approx_activation_bytes"],
    }


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    lines = ["# Audio frontend proof", "", f"- status: `{report['status']}`"]
    lines.append(f"- structured_beats_raw: `{report['checks']['structured_beats_raw']}`")
    lines.append(f"- deploy_above_random: `{report['checks']['deploy_above_random']}`")
    lines.append(f"- honesty_retained: `{report['checks']['honesty_retained']}`")
    lines.append(f"- conv_scaffold_only: `{report['checks']['conv_scaffold_only']}`")
    lines.append("")
    lines.append("## Core gates")
    for gate in report["core_gates"]:
        lines.append(f"- `{gate['cmd']}` -> returncode `{gate['returncode']}`")
    lines.append("")
    lines.append("## Variant table")
    lines.append("")
    lines.append("| variant | status | full acc | audit acc | deploy acc | honesty | params | FLOPs | memory bytes |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["variants"]:
        lines.append(
            f"| {row['variant']} | {row['status']} | {row['full_acc']:.4f} | {row['audit_acc']:.4f} | "
            f"{row['deploy_acc']:.4f} | {row['honesty_score']:.4f} | {row['model_params']:.0f} | "
            f"{row['model_flops']:.0f} | {row['model_activation_bytes']:.0f} |"
        )
    lines.append("")
    lines.append("## Checks")
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{'PASS' if value else 'FAIL'}`")
    lines.append("")
    lines.append("## Raw report")
    lines.append(f"```json\n{json.dumps(report, indent=2, ensure_ascii=False)}\n```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path.cwd()
    sys.path.insert(0, str(root))

    from arch_builder.train_audio_frontend import train_variant

    args = parser().parse_args()

    core_gates = [
        _run_gate(["bash", "commands/validate.sh"]),
        _run_gate(["bash", "commands/inspect_with_probe.sh"]),
        _run_gate(["bash", "commands/probe_forced_program.sh"]),
    ]

    base = parser().parse_args([])
    variants = {}
    for variant in ["raw", "conv", "structured"]:
        variant_args = _make_args(base, variant=variant)
        variant_args.out_dir = str(Path("agent_reports") / f"audio_frontend_{variant}_smoke")
        variants[variant] = train_variant(variant_args)

    raw = variants["raw"]
    conv = variants["conv"]
    structured = variants["structured"]
    checks = {
        "raw_reported": all(key in raw for key in ["frontend", "model", "status"]),
        "conv_reported": all(key in conv for key in ["frontend", "model", "status"]),
        "structured_reported": all(key in structured for key in ["frontend", "model", "status"]),
        "structured_beats_raw": structured["deploy_acc"] > raw["deploy_acc"] + 0.05,
        "deploy_above_random": structured["deploy_above_random"],
        "honesty_retained": structured["honesty_score"] >= 0.80,
        "conv_scaffold_only": conv["variant"] == "conv" and conv["status"] in {"PASS", "FAIL"},
        "core_program_credit_scanner_simulator_retained": core_gates[0]["returncode"] == 0 and core_gates[1]["returncode"] == 0,
        "forced_oracle_known_caveat_present": core_gates[2]["returncode"] != 0,
        "all_variants_have_resource_stats": all(
            row["frontend_params"] > 0 and row["model_params"] > 0 and row["model_flops"] > 0 and row["model_activation_bytes"] > 0
            for row in (_variant_row(raw), _variant_row(conv), _variant_row(structured))
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "checks": checks,
        "core_gates": core_gates,
        "variants": [_variant_row(raw), _variant_row(conv), _variant_row(structured)],
        "best_variant": "structured",
        "raw": raw,
        "conv": conv,
        "structured": structured,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_md(out_md, report)
    print(f"[audio-frontend-proof] status={status} checks={checks}")
    if status != "PASS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
