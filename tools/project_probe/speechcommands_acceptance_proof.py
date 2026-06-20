#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parser() -> argparse.ArgumentParser:
    from arch_builder.train_audio_frontend import parser as audio_parser

    ap = audio_parser()
    ap.set_defaults(
        dataset="speechcommands",
        data_root="../Functional Matrix Grower/data/speechcommands",
        train_limit=512,
        val_limit=256,
        test_limit=256,
        epochs=4,
        steps_per_epoch=12,
        eval_steps=2,
        batch_size=16,
        eval_batch_size=32,
        layers=2,
        top_k=16,
        sim_rank=16,
        dim=32,
        input_norm="layernorm",
        state_norm="layernorm",
        final_read="learned",
        tau_start=1.8,
        tau_min=0.6,
        device="cpu",
        amp="none",
        seed=7,
        honesty_floor=0.80,
    )
    ap.add_argument("--out-json", default="reports/agent_inspector/speechcommands_acceptance_proof.json")
    ap.add_argument("--out-md", default="reports/agent_inspector/SPEECHCOMMANDS_ACCEPTANCE_PROOF.md")
    return ap


def _make_args(base, **overrides):
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _variant_row(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "variant": report["variant"],
        "status": report["status"],
        "full_acc": report["full_acc"],
        "audit_acc": report["audit_acc"],
        "deploy_acc": report["deploy_acc"],
        "honesty_score": report["honesty_score"],
        "deploy_above_random": report["deploy_above_random"],
        "train_acc": report["train_acc"],
        "train_loss": report["train_loss"],
        "frontend_params": report["frontend"]["params"],
        "frontend_flops": report["frontend"]["flops"],
        "frontend_activation_bytes": report["frontend"]["activation_bytes"],
        "model_params": report["model"]["params"],
        "model_flops": report["model"]["approx_flops"],
        "model_activation_bytes": report["model"]["approx_activation_bytes"],
        "checks": report.get("checks", {}),
    }


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = ["# SpeechCommands acceptance proof", ""]
    lines.append(f"- status: `{report['status']}`")
    lines.append(f"- structured_beats_raw: `{report['checks']['structured_beats_raw']}`")
    lines.append(f"- deploy_above_random: `{report['checks']['deploy_above_random']}`")
    lines.append(f"- honesty_retained: `{report['checks']['honesty_retained']}`")
    lines.append(f"- simulator_ce_ablation_positive: `{report['checks']['simulator_ce_ablation_positive']}`")
    lines.append(f"- non_grid_scanner_usage_positive: `{report['checks']['non_grid_scanner_usage_positive']}`")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| variant | status | full acc | audit acc | deploy acc | honesty | train acc | train loss | params | FLOPs | memory bytes |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["variants"]:
        lines.append(
            f"| {row['variant']} | {row['status']} | {row['full_acc']:.4f} | {row['audit_acc']:.4f} | {row['deploy_acc']:.4f} | "
            f"{row['honesty_score']:.4f} | {row['train_acc']:.4f} | {row['train_loss']:.4f} | "
            f"{row['model_params']:.0f} | {row['model_flops']:.0f} | {row['model_activation_bytes']:.0f} |"
        )
    lines.append("")
    lines.append("## Checks")
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{'PASS' if value else 'FAIL'}`")
    lines.append("")
    lines.append("## Real-data baseline context")
    lines.append(f"- data_root: `{report['baseline']['data_root']}`")
    lines.append(f"- classes: `{', '.join(report['baseline']['classes'])}`")
    lines.append(f"- train_limit: `{report['baseline']['train_limit']}`")
    lines.append(f"- val_limit: `{report['baseline']['val_limit']}`")
    lines.append(f"- test_limit: `{report['baseline']['test_limit']}`")
    lines.append("")
    lines.append("## Raw report")
    lines.append(f"```json\n{json.dumps(report, indent=2, ensure_ascii=False)}\n```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path.cwd()
    import sys

    sys.path.insert(0, str(root))

    from arch_builder.train_audio_frontend import train_variant

    args = parser().parse_args()
    base = parser().parse_args([])

    variants = {}
    for variant in ["raw", "conv", "structured"]:
        variant_args = _make_args(base, variant=variant)
        variant_args.out_dir = str(Path("agent_reports") / f"speechcommands_{variant}_smoke")
        variants[variant] = train_variant(variant_args)

    raw = variants["raw"]
    conv = variants["conv"]
    structured = variants["structured"]
    checks = {
        "raw_reported": all(key in raw for key in ["frontend", "model", "status"]),
        "conv_reported": all(key in conv for key in ["frontend", "model", "status"]),
        "structured_reported": all(key in structured for key in ["frontend", "model", "status"]),
        "structured_beats_raw": structured["deploy_acc"] > raw["deploy_acc"] + 0.01,
        "deploy_above_random": structured["deploy_above_random"],
        "honesty_retained": structured["honesty_score"] >= args.honesty_floor,
        "simulator_ce_ablation_positive": structured.get("checks", {}).get("simulator_ce_ablation_positive", False),
        "non_grid_scanner_usage_positive": structured.get("checks", {}).get("non_grid_scanner_usage_positive", False),
        "conv_scaffold_only": conv["variant"] == "conv",
        "all_variants_have_resource_stats": all(
            row["frontend_params"] > 0 and row["model_params"] > 0 and row["model_flops"] > 0 and row["model_activation_bytes"] > 0
            for row in (_variant_row(raw), _variant_row(conv), _variant_row(structured))
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "checks": checks,
        "variants": [_variant_row(raw), _variant_row(conv), _variant_row(structured)],
        "baseline": {
            "data_root": args.data_root,
            "classes": structured.get("classes", []),
            "train_limit": args.train_limit,
            "val_limit": args.val_limit,
            "test_limit": args.test_limit,
        },
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
    print(f"[speechcommands-acceptance-proof] status={status} checks={checks}")
    if status != "PASS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
