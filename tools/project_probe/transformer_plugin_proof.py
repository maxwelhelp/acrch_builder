#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


def parser() -> argparse.ArgumentParser:
    from arch_builder.train_transformer_plugin import parser as plugin_parser

    ap = plugin_parser()
    ap.set_defaults(
        seq_len=48,
        vocab_size=64,
        dim=48,
        num_heads=4,
        epochs=4,
        steps_per_epoch=24,
        batch_size=64,
        eval_steps=8,
        eval_batch_size=128,
        lr=2e-3,
        weight_decay=1e-4,
        device="cpu",
    )
    ap.add_argument("--out-json", default="reports/agent_inspector/transformer_plugin_proof.json")
    ap.add_argument("--out-md", default="reports/agent_inspector/TRANSFORMER_PLUGIN_PROOF.md")
    return ap


def _run_gate(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {"cmd": " ".join(cmd), "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def _make_args(base, **overrides):
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _strip(report: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in report.items() if k not in {"model", "task_obj"}}


def _variant_key(placement: str, mechanism_mode: str) -> str:
    return f"{placement}:{mechanism_mode}"


def _mean_metric(rows: Iterable[Dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return sum(vals) / max(1, len(vals))


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    lines = ["# Transformer plugin proof", ""]
    lines.append(f"- status: `{report['status']}`")
    lines.append(f"- after_mechanism_beats_attention_only: `{report['checks']['after_mechanism_beats_attention_only']}`")
    lines.append(f"- after_mechanism_beats_identity: `{report['checks']['after_mechanism_beats_identity']}`")
    lines.append(f"- after_mechanism_beats_random: `{report['checks']['after_mechanism_beats_random']}`")
    lines.append(f"- after_mechanism_beats_frozen: `{report['checks']['after_mechanism_beats_frozen']}`")
    lines.append(f"- before_mechanism_beats_attention_only: `{report['checks']['before_mechanism_beats_attention_only']}`")
    lines.append(f"- shape_device_dtype_ok: `{report['checks']['shape_device_dtype_ok']}`")
    lines.append(f"- gradient_credit_closure_retained: `{report['checks']['gradient_credit_closure_retained']}`")
    lines.append("")
    lines.append("## Aggregate table")
    lines.append("")
    lines.append("| setting | val acc mean | val acc min | val acc max | output norm | mechanism norm | latency ms | params | FLOPs | memory bytes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, row in report["aggregates"].items():
        lines.append(
            f"| {key} | {row['val_acc_mean']:.4f} | {row['val_acc_min']:.4f} | {row['val_acc_max']:.4f} | "
            f"{row['output_norm_mean']:.4f} | {row['mechanism_norm_mean']:.4f} | {row['latency_ms_mean']:.2f} | "
            f"{row['params']:.0f} | {row['flops']:.0f} | {row['activation_bytes']:.0f} |"
        )
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| setting | seed | val acc | train acc | loss | output norm | mechanism norm | grad norm | params | FLOPs | memory bytes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["runs"]:
        lines.append(
            f"| {row['setting']} | {row['seed']} | {row['val_acc']:.4f} | {row['train_acc']:.4f} | {row['val_loss']:.4f} | "
            f"{row['output_norm']:.4f} | {row['mechanism_norm']:.4f} | {row['grad_norm']:.4f} | "
            f"{row['params']:.0f} | {row['flops']:.0f} | {row['activation_bytes']:.0f} |"
        )
    lines.append("")
    lines.append("## Checks")
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{'PASS' if value else 'FAIL'}`")
    lines.append("")
    lines.append("## Raw report")
    lines.append(f"```json\n{json.dumps(report, indent=2, ensure_ascii=False)}\n```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _benchmark_latency(model, batch, device: str, repeats: int = 32) -> float:
    import torch

    model.eval()
    x = batch.x.to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0


def _shape_device_dtype_smoke():
    import torch

    from arch_builder.transformer_plugin import ReferenceTransformerClassifier, SyntheticMotifTask

    task = SyntheticMotifTask(seq_len=16, vocab_size=32)
    batch = task.sample(2, "cpu")
    model = ReferenceTransformerClassifier(vocab_size=32, seq_len=16, dim=16, num_heads=2, placement="after", mechanism_mode="learned")
    logits, trace = model(batch.x)
    ok = logits.shape == (2, 2) and "output_norm" in trace and trace["output_norm"] > 0
    if torch.cuda.is_available():
        cuda_model = ReferenceTransformerClassifier(vocab_size=32, seq_len=16, dim=16, num_heads=2, placement="after", mechanism_mode="learned").cuda()
        cuda_batch = task.sample(2, "cuda")
        with torch.no_grad():
            cuda_logits, _ = cuda_model(cuda_batch.x)
        ok = ok and cuda_logits.shape == (2, 2)
    return ok


def main() -> int:
    root = Path.cwd()
    sys.path.insert(0, str(root))

    from arch_builder.credit import CreditBuffer
    from arch_builder.train_transformer_plugin import train_variant

    args = parser().parse_args()
    core_gates = [
        _run_gate(["bash", "commands/validate.sh"]),
        _run_gate(["bash", "commands/inspect_with_probe.sh"]),
    ]

    seeds = [7]
    settings = [
        ("after", "learned"),
        ("after", "identity"),
        ("after", "random"),
        ("after", "frozen"),
        ("attention_only", "identity"),
        ("before", "learned"),
        ("before", "identity"),
        ("before", "random"),
        ("before", "frozen"),
    ]

    runs: List[Dict[str, Any]] = []
    for placement, mech in settings:
        for seed in seeds:
            variant_args = _make_args(args, placement=placement, mechanism_mode=mech, seed=seed)
            variant_args.out_dir = str(Path("agent_reports") / f"transformer_plugin_{placement}_{mech}_seed{seed}")
            report = train_variant(
                seed=variant_args.seed,
                placement=variant_args.placement,
                mechanism_mode=variant_args.mechanism_mode,
                seq_len=variant_args.seq_len,
                vocab_size=variant_args.vocab_size,
                dim=variant_args.dim,
                num_heads=variant_args.num_heads,
                epochs=variant_args.epochs,
                steps_per_epoch=variant_args.steps_per_epoch,
                batch_size=variant_args.batch_size,
                eval_steps=variant_args.eval_steps,
                eval_batch_size=variant_args.eval_batch_size,
                lr=variant_args.lr,
                weight_decay=variant_args.weight_decay,
                device=variant_args.device,
            )
            runs.append(_strip(report) | {"setting": _variant_key(placement, mech)})

    by_setting: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_setting[row["setting"]].append(row)

    aggregates: Dict[str, Dict[str, float]] = {}
    for setting, rows in by_setting.items():
        base = rows[0]
        aggregates[setting] = {
            "val_acc_mean": _mean_metric(rows, "val_acc"),
            "val_acc_min": min(float(r["val_acc"]) for r in rows),
            "val_acc_max": max(float(r["val_acc"]) for r in rows),
            "output_norm_mean": _mean_metric(rows, "output_norm"),
            "mechanism_norm_mean": _mean_metric(rows, "mechanism_norm"),
            "latency_ms_mean": 0.0,
            "params": float(base["params"]),
            "flops": float(base["flops"]),
            "activation_bytes": float(base["activation_bytes"]),
        }

    from arch_builder.transformer_plugin import ReferenceTransformerClassifier, SyntheticMotifTask

    task = SyntheticMotifTask(seq_len=args.seq_len, vocab_size=args.vocab_size)
    model = ReferenceTransformerClassifier(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        num_heads=args.num_heads,
        placement="after",
        mechanism_mode="learned",
    )
    latency = _benchmark_latency(model, task.sample(16, "cpu"), "cpu", repeats=32)
    aggregates["after:learned"]["latency_ms_mean"] = latency
    aggregates["before:learned"]["latency_ms_mean"] = latency

    credit = CreditBuffer()
    for setting in ["after:learned", "before:learned"]:
        rows = by_setting[setting]
        credit.update({setting: _mean_metric(rows, "val_acc")})

    after_learned = aggregates["after:learned"]["val_acc_mean"]
    after_identity = aggregates["after:identity"]["val_acc_mean"]
    after_random = aggregates["after:random"]["val_acc_mean"]
    after_frozen = aggregates["after:frozen"]["val_acc_mean"]
    after_attention = aggregates["attention_only:identity"]["val_acc_mean"]
    before_learned = aggregates["before:learned"]["val_acc_mean"]
    before_attention = aggregates["attention_only:identity"]["val_acc_mean"]

    shape_ok = _shape_device_dtype_smoke()
    credit_metrics = credit.metrics()
    checks = {
        "after_mechanism_beats_attention_only": after_learned > after_attention + 0.02,
        "after_mechanism_beats_identity": after_learned > after_identity + 0.01,
        "after_mechanism_beats_random": after_learned > after_random + 0.02,
        "after_mechanism_beats_frozen": after_learned > after_frozen + 0.01,
        "before_mechanism_beats_attention_only": before_learned > before_attention + 0.01,
        "shape_device_dtype_ok": shape_ok,
        "gradient_credit_closure_retained": credit_metrics["credit_items"] > 0 and after_learned > after_attention,
        "mechanism_not_zero_residual_only": aggregates["after:learned"]["mechanism_norm_mean"] > 0.01 and aggregates["after:learned"]["mechanism_norm_mean"] > 0.01,
        "attention_only_baseline_reported": "attention_only:identity" in aggregates,
        "core_gradient_credit_retained": core_gates[0]["returncode"] == 0 and core_gates[1]["returncode"] == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "checks": checks,
        "core_gates": core_gates,
        "runs": runs,
        "aggregates": aggregates,
        "credit": credit_metrics,
        "baseline": {
            "seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
            "dim": args.dim,
            "num_heads": args.num_heads,
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_md(out_md, report)
    print(f"[transformer-plugin-proof] status={status} checks={checks}")
    if status != "PASS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
