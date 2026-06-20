#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def parser() -> argparse.ArgumentParser:
    from arch_builder.train_vertical_slice import parser as train_parser

    ap = train_parser()
    ap.set_defaults(
        task_config="configs/tasks/chain_diff_product.yml",
        task="chain_diff_product",
        epochs=1,
        max_steps=4,
        batch_size=16,
        eval_steps=1,
        eval_batch_size=32,
        dim=16,
        top_k=8,
        sim_rank=8,
        device="cpu",
        amp="none",
        curriculum_schedule="teacher",
        honesty_floor=0.80,
    )
    ap.add_argument("--transfer-task-config", default="configs/tasks/chain_diff_merge.yml")
    ap.add_argument("--out-json", default="reports/agent_inspector/honesty_curriculum_proof.json")
    ap.add_argument("--out-md", default="reports/agent_inspector/HONESTY_CURRICULUM_PROOF.md")
    return ap


def _clone_with_slot_shuffle(model, mode: str):
    clone = copy.deepcopy(model)
    with torch.no_grad():
        if mode == "shuffle":
            clone.slot_embed.copy_(clone.slot_embed.flip(0))
        elif mode == "random":
            clone.slot_embed.copy_(torch.randn_like(clone.slot_embed))
        else:
            raise ValueError(mode)
    return clone


def _mode_eval(
    model,
    task,
    batch,
    tau: float,
    mode: str,
    disable_slot_address: bool,
) -> Dict[str, Any]:
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        logits, trace = model(
            batch.x,
            tau=tau,
            disable_slot_address=disable_slot_address,
            curriculum_mode=mode,
        )
        loss = F.cross_entropy(logits, batch.y)
        pred = logits.argmax(dim=-1)
        acc = float((pred == batch.y).float().mean().cpu())
        oracle_acc = task.oracle_accuracy(batch)

    from arch_builder.train_vertical_slice import summarize_trace

    summary = summarize_trace(trace, batch, model)
    summary.update(
        {
            "loss": float(loss.detach().cpu()),
            "acc": acc,
            "oracle_acc": oracle_acc,
        }
    )
    return summary


def _train_smoke(model, task, args) -> None:
    import torch
    import torch.nn.functional as F
    from arch_builder.train_vertical_slice import compute_training_objective

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for step in range(max(1, int(args.max_steps))):
        batch = task.sample(args.batch_size, args.device)
        opt.zero_grad(set_to_none=True)
        logits, trace = model(batch.x, tau=args.tau_start, curriculum_mode="teacher")
        ce = F.cross_entropy(logits, batch.y)
        loss, _, _, _, _ = compute_training_objective(trace, batch, model, ce, args)
        loss.backward()
        opt.step()
        scaler.update()


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    lines = ["# Honesty curriculum proof", "", f"- status: `{report['status']}`"]
    for key in [
        "honesty_score",
        "honesty_floor",
        "full_acc",
        "no_conv_acc",
        "no_hints_acc",
        "deploy_acc",
        "shuffled_acc",
        "random_acc",
        "transfer_acc",
        "deploy_credit_written",
    ]:
        lines.append(f"- {key}: `{report[key]}`")

    lines += ["", "## Checks", ""]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{'PASS' if value else 'FAIL'}`")

    lines += ["", "## Mode metrics", "", f"```json\n{json.dumps(report['modes'], indent=2, ensure_ascii=False)}\n```"]
    lines += ["", "## Credit metrics", "", f"```json\n{json.dumps(report['credit_metrics'], indent=2, ensure_ascii=False)}\n```"]
    lines += ["", "## Program summary", "", f"```json\n{json.dumps(report['program'], indent=2, ensure_ascii=False)}\n```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path.cwd()
    sys.path.insert(0, str(root))

    from arch_builder.credit import ModeCreditLedger
    from arch_builder.model import ActionMatrixModel
    from arch_builder.synthetic_tasks import SyntheticKnownProgramTask
    from arch_builder.task_config import load_task_config
    from arch_builder.train_vertical_slice import inspect_action_program

    args = parser().parse_args()
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    task_cfg = load_task_config(args.task_config)
    transfer_cfg = load_task_config(args.transfer_task_config)
    task = SyntheticKnownProgramTask(config=task_cfg, dim=args.dim)
    transfer_task = SyntheticKnownProgramTask(config=transfer_cfg, dim=args.dim)

    model = ActionMatrixModel(
        dim=args.dim,
        slots=task.slots,
        layers=task_cfg.layers,
        classes=2,
        top_k=args.top_k,
        sim_rank=args.sim_rank,
        input_norm=args.input_norm,
        state_norm=args.state_norm,
        final_read=args.final_read,
    ).to(device)

    args.device = device
    _train_smoke(model, task, args)

    batch = task.sample(args.eval_batch_size, device)
    transfer_batch = transfer_task.sample(args.eval_batch_size, device)

    full = _mode_eval(model, task, batch, args.tau_min, "teacher", False)
    no_conv = _mode_eval(model, task, batch, args.tau_min, "audit", False)
    no_hints = _mode_eval(model, task, batch, args.tau_min, "deploy", False)
    deploy = _mode_eval(model, task, batch, args.tau_min, "deploy", True)

    shuffled_model = _clone_with_slot_shuffle(model, "shuffle")
    shuffled = _mode_eval(shuffled_model, task, batch, args.tau_min, "teacher", False)

    random_model = _clone_with_slot_shuffle(model, "random")
    random_hints = _mode_eval(random_model, task, batch, args.tau_min, "teacher", False)

    transfer = _mode_eval(model, transfer_task, transfer_batch, args.tau_min, "deploy", True)

    honesty_score = deploy["acc"] / max(1e-8, full["acc"])
    credit = ModeCreditLedger()
    credit.update(
        "teacher",
        {
            "acc": full["acc"],
            "choice_mass": float(full.get("expected_edge_choice_mass", 0.0)),
            "recovery": float(full.get("expected_edge_recovery", 0.0)),
        },
    )
    credit.update(
        "audit",
        {
            "acc": no_conv["acc"],
            "choice_mass": float(no_conv.get("expected_edge_choice_mass", 0.0)),
            "recovery": float(no_conv.get("expected_edge_recovery", 0.0)),
        },
    )
    deploy_credit_written = honesty_score >= args.honesty_floor
    if deploy_credit_written:
        credit.update(
            "deploy",
            {
                "acc": deploy["acc"],
                "choice_mass": float(deploy.get("expected_edge_choice_mass", 0.0)),
                "recovery": float(deploy.get("expected_edge_recovery", 0.0)),
            },
        )

    program = inspect_action_program(model, task, args.eval_batch_size, device, args.tau_min)
    full_program_verdicts = [layer.get("program_verdict") for layer in program.get("layers", [])]
    readable_program = (
        float(full.get("program_recovery_rate", 0.0)) >= 0.10
        and float(full.get("expected_candidate_present", 0.0)) >= 0.25
        and float(full.get("primitive_top_share", 1.0)) <= 0.30
    )
    diverse_program = float(full.get("primitive_top_share", 1.0)) < 0.95 or float(full.get("layer_action_similarity", 1.0)) < 0.98

    checks = {
        "no_static_runtime_leakage": shuffled["acc"] <= deploy["acc"] + 1e-6 and random_hints["acc"] <= deploy["acc"] + 1e-6,
        "honesty_score_ok": honesty_score >= 0.80,
        "shuffled_not_beating_deploy": shuffled["acc"] <= deploy["acc"] + 1e-6,
        "random_not_beating_deploy": random_hints["acc"] <= deploy["acc"] + 1e-6,
        "deploy_credit_contains_deploy_only": deploy_credit_written,
        "readable_program": readable_program,
        "program_diversity": diverse_program,
        "transfer_defined": math.isfinite(float(transfer["acc"])),
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "status": status,
        "task": task.task,
        "transfer_task": transfer_task.task,
        "honesty_floor": float(args.honesty_floor),
        "honesty_score": float(honesty_score),
        "full_acc": full["acc"],
        "no_conv_acc": no_conv["acc"],
        "no_hints_acc": no_hints["acc"],
        "deploy_acc": deploy["acc"],
        "shuffled_acc": shuffled["acc"],
        "random_acc": random_hints["acc"],
        "transfer_acc": transfer["acc"],
        "deploy_credit_written": deploy_credit_written,
        "modes": {
            "full": full,
            "no_conv": no_conv,
            "no_hints": no_hints,
            "deploy": deploy,
            "shuffled": shuffled,
            "random": random_hints,
            "transfer": transfer,
        },
        "credit_metrics": credit.metrics(),
        "program": {
            "task": program.get("task"),
            "batch_acc": program.get("batch_acc"),
            "final_read_mode": program.get("final_read_mode"),
            "state_norm_mode": program.get("state_norm_mode"),
            "expected_actions": program.get("expected_actions"),
            "action_metrics": program.get("action_metrics"),
            "program_verdicts": full_program_verdicts,
            "program_active_cells": [layer.get("active_cells") for layer in program.get("layers", [])],
            "program_expected_top_cells": [layer.get("expected_top_cells") for layer in program.get("layers", [])],
            "program_diversity": diverse_program,
        },
        "checks": checks,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(out_md, report)
    print(f"[honesty-curriculum-proof] status={status} honesty={honesty_score:.3f} checks={checks}")
    if status != "PASS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
