#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def _params(module_or_params: Any) -> list[Any]:
    if hasattr(module_or_params, "parameters"):
        return list(module_or_params.parameters())
    return list(module_or_params)


def _grad_norm_from_grads(grads: Iterable[Any]) -> float:
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().float().norm().cpu()) ** 2
    return total**0.5


def params_grad_norm(params: Iterable[Any]) -> float | None:
    grads = [getattr(param, "grad", None) for param in params]
    if not any(grad is not None for grad in grads):
        return None
    return _grad_norm_from_grads(grads)


def loss_grad_norm(loss: Any, params: Iterable[Any]) -> float:
    params = [param for param in params if getattr(param, "requires_grad", False)]
    if not getattr(loss, "requires_grad", False) or not params:
        return 0.0
    grads = loss_gradients(loss, params)
    return _grad_norm_from_grads(grads)


def loss_gradients(loss: Any, params: list[Any]) -> tuple[Any, ...]:
    import torch

    return torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="One-batch autograd probe for the current acrch_builder API.")
    ap.add_argument("--out", default="reports/agent_inspector/runtime_probe.json")
    ap.add_argument("--task", default="diff", choices=["diff", "two_diff", "merge", "product", "chain_diff_product", "semantic_rescue"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--sim-rank", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--input-norm", default="none", choices=["none", "layernorm"])
    ap.add_argument("--seed", type=int, default=123)

    # Keep these defaults aligned with train_vertical_slice.parser().
    ap.add_argument("--lambda-sim", type=float, default=0.1)
    ap.add_argument("--lambda-choice", type=float, default=1.0)
    ap.add_argument("--lambda-non-expected-primitive", type=float, default=0.25)
    ap.add_argument("--lambda-expected-active", type=float, default=0.05)
    ap.add_argument("--lambda-non-expected-active", type=float, default=0.02)
    ap.add_argument("--lambda-non-expected-tape", type=float, default=0.05)
    ap.add_argument("--lambda-non-expected-transform", type=float, default=0.02)
    ap.add_argument("--lambda-primitive-usage-diversity", type=float, default=0.05)
    ap.add_argument("--lambda-cell-choice-diversity", type=float, default=0.05)
    ap.add_argument("--lambda-active-budget", type=float, default=0.02)
    ap.add_argument("--lambda-tape-budget", type=float, default=0.02)
    ap.add_argument("--lambda-layer-action-diversity", type=float, default=0.05)
    ap.add_argument("--lambda-collapse", type=float, default=0.01)
    ap.add_argument("--min-transform-mass", type=float, default=0.15)
    ap.add_argument("--target-active-fraction", type=float, default=0.18)
    ap.add_argument("--target-tape-fraction", type=float, default=0.015)
    ap.add_argument("--target-active-cells", type=float, default=3.0)
    ap.add_argument("--adapt-choice-floor", type=float, default=0.45)
    ap.add_argument("--adapt-top-share-floor", type=float, default=0.55)
    ap.add_argument("--adapt-sharpness", type=float, default=0.08)
    ap.add_argument("--adapt-cell-sharpness", type=float, default=1.5)
    ap.add_argument("--adapt-choice-boost", type=float, default=1.5)
    return ap


def main() -> int:
    args = parser().parse_args()

    import torch
    import torch.nn.functional as F

    root = Path.cwd()
    sys.path.insert(0, str(root))

    from arch_builder.model import ActionMatrixModel
    from arch_builder.credit import simulator_ablation_metrics
    from arch_builder.synthetic_tasks import SyntheticKnownProgramTask
    from arch_builder.train_vertical_slice import (
        compute_training_objective,
        summarize_trace,
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)

    task = SyntheticKnownProgramTask(task=args.task, slots=args.slots, dim=args.dim)
    model = ActionMatrixModel(
        dim=args.dim,
        slots=args.slots,
        layers=args.layers,
        classes=2,
        top_k=args.top_k,
        sim_rank=args.sim_rank,
        input_norm=args.input_norm,
    ).to(device)
    model.train()
    batch = task.sample(args.batch_size, device)
    logits, trace = model(batch.x, tau=args.tau)
    ce = F.cross_entropy(logits, batch.y)
    loss, raw_losses, signals, gates, weighted = compute_training_objective(
        trace,
        batch,
        model,
        ce,
        args,
    )

    first_layer = model.layers[0]
    choice_controller_params = _params(first_layer.context_logits)
    gate_controller_params: list[Any] = []
    for module in [
        first_layer.mode_head,
        first_layer.edge_gate,
        first_layer.write_gate,
        first_layer.phase_gate,
        first_layer.edge_op,
        first_layer.output_gate,
        first_layer.cell_output_gate,
    ]:
        gate_controller_params.extend(_params(module))
    gate_controller_params.extend([
        first_layer.edge_pair_bias,
        first_layer.write_pair_bias,
        first_layer.phase_pair_bias,
        first_layer.cell_output_pair_bias,
    ])
    simulator_params = _params(first_layer.simulator) + _params(first_layer.sim_logits)
    param_groups = {
        "choice_controller": choice_controller_params,
        "gate_controller": gate_controller_params,
        "scanner": _params(first_layer.scanner),
        "simulator": simulator_params,
        "executor": _params(first_layer.executor),
        "classifier": _params(model.classifier),
    }

    target_groups = {
        "ce_loss": list(param_groups),
        "sim_loss": ["simulator"],
        "expected_choice_loss": ["choice_controller", "scanner", "simulator"],
        "non_expected_primitive_loss": ["choice_controller", "scanner", "simulator"],
        "expected_active_loss": ["gate_controller"],
        "non_expected_active_loss": ["gate_controller"],
        "non_expected_tape_loss": ["gate_controller"],
        "non_expected_transform_loss": ["gate_controller"],
        "primitive_usage_diversity": ["choice_controller", "scanner", "simulator"],
        "cell_choice_diversity": ["choice_controller", "scanner", "simulator"],
        "active_budget": ["gate_controller"],
        "tape_budget": ["gate_controller"],
        "layer_action_diversity": ["choice_controller", "scanner", "simulator"],
        "min_transform_loss": ["gate_controller"],
    }
    loss_connectivity: dict[str, Any] = {}
    applicability = {name: True for name in raw_losses}
    applicability["layer_action_diversity"] = args.layers > 1
    for name, value in raw_losses.items():
        groups = target_groups[name]
        group_norms = {group: loss_grad_norm(value, param_groups[group]) for group in groups}
        loss_connectivity[name] = {
            "value": safe_float(value.detach().cpu()),
            "requires_grad": bool(value.requires_grad),
            "grad_fn": type(value.grad_fn).__name__ if value.grad_fn is not None else None,
            "applicable": applicability[name],
            "target_grad_norms": group_norms,
            "connected_to_intended_target": any(norm > 1e-12 for norm in group_norms.values()),
        }

    recovery_norms = loss_connectivity["expected_choice_loss"]["target_grad_norms"]
    recovery_loss_connected = bool(
        raw_losses["expected_choice_loss"].requires_grad
        and recovery_norms.get("choice_controller", 0.0) > 1e-12
    )

    model.zero_grad(set_to_none=True)
    loss.backward()
    controller_params = choice_controller_params + gate_controller_params
    grad_norms = {
        "controller": params_grad_norm(controller_params),
        "scanner": params_grad_norm(_params(first_layer.scanner)),
        "simulator": params_grad_norm(simulator_params),
        "executor": params_grad_norm(_params(first_layer.executor)),
        "memory": None,
        "classifier": params_grad_norm(_params(model.classifier)),
        "primitive_matrix": params_grad_norm(_params(model.pm)),
        "model_core": params_grad_norm(_params(model)),
    }

    ablations = simulator_ablation_metrics(model, task, args.batch_size, device, args.tau)
    model.eval()
    with torch.no_grad():
        train_acc = float((logits.argmax(dim=-1) == batch.y).float().mean().detach().cpu())
        trace_metrics = summarize_trace(trace, batch, model)

    metrics = {
        "loss": safe_float(loss.detach().cpu()),
        "train_acc": train_acc,
        **ablations,
        **{name: safe_float(value.detach().cpu()) for name, value in raw_losses.items()},
        **{k: safe_float(v) for k, v in signals.items()},
        **{k: safe_float(v) for k, v in gates.items()},
        **{k: safe_float(v) for k, v in trace_metrics.items()},
    }
    weighted_contributions = {name: safe_float(value.detach().cpu()) for name, value in weighted.items()}
    metrics["loss_accounting_error"] = abs(metrics["loss"] - sum(weighted_contributions.values()))

    credit_path = {
        "expected_edge_present": bool(metrics.get("expected_candidate_present", 0.0) > 0),
        "candidate_present": bool(metrics.get("expected_candidate_present", 0.0) > 0),
        "choice_mass": metrics.get("expected_edge_choice_mass", 0.0),
        "edge_active": metrics.get("expected_edge_active", 0.0),
        "recovery": metrics.get("expected_edge_recovery", 0.0),
        "recovery_loss_connected": recovery_loss_connected,
        "simulator_influence": ablations["choice_without_sim_delta"],
    }
    threshold = 1e-12
    critical_groups = ["controller", "scanner", "simulator", "classifier", "model_core"]
    gradient_closed = all((grad_norms.get(key) or 0.0) > threshold for key in critical_groups)
    credit_closed = bool(
        credit_path["candidate_present"]
        and credit_path["choice_mass"] > 1e-3
        and credit_path["edge_active"] > 1e-3
        and credit_path["recovery"] > 1e-3
        and recovery_loss_connected
    )

    report = {
        "schema": "runtime_probe_contract.v3",
        "project": "acrch_builder",
        "producer": "tools/project_probe/probe_learning_loop.py",
        "timestamp": time.time(),
        "run_type": "one_batch_backward_probe",
        "note": "Uses the same canonical objective as training; reports aggregate and per-loss autograd connectivity separately.",
        "config": {
            "task": args.task,
            "device": device,
            "dim": args.dim,
            "slots": args.slots,
            "layers": args.layers,
            "top_k": args.top_k,
            "sim_rank": args.sim_rank,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "input_norm": args.input_norm,
        },
        "metrics": metrics,
        "weighted_loss_contributions": weighted_contributions,
        "loss_connectivity": loss_connectivity,
        "grad_norms": grad_norms,
        "credit_path": credit_path,
        "ablations": {
            **ablations,
        },
        "health": {
            "probe_api_ok": True,
            "gradient_closed": gradient_closed,
            "credit_closed": credit_closed,
            "recovery_loss_connected": recovery_loss_connected,
            "recovery_ok": credit_path["recovery"] > 1e-3,
            "detached_enabled_losses": [
                name
                for name, item in loss_connectivity.items()
                if item["applicable"] and not item["requires_grad"]
            ],
        },
    }
    if not math.isfinite(report["metrics"]["loss"]):
        raise RuntimeError("probe produced a non-finite total loss")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[probe] wrote {out}")
    print(f"[probe] gradient_closed={gradient_closed} credit_closed={credit_closed}")
    print(f"[probe] recovery_loss_connected={recovery_loss_connected}")
    print(f"[probe] detached_enabled_losses={report['health']['detached_enabled_losses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
