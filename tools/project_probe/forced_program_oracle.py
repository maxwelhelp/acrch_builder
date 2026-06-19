#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


AUDIT_TASKS = ("diff", "merge", "product", "two_diff", "chain_diff_merge", "chain_diff_product")
PRIMITIVE_TASKS = ("diff", "merge", "product")


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "mean_abs_error": float(delta.mean().cpu()),
        "max_abs_error": float(delta.max().cpu()),
    }


def label_accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    pred = (output.float().mean(dim=-1) > 0).long()
    return float((pred == labels).float().mean().cpu())


def expected_formula(name: str, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if name == "diff":
        return src - tgt
    if name == "merge":
        return 0.5 * (src + tgt)
    if name == "product":
        return src * tgt
    raise ValueError(f"no oracle formula for primitive {name!r}")


def execute_action(layer, state: torch.Tensor, memory: torch.Tensor, action: dict[str, Any]) -> torch.Tensor:
    src = int(action["src"])
    tgt = int(action["tgt"])
    name = str(action["primitive"])
    pid = layer.pm.name_to_id[name]
    candidate_ids = torch.full((state.shape[0], 1), pid, dtype=torch.long, device=state.device)
    return layer.executor(state[:, src], state[:, tgt], memory, candidate_ids)[:, 0]


def actions_by_layer(actions: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        grouped[int(action["layer"])].append(action)
    return dict(grouped)


def validate_actions(model, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for action in actions:
        layer = int(action["layer"])
        src = int(action["src"])
        tgt = int(action["tgt"])
        primitive = str(action["primitive"])
        check = {
            "action": action,
            "primitive_available": primitive in model.pm.name_to_id,
            "layer_valid": 0 <= layer < model.num_layers,
            "source_valid": 0 <= src < model.slots,
            "target_valid": 0 <= tgt < model.slots,
        }
        check["valid"] = all(value for key, value in check.items() if key != "action")
        checks.append(check)
    return checks


@torch.no_grad()
def raw_primitive_audit(model, batch_size: int, device: str, seed: int) -> list[dict[str, Any]]:
    from arch_builder.synthetic_tasks import SyntheticKnownProgramTask

    reports = []
    for offset, task_name in enumerate(PRIMITIVE_TASKS):
        torch.manual_seed(seed + offset)
        task = SyntheticKnownProgramTask(task=task_name, slots=model.slots, dim=model.dim)
        batch = task.sample(batch_size, device)
        action = batch.expected_actions[0]
        name = str(action["primitive"])
        src = batch.x[:, int(action["src"])]
        tgt = batch.x[:, int(action["tgt"])]
        memory = batch.x.mean(dim=1)
        actual = execute_action(model.layers[0], batch.x, memory, action)
        expected = expected_formula(name, src, tgt)
        reports.append({
            "task": task_name,
            "primitive": name,
            "primitive_available": name in model.pm.name_to_id,
            "primitive_executed": bool(torch.isfinite(actual).all().item()),
            **tensor_error(actual, expected),
            "label_accuracy_from_primitive_output": label_accuracy(actual, batch.y),
        })
    return reports


@torch.no_grad()
def address_audit(model, batch_size: int, device: str, seed: int) -> list[dict[str, Any]]:
    from arch_builder.synthetic_tasks import SyntheticKnownProgramTask

    reports = []
    for offset, task_name in enumerate(PRIMITIVE_TASKS):
        torch.manual_seed(seed + 100 + offset)
        task = SyntheticKnownProgramTask(task=task_name, slots=model.slots, dim=model.dim)
        batch = task.sample(batch_size, device)
        action = batch.expected_actions[0]
        raw_memory = batch.x.mean(dim=1)
        raw_output = execute_action(model.layers[0], batch.x, raw_memory, action)
        # In Stage 3 slot address is controller-only, so executor operands are
        # identical with address enabled or disabled.
        addressed_output = execute_action(model.layers[0], batch.x, raw_memory, action)
        raw_pred = raw_output.float().mean(dim=-1) > 0
        addressed_pred = addressed_output.float().mean(dim=-1) > 0
        reports.append({
            "task": task_name,
            "primitive": str(action["primitive"]),
            "raw_label_accuracy": label_accuracy(raw_output, batch.y),
            "addressed_label_accuracy": label_accuracy(addressed_output, batch.y),
            "address_output_delta": float((addressed_output.float() - raw_output.float()).abs().mean().cpu()),
            "address_label_flip_rate": float((raw_pred != addressed_pred).float().mean().cpu()),
        })
    return reports


@torch.no_grad()
def controller_address_audit(model, batch_size: int, device: str, seed: int) -> dict[str, Any]:
    from arch_builder.synthetic_tasks import SyntheticKnownProgramTask

    torch.manual_seed(seed + 150)
    task = SyntheticKnownProgramTask(task="diff", slots=model.slots, dim=model.dim)
    batch = task.sample(batch_size, device)

    torch.manual_seed(seed + 151)
    logits_with_address, with_address = model(batch.x, disable_slot_address=False)
    torch.manual_seed(seed + 151)
    _, without_address = model(batch.x, disable_slot_address=True)
    torch.manual_seed(seed + 151)
    logits_layer0_output_ablated, _ = model(
        batch.x,
        disable_slot_address=False,
        ablate_layer_output=0,
    )

    deltas = []
    for layer_with, layer_without in zip(with_address["layers"], without_address["layers"]):
        rows = layer_with["candidate_ids"].shape[0]
        p = model.pm.num_primitives
        dist_with = torch.zeros(rows, p, device=device, dtype=layer_with["choice"].dtype)
        dist_without = torch.zeros_like(dist_with)
        dist_with.scatter_add_(1, layer_with["candidate_ids"], layer_with["choice"])
        dist_without.scatter_add_(1, layer_without["candidate_ids"], layer_without["choice"])
        deltas.append((dist_with.float() - dist_without.float()).abs().mean())
    delta = torch.stack(deltas).mean() if deltas else torch.zeros((), device=device)
    return {
        "controller_address_choice_delta": float(delta.cpu()),
        "slot_address_used_by_controller": bool(with_address["slot_address_used_by_controller"]),
        "slot_address_used_by_executor": bool(with_address["slot_address_used_by_executor"]),
        "final_read_last_layer0_output_bypass_delta": float(
            (logits_with_address.float() - logits_layer0_output_ablated.float()).abs().max().cpu()
        ),
    }


@torch.no_grad()
def forced_path(model, batch, addressed: bool) -> dict[str, Any]:
    state = batch.x.clone()
    # Address is controller-only. Forced execution bypasses the controller, so
    # raw and addressed runs must have identical executor/state operands.
    ideal_state = batch.x.clone()
    memory = state.mean(dim=1)
    ideal_memory = ideal_state.mean(dim=1)
    grouped = actions_by_layer(batch.expected_actions)
    action_reports = []
    final_outputs: list[torch.Tensor] = []
    final_layer = max(grouped)

    for layer_idx in range(final_layer + 1):
        layer = model.layers[layer_idx]
        cell_value_grid = torch.zeros(
            state.shape[0], model.slots, model.slots, model.dim,
            dtype=state.dtype, device=state.device,
        )
        cell_write_mass_grid = torch.zeros(
            state.shape[0], model.slots, model.slots, 1,
            dtype=state.dtype, device=state.device,
        )
        ideal_next = ideal_state.clone()
        layer_outputs = []

        for action in grouped.get(layer_idx, []):
            src = int(action["src"])
            tgt = int(action["tgt"])
            name = str(action["primitive"])
            actual = execute_action(layer, state, memory, action)
            ideal = expected_formula(name, ideal_state[:, src], ideal_state[:, tgt])
            cell_value_grid[:, src, tgt] = actual
            cell_write_mass_grid[:, src, tgt] = 1.0
            ideal_next[:, tgt] = ideal
            layer_outputs.append(actual)
            action_reports.append({
                "layer": layer_idx,
                "src": src,
                "tgt": tgt,
                "primitive": name,
                "input_src_error": tensor_error(state[:, src], ideal_state[:, src]),
                "input_tgt_error": tensor_error(state[:, tgt], ideal_state[:, tgt]),
                "primitive_output_error": tensor_error(actual, ideal),
            })

        next_state, _, _ = layer.apply_state_update(
            state,
            cell_value_grid,
            cell_write_mass_grid,
        )
        for item, action in zip(action_reports[-len(layer_outputs):], grouped.get(layer_idx, [])):
            tgt = int(action["tgt"])
            item["target_slot_error_after_state_update"] = tensor_error(next_state[:, tgt], ideal_next[:, tgt])

        state = next_state
        ideal_state = ideal_next
        memory = 0.95 * memory + 0.05 * state.mean(dim=1)
        ideal_memory = 0.95 * ideal_memory + 0.05 * ideal_state.mean(dim=1)
        if layer_idx == final_layer:
            final_outputs = layer_outputs

    if not final_outputs:
        raise RuntimeError("forced path produced no terminal outputs")
    first_divergence = None
    for action in action_reports:
        ordered_checks = [
            ("input_src", action["input_src_error"]),
            ("input_tgt", action["input_tgt_error"]),
            ("primitive_output", action["primitive_output_error"]),
            ("target_slot_after_state_update", action["target_slot_error_after_state_update"]),
        ]
        for stage, error in ordered_checks:
            if error["max_abs_error"] > 1e-6:
                first_divergence = {
                    "layer": action["layer"],
                    "src": action["src"],
                    "tgt": action["tgt"],
                    "primitive": action["primitive"],
                    "stage": stage,
                    **error,
                }
                break
        if first_divergence is not None:
            break
    # The current proof tasks either have one terminal action or use the mean of
    # terminal actions; mean preserves the sign of the declared sum readout.
    final_output = torch.stack(final_outputs, dim=0).mean(dim=0)
    return {
        "path": "addressed" if addressed else "raw_content",
        "read_rule": "mean_terminal_action_outputs",
        "final_label_accuracy": label_accuracy(final_output, batch.y),
        "first_divergence": first_divergence,
        "actions": action_reports,
    }


@torch.no_grad()
def sequential_audit(model, batch_size: int, device: str, seed: int) -> list[dict[str, Any]]:
    from arch_builder.synthetic_tasks import SyntheticKnownProgramTask

    reports = []
    for offset, task_name in enumerate(AUDIT_TASKS):
        torch.manual_seed(seed + 200 + offset)
        task = SyntheticKnownProgramTask(task=task_name, slots=model.slots, dim=model.dim)
        batch = task.sample(batch_size, device)
        checks = validate_actions(model, batch.expected_actions)
        reports.append({
            "task": task_name,
            "action_invariants": checks,
            "invariants_pass": all(check["valid"] for check in checks),
            "raw_content": forced_path(model, batch, addressed=False),
            "addressed": forced_path(model, batch, addressed=True),
        })
    return reports


@torch.no_grad()
def state_update_contract(model, batch_size: int, device: str, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed + 999)
    layer = model.layers[0]
    state = torch.randn(batch_size, model.slots, model.dim, device=device)
    values = torch.randn(batch_size, model.slots, model.slots, model.dim, device=device)
    zero_mass = torch.zeros(batch_size, model.slots, model.slots, 1, device=device)

    no_write, _, no_write_gate = layer.apply_state_update(state, values, zero_mass)
    no_write_error = tensor_error(no_write, state)

    unit_values = torch.zeros_like(values)
    unit_mass = torch.zeros_like(zero_mass)
    forced_value = torch.randn(batch_size, model.dim, device=device)
    unit_values[:, 2, 1] = forced_value
    unit_mass[:, 2, 1] = 1.0
    unit_out, unit_write_value, unit_gate = layer.apply_state_update(state, unit_values, unit_mass)
    unit_target_error = tensor_error(unit_out[:, 1], forced_value)
    unit_value_error = tensor_error(unit_write_value[:, 1], forced_value)

    multi_values = torch.zeros_like(values)
    multi_mass = torch.zeros_like(zero_mass)
    value0 = torch.randn(batch_size, model.dim, device=device)
    value1 = torch.randn(batch_size, model.dim, device=device)
    multi_values[:, 0, 2] = value0
    multi_values[:, 1, 2] = value1
    multi_mass[:, 0, 2] = 0.25
    multi_mass[:, 1, 2] = 0.50
    multi_out, multi_write_value, multi_gate = layer.apply_state_update(state, multi_values, multi_mass)
    expected_value = (0.25 * value0 + 0.50 * value1) / 0.75
    expected_gate = torch.full((batch_size, 1), 1.0 - (1.0 - 0.25) * (1.0 - 0.50), device=device)
    expected_target = (1.0 - expected_gate) * state[:, 2] + expected_gate * expected_value
    multi_value_error = tensor_error(multi_write_value[:, 2], expected_value)
    multi_target_error = tensor_error(multi_out[:, 2], expected_target)
    multi_gate_error = tensor_error(multi_gate[:, 2], expected_gate)

    passed = all(
        item["max_abs_error"] <= 1e-6
        for item in [
            no_write_error,
            unit_target_error,
            unit_value_error,
            multi_value_error,
            multi_target_error,
            multi_gate_error,
        ]
    ) and float(no_write_gate.abs().max().cpu()) == 0.0 and float(unit_gate[:, 1].min().cpu()) == 1.0
    return {
        "state_norm_mode": layer.state_norm_mode,
        "no_write_preserve_error": no_write_error,
        "no_write_gate_max": float(no_write_gate.abs().max().cpu()),
        "unit_write_target_error": unit_target_error,
        "unit_write_value_error": unit_value_error,
        "unit_write_gate_min": float(unit_gate[:, 1].min().cpu()),
        "multiple_write_value_error": multi_value_error,
        "multiple_write_target_error": multi_target_error,
        "multiple_write_gate_error": multi_gate_error,
        "contract_pass": passed,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Forced Program Oracle", ""]
    contract = report["state_update_contract"]
    lines.append(f"- state_update_contract_pass: `{contract['contract_pass']}`")
    lines.append(f"- raw_primitive_pass: `{report['summary']['raw_primitive_pass']}`")
    lines.append(f"- invariants_pass: `{report['summary']['invariants_pass']}`")
    lines.append(f"- controller_address_choice_delta: `{report['controller_address_audit']['controller_address_choice_delta']}`")
    lines.append(
        f"- final_read_last_layer0_output_bypass_delta: "
        f"`{report['controller_address_audit']['final_read_last_layer0_output_bypass_delta']}`"
    )
    lines.append("")
    lines.append("## Raw primitive audit")
    lines.append("")
    for item in report["raw_primitive_audit"]:
        lines.append(
            f"- `{item['task']}`/{item['primitive']}: accuracy=`{item['label_accuracy_from_primitive_output']:.4f}` "
            f"max_error=`{item['max_abs_error']:.6g}`"
        )
    lines.append("")
    lines.append("## Address contamination")
    lines.append("")
    for item in report["address_contamination_audit"]:
        lines.append(
            f"- `{item['task']}`: raw=`{item['raw_label_accuracy']:.4f}` "
            f"addressed=`{item['addressed_label_accuracy']:.4f}` "
            f"flip_rate=`{item['address_label_flip_rate']:.4f}` "
            f"output_delta=`{item['address_output_delta']:.6g}`"
        )
    lines.append("")
    lines.append("## Sequential state audit")
    lines.append("")
    for item in report["sequential_state_audit"]:
        raw = item["raw_content"]["final_label_accuracy"]
        addressed = item["addressed"]["final_label_accuracy"]
        lines.append(
            f"- `{item['task']}`: invariants=`{item['invariants_pass']}` "
            f"raw_path_accuracy=`{raw:.4f}` addressed_path_accuracy=`{addressed:.4f}` "
            f"raw_first_divergence=`{item['raw_content']['first_divergence']}` "
            f"addressed_first_divergence=`{item['addressed']['first_divergence']}`"
        )
        for action in item["raw_content"]["actions"]:
            lines.append(
                f"  - L{action['layer']} {action['src']}->{action['tgt']} {action['primitive']}: "
                f"input_src_mae=`{action['input_src_error']['mean_abs_error']:.6g}` "
                f"input_tgt_mae=`{action['input_tgt_error']['mean_abs_error']:.6g}` "
                f"primitive_mae=`{action['primitive_output_error']['mean_abs_error']:.6g}` "
                f"target_after_update_mae=`{action['target_slot_error_after_state_update']['mean_abs_error']:.6g}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Forced known-program contract audit without training.")
    ap.add_argument("--out", default="reports/agent_inspector/forced_program_oracle.json")
    ap.add_argument("--markdown", default="reports/agent_inspector/FORCED_PROGRAM_ORACLE.md")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    root = Path.cwd()
    sys.path.insert(0, str(root))
    from arch_builder.model import ActionMatrixModel

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = ActionMatrixModel(
        dim=args.dim,
        slots=args.slots,
        layers=2,
        classes=2,
        top_k=25,
        sim_rank=8,
        input_norm="none",
        state_norm="none",
        final_read="last",
    ).to(device).eval()

    raw = raw_primitive_audit(model, args.batch_size, device, args.seed)
    address = address_audit(model, args.batch_size, device, args.seed)
    controller_address = controller_address_audit(model, min(args.batch_size, 128), device, args.seed)
    sequential = sequential_audit(model, args.batch_size, device, args.seed)
    contract = state_update_contract(model, min(args.batch_size, 32), device, args.seed)
    report = {
        "schema": "forced_program_oracle.v2",
        "config": {
            "device": device,
            "dim": args.dim,
            "slots": args.slots,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "state_update_contract": contract,
        "raw_primitive_audit": raw,
        "address_contamination_audit": address,
        "controller_address_audit": controller_address,
        "sequential_state_audit": sequential,
        "summary": {
            "state_update_contract_pass": contract["contract_pass"],
            "raw_primitive_pass": all(
                item["primitive_available"]
                and item["primitive_executed"]
                and item["max_abs_error"] <= 1e-7
                and item["label_accuracy_from_primitive_output"] == 1.0
                for item in raw
            ),
            "invariants_pass": all(item["invariants_pass"] for item in sequential),
            "address_content_separated": all(
                item["address_output_delta"] == 0.0
                and item["address_label_flip_rate"] == 0.0
                for item in address
            ),
            "controller_address_alive": controller_address["controller_address_choice_delta"] > 0.0,
            "final_read_last_has_no_layer0_output_bypass": (
                controller_address["final_read_last_layer0_output_bypass_delta"] == 0.0
            ),
            "sequential_contract_pass": all(
                item["raw_content"]["final_label_accuracy"] == 1.0
                and item["addressed"]["final_label_accuracy"] == 1.0
                for item in sequential
            ),
        },
    }

    out = Path(args.out)
    md = Path(args.markdown)
    out.parent.mkdir(parents=True, exist_ok=True)
    md.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md, report)
    print(f"[forced-oracle] wrote {out}")
    print(f"[forced-oracle] wrote {md}")
    print(f"[forced-oracle] summary={report['summary']}")
    return 0 if all(report["summary"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
