from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .model import ActionMatrixModel
from .synthetic_tasks import SyntheticKnownProgramTask
from .credit import CreditBuffer, ModeCreditLedger, simulator_ablation_metrics
from .reporting import ensure_dir, write_json, append_csv, write_latest_report
from .task_config import evaluate_pass_thresholds, legacy_task_path, load_task_config


def amp_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _actions_by_layer(batch) -> Dict[int, List[Dict[str, object]]]:
    out: Dict[int, List[Dict[str, object]]] = {}
    for act in getattr(batch, "expected_actions", []):
        out.setdefault(int(act["layer"]), []).append(act)
    if not out:
        out = {0: [{"layer": 0, "src": batch.expected_src, "tgt": batch.expected_tgt, "primitive": batch.expected_primitive}]}
    return out


def _action_metric_prefix(action: Dict[str, object]) -> str:
    primitive = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in str(action["primitive"]).lower()
    )
    return f"action_L{int(action['layer'])}_{int(action['src'])}_{int(action['tgt'])}_{primitive}"


def curriculum_phase(epoch: int, total_epochs: int, schedule: str) -> str:
    if schedule == "teacher":
        return "teacher"
    if schedule != "phased":
        raise ValueError(f"unknown curriculum schedule: {schedule!r}")
    teacher_end = max(1, int(total_epochs * 0.33))
    audit_end = max(teacher_end + 1, int(total_epochs * 0.66))
    if epoch <= teacher_end:
        return "teacher"
    if epoch <= audit_end:
        return "audit"
    return "deploy"


def _next_layer_context(trace_layer: Dict[str, object], out: torch.Tensor, model: ActionMatrixModel) -> Dict[str, torch.Tensor]:
    slots = model.slots
    b = trace_layer["candidate_ids"].shape[0] // (slots * slots)
    action_dist = _primitive_distribution(trace_layer, model.pm.num_primitives)
    action_dist = action_dist.view(b, slots * slots, -1).mean(dim=1)
    active = trace_layer["active"].view(b, slots * slots)
    write_mass = trace_layer["cell_write_mass"].view(b, slots * slots)
    return {
        "output": out,
        "action_dist": action_dist,
        "active_mass": active.mean(dim=1, keepdim=True),
        "write_mass": write_mass.mean(dim=1, keepdim=True),
    }


def _primitive_distribution(
    layer0: Dict[str, torch.Tensor],
    num_primitives: int,
    choice_key: str = "choice",
) -> torch.Tensor:
    cand = layer0["candidate_ids"]
    choice = layer0[choice_key]
    rows = cand.shape[0]
    out = torch.zeros(rows, num_primitives, device=choice.device, dtype=choice.dtype)
    out.scatter_add_(1, cand, choice)
    return out


def summarize_trace(trace: Dict[str, object], batch, model: ActionMatrixModel) -> Dict[str, float]:
    slots = model.slots
    actions_by_layer = _actions_by_layer(batch)

    recoveries, candidate_present, choice_masses, actives, expected_any = [], [], [], [], []
    choice_entropy, transform_mass, skip_mass, disable_mass = [], [], [], []
    listen_scores, layer_action_dists = [], []
    edge_scale_mean, cell_output_gate_mean, cell_tape_weight_mean = [], [], []
    cell_write_mass_mean, target_write_gate_mean = [], []
    slot_alive_mean, slot_alive_count = [], []
    split_none_mean, split_one_mean, split_two_mean = [], [], []
    child_gate_mean, merge_gate_mean, collector_mass_mean = [], [], []
    active_cells_by_layer, expected_top_cells_by_layer, primitive_top_share_by_layer = [], [], []
    active_edges_per_target_by_layer = []
    action_metrics: Dict[str, float] = {}

    for layer_idx, layer0 in enumerate(trace["layers"]):
        chosen_flat = layer0["chosen"]
        chosen_edges = chosen_flat.view(-1, slots, slots)
        cand = layer0["candidate_ids"]
        choice = layer0["choice"]

        prim_dist_rows = _primitive_distribution(layer0, model.pm.num_primitives)
        cell_dist = prim_dist_rows.view(chosen_edges.shape[0], slots * slots, -1).mean(dim=0)
        global_prim = cell_dist.mean(dim=0)
        primitive_top_share_by_layer.append(global_prim.max().item())
        layer_action_dists.append(global_prim)

        mode = layer0["mode"]
        transform_mass.append(mode[:, 0].mean().item())
        skip_mass.append(mode[:, 1].mean().item())
        disable_mass.append(mode[:, 2].mean().item())
        choice_entropy.append(float((-(choice + 1e-8) * (choice + 1e-8).log()).sum(dim=-1).mean().cpu()))
        if layer_idx > 0 and layer0.get("listen_gate") is not None:
            listen_scores.append(float(layer0["listen_gate"].mean().cpu()))

        edge = layer0["edge"].view(-1, slots, slots, 1)
        write = layer0["write"].view(-1, slots, slots, 1)
        phase = layer0["phase"].view(-1, slots, slots, 1)
        active = (edge * write * phase).squeeze(-1)
        tape_grid = layer0.get("cell_tape_weight")
        if tape_grid is not None:
            tape_grid = tape_grid.view(-1, slots, slots)
        active_mean_grid = active.mean(dim=0)
        active_cells_by_layer.append((active_mean_grid > 0.05).float().sum().item())
        active_edges_per_target_by_layer.append((active_mean_grid > 0.05).float().sum(dim=0).mean().item())

        if "edge_scale" in layer0:
            edge_scale_mean.append(layer0["edge_scale"].mean().item())
        if "cell_output_gate" in layer0:
            cell_output_gate_mean.append(layer0["cell_output_gate"].mean().item())
        if "cell_tape_weight" in layer0:
            cell_tape_weight_mean.append(layer0["cell_tape_weight"].mean().item())
        if "cell_write_mass" in layer0:
            cell_write_mass_mean.append(layer0["cell_write_mass"].mean().item())
        if "target_write_gate" in layer0:
            target_write_gate_mean.append(layer0["target_write_gate"].mean().item())
        if "slot_alive" in layer0:
            slot_alive = layer0["slot_alive"].view(-1, slots)
            slot_alive_mean.append(slot_alive.mean().item())
            slot_alive_count.append((slot_alive > 0.5).float().sum(dim=1).mean().item())
        if "split_count" in layer0:
            split_count = layer0["split_count"].view(-1, slots, 3)
            split_none_mean.append(split_count[..., 0].mean().item())
            split_one_mean.append(split_count[..., 1].mean().item())
            split_two_mean.append(split_count[..., 2].mean().item())
        if "child_gate" in layer0:
            child_gate_mean.append(layer0["child_gate"].mean().item())
        if "merge_gate" in layer0:
            merge_gate_mean.append(layer0["merge_gate"].mean().item())
        if "collector_mass" in layer0:
            collector_mass_mean.append(layer0["collector_mass"].mean().item())

        layer_actions = actions_by_layer.get(layer_idx, [])
        expected_names = {str(a["primitive"]) for a in layer_actions}
        exp_top_cells = 0
        for src in range(slots):
            for tgt in range(slots):
                edge_idx = src * slots + tgt
                rows = torch.arange(chosen_edges.shape[0], device=chosen_flat.device) * (slots * slots) + edge_idx
                prim_mass = torch.zeros(model.pm.num_primitives, device=chosen_flat.device)
                for kk in range(cand.shape[1]):
                    prim_mass.scatter_add_(0, cand[rows, kk], choice[rows, kk])
                top_id = int(prim_mass.argmax().detach().cpu())
                if expected_names and model.pm.names[top_id] in expected_names:
                    exp_top_cells += 1
        if expected_names:
            expected_top_cells_by_layer.append(float(exp_top_cells))

        for act in layer_actions:
            src = int(act["src"])
            tgt = int(act["tgt"])
            pid = model.pm.name_to_id[str(act["primitive"])]
            rows = torch.arange(chosen_edges.shape[0], device=chosen_flat.device) * (slots * slots) + src * slots + tgt
            edge_choice = choice[rows]
            edge_cands = cand[rows]
            mask = edge_cands == pid
            present_value = mask.any(dim=-1).float().mean().item()
            choice_value = (edge_choice * mask.float()).sum(dim=-1).mean().item()
            recovery_value = (chosen_edges[:, src, tgt] == pid).float().mean().item()
            active_value = active[:, src, tgt].mean().item()
            tape_value = tape_grid[:, src, tgt].mean().item() if tape_grid is not None else 0.0
            candidate_present.append(present_value)
            choice_masses.append(choice_value)
            recoveries.append(recovery_value)
            actives.append(active_value)
            expected_any.append((chosen_edges == pid).float().mean().item())
            prefix = _action_metric_prefix(act)
            action_metrics.update({
                f"{prefix}_present": present_value,
                f"{prefix}_choice_mass": choice_value,
                f"{prefix}_recovery": recovery_value,
                f"{prefix}_active": active_value,
                f"{prefix}_tape": tape_value,
            })

    out = {
        "program_recovery_rate": float(sum(recoveries) / max(1, len(recoveries))),
        "expected_edge_recovery": float(sum(recoveries) / max(1, len(recoveries))),
        "expected_any_recovery": float(sum(expected_any) / max(1, len(expected_any))),
        "expected_candidate_present": float(sum(candidate_present) / max(1, len(candidate_present))),
        "expected_edge_choice_mass": float(sum(choice_masses) / max(1, len(choice_masses))),
        "layer_output_credit": float(sum(choice_masses) / max(1, len(choice_masses))),
        "expected_edge_active": float(sum(actives) / max(1, len(actives))),
        "transform_mass": float(sum(transform_mass) / max(1, len(transform_mass))),
        "skip_mass": float(sum(skip_mass) / max(1, len(skip_mass))),
        "disable_mass": float(sum(disable_mass) / max(1, len(disable_mass))),
        "choice_entropy": float(sum(choice_entropy) / max(1, len(choice_entropy))),
        "active_cells": float(sum(active_cells_by_layer) / max(1, len(active_cells_by_layer))),
        "expected_top_cells": float(sum(expected_top_cells_by_layer) / max(1, len(expected_top_cells_by_layer))),
        "primitive_top_share": float(sum(primitive_top_share_by_layer) / max(1, len(primitive_top_share_by_layer))),
        "active_edges_per_target": float(sum(active_edges_per_target_by_layer) / max(1, len(active_edges_per_target_by_layer))),
        "final_read_last": 1.0 if trace.get("final_read_mode") == "last" else 0.0,
        "final_read_mean": 1.0 if trace.get("final_read_mode") == "mean" else 0.0,
        "final_read_learned": 1.0 if trace.get("final_read_mode") == "learned" else 0.0,
        "state_norm_none": 1.0 if trace.get("state_norm_mode") == "none" else 0.0,
        "state_norm_layernorm": 1.0 if trace.get("state_norm_mode") == "layernorm" else 0.0,
        "slot_address_used_by_controller": 1.0 if trace.get("slot_address_used_by_controller") else 0.0,
        "slot_address_used_by_executor": 1.0 if trace.get("slot_address_used_by_executor") else 0.0,
        **{k: float(v) for k, v in trace.get("primitive_metrics", {}).items()},
        **action_metrics,
    }

    if listen_scores:
        out["layer_listen_score"] = float(sum(listen_scores) / len(listen_scores))
    if len(layer_action_dists) >= 2:
        sims = [
            F.cosine_similarity(layer_action_dists[i].float(), layer_action_dists[i + 1].float(), dim=0).item()
            for i in range(len(layer_action_dists) - 1)
        ]
        out["layer_action_similarity"] = float(sum(sims) / len(sims))
    else:
        out["layer_action_similarity"] = 1.0 if layer_action_dists else 0.0

    for key in ["semantic_grid_mismatch", "semantic_neighbor_entropy", "scanner_full_scan", "grid_candidate_usage", "semantic_candidate_usage", "usage_candidate_usage", "random_candidate_usage", "global_candidate_usage", "scanner_source_mass_sum"]:
        vals = [layer["scan_metrics"].get(key, 0.0) for layer in trace["layers"] if "scan_metrics" in layer]
        if vals:
            out[key] = float(sum(vals) / len(vals))

    if edge_scale_mean:
        out["edge_scale_mean"] = float(sum(edge_scale_mean) / len(edge_scale_mean))
    if cell_output_gate_mean:
        out["cell_output_gate_mean"] = float(sum(cell_output_gate_mean) / len(cell_output_gate_mean))
    if cell_tape_weight_mean:
        out["cell_tape_weight_mean"] = float(sum(cell_tape_weight_mean) / len(cell_tape_weight_mean))
    if cell_write_mass_mean:
        out["cell_write_mass_mean"] = float(sum(cell_write_mass_mean) / len(cell_write_mass_mean))
    if target_write_gate_mean:
        out["target_write_gate_mean"] = float(sum(target_write_gate_mean) / len(target_write_gate_mean))
    if slot_alive_mean:
        out["slot_alive_mean"] = float(sum(slot_alive_mean) / len(slot_alive_mean))
    if slot_alive_count:
        out["slot_alive_count"] = float(sum(slot_alive_count) / len(slot_alive_count))
    if split_none_mean:
        out["split_none_mass"] = float(sum(split_none_mean) / len(split_none_mean))
        out["split_one_mass"] = float(sum(split_one_mean) / len(split_one_mean))
        out["split_two_mass"] = float(sum(split_two_mean) / len(split_two_mean))
    if child_gate_mean:
        out["child_gate_mean"] = float(sum(child_gate_mean) / len(child_gate_mean))
    if merge_gate_mean:
        out["merge_gate_mean"] = float(sum(merge_gate_mean) / len(merge_gate_mean))
    if collector_mass_mean:
        out["collector_mass_mean"] = float(sum(collector_mass_mean) / len(collector_mass_mean))

    first_layer = trace["layers"][0]
    first = _actions_by_layer(batch)[0][0]
    src = int(first["src"])
    tgt = int(first["tgt"])
    for key in ["edge_pair_bias", "write_pair_bias", "phase_pair_bias", "cell_output_pair_bias"]:
        if key in first_layer:
            mat = first_layer[key]
            out[f"{key}_expected"] = mat[src, tgt].item()
            out[f"{key}_std"] = mat.float().std().item()

    return out


@torch.no_grad()
def evaluate(
    model,
    task,
    steps: int,
    batch_size: int,
    device: str,
    tau: float,
    args=None,
    ablate_layer_output: Optional[int] = None,
    ablate_state_after: Optional[int] = None,
    curriculum_mode: str = "teacher",
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss = 0.0
    oracle_acc_sum = 0.0
    last_trace = None
    last_batch = None

    for _ in range(steps):
        batch = task.sample(batch_size, device)
        logits, trace = model(
            batch.x,
            tau=tau,
            ablate_layer_output=ablate_layer_output,
            ablate_state_after=ablate_state_after,
            curriculum_mode=curriculum_mode,
        )
        loss += F.cross_entropy(logits, batch.y).item() * batch_size
        correct += (logits.argmax(dim=-1) == batch.y).sum().item()
        oracle_acc_sum += task.oracle_accuracy(batch) * batch_size
        total += batch_size
        last_trace = trace
        last_batch = batch

    out = summarize_trace(last_trace, last_batch, model)
    signals = structure_signal_stats(last_trace, last_batch, model)
    metric_args = args if args is not None else argparse.Namespace(
        adapt_choice_floor=0.45,
        adapt_top_share_floor=0.55,
        adapt_sharpness=0.08,
        adapt_cell_sharpness=1.5,
        target_active_cells=3.0,
        adapt_choice_boost=1.5,
        lambda_choice=1.0,
        lambda_non_expected_primitive=1.0,
        lambda_primitive_usage_diversity=1.0,
        lambda_cell_choice_diversity=1.0,
        lambda_non_expected_active=1.0,
        lambda_non_expected_tape=1.0,
        lambda_non_expected_transform=1.0,
        lambda_active_budget=1.0,
        lambda_tape_budget=1.0,
        lambda_layer_action_diversity=1.0,
    )
    gates = adaptive_loss_weights(signals, metric_args)
    out.update({k: float(v.detach().cpu()) for k, v in signals.items()})
    out.update({k: float(v.detach().cpu()) for k, v in gates.items() if k.startswith("adaptive_")})
    out.update({"val_loss": loss / total, "val_acc": correct / total, "oracle_acc": oracle_acc_sum / total})
    return out


@torch.no_grad()
def evaluate_with_deleted_layer(
    model: ActionMatrixModel,
    task,
    steps: int,
    batch_size: int,
    device: str,
    tau: float,
    delete_layer: int,
    args=None,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss = 0.0
    oracle_acc_sum = 0.0
    last_trace = None
    last_batch = None

    for _ in range(steps):
        batch = task.sample(batch_size, device)
        state = model.input_norm(batch.x)
        memory = state.mean(dim=1)
        slot_address = model.slot_embed.to(dtype=state.dtype, device=state.device)
        outputs = []
        traces = []
        prev_context = None
        for idx, layer in enumerate(model.layers):
            if idx == delete_layer:
                continue
            state, out, memory, tr = layer(
                state,
                memory,
                slot_address,
                prev_context=prev_context,
                tau=tau,
            )
            outputs.append(out)
            traces.append(tr)
            prev_context = _next_layer_context(tr, out, model)
        if not outputs:
            outputs = [state.mean(dim=1)]
        final, _ = model._merge_outputs(outputs)
        logits = model.classifier(final)
        loss += F.cross_entropy(logits, batch.y).item() * batch_size
        correct += (logits.argmax(dim=-1) == batch.y).sum().item()
        oracle_acc_sum += task.oracle_accuracy(batch) * batch_size
        total += batch_size
        last_trace = {"layers": traces, "primitive_metrics": model.pm.metrics(), "final_read_mode": model.final_read, "state_norm_mode": model.state_norm_mode}
        last_batch = batch

    out = summarize_trace(last_trace, last_batch, model) if last_trace is not None else {}
    signals = structure_signal_stats(last_trace, last_batch, model) if last_trace is not None else {}
    metric_args = args if args is not None else argparse.Namespace(
        adapt_choice_floor=0.45,
        adapt_listen_floor=0.45,
        adapt_top_share_floor=0.55,
        adapt_sharpness=0.08,
        adapt_cell_sharpness=1.5,
        target_active_cells=3.0,
        adapt_choice_boost=1.5,
        lambda_choice=1.0,
        lambda_non_expected_primitive=1.0,
        lambda_primitive_usage_diversity=1.0,
        lambda_cell_choice_diversity=1.0,
        lambda_non_expected_active=1.0,
        lambda_non_expected_tape=1.0,
        lambda_non_expected_transform=1.0,
        lambda_active_budget=1.0,
        lambda_tape_budget=1.0,
        lambda_layer_action_diversity=1.0,
    )
    gates = adaptive_loss_weights(signals, metric_args) if signals else {}
    out.update({k: float(v.detach().cpu()) for k, v in signals.items()})
    out.update({k: float(v.detach().cpu()) for k, v in gates.items() if k.startswith("adaptive_")})
    out.update({"val_loss": loss / total, "val_acc": correct / total, "oracle_acc": oracle_acc_sum / total})
    return out


def proof_slice_structure_losses(trace: Dict[str, object], batch, model: ActionMatrixModel, expected_id: int) -> Dict[str, torch.Tensor]:
    total_expected_choice, total_non_expected_primitive = [], []
    total_expected_active, total_non_expected_active = [], []
    total_non_expected_tape, total_non_expected_transform = [], []

    b = batch.x.shape[0]
    s = model.slots
    edges_per_sample = s * s
    actions_by_layer = _actions_by_layer(batch)

    for layer_idx, layer0 in enumerate(trace["layers"]):
        cand = layer0["candidate_ids"]
        choice = layer0["choice_for_loss"]
        mode = layer0["mode_for_loss"]
        edge = layer0["edge_for_loss"]
        write = layer0["write_for_loss"]
        phase = layer0["phase_for_loss"]
        cell_tape = layer0["cell_tape_weight_for_loss"]

        active = (edge * write * phase).squeeze(-1)
        transform = mode[:, 0]
        expected_rows_mask = torch.zeros(cand.shape[0], dtype=torch.bool, device=cand.device)
        expected_prim_ids = set()

        for act in actions_by_layer.get(layer_idx, []):
            src = int(act["src"])
            tgt = int(act["tgt"])
            prim = str(act["primitive"])
            pid = model.pm.name_to_id[prim]
            expected_prim_ids.add(pid)
            rows = torch.arange(b, device=cand.device) * edges_per_sample + src * s + tgt
            expected_rows_mask[rows] = True

            expected_mask = cand[rows] == pid
            choice_mass = (choice[rows] * expected_mask.float()).sum(dim=-1).clamp_min(1e-8)
            present = expected_mask.any(dim=-1).float()
            total_expected_choice.append(-(present * choice_mass.log()).sum() / present.sum().clamp_min(1.0))
            total_expected_active.append(-active[rows].clamp_min(1e-8).log().mean())

        non_expected_rows = ~expected_rows_mask
        if non_expected_rows.any():
            if expected_prim_ids:
                prim_mass = torch.zeros_like(active[non_expected_rows])
                for pid in expected_prim_ids:
                    prim_mass = prim_mass + (choice[non_expected_rows] * (cand[non_expected_rows] == pid).float()).sum(dim=-1)
                total_non_expected_primitive.append(prim_mass.mean())
            total_non_expected_active.append(active[non_expected_rows].mean())
            total_non_expected_tape.append(cell_tape.squeeze(-1)[non_expected_rows].mean())
            total_non_expected_transform.append(transform[non_expected_rows].mean())

    z = torch.zeros((), device=batch.x.device)
    def mean_or_zero(xs):
        return torch.stack(xs).mean() if xs else z

    return {
        "expected_choice_loss": mean_or_zero(total_expected_choice),
        "non_expected_primitive_loss": mean_or_zero(total_non_expected_primitive),
        "expected_active_loss": mean_or_zero(total_expected_active),
        "non_expected_active_loss": mean_or_zero(total_non_expected_active),
        "non_expected_tape_loss": mean_or_zero(total_non_expected_tape),
        "non_expected_transform_loss": mean_or_zero(total_non_expected_transform),
    }


def generic_anti_collapse_losses(trace: Dict[str, object], model: ActionMatrixModel, target_active_fraction: float, target_tape_fraction: float, target_active_cells: Optional[float] = None) -> Dict[str, torch.Tensor]:
    z = torch.zeros((), device=next(model.parameters()).device)
    same_primitive_losses, cell_similarity_losses = [], []
    active_budget_losses, tape_budget_losses, layer_distributions = [], [], []

    for layer0 in trace["layers"]:
        prim_dist_rows = _primitive_distribution(
            layer0,
            model.pm.num_primitives,
            choice_key="choice_for_loss",
        )
        b_edges = prim_dist_rows.shape[0]
        s = model.slots
        b = b_edges // (s * s)

        cell_dist = prim_dist_rows.view(b, s * s, -1).mean(dim=0)
        cell_dist = cell_dist / cell_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        global_prim = cell_dist.mean(dim=0)
        top_share = global_prim.max()
        same_primitive_losses.append(F.relu(top_share - 0.45).pow(2))

        normed = F.normalize(cell_dist.float(), dim=-1)
        sim = normed @ normed.t()
        off = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
        cell_similarity_losses.append(F.relu(off - 0.70).pow(2).mean())

        edge = layer0["edge_for_loss"]
        write = layer0["write_for_loss"]
        phase = layer0["phase_for_loss"]

        # Match the acceptance metric: penalize soft active-cell count, not only
        # mean activity. The old mean-only budget allowed all 16 cells to stay
        # weakly active, which passed loss but failed Task 02.
        active_grid = (edge * write * phase).view(b, s * s)
        active_mean = active_grid.mean()
        active_cell_count = active_grid.mean(dim=0).sum()
        active_cell_target = (
            float(target_active_cells)
            if target_active_cells is not None
            else float(target_active_fraction) * float(s * s)
        )
        active_budget_losses.append(
            F.relu(active_cell_count - active_cell_target).pow(2) / float(s * s) ** 2
            + F.relu(active_mean - target_active_fraction).pow(2)
        )

        tape = layer0["cell_tape_weight_for_loss"].mean()
        tape_budget_losses.append((tape - target_tape_fraction).pow(2))

        layer_distributions.append(global_prim)

    layer_div_losses = []
    for a, b in zip(layer_distributions, layer_distributions[1:]):
        cos = F.cosine_similarity(a.float(), b.float(), dim=0)
        layer_div_losses.append(F.relu(cos - 0.80).pow(2))

    def mean_or_zero(xs):
        return torch.stack(xs).mean() if xs else z

    return {
        "primitive_usage_diversity": mean_or_zero(same_primitive_losses),
        "cell_choice_diversity": mean_or_zero(cell_similarity_losses),
        "active_budget": mean_or_zero(active_budget_losses),
        "tape_budget": mean_or_zero(tape_budget_losses),
        "layer_action_diversity": mean_or_zero(layer_div_losses),
    }



def structure_signal_stats(
    trace: Dict[str, object],
    batch,
    model: ActionMatrixModel,
    use_expected_actions: bool = True,
) -> Dict[str, torch.Tensor]:
    """Signal-based controller for regularization.

    No epoch calendar. We read the current program state:
      - are expected candidates present?
      - is expected choice mass already alive?
      - is one primitive dominating all cells?
      - are too many cells active/tape-writing?

    The structure losses are enabled only when the model has enough recovery
    signal to not destroy learning.
    """
    s = model.slots
    b = batch.x.shape[0]
    actions_by_layer = _actions_by_layer(batch)

    choice_masses = []
    candidate_present = []
    primitive_top_shares = []
    active_means = []
    tape_means = []
    active_cells_soft = []
    listen_scores = []
    layer_action_dists = []
    slot_alive_means = []
    split_two_masses = []
    merge_gates = []

    for layer_idx, layer0 in enumerate(trace["layers"]):
        cand = layer0["candidate_ids"]
        choice = layer0["choice"]
        prim_dist_rows = _primitive_distribution(layer0, model.pm.num_primitives)
        cell_dist = prim_dist_rows.view(b, s * s, -1).mean(dim=0)
        global_prim = cell_dist.mean(dim=0)
        primitive_top_shares.append(global_prim.max())
        layer_action_dists.append(global_prim)
        if layer_idx > 0 and layer0.get("listen_gate") is not None:
            listen_scores.append(layer0["listen_gate"].mean())

        edge = layer0["edge_for_loss"]
        write = layer0["write_for_loss"]
        phase = layer0["phase_for_loss"]
        active = (edge * write * phase).view(b, s, s)
        active_means.append(active.mean())
        active_cells_soft.append(active.mean(dim=0).sum())

        tape = layer0["cell_tape_weight_for_loss"].view(b, s, s)
        tape_means.append(tape.mean())
        if "slot_alive" in layer0:
            slot_alive_means.append(layer0["slot_alive"].view(b, s).mean())
        if "split_count" in layer0:
            split_two_masses.append(layer0["split_count"].view(b, s, 3)[..., 2].mean())
        if "merge_gate" in layer0:
            merge_gates.append(layer0["merge_gate"].mean())

        for act in actions_by_layer.get(layer_idx, []) if use_expected_actions else []:
            pid = model.pm.name_to_id[str(act["primitive"])]
            src = int(act["src"])
            tgt = int(act["tgt"])
            rows = torch.arange(b, device=cand.device) * (s * s) + src * s + tgt
            expected_mask = cand[rows] == pid
            candidate_present.append(expected_mask.any(dim=-1).float().mean())
            choice_masses.append((choice[rows] * expected_mask.float()).sum(dim=-1).mean())

    device = next(model.parameters()).device
    z = torch.zeros((), device=device)

    def mean_or_zero(xs):
        return torch.stack(xs).mean() if xs else z

    return {
        "signal_expected_choice_mass": mean_or_zero(choice_masses),
        "signal_candidate_present": mean_or_zero(candidate_present),
        "signal_primitive_top_share": mean_or_zero(primitive_top_shares),
        "signal_active_mean": mean_or_zero(active_means),
        "signal_tape_mean": mean_or_zero(tape_means),
        "signal_active_cells_soft": mean_or_zero(active_cells_soft),
        "signal_layer_listen_score": mean_or_zero(listen_scores),
        "signal_slot_alive_mean": mean_or_zero(slot_alive_means),
        "signal_split_two_mass": mean_or_zero(split_two_masses),
        "signal_merge_gate_mean": mean_or_zero(merge_gates),
        "signal_layer_action_similarity": (
            torch.stack([
                F.cosine_similarity(layer_action_dists[i].float(), layer_action_dists[i + 1].float(), dim=0)
                for i in range(len(layer_action_dists) - 1)
            ]).mean()
            if len(layer_action_dists) >= 2
            else torch.ones((), device=model.classifier.weight.device)
        ),
    }


def generic_discovery_losses(
    trace: Dict[str, object],
    model: ActionMatrixModel,
    min_active_mass: float,
    min_write_mass: float,
    min_choice_entropy: float,
    target_active_cells: float,
) -> Dict[str, torch.Tensor]:
    """Oracle-free pressure that keeps the differentiable program path alive.

    These terms name no cell or primitive.  They only prevent the controller,
    write path, and candidate distribution from becoming irrecoverably silent
    before task CE has found a useful program.
    """
    active_floors, write_floors, entropy_floors, coverage_losses = [], [], [], []
    active_tail_losses, topology_consistency_losses = [], []
    for layer0 in trace["layers"]:
        active = (
            layer0["edge_for_loss"]
            * layer0["write_for_loss"]
            * layer0["phase_for_loss"]
        ).mean()
        write_mass = (
            active
            * (layer0["mode_for_loss"][:, 0:1] + layer0["mode_for_loss"][:, 1:2])
        ).mean()
        choice = layer0["choice_for_loss"].clamp_min(1e-8)
        entropy = -(choice * choice.log()).sum(dim=-1).mean()
        active_floors.append(F.relu(min_active_mass - active).pow(2))
        write_floors.append(F.relu(min_write_mass - write_mass).pow(2))
        entropy_floors.append(F.relu(min_choice_entropy - entropy).pow(2))
        # Entropy alone permits individual primitives to disappear. The log
        # barrier keeps every full-scan candidate recoverable during discovery.
        coverage_losses.append(-choice.log().mean())
        s = model.slots
        active_grid = (
            layer0["edge_for_loss"]
            * layer0["write_for_loss"]
            * layer0["phase_for_loss"]
        ).view(-1, s * s)
        keep = max(1, min(s * s, int(round(target_active_cells))))
        if keep < s * s:
            mean_grid = active_grid.mean(dim=0)
            top_mask = torch.zeros_like(mean_grid)
            top_pos = mean_grid.topk(k=keep, dim=-1).indices
            top_mask.scatter_(0, top_pos, 1.0)
            # Select one stable topology for the batch, not a different set of
            # cells per sample. Otherwise no reusable matrix program can form.
            active_tail_losses.append((active_grid * (1.0 - top_mask[None, :])).mean())
        topology_consistency_losses.append(active_grid.var(dim=0, unbiased=False).mean())

    z = torch.zeros((), device=next(model.parameters()).device)

    def mean_or_zero(values):
        return torch.stack(values).mean() if values else z

    return {
        "discovery_active_floor_loss": mean_or_zero(active_floors),
        "discovery_write_floor_loss": mean_or_zero(write_floors),
        "discovery_choice_exploration_loss": mean_or_zero(entropy_floors),
        "discovery_choice_coverage_loss": mean_or_zero(coverage_losses),
        "discovery_active_tail_loss": mean_or_zero(active_tail_losses),
        "discovery_topology_consistency_loss": mean_or_zero(topology_consistency_losses),
    }


def adaptive_loss_weights(signals: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    """Convert current program health signals into loss gates.

    This is deliberately not epoch-based:
      low expected_choice_mass  -> protect learning, disable heavy sparsity/diversity
      high expected_choice_mass + high primitive_top_share -> enable anti-collapse
      high expected_choice_mass + too many active cells -> enable sparse budget
    """
    choice = signals["signal_expected_choice_mass"].detach()
    present = signals["signal_candidate_present"].detach()
    top = signals["signal_primitive_top_share"].detach()
    active_cells = signals["signal_active_cells_soft"].detach()
    listen = signals.get("signal_layer_listen_score", torch.tensor(0.0, device=choice.device)).detach()

    # Candidate must exist and expected choice must be alive before structure
    # pressure is trusted.
    recovery_gate = present * torch.sigmoid((choice - args.adapt_choice_floor) / args.adapt_sharpness)

    dependency_gate = torch.sigmoid((listen - args.adapt_listen_floor) / args.adapt_sharpness)

    # Only fight primitive collapse if recovery exists AND one primitive dominates.
    collapse_gate = recovery_gate * torch.sigmoid((top - args.adapt_top_share_floor) / args.adapt_sharpness)

    # Only sparsify if recovery exists AND active cells exceed target.
    sparse_gate = recovery_gate * torch.sigmoid((active_cells - args.target_active_cells) / max(args.adapt_cell_sharpness, 1e-6))

    # If expected choice is weak, expected-choice teacher gets boosted.
    choice_boost = 1.0 + args.adapt_choice_boost * (1.0 - recovery_gate)

    return {
        "adaptive_recovery_gate": recovery_gate,
        "adaptive_collapse_gate": collapse_gate,
        "adaptive_sparse_gate": sparse_gate,
        "adaptive_dependency_gate": dependency_gate,
        "adaptive_choice_boost": choice_boost,
        "eff_lambda_choice": args.lambda_choice * choice_boost,
        "eff_lambda_non_expected_primitive": args.lambda_non_expected_primitive * collapse_gate,
        "eff_lambda_primitive_usage_diversity": args.lambda_primitive_usage_diversity * collapse_gate,
        "eff_lambda_cell_choice_diversity": args.lambda_cell_choice_diversity * collapse_gate,
        "eff_lambda_non_expected_active": args.lambda_non_expected_active * sparse_gate,
        "eff_lambda_non_expected_tape": args.lambda_non_expected_tape * sparse_gate,
        "eff_lambda_non_expected_transform": args.lambda_non_expected_transform * sparse_gate,
        "eff_lambda_active_budget": args.lambda_active_budget * sparse_gate,
        "eff_lambda_tape_budget": args.lambda_tape_budget * sparse_gate,
        "eff_lambda_layer_action_diversity": args.lambda_layer_action_diversity * collapse_gate * dependency_gate,
    }



def _bce_prob_safe(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE on probabilities is unsafe under CUDA autocast; force fp32 locally."""
    device_type = "cuda" if input.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        return F.binary_cross_entropy(input.float(), target.float())

def branch_structure_losses(trace: Dict[str, object], batch, model: ActionMatrixModel) -> Dict[str, torch.Tensor]:
    """Light branch supervision for split/merge/collector tasks."""
    device = next(model.parameters()).device
    z = torch.zeros((), device=device)
    b = batch.x.shape[0]
    s = model.slots
    actions_by_layer = _actions_by_layer(batch)

    split_losses, merge_losses, alive_losses, collector_losses, child_losses = [], [], [], [], []

    for layer_idx, layer0 in enumerate(trace["layers"]):
        split = layer0.get("split_count_for_loss")
        alive = layer0.get("slot_alive_for_loss")
        child_gate = layer0.get("child_gate_for_loss")
        merge_gate = layer0.get("merge_gate_for_loss")
        collector_mass = layer0.get("collector_mass_for_loss")
        if split is None or alive is None or child_gate is None or merge_gate is None or collector_mass is None:
            continue

        layer_actions = actions_by_layer.get(layer_idx, [])
        split_target = torch.zeros(b, s, dtype=torch.long, device=split.device)
        merge_target = torch.zeros(b, s, dtype=torch.float32, device=split.device)
        child_target = torch.zeros(b, s, dtype=torch.float32, device=split.device)
        alive_target = torch.zeros(b, s, dtype=torch.float32, device=split.device)
        touched_slots = set()

        split_sources: Dict[int, int] = {}
        merge_targets: Dict[int, int] = {}
        child_sources: Dict[int, int] = {}
        for act in layer_actions:
            src = int(act["src"])
            tgt = int(act["tgt"])
            prim = str(act["primitive"])
            touched_slots.add(src)
            touched_slots.add(tgt)
            if prim == "split":
                split_sources[src] = split_sources.get(src, 0) + 1
                child_sources[src] = child_sources.get(src, 0) + 1
            if prim == "merge":
                merge_targets[tgt] = merge_targets.get(tgt, 0) + 1

        for slot in touched_slots:
            alive_target[:, slot] = 1.0
        for src, count in split_sources.items():
            split_target[:, src] = min(2, count)
        for src, count in child_sources.items():
            child_target[:, src] = 1.0 if count > 0 else 0.0
        for tgt, count in merge_targets.items():
            merge_target[:, tgt] = 1.0 if count > 0 else 0.0

        if split_sources:
            split_losses.append(
                F.nll_loss(
                    torch.log(split.clamp_min(1e-8)).reshape(b * s, 3),
                    split_target.reshape(b * s),
                )
            )
        if touched_slots:
            alive_losses.append(_bce_prob_safe(alive, alive_target))
            child_losses.append(_bce_prob_safe(child_gate, child_target))
            merge_losses.append(_bce_prob_safe(merge_gate, merge_target))
            collector_losses.append(_bce_prob_safe(collector_mass, merge_target))

    def mean_or_zero(xs):
        return torch.stack(xs).mean() if xs else z

    return {
        "branch_split_loss": mean_or_zero(split_losses),
        "branch_alive_loss": mean_or_zero(alive_losses),
        "branch_child_loss": mean_or_zero(child_losses),
        "branch_merge_loss": mean_or_zero(merge_losses),
        "branch_collector_loss": mean_or_zero(collector_losses),
    }

def sim_targets_for_expected_actions(trace: Dict[str, object], batch, model: ActionMatrixModel) -> torch.Tensor:
    losses = []
    b = batch.x.shape[0]
    s = model.slots
    edges_per_sample = s * s
    actions_by_layer = _actions_by_layer(batch)

    for layer_idx, layer0 in enumerate(trace["layers"]):
        chosen_candidates = layer0["candidate_ids"]
        pred_gain = layer0["predicted_gain_for_loss"]
        sim_target = -0.2 * torch.ones_like(pred_gain)

        for act in actions_by_layer.get(layer_idx, []):
            pid = model.pm.name_to_id[str(act["primitive"])]
            src = int(act["src"])
            tgt = int(act["tgt"])
            rows = torch.arange(b, device=chosen_candidates.device) * edges_per_sample + src * s + tgt
            expected_mask = chosen_candidates[rows] == pid
            sim_target[rows] = torch.where(expected_mask, torch.ones_like(sim_target[rows]), -0.2 * torch.ones_like(sim_target[rows]))

        losses.append(F.mse_loss(pred_gain.float(), sim_target.float()))

    return torch.stack(losses).mean()


def compute_training_objective(
    trace: Dict[str, object],
    batch,
    model: ActionMatrixModel,
    ce_loss: torch.Tensor,
    args,
):
    """Canonical objective used by both real training and runtime probes."""
    supervision_mode = getattr(args, "supervision_mode", "oracle")
    if supervision_mode not in {"oracle", "discovery"}:
        raise ValueError(f"unknown supervision mode: {supervision_mode!r}")

    z = torch.zeros((), device=ce_loss.device)
    if supervision_mode == "oracle":
        expected_id = model.pm.name_to_id[batch.expected_primitive]
        sim_loss = sim_targets_for_expected_actions(trace, batch, model)
        struct = proof_slice_structure_losses(trace, batch, model, expected_id)
        branch = branch_structure_losses(trace, batch, model)
    else:
        # Strict contract: expected_actions are not even read while constructing
        # the discovery loss graph. They remain available to evaluation/reporting.
        sim_loss = z
        struct = {
            "expected_choice_loss": z,
            "non_expected_primitive_loss": z,
            "expected_active_loss": z,
            "non_expected_active_loss": z,
            "non_expected_tape_loss": z,
            "non_expected_transform_loss": z,
        }
        branch = {
            "branch_split_loss": z,
            "branch_alive_loss": z,
            "branch_child_loss": z,
            "branch_merge_loss": z,
            "branch_collector_loss": z,
        }
    generic = generic_anti_collapse_losses(
        trace,
        model,
        args.target_active_fraction,
        args.target_tape_fraction,
        args.target_active_cells,
    )
    discovery = generic_discovery_losses(
        trace,
        model,
        getattr(args, "discovery_min_active_mass", 0.08),
        getattr(args, "discovery_min_write_mass", 0.06),
        getattr(args, "discovery_min_choice_entropy", 1.0),
        args.target_active_cells,
    )
    signals = structure_signal_stats(
        trace,
        batch,
        model,
        use_expected_actions=(supervision_mode == "oracle"),
    )
    gates = adaptive_loss_weights(signals, args)
    mode = trace["layers"][0]["mode_for_loss"]
    min_transform = F.relu(args.min_transform_mass - mode[:, 0].mean())

    raw_losses = {
        "ce_loss": ce_loss,
        "sim_loss": sim_loss,
        **struct,
        **generic,
        **branch,
        **discovery,
        "min_transform_loss": min_transform,
    }
    weighted = {
        "weighted_ce_loss": ce_loss,
        "weighted_sim_loss": args.lambda_sim * sim_loss,
        "weighted_expected_choice_loss": gates["eff_lambda_choice"] * struct["expected_choice_loss"],
        "weighted_expected_active_loss": args.lambda_expected_active * struct["expected_active_loss"],
        "weighted_non_expected_primitive_loss": gates["eff_lambda_non_expected_primitive"] * struct["non_expected_primitive_loss"],
        "weighted_non_expected_active_loss": gates["eff_lambda_non_expected_active"] * struct["non_expected_active_loss"],
        "weighted_non_expected_tape_loss": gates["eff_lambda_non_expected_tape"] * struct["non_expected_tape_loss"],
        "weighted_non_expected_transform_loss": gates["eff_lambda_non_expected_transform"] * struct["non_expected_transform_loss"],
        "weighted_primitive_usage_diversity": gates["eff_lambda_primitive_usage_diversity"] * generic["primitive_usage_diversity"],
        "weighted_cell_choice_diversity": gates["eff_lambda_cell_choice_diversity"] * generic["cell_choice_diversity"],
        "weighted_active_budget": gates["eff_lambda_active_budget"] * generic["active_budget"],
        "weighted_tape_budget": gates["eff_lambda_tape_budget"] * generic["tape_budget"],
        "weighted_layer_action_diversity": gates["eff_lambda_layer_action_diversity"] * generic["layer_action_diversity"],
        "weighted_branch_split_loss": args.lambda_branch * branch["branch_split_loss"],
        "weighted_branch_alive_loss": args.lambda_branch * branch["branch_alive_loss"],
        "weighted_branch_child_loss": args.lambda_branch * branch["branch_child_loss"],
        "weighted_branch_merge_loss": args.lambda_branch * branch["branch_merge_loss"],
        "weighted_branch_collector_loss": args.lambda_branch * branch["branch_collector_loss"],
        "weighted_min_transform_loss": args.lambda_collapse * min_transform,
        "weighted_discovery_active_floor_loss": (
            getattr(args, "lambda_discovery_alive", 0.20) * discovery["discovery_active_floor_loss"]
            if supervision_mode == "discovery" else z
        ),
        "weighted_discovery_write_floor_loss": (
            getattr(args, "lambda_discovery_alive", 0.20) * discovery["discovery_write_floor_loss"]
            if supervision_mode == "discovery" else z
        ),
        "weighted_discovery_choice_exploration_loss": (
            getattr(args, "lambda_discovery_exploration", 0.02) * discovery["discovery_choice_exploration_loss"]
            if supervision_mode == "discovery" else z
        ),
        "weighted_discovery_choice_coverage_loss": (
            getattr(args, "lambda_discovery_coverage", 0.001)
            * torch.sigmoid((ce_loss.detach() - 0.66) / 0.02)
            * discovery["discovery_choice_coverage_loss"]
            if supervision_mode == "discovery" else z
        ),
        "weighted_discovery_active_tail_loss": (
            getattr(args, "lambda_discovery_sparsity", 0.10)
            * discovery["discovery_active_tail_loss"]
            if supervision_mode == "discovery" else z
        ),
        "weighted_discovery_topology_consistency_loss": (
            getattr(args, "lambda_discovery_topology_consistency", 0.05)
            * discovery["discovery_topology_consistency_loss"]
            if supervision_mode == "discovery" else z
        ),
    }
    total_loss = torch.stack([value.float() for value in weighted.values()]).sum()
    return total_loss, raw_losses, signals, gates, weighted


@torch.no_grad()
def inspect_action_program(model: ActionMatrixModel, task: SyntheticKnownProgramTask, batch_size: int, device: str, tau: float) -> Dict[str, object]:
    model.eval()
    batch = task.sample(batch_size, device)
    logits, trace = model(batch.x, tau=tau)
    slots = model.slots
    names = model.pm.names
    num_prims = len(names)
    actions_by_layer = _actions_by_layer(batch)

    layers_report = []
    action_metrics: Dict[str, float] = {}
    for layer_idx, layer0 in enumerate(trace["layers"]):
        cand = layer0["candidate_ids"]
        choice = layer0["choice"]
        mode = layer0["mode"].view(-1, slots, slots, 3)
        edge = layer0["edge"].view(-1, slots, slots)
        write = layer0["write"].view(-1, slots, slots)
        phase = layer0["phase"].view(-1, slots, slots)
        active = edge * write * phase
        cell_tape_weight = layer0.get("cell_tape_weight")
        if cell_tape_weight is not None:
            cell_tape_weight = cell_tape_weight.view(-1, slots, slots)

        expected_cells = {(int(a["src"]), int(a["tgt"])): str(a["primitive"]) for a in actions_by_layer.get(layer_idx, [])}

        cells = []
        top_table = []
        for src in range(slots):
            row_names = []
            for tgt in range(slots):
                edge_idx = src * slots + tgt
                rows = torch.arange(batch_size, device=cand.device) * (slots * slots) + edge_idx
                prim_mass = torch.zeros(num_prims, device=cand.device)
                for kk in range(cand.shape[1]):
                    prim_mass.scatter_add_(0, cand[rows, kk], choice[rows, kk])
                prim_mass = prim_mass / float(batch_size)

                top_id = int(prim_mass.argmax().detach().cpu())
                top_mass = float(prim_mass[top_id].detach().cpu())
                expected_prim = expected_cells.get((src, tgt))
                expected_mass = float(prim_mass[model.pm.name_to_id[expected_prim]].detach().cpu()) if expected_prim else 0.0

                mode_mean = mode[:, src, tgt, :].mean(dim=0)
                active_mean = float(active[:, src, tgt].mean().detach().cpu())
                tape_mean = float(cell_tape_weight[:, src, tgt].mean().detach().cpu()) if cell_tape_weight is not None else 0.0

                cell = {
                    "src": src,
                    "tgt": tgt,
                    "top_primitive": names[top_id],
                    "top_mass": top_mass,
                    "expected_primitive": expected_prim or "",
                    "expected_primitive_mass": expected_mass,
                    "active": active_mean,
                    "transform": float(mode_mean[0].detach().cpu()),
                    "skip": float(mode_mean[1].detach().cpu()),
                    "disable": float(mode_mean[2].detach().cpu()),
                    "cell_tape_weight": tape_mean,
                    "is_expected_edge": bool(expected_prim),
                }
                cells.append(cell)
                if expected_prim:
                    action = {
                        "layer": layer_idx,
                        "src": src,
                        "tgt": tgt,
                        "primitive": expected_prim,
                    }
                    pid = model.pm.name_to_id[expected_prim]
                    prefix = _action_metric_prefix(action)
                    action_metrics.update({
                        f"{prefix}_present": float((cand[rows] == pid).any(dim=-1).float().mean().cpu()),
                        f"{prefix}_choice_mass": expected_mass,
                        f"{prefix}_recovery": float((layer0["chosen"].view(-1, slots, slots)[:, src, tgt] == pid).float().mean().cpu()),
                        f"{prefix}_active": active_mean,
                        f"{prefix}_tape": tape_mean,
                    })
                mark = "*" if expected_prim else ""
                row_names.append(f"{mark}{names[top_id]}:{top_mass:.2f}/a{active_mean:.3f}")
            top_table.append(row_names)

        expected_names = {str(a["primitive"]) for a in actions_by_layer.get(layer_idx, [])}
        expected_top_cells = sum(1 for c in cells if c["top_primitive"] in expected_names) if expected_names else 0
        active_cells = sum(1 for c in cells if c["active"] > 0.05)
        active_edges_per_target = sum(1 for c in cells if c["active"] > 0.05) / float(slots)
        expected_count = len(expected_cells)
        expected_ok = all(c["top_primitive"] == c["expected_primitive"] for c in cells if c["is_expected_edge"])
        expected_active_vals = [c["active"] for c in cells if c["is_expected_edge"]]
        non_expected_cells = [c for c in cells if not c["is_expected_edge"]]
        non_expected_active_vals = [c["active"] for c in non_expected_cells]
        non_expected_top_split = sum(1 for c in non_expected_cells if c["top_primitive"] == "split")
        non_expected_top_skip = sum(1 for c in non_expected_cells if c["top_primitive"] == "skip")
        non_expected_top_disable = sum(1 for c in non_expected_cells if c["top_primitive"] == "disable")

        if expected_names and expected_top_cells > expected_count + max(2, slots // 2):
            verdict = "primitive_collapse"
        elif expected_count and expected_ok:
            verdict = "sparse_or_partly_sparse_program"
        elif expected_count == 0:
            verdict = "no_expected_actions_for_this_layer"
        else:
            verdict = "not_recovered"

        layers_report.append({
            "layer": layer_idx,
            "expected_actions": actions_by_layer.get(layer_idx, []),
            "expected_top_cells": expected_top_cells,
            "active_cells": active_cells,
            "active_edges_per_target": active_edges_per_target,
            "expected_active_mean": float(sum(expected_active_vals) / max(1, len(expected_active_vals))) if expected_active_vals else 0.0,
            "non_expected_active_mean": float(sum(non_expected_active_vals) / max(1, len(non_expected_active_vals))) if non_expected_active_vals else 0.0,
            "non_expected_top_split": float(non_expected_top_split),
            "non_expected_top_skip": float(non_expected_top_skip),
            "non_expected_top_disable": float(non_expected_top_disable),
            "program_verdict": verdict,
            "top_table": top_table,
            "cells": cells,
        })

    pred = logits.argmax(dim=-1)
    acc = float((pred == batch.y).float().mean().detach().cpu())
    return {
        "task": task.task,
        "batch_acc": acc,
        "expected_actions": getattr(batch, "expected_actions", []),
        "final_read_mode": trace.get("final_read_mode", ""),
        "state_norm_mode": trace.get("state_norm_mode", ""),
        "slot_address_used_by_controller": bool(trace.get("slot_address_used_by_controller", False)),
        "slot_address_used_by_executor": bool(trace.get("slot_address_used_by_executor", False)),
        "action_metrics": action_metrics,
        "final_read_weights": trace.get("final_read_weights", torch.tensor([])).detach().cpu().tolist() if hasattr(trace.get("final_read_weights", None), "detach") else [],
        "layers": layers_report,
        "legend": "cell format: primitive:choice_mass/active; * marks expected edge",
    }


def write_program_report(out_dir: Path, program: Dict[str, object], epoch: int) -> None:
    json_path = out_dir / f"program_epoch_{epoch:03d}.json"
    md_path = out_dir / "PROGRAM_REPORT.md"
    json_path.write_text(json.dumps(program, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# ActionMatrix Program Report", ""]
    lines.append(f"- task: `{program['task']}`")
    lines.append(f"- batch_acc: `{program['batch_acc']}`")
    lines.append(f"- final_read: `{program.get('final_read_mode', '')}`")
    lines.append(f"- state_norm: `{program.get('state_norm_mode', '')}`")
    lines.append(f"- slot_address_used_by_controller: `{program.get('slot_address_used_by_controller', False)}`")
    lines.append(f"- slot_address_used_by_executor: `{program.get('slot_address_used_by_executor', False)}`")
    lines.append(f"- final_read_weights: `{program.get('final_read_weights', [])}`")
    lines.append(f"- expected_actions: `{program.get('expected_actions', [])}`")
    lines.append(f"- action_metrics: `{program.get('action_metrics', {})}`")
    lines.append("")
    lines.append("Flow:")
    lines.append("```text")
    layers = len(program.get("layers", []))
    flow = "input"
    for i in range(layers):
        flow += f" -> Layer {i}"
    flow += f" -> final_read:{program.get('final_read_mode', '')}"
    lines.append(flow)
    lines.append("```")
    lines.append("")
    lines.append("Cell format: `primitive:choice_mass/active`. `*` marks expected edge.")
    lines.append("")

    for layer in program["layers"]:
        lines.append(f"## Layer {layer['layer']}")
        lines.append("")
        lines.append(f"- verdict: `{layer.get('program_verdict', 'unknown')}`")
        lines.append(f"- expected_top_cells: `{layer.get('expected_top_cells', 'NA')}`")
        lines.append(f"- active_cells: `{layer.get('active_cells', 'NA')}`")
        lines.append(f"- active_edges_per_target: `{layer.get('active_edges_per_target', 'NA')}`")
        lines.append(f"- expected_active_mean: `{layer.get('expected_active_mean', 'NA')}`")
        lines.append(f"- non_expected_active_mean: `{layer.get('non_expected_active_mean', 'NA')}`")
        lines.append(f"- non_expected_top_split: `{layer.get('non_expected_top_split', 'NA')}`")
        lines.append(f"- non_expected_top_skip: `{layer.get('non_expected_top_skip', 'NA')}`")
        lines.append(f"- non_expected_top_disable: `{layer.get('non_expected_top_disable', 'NA')}`")
        lines.append(f"- expected_actions: `{layer.get('expected_actions', [])}`")
        lines.append("")
        n = len(layer["top_table"])
        lines.append("| src\\\\tgt | " + " | ".join([f"t{j}" for j in range(n)]) + " |")
        lines.append("|---|" + "|".join(["---"] * n) + "|")
        for i, row in enumerate(layer["top_table"]):
            lines.append(f"| s{i} | " + " | ".join(row) + " |")
        lines.append("")
        lines.append("### Cells")
        for c in layer["cells"]:
            mark = " EXPECTED" if c["is_expected_edge"] else ""
            exp = f", expected={c['expected_primitive']} mass={c['expected_primitive_mass']:.4f}" if c["is_expected_edge"] else ""
            lines.append(
                f"- edge {c['src']}->{c['tgt']}{mark}: "
                f"top={c['top_primitive']} mass={c['top_mass']:.4f}{exp}, "
                f"active={c['active']:.6f}, transform={c['transform']:.4f}, "
                f"skip={c['skip']:.4f}, disable={c['disable']:.4f}, "
                f"tape={c['cell_tape_weight']:.6f}"
            )
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def dependency_metrics(model, task, args, device: str, tau: float, normal_acc: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if args.layers >= 1:
        last_ablate = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau, args=args, ablate_layer_output=args.layers - 1)
        out["layer_ablation_delta"] = normal_acc - last_ablate["val_acc"]
    else:
        out["layer_ablation_delta"] = 0.0
    if args.layers >= 2:
        dep = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau, args=args, ablate_state_after=0)
        out["layer_dependency_delta"] = normal_acc - dep["val_acc"]
    else:
        out["layer_dependency_delta"] = 0.0
    return out


@torch.no_grad()
def specialization_metrics(model, task, args, device: str, tau: float, normal_acc: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if args.layers >= 2:
        deleted = evaluate_with_deleted_layer(model, task, args.eval_steps, args.eval_batch_size, device, tau, delete_layer=0, args=args)
        out["external_delete_delta"] = normal_acc - deleted["val_acc"]
        out["external_delete_acc"] = deleted["val_acc"]
    else:
        out["external_delete_delta"] = 0.0
        out["external_delete_acc"] = normal_acc

    internal = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau, args=args, ablate_layer_output=0)
    out["internal_skip_delta"] = normal_acc - internal["val_acc"]
    out["internal_skip_acc"] = internal["val_acc"]

    state_ablate = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau, args=args, ablate_state_after=0)
    out["state_ablation_delta"] = normal_acc - state_ablate["val_acc"]
    out["state_ablation_acc"] = state_ablate["val_acc"]
    return out


def train(args) -> None:
    config = load_task_config(args.task_config if args.task_config else legacy_task_path(args.task))
    args.task = config.name
    args.slots = config.slots
    args.layers = config.layers
    torch.manual_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    out_dir = ensure_dir(Path(args.out_dir))
    latest_report = Path(args.latest_report)

    task = SyntheticKnownProgramTask(dim=args.dim, config=config)
    model = ActionMatrixModel(
        dim=args.dim,
        slots=args.slots,
        layers=args.layers,
        classes=2,
        top_k=args.top_k,
        sim_rank=args.sim_rank,
        input_norm=args.input_norm,
        state_norm=args.state_norm,
        final_read=args.final_read,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and args.amp == "fp16"))
    dtype = amp_dtype(args.amp)
    credit = ModeCreditLedger()
    honesty_floor = float(args.honesty_floor)

    best = 0.0
    last_eval: Dict[str, float] = {}
    last_train_diagnostics: Dict[str, float] = {}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tau = max(args.tau_min, args.tau_start * (args.tau_decay ** (epoch - 1)))
        phase = curriculum_phase(epoch, args.epochs, args.curriculum_schedule)
        total = 0
        correct = 0
        loss_sum = 0.0
        diagnostic_sums: Dict[str, float] = {}
        optimizer_steps = 0
        max_steps = args.max_steps if args.max_steps > 0 else args.steps_per_epoch

        for _ in range(max_steps):
            batch = task.sample(args.batch_size, device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda") and dtype != torch.float32):
                choice_sampling = args.choice_sampling
                if choice_sampling == "auto":
                    choice_sampling = "softmax" if args.supervision_mode == "discovery" else "gumbel"
                logits, trace = model(
                    batch.x,
                    tau=tau,
                    curriculum_mode=phase,
                    choice_sampling=choice_sampling,
                )
                ce = F.cross_entropy(logits, batch.y)
                loss, raw_losses, signals, gates, weighted = compute_training_objective(
                    trace,
                    batch,
                    model,
                    ce,
                    args,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            # Usage ranking is delayed until a real task outcome exists. Selection
            # alone is not credit and eval forwards never mutate usage history.
            with torch.no_grad():
                sample_credit = (logits.argmax(dim=-1) == batch.y).to(logits.dtype)
                for layer_trace in trace["layers"]:
                    edge_credit = sample_credit.repeat_interleave(args.slots * args.slots)
                    model.pm.update_usage_credit(layer_trace["chosen"], edge_credit)

            total += args.batch_size
            correct += (logits.argmax(dim=-1) == batch.y).sum().item()
            loss_sum += loss.item() * args.batch_size
            optimizer_steps += 1
            for mapping in (raw_losses, signals, gates, weighted):
                for key, value in mapping.items():
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value.detach().cpu())

        ev = evaluate(
            model,
            task,
            args.eval_steps,
            args.eval_batch_size,
            device,
            tau,
            args=args,
            curriculum_mode=phase,
        )
        ablations = simulator_ablation_metrics(model, task, args.eval_batch_size, device, tau)
        ev.update(ablations)
        ev["final_read_mode"] = args.final_read
        honesty_score = float(ev.get("val_acc", 0.0) / max(1e-8, ev.get("oracle_acc", ev.get("val_acc", 1.0))))
        phase_credit = {
            "val_acc": float(ev.get("val_acc", 0.0)),
            "oracle_acc": float(ev.get("oracle_acc", 0.0)),
            "sim_disabled_delta": float(ablations["sim_disabled_delta"]),
            "expected_edge_choice_mass": float(ev.get("expected_edge_choice_mass", 0.0)),
        }
        if phase == "deploy":
            if honesty_score >= honesty_floor:
                credit.update("deploy", phase_credit)
        else:
            credit.update(phase, phase_credit)
        ev.update(credit.metrics())
        ev["curriculum_phase"] = phase
        ev["honesty_score"] = honesty_score
        best = max(best, ev["val_acc"])
        last_eval = ev

        train_diagnostics = {
            key: value / max(1, optimizer_steps)
            for key, value in diagnostic_sums.items()
        }
        weighted_sum = sum(
            value for key, value in train_diagnostics.items() if key.startswith("weighted_")
        )
        train_diagnostics["loss_accounting_error"] = abs(loss_sum / total - weighted_sum)
        last_train_diagnostics = train_diagnostics

        row = {
            "epoch": epoch,
            "tau": tau,
            "curriculum_phase": phase,
            "train_loss": loss_sum / total,
            "train_acc": correct / total,
            "best_acc": best,
            **ev,
            **train_diagnostics,
        }
        append_csv(out_dir / "metrics.csv", row)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train={row['train_loss']:.4f}/{100*row['train_acc']:.2f}% "
            f"val={ev['val_loss']:.4f}/{100*ev['val_acc']:.2f}% "
            f"edge_prog={ev['expected_edge_recovery']:.3f} "
            f"any_prog={ev['expected_any_recovery']:.3f} "
            f"active_cells={ev.get('active_cells', 0):.1f} "
            f"top_share={ev.get('primitive_top_share', 0):.3f} adaptR={ev.get('adaptive_recovery_gate', 0):.2f} adaptC={ev.get('adaptive_collapse_gate', 0):.2f} adaptS={ev.get('adaptive_sparse_gate', 0):.2f} "
            f"write_mass={ev.get('cell_write_mass_mean', 0):.3f} write_gate={ev.get('target_write_gate_mean', 0):.3f} "
            f"phase={phase} honesty={honesty_score:.3f} "
            f"final={args.final_read} "
            f"cand={ev.get('expected_candidate_present', 0):.3f} "
            f"choice_mass={ev.get('expected_edge_choice_mass', 0):.3f} "
            f"oracle={100*ev.get('oracle_acc', 0):.1f}% "
            f"sim_delta={ablations['sim_disabled_delta']:+.4f}",
            flush=True,
        )

    dep = dependency_metrics(model, task, args, device, tau, last_eval.get("val_acc", 0.0))
    last_eval.update(dep)
    spec = specialization_metrics(model, task, args, device, tau, last_eval.get("val_acc", 0.0))
    last_eval.update(spec)

    program = inspect_action_program(model, task, args.eval_batch_size, device, tau)
    write_program_report(out_dir, program, args.epochs)

    conclusion = "vertical slice completed"
    if any(layer.get("program_verdict") == "primitive_collapse" for layer in program.get("layers", [])):
        conclusion = "WARNING: primitive collapse in PROGRAM_REPORT"
    if last_eval.get("sim_disabled_delta", 0) <= 0:
        conclusion = "WARNING: simulator may be decorative"

    summary = {
        "task": args.task,
        "task_config_path": str(config.path),
        "task_config_name": config.name,
        "task_config_digest": config.digest,
        "expected_actions": config.expected_actions,
        "pass_thresholds": config.pass_thresholds,
        "epochs": args.epochs,
        "curriculum_schedule": args.curriculum_schedule,
        "supervision_mode": args.supervision_mode,
        "expected_actions_used_for_training": args.supervision_mode == "oracle",
        "layers": args.layers,
        "final_read_mode": args.final_read,
        "state_norm_mode": args.state_norm,
        "best_acc": best,
        "last_acc": last_eval.get("val_acc"),
        "honesty_floor": honesty_floor,
        **last_eval,
        **last_train_diagnostics,
        **credit.metrics(),
        "program_verdicts": [layer.get("program_verdict") for layer in program.get("layers", [])],
        "program_expected_top_cells": [layer.get("expected_top_cells") for layer in program.get("layers", [])],
        "program_active_cells": [layer.get("active_cells") for layer in program.get("layers", [])],
        "program_active_edges_per_target": [layer.get("active_edges_per_target") for layer in program.get("layers", [])],
        "program_action_metrics": program.get("action_metrics", {}),
        "conclusion": conclusion,
        "seconds": time.time() - start,
    }
    summary["pass_threshold_results"] = evaluate_pass_thresholds(config.pass_thresholds, summary)
    summary["pass_thresholds_met"] = all(
        result["passed"] for result in summary["pass_threshold_results"].values()
    )
    write_json(out_dir / "final_report.json", summary)
    write_json(out_dir / "credit_ablation_epoch_final.json", credit.state_dict())
    write_latest_report(out_dir / "REPORT_TO_CHATGPT.txt", str(out_dir), summary)
    write_latest_report(latest_report, str(out_dir), summary)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="diff", help="legacy task name resolved through configs/tasks/<name>.yml")
    p.add_argument("--task-config", help="path to a YAML task config; overrides --task/--slots/--layers")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--sim-rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-steps", type=int, default=10)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--input-norm", default="none", choices=["none", "layernorm"])
    p.add_argument("--state-norm", default="none", choices=["none", "layernorm"])
    p.add_argument("--final-read", default="last", choices=["last", "mean", "learned"])
    p.add_argument("--amp", default="fp16", choices=["none", "fp16", "bf16"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-min", type=float, default=0.3)
    p.add_argument("--tau-decay", type=float, default=0.92)
    p.add_argument("--lambda-sim", type=float, default=0.1)
    p.add_argument("--lambda-choice", type=float, default=1.0)
    p.add_argument("--lambda-non-expected-primitive", type=float, default=0.25)
    p.add_argument("--lambda-expected-active", type=float, default=0.05)
    p.add_argument("--lambda-non-expected-active", type=float, default=0.02)
    p.add_argument("--lambda-non-expected-tape", type=float, default=0.05)
    p.add_argument("--lambda-non-expected-transform", type=float, default=0.02)
    p.add_argument("--lambda-primitive-usage-diversity", type=float, default=0.05)
    p.add_argument("--lambda-cell-choice-diversity", type=float, default=0.05)
    p.add_argument("--lambda-active-budget", type=float, default=0.02)
    p.add_argument("--lambda-tape-budget", type=float, default=0.02)
    p.add_argument("--lambda-layer-action-diversity", type=float, default=0.05)
    p.add_argument("--lambda-branch", type=float, default=0.05)
    p.add_argument("--curriculum-schedule", default="teacher", choices=["teacher", "phased"])
    p.add_argument("--supervision-mode", default="oracle", choices=["oracle", "discovery"])
    p.add_argument("--choice-sampling", default="auto", choices=["auto", "gumbel", "softmax"])
    p.add_argument("--honesty-floor", type=float, default=0.80)
    p.add_argument("--target-active-fraction", type=float, default=0.18)
    p.add_argument("--target-tape-fraction", type=float, default=0.015)
    p.add_argument("--target-active-cells", type=float, default=3.0)
    p.add_argument("--adapt-choice-floor", type=float, default=0.45)
    p.add_argument("--adapt-listen-floor", type=float, default=0.45)
    p.add_argument("--adapt-top-share-floor", type=float, default=0.55)
    p.add_argument("--adapt-sharpness", type=float, default=0.08)
    p.add_argument("--adapt-cell-sharpness", type=float, default=1.5)
    p.add_argument("--adapt-choice-boost", type=float, default=1.5)
    p.add_argument("--lambda-collapse", type=float, default=0.01)
    p.add_argument("--min-transform-mass", type=float, default=0.15)
    p.add_argument("--lambda-discovery-alive", type=float, default=0.20)
    p.add_argument("--lambda-discovery-exploration", type=float, default=0.02)
    p.add_argument("--lambda-discovery-coverage", type=float, default=0.001)
    p.add_argument("--lambda-discovery-sparsity", type=float, default=0.10)
    p.add_argument("--lambda-discovery-topology-consistency", type=float, default=0.05)
    p.add_argument("--discovery-min-active-mass", type=float, default=0.08)
    p.add_argument("--discovery-min-write-mass", type=float, default=0.06)
    p.add_argument("--discovery-min-choice-entropy", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="agent_reports/vertical_slice")
    p.add_argument("--latest-report", default="LATEST_RUN_REPORT.md")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
