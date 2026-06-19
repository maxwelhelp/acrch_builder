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
from .credit import CreditBuffer, simulator_ablation_metrics
from .reporting import ensure_dir, write_json, append_csv, write_latest_report


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
    edge_scale_mean, cell_output_gate_mean, cell_tape_weight_mean = [], [], []
    cell_write_mass_mean, target_write_gate_mean = [], []
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

        mode = layer0["mode"]
        transform_mass.append(mode[:, 0].mean().item())
        skip_mass.append(mode[:, 1].mean().item())
        disable_mass.append(mode[:, 2].mean().item())
        choice_entropy.append(float((-(choice + 1e-8) * (choice + 1e-8).log()).sum(dim=-1).mean().cpu()))

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

    for key in ["semantic_grid_mismatch", "grid_candidate_usage", "semantic_candidate_usage", "usage_candidate_usage", "random_candidate_usage", "scanner_source_mass_sum"]:
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
def evaluate(model, task, steps: int, batch_size: int, device: str, tau: float, args=None, ablate_layer_output: Optional[int] = None, ablate_state_after: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss = 0.0
    oracle_acc_sum = 0.0
    last_trace = None
    last_batch = None

    for _ in range(steps):
        batch = task.sample(batch_size, device)
        logits, trace = model(batch.x, tau=tau, ablate_layer_output=ablate_layer_output, ablate_state_after=ablate_state_after)
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


def generic_anti_collapse_losses(trace: Dict[str, object], model: ActionMatrixModel, target_active_fraction: float, target_tape_fraction: float) -> Dict[str, torch.Tensor]:
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
        active = (edge * write * phase).mean()
        active_budget_losses.append((active - target_active_fraction).pow(2))

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



def structure_signal_stats(trace: Dict[str, object], batch, model: ActionMatrixModel) -> Dict[str, torch.Tensor]:
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

    for layer_idx, layer0 in enumerate(trace["layers"]):
        cand = layer0["candidate_ids"]
        choice = layer0["choice"]
        prim_dist_rows = _primitive_distribution(layer0, model.pm.num_primitives)
        cell_dist = prim_dist_rows.view(b, s * s, -1).mean(dim=0)
        global_prim = cell_dist.mean(dim=0)
        primitive_top_shares.append(global_prim.max())

        edge = layer0["edge_for_loss"]
        write = layer0["write_for_loss"]
        phase = layer0["phase_for_loss"]
        active = (edge * write * phase).view(b, s, s)
        active_means.append(active.mean())
        active_cells_soft.append(active.mean(dim=0).sum())

        tape = layer0["cell_tape_weight_for_loss"].view(b, s, s)
        tape_means.append(tape.mean())

        for act in actions_by_layer.get(layer_idx, []):
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

    # Candidate must exist and expected choice must be alive before structure
    # pressure is trusted.
    recovery_gate = present * torch.sigmoid((choice - args.adapt_choice_floor) / args.adapt_sharpness)

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
        "eff_lambda_layer_action_diversity": args.lambda_layer_action_diversity * collapse_gate,
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
    expected_id = model.pm.name_to_id[batch.expected_primitive]
    sim_loss = sim_targets_for_expected_actions(trace, batch, model)
    struct = proof_slice_structure_losses(trace, batch, model, expected_id)
    generic = generic_anti_collapse_losses(
        trace,
        model,
        args.target_active_fraction,
        args.target_tape_fraction,
    )
    signals = structure_signal_stats(trace, batch, model)
    gates = adaptive_loss_weights(signals, args)
    mode = trace["layers"][0]["mode_for_loss"]
    min_transform = F.relu(args.min_transform_mass - mode[:, 0].mean())

    raw_losses = {
        "ce_loss": ce_loss,
        "sim_loss": sim_loss,
        **struct,
        **generic,
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
        "weighted_min_transform_loss": args.lambda_collapse * min_transform,
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


def train(args) -> None:
    torch.manual_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    out_dir = ensure_dir(Path(args.out_dir))
    latest_report = Path(args.latest_report)

    task = SyntheticKnownProgramTask(task=args.task, slots=args.slots, dim=args.dim)
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
    credit = CreditBuffer()

    best = 0.0
    last_eval: Dict[str, float] = {}
    last_train_diagnostics: Dict[str, float] = {}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tau = max(args.tau_min, args.tau_start * (args.tau_decay ** (epoch - 1)))
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
                logits, trace = model(batch.x, tau=tau)
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

            total += args.batch_size
            correct += (logits.argmax(dim=-1) == batch.y).sum().item()
            loss_sum += loss.item() * args.batch_size
            optimizer_steps += 1
            for mapping in (raw_losses, signals, gates, weighted):
                for key, value in mapping.items():
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value.detach().cpu())

        ev = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau, args=args)
        ablations = simulator_ablation_metrics(model, task, args.eval_batch_size, device, tau)
        ev.update(ablations)
        ev["final_read_mode"] = args.final_read
        credit.update({"sim_disabled_delta": ablations["sim_disabled_delta"]})
        ev.update(credit.metrics())
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
            f"top_share={ev.get('primitive_top_share', 0):.3f} gateR={ev.get('adaptive_recovery_gate', 0):.2f} gateC={ev.get('adaptive_collapse_gate', 0):.2f} gateS={ev.get('adaptive_sparse_gate', 0):.2f} "
            f"final={args.final_read} "
            f"cand={ev.get('expected_candidate_present', 0):.3f} "
            f"choice_mass={ev.get('expected_edge_choice_mass', 0):.3f} "
            f"oracle={100*ev.get('oracle_acc', 0):.1f}% "
            f"sim_delta={ablations['sim_disabled_delta']:+.4f}",
            flush=True,
        )

    dep = dependency_metrics(model, task, args, device, tau, last_eval.get("val_acc", 0.0))
    last_eval.update(dep)

    program = inspect_action_program(model, task, args.eval_batch_size, device, tau)
    write_program_report(out_dir, program, args.epochs)

    conclusion = "vertical slice completed"
    if any(layer.get("program_verdict") == "primitive_collapse" for layer in program.get("layers", [])):
        conclusion = "WARNING: primitive collapse in PROGRAM_REPORT"
    if last_eval.get("sim_disabled_delta", 0) <= 0:
        conclusion = "WARNING: simulator may be decorative"

    summary = {
        "task": args.task,
        "epochs": args.epochs,
        "layers": args.layers,
        "final_read_mode": args.final_read,
        "state_norm_mode": args.state_norm,
        "best_acc": best,
        "last_acc": last_eval.get("val_acc"),
        **last_eval,
        **last_train_diagnostics,
        "program_verdicts": [layer.get("program_verdict") for layer in program.get("layers", [])],
        "program_expected_top_cells": [layer.get("expected_top_cells") for layer in program.get("layers", [])],
        "program_active_cells": [layer.get("active_cells") for layer in program.get("layers", [])],
        "program_active_edges_per_target": [layer.get("active_edges_per_target") for layer in program.get("layers", [])],
        "conclusion": conclusion,
        "seconds": time.time() - start,
    }
    write_json(out_dir / "final_report.json", summary)
    write_json(out_dir / "credit_ablation_epoch_final.json", credit.values)
    write_latest_report(out_dir / "REPORT_TO_CHATGPT.txt", str(out_dir), summary)
    write_latest_report(latest_report, str(out_dir), summary)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="diff", choices=["diff", "two_diff", "merge", "product", "chain_diff_product", "semantic_rescue"])
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
    p.add_argument("--target-active-fraction", type=float, default=0.18)
    p.add_argument("--target-tape-fraction", type=float, default=0.015)
    p.add_argument("--target-active-cells", type=float, default=3.0)
    p.add_argument("--adapt-choice-floor", type=float, default=0.45)
    p.add_argument("--adapt-top-share-floor", type=float, default=0.55)
    p.add_argument("--adapt-sharpness", type=float, default=0.08)
    p.add_argument("--adapt-cell-sharpness", type=float, default=1.5)
    p.add_argument("--adapt-choice-boost", type=float, default=1.5)
    p.add_argument("--lambda-collapse", type=float, default=0.01)
    p.add_argument("--min-transform-mass", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="agent_reports/vertical_slice")
    p.add_argument("--latest-report", default="LATEST_RUN_REPORT.md")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
