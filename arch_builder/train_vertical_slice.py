from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from .model import ActionMatrixModel
from .synthetic_tasks import SyntheticKnownProgramTask
from .credit import CreditBuffer, sim_disabled_delta
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


def summarize_trace(trace: Dict[str, object], batch, model: ActionMatrixModel) -> Dict[str, float]:
    slots = model.slots
    actions_by_layer = _actions_by_layer(batch)

    recoveries = []
    candidate_present = []
    choice_masses = []
    actives = []
    expected_any = []
    choice_entropy = []
    transform_mass = []
    skip_mass = []
    disable_mass = []
    edge_scale_mean = []
    cell_output_gate_mean = []
    cell_tape_weight_mean = []

    for layer_idx, layer0 in enumerate(trace["layers"]):
        chosen_flat = layer0["chosen"]
        chosen_edges = chosen_flat.view(-1, slots, slots)
        cand = layer0["candidate_ids"]
        choice = layer0["choice"]

        mode = layer0["mode"]
        transform_mass.append(mode[:, 0].mean().item())
        skip_mass.append(mode[:, 1].mean().item())
        disable_mass.append(mode[:, 2].mean().item())
        choice_entropy.append(float((-(choice + 1e-8) * (choice + 1e-8).log()).sum(dim=-1).mean().cpu()))

        edge = layer0["edge"].view(-1, slots, slots, 1)
        write = layer0["write"].view(-1, slots, slots, 1)
        phase = layer0["phase"].view(-1, slots, slots, 1)
        active = (edge * write * phase).squeeze(-1)

        if "edge_scale" in layer0:
            edge_scale_mean.append(layer0["edge_scale"].mean().item())
        if "cell_output_gate" in layer0:
            cell_output_gate_mean.append(layer0["cell_output_gate"].mean().item())
        if "cell_tape_weight" in layer0:
            cell_tape_weight_mean.append(layer0["cell_tape_weight"].mean().item())

        layer_actions = actions_by_layer.get(layer_idx, [])
        for act in layer_actions:
            src = int(act["src"])
            tgt = int(act["tgt"])
            pid = model.pm.name_to_id[str(act["primitive"])]
            rows = torch.arange(chosen_edges.shape[0], device=chosen_flat.device) * (slots * slots) + src * slots + tgt
            edge_choice = choice[rows]
            edge_cands = cand[rows]
            mask = edge_cands == pid
            candidate_present.append(mask.any(dim=-1).float().mean().item())
            choice_masses.append((edge_choice * mask.float()).sum(dim=-1).mean().item())
            recoveries.append((chosen_edges[:, src, tgt] == pid).float().mean().item())
            actives.append(active[:, src, tgt].mean().item())
            expected_any.append((chosen_edges == pid).float().mean().item())

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
        **{k: float(v) for k, v in trace.get("primitive_metrics", {}).items()},
    }

    for key in ["semantic_grid_mismatch", "grid_candidate_usage", "semantic_candidate_usage", "usage_candidate_usage", "random_candidate_usage"]:
        vals = [layer["scan_metrics"].get(key, 0.0) for layer in trace["layers"] if "scan_metrics" in layer]
        if vals:
            out[key] = float(sum(vals) / len(vals))

    if edge_scale_mean:
        out["edge_scale_mean"] = float(sum(edge_scale_mean) / len(edge_scale_mean))
    if cell_output_gate_mean:
        out["cell_output_gate_mean"] = float(sum(cell_output_gate_mean) / len(cell_output_gate_mean))
    if cell_tape_weight_mean:
        out["cell_tape_weight_mean"] = float(sum(cell_tape_weight_mean) / len(cell_tape_weight_mean))

    # Pair-bias metrics for first expected action if available.
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
def evaluate(model, task, steps: int, batch_size: int, device: str, tau: float) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss = 0.0
    oracle_acc_sum = 0.0
    last_trace = None
    last_batch = None

    for _ in range(steps):
        batch = task.sample(batch_size, device)
        logits, trace = model(batch.x, tau=tau)
        loss += F.cross_entropy(logits, batch.y).item() * batch_size
        correct += (logits.argmax(dim=-1) == batch.y).sum().item()
        oracle_acc_sum += task.oracle_accuracy(batch) * batch_size
        total += batch_size
        last_trace = trace
        last_batch = batch

    out = summarize_trace(last_trace, last_batch, model)
    out.update({"val_loss": loss / total, "val_acc": correct / total, "oracle_acc": oracle_acc_sum / total})
    return out


def proof_slice_structure_losses(trace: Dict[str, object], batch, model: ActionMatrixModel, expected_id: int) -> Dict[str, torch.Tensor]:
    total_expected_choice = []
    total_non_expected_primitive = []
    total_expected_active = []
    total_non_expected_active = []
    total_non_expected_tape = []
    total_non_expected_transform = []

    b = batch.x.shape[0]
    s = model.slots
    edges_per_sample = s * s
    actions_by_layer = _actions_by_layer(batch)

    for layer_idx, layer0 in enumerate(trace["layers"]):
        cand = layer0["candidate_ids"]
        choice = layer0["choice"]
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



def _primitive_distribution(layer0: Dict[str, torch.Tensor], num_primitives: int) -> torch.Tensor:
    """Return [rows, P] primitive probability mass from candidate choice."""
    cand = layer0["candidate_ids"]
    choice = layer0["choice"]
    rows = cand.shape[0]
    out = torch.zeros(rows, num_primitives, device=choice.device, dtype=choice.dtype)
    out.scatter_add_(1, cand, choice)
    return out


def generic_anti_collapse_losses(trace: Dict[str, object], model: ActionMatrixModel) -> Dict[str, torch.Tensor]:
    """Generic anti-collapse losses.

    These do NOT know that the task uses diff or which edge is expected.
    They are inspired by v3:
      - class/read diversity -> here: cell primitive diversity
      - phase balance -> here: active/output budget
      - slot_div -> here: action diversity across cells and layers
    """
    z = torch.zeros((), device=next(model.parameters()).device)
    same_primitive_losses = []
    cell_similarity_losses = []
    active_budget_losses = []
    tape_budget_losses = []
    layer_distributions = []

    for layer0 in trace["layers"]:
        prim_dist_rows = _primitive_distribution(layer0, model.pm.num_primitives)  # [B*S*S, P]
        b_edges = prim_dist_rows.shape[0]
        s = model.slots
        b = b_edges // (s * s)

        # Distribution averaged per ActionMatrix cell: [S*S, P]
        cell_dist = prim_dist_rows.view(b, s * s, -1).mean(dim=0)
        cell_dist = cell_dist / cell_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # 1) Cap one primitive dominating all cells.
        global_prim = cell_dist.mean(dim=0)
        top_share = global_prim.max()
        same_primitive_losses.append(F.relu(top_share - 0.45).pow(2))

        # 2) Prevent all cells having identical choice distributions.
        normed = F.normalize(cell_dist.float(), dim=-1)
        sim = normed @ normed.t()
        off = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
        cell_similarity_losses.append(F.relu(off - 0.70).pow(2).mean())

        # 3) Active budget: not all cells active, not all dead.
        edge = layer0["edge_for_loss"]
        write = layer0["write_for_loss"]
        phase = layer0["phase_for_loss"]
        active = (edge * write * phase).mean()
        active_budget_losses.append((active - 0.25).pow(2))

        # 4) Output tape budget: small but alive.
        tape = layer0["cell_tape_weight_for_loss"].mean()
        tape_budget_losses.append((tape - 0.025).pow(2))

        layer_distributions.append(global_prim)

    # 5) Adjacent layers should not use identical primitive mix.
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
    for layer_idx, layer0 in enumerate(trace["layers"]):
        cand = layer0["candidate_ids"]
        choice = layer0["choice"]
        chosen = layer0["chosen"].view(-1, slots, slots)

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
                mark = "*" if expected_prim else ""
                row_names.append(f"{mark}{names[top_id]}:{top_mass:.2f}/a{active_mean:.3f}")
            top_table.append(row_names)

        expected_names = {str(a["primitive"]) for a in actions_by_layer.get(layer_idx, [])}
        expected_top_cells = sum(1 for c in cells if c["top_primitive"] in expected_names) if expected_names else 0
        active_cells = sum(1 for c in cells if c["active"] > 0.05)
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
    lines.append(f"- expected_actions: `{program.get('expected_actions', [])}`")
    lines.append("")
    lines.append("Cell format: `primitive:choice_mass/active`. `*` marks expected edge.")
    lines.append("")

    for layer in program["layers"]:
        lines.append(f"## Layer {layer['layer']}")
        lines.append("")
        lines.append(f"- verdict: `{layer.get('program_verdict', 'unknown')}`")
        lines.append(f"- expected_top_cells: `{layer.get('expected_top_cells', 'NA')}`")
        lines.append(f"- active_cells: `{layer.get('active_cells', 'NA')}`")
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
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and args.amp == "fp16"))
    dtype = amp_dtype(args.amp)
    credit = CreditBuffer()

    best = 0.0
    last_eval: Dict[str, float] = {}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tau = max(args.tau_min, args.tau_start * (args.tau_decay ** (epoch - 1)))
        total = 0
        correct = 0
        loss_sum = 0.0
        max_steps = args.max_steps if args.max_steps > 0 else args.steps_per_epoch

        for _ in range(max_steps):
            batch = task.sample(args.batch_size, device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda") and dtype != torch.float32):
                logits, trace = model(batch.x, tau=tau)
                ce = F.cross_entropy(logits, batch.y)
                expected_id = model.pm.name_to_id[batch.expected_primitive]

                sim_loss = sim_targets_for_expected_actions(trace, batch, model)
                struct_losses = proof_slice_structure_losses(trace, batch, model, expected_id)
                generic_losses = generic_anti_collapse_losses(trace, model)
                mode = trace["layers"][0]["mode_for_loss"]
                min_transform = F.relu(args.min_transform_mass - mode[:, 0].mean())

                loss = (
                    ce
                    + args.lambda_sim * sim_loss
                    + args.lambda_choice * struct_losses["expected_choice_loss"]
                    + args.lambda_non_expected_primitive * struct_losses["non_expected_primitive_loss"]
                    + args.lambda_expected_active * struct_losses["expected_active_loss"]
                    + args.lambda_non_expected_active * struct_losses["non_expected_active_loss"]
                    + args.lambda_non_expected_tape * struct_losses["non_expected_tape_loss"]
                    + args.lambda_non_expected_transform * struct_losses["non_expected_transform_loss"]
                    + args.lambda_primitive_usage_diversity * generic_losses["primitive_usage_diversity"]
                    + args.lambda_cell_choice_diversity * generic_losses["cell_choice_diversity"]
                    + args.lambda_active_budget * generic_losses["active_budget"]
                    + args.lambda_tape_budget * generic_losses["tape_budget"]
                    + args.lambda_layer_action_diversity * generic_losses["layer_action_diversity"]
                    + args.lambda_collapse * min_transform
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            total += args.batch_size
            correct += (logits.argmax(dim=-1) == batch.y).sum().item()
            loss_sum += loss.item() * args.batch_size

        ev = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau)
        sdelta = sim_disabled_delta(model, task, args.eval_batch_size, device, tau)
        ev["sim_disabled_delta"] = sdelta
        ev["choice_without_sim_delta"] = sdelta
        credit.update({"sim_disabled_delta": sdelta})
        ev.update(credit.metrics())
        best = max(best, ev["val_acc"])
        last_eval = ev

        row = {
            "epoch": epoch,
            "tau": tau,
            "train_loss": loss_sum / total,
            "train_acc": correct / total,
            "best_acc": best,
            **ev,
        }
        append_csv(out_dir / "metrics.csv", row)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train={row['train_loss']:.4f}/{100*row['train_acc']:.2f}% "
            f"val={ev['val_loss']:.4f}/{100*ev['val_acc']:.2f}% "
            f"edge_prog={ev['expected_edge_recovery']:.3f} "
            f"any_prog={ev['expected_any_recovery']:.3f} "
            f"cand={ev.get('expected_candidate_present', 0):.3f} "
            f"choice_mass={ev.get('expected_edge_choice_mass', 0):.3f} "
            f"oracle={100*ev.get('oracle_acc', 0):.1f}% "
            f"sim_delta={sdelta:+.4f}",
            flush=True,
        )

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
        "best_acc": best,
        "last_acc": last_eval.get("val_acc"),
        **last_eval,
        "program_verdicts": [layer.get("program_verdict") for layer in program.get("layers", [])],
        "program_expected_top_cells": [layer.get("expected_top_cells") for layer in program.get("layers", [])],
        "program_active_cells": [layer.get("active_cells") for layer in program.get("layers", [])],
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
    p.add_argument("--lambda-collapse", type=float, default=0.01)
    p.add_argument("--min-transform-mass", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="agent_reports/vertical_slice")
    p.add_argument("--latest-report", default="LATEST_RUN_REPORT.md")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
