from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

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


def summarize_trace(trace: Dict[str, object], expected_id: int, slots: int, expected_src: int, expected_tgt: int) -> Dict[str, float]:
    layer0 = trace["layers"][0]
    chosen_flat = layer0["chosen"]
    chosen_edges = chosen_flat.view(-1, slots, slots)
    expected_edge_chosen = chosen_edges[:, expected_src, expected_tgt]

    mode = layer0["mode"]
    choice = layer0["choice"]
    cand = layer0["candidate_ids"]
    scan = layer0["scan_metrics"]

    edge = layer0["edge"].view(-1, slots, slots, 1)
    write = layer0["write"].view(-1, slots, slots, 1)
    phase = layer0["phase"].view(-1, slots, slots, 1)
    active = (edge * write * phase).squeeze(-1)

    edge_scale = layer0.get("edge_scale")
    cell_output_gate = layer0.get("cell_output_gate")
    cell_tape_weight = layer0.get("cell_tape_weight")

    batch_n = chosen_edges.shape[0]
    expected_rows = torch.arange(batch_n, device=chosen_flat.device) * (slots * slots) + expected_src * slots + expected_tgt
    expected_choice = choice[expected_rows]
    expected_cands = cand[expected_rows]
    expected_mask = expected_cands == expected_id

    expected_candidate_present = expected_mask.any(dim=-1).float().mean().item()
    expected_edge_choice_mass = (expected_choice * expected_mask.float()).sum(dim=-1).mean().item()
    expected_edge_recovery = (expected_edge_chosen == expected_id).float().mean().item()
    expected_any_recovery = (chosen_edges == expected_id).float().mean().item()
    expected_edge_active = active[:, expected_src, expected_tgt].mean().item()

    out = {
        "program_recovery_rate": expected_edge_recovery,
        "expected_edge_recovery": expected_edge_recovery,
        "expected_any_recovery": expected_any_recovery,
        "expected_candidate_present": expected_candidate_present,
        "expected_edge_choice_mass": expected_edge_choice_mass,
        "expected_edge_active": expected_edge_active,
        "transform_mass": mode[:, 0].mean().item(),
        "skip_mass": mode[:, 1].mean().item(),
        "disable_mass": mode[:, 2].mean().item(),
        "choice_entropy": float((-(choice + 1e-8) * (choice + 1e-8).log()).sum(dim=-1).mean().cpu()),
        **{k: float(v) for k, v in scan.items()},
        **{k: float(v) for k, v in trace.get("primitive_metrics", {}).items()},
    }
    if edge_scale is not None:
        out["edge_scale_mean"] = edge_scale.mean().item()
    if cell_output_gate is not None:
        out["cell_output_gate_mean"] = cell_output_gate.mean().item()
    if cell_tape_weight is not None:
        out["cell_tape_weight_mean"] = cell_tape_weight.mean().item()

    for key in ["edge_pair_bias", "write_pair_bias", "phase_pair_bias", "cell_output_pair_bias"]:
        if key in layer0:
            mat = layer0[key]
            out[f"{key}_expected"] = mat[expected_src, expected_tgt].item()
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
        total += batch_size
        oracle_acc_sum += task.oracle_accuracy(batch) * batch_size
        last_trace = trace
        last_batch = batch

    expected_id = model.pm.name_to_id[last_batch.expected_primitive]
    out = summarize_trace(last_trace, expected_id, model.slots, last_batch.expected_src, last_batch.expected_tgt)
    out.update({"val_loss": loss / total, "val_acc": correct / total, "oracle_acc": oracle_acc_sum / total})
    return out


def expected_edge_choice_loss(trace: Dict[str, object], batch, model: ActionMatrixModel, expected_id: int) -> torch.Tensor:
    """Known-program vertical-slice supervision.

    This is not the final learning rule. It is a proof-slice teacher signal that
    checks whether the ActionMatrix path can execute the known primitive on the
    known edge. Later stages must anneal/remove it.
    """
    layer0 = trace["layers"][0]
    cand = layer0["candidate_ids"]
    choice = layer0["choice"]
    edges_per_sample = model.slots * model.slots
    edge_index = batch.expected_src * model.slots + batch.expected_tgt
    rows = torch.arange(batch.x.shape[0], device=cand.device) * edges_per_sample + edge_index
    expected_mask = cand[rows] == expected_id
    choice_mass = (choice[rows] * expected_mask.float()).sum(dim=-1).clamp_min(1e-8)
    # If candidate is absent, do not create infinite loss; the candidate-presence
    # metric will expose that separately.
    present = expected_mask.any(dim=-1).float()
    loss = -(present * choice_mass.log()).sum() / present.sum().clamp_min(1.0)
    return loss


@torch.no_grad()
def inspect_action_program(model: ActionMatrixModel, task: SyntheticKnownProgramTask, batch_size: int, device: str, tau: float) -> Dict[str, object]:
    """Build a readable ActionMatrix report from one eval batch.

    This answers: what primitive/action did each source->target cell choose,
    how strong was that choice, and how active was the cell.
    """
    model.eval()
    batch = task.sample(batch_size, device)
    logits, trace = model(batch.x, tau=tau)
    layer0 = trace["layers"][0]
    slots = model.slots
    names = model.pm.names
    num_prims = len(names)

    cand = layer0["candidate_ids"]          # [B*S*S, K]
    choice = layer0["choice"]               # [B*S*S, K]
    chosen = layer0["chosen"].view(-1, slots, slots)

    mode = layer0["mode"].view(-1, slots, slots, 3)
    edge = layer0["edge"].view(-1, slots, slots)
    write = layer0["write"].view(-1, slots, slots)
    phase = layer0["phase"].view(-1, slots, slots)
    active = edge * write * phase

    cell_tape_weight = layer0.get("cell_tape_weight")
    if cell_tape_weight is not None:
        cell_tape_weight = cell_tape_weight.view(-1, slots, slots)

    expected_id = model.pm.name_to_id[batch.expected_primitive]
    expected_src = batch.expected_src
    expected_tgt = batch.expected_tgt

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
            expected_mass = float(prim_mass[expected_id].detach().cpu())

            mode_mean = mode[:, src, tgt, :].mean(dim=0)
            active_mean = float(active[:, src, tgt].mean().detach().cpu())
            tape_mean = float(cell_tape_weight[:, src, tgt].mean().detach().cpu()) if cell_tape_weight is not None else 0.0
            chosen_top_frac = float((chosen[:, src, tgt] == top_id).float().mean().detach().cpu())
            expected_frac = float((chosen[:, src, tgt] == expected_id).float().mean().detach().cpu())

            cell = {
                "src": src,
                "tgt": tgt,
                "top_primitive": names[top_id],
                "top_mass": top_mass,
                "chosen_top_fraction": chosen_top_frac,
                "expected_primitive_mass": expected_mass,
                "expected_primitive_fraction": expected_frac,
                "active": active_mean,
                "transform": float(mode_mean[0].detach().cpu()),
                "skip": float(mode_mean[1].detach().cpu()),
                "disable": float(mode_mean[2].detach().cpu()),
                "cell_tape_weight": tape_mean,
                "is_expected_edge": bool(src == expected_src and tgt == expected_tgt),
            }
            cells.append(cell)
            mark = "*" if cell["is_expected_edge"] else ""
            row_names.append(f"{mark}{names[top_id]}:{top_mass:.2f}/a{active_mean:.3f}")
        top_table.append(row_names)

    pred = logits.argmax(dim=-1)
    acc = float((pred == batch.y).float().mean().detach().cpu())

    expected_top_cells = sum(1 for c in cells if c["top_primitive"] == batch.expected_primitive)
    active_cells = sum(1 for c in cells if c["active"] > 0.05)
    expected_edge = next(c for c in cells if c["is_expected_edge"])
    if expected_top_cells > max(2, slots):
        verdict = "primitive_collapse"
    elif expected_edge["top_primitive"] == batch.expected_primitive and expected_edge["active"] > 0.05:
        verdict = "sparse_or_partly_sparse_program"
    else:
        verdict = "not_recovered"

    return {
        "task": task.task,
        "batch_acc": acc,
        "expected_primitive": batch.expected_primitive,
        "expected_src": expected_src,
        "expected_tgt": expected_tgt,
        "expected_top_cells": expected_top_cells,
        "active_cells": active_cells,
        "program_verdict": verdict,
        "top_table": top_table,
        "cells": cells,
        "legend": "cell format: primitive:choice_mass/active; * marks expected edge",
    }


def write_program_report(out_dir: Path, program: Dict[str, object], epoch: int) -> None:
    json_path = out_dir / f"program_epoch_{epoch:03d}.json"
    md_path = out_dir / "PROGRAM_REPORT.md"
    json_path.write_text(json.dumps(program, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# ActionMatrix Program Report")
    lines.append("")
    lines.append(f"- task: `{program['task']}`")
    lines.append(f"- batch_acc: `{program['batch_acc']}`")
    lines.append(f"- expected: `{program['expected_primitive']}` on edge `{program['expected_src']}->{program['expected_tgt']}`")
    lines.append(f"- verdict: `{program.get('program_verdict', 'unknown')}`")
    lines.append(f"- expected_top_cells: `{program.get('expected_top_cells', 'NA')}`")
    lines.append(f"- active_cells: `{program.get('active_cells', 'NA')}`")
    lines.append("")
    lines.append("Cell format: `primitive:choice_mass/active`. `*` marks expected edge.")
    lines.append("")
    lines.append("| src\\\\tgt | " + " | ".join([f"t{j}" for j in range(len(program['top_table']))]) + " |")
    lines.append("|---|" + "|".join(["---"] * len(program["top_table"])) + "|")
    for i, row in enumerate(program["top_table"]):
        lines.append(f"| s{i} | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    for c in program["cells"]:
        mark = " EXPECTED" if c["is_expected_edge"] else ""
        lines.append(
            f"- edge {c['src']}->{c['tgt']}{mark}: "
            f"top={c['top_primitive']} mass={c['top_mass']:.4f}, "
            f"expected_mass={c['expected_primitive_mass']:.4f}, "
            f"active={c['active']:.6f}, transform={c['transform']:.4f}, "
            f"skip={c['skip']:.4f}, disable={c['disable']:.4f}, "
            f"tape={c['cell_tape_weight']:.6f}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def proof_slice_structure_losses(trace: Dict[str, object], batch, model: ActionMatrixModel, expected_id: int) -> Dict[str, torch.Tensor]:
    """Small synthetic-proof losses that prevent the trivial 'diff everywhere' program.

    These are NOT final real-task losses. They enforce the known-program target:
      - expected primitive should be selected on expected edge;
      - the same primitive should not be selected on all other edges;
      - expected edge should stay active enough to feed the output tape;
      - non-expected edges should not dominate active/output mass.
    """
    layer0 = trace["layers"][0]
    cand = layer0["candidate_ids"]            # [B*S*S, K]
    choice = layer0["choice"]                 # [B*S*S, K], non-detached in train path
    mode = layer0["mode_for_loss"]            # [B*S*S, 3]
    edge = layer0["edge_for_loss"]            # [B*S*S, 1]
    write = layer0["write_for_loss"]          # [B*S*S, 1]
    phase = layer0["phase_for_loss"]          # [B*S*S, 1]
    cell_tape = layer0["cell_tape_weight_for_loss"]  # [B*S*S, 1]

    b = batch.x.shape[0]
    s = model.slots
    edges_per_sample = s * s
    edge_index = batch.expected_src * s + batch.expected_tgt
    rows = torch.arange(b, device=cand.device) * edges_per_sample + edge_index

    expected_mask = cand == expected_id
    expected_edge_rows = torch.zeros(cand.shape[0], dtype=torch.bool, device=cand.device)
    expected_edge_rows[rows] = True
    non_expected_rows = ~expected_edge_rows

    # How much the expected primitive is chosen per row.
    expected_prim_mass_per_row = (choice * expected_mask.float()).sum(dim=-1)

    # Direct teacher for expected edge.
    expected_choice_mass = expected_prim_mass_per_row[expected_edge_rows].clamp_min(1e-8)
    expected_choice_loss = -expected_choice_mass.log().mean()

    # Anti-collapse: choosing the expected primitive on non-expected edges is bad.
    non_expected_expected_prim_mass = expected_prim_mass_per_row[non_expected_rows]
    non_expected_primitive_loss = non_expected_expected_prim_mass.mean()

    active = (edge * write * phase).squeeze(-1)
    expected_active = active[expected_edge_rows].clamp_min(1e-8)
    expected_active_loss = -expected_active.log().mean()

    # Penalize active/output mass outside the known useful edge.
    non_expected_active_loss = active[non_expected_rows].mean()
    non_expected_tape_loss = cell_tape.squeeze(-1)[non_expected_rows].mean()

    # Avoid transform collapse everywhere.
    transform = mode[:, 0]
    non_expected_transform_loss = transform[non_expected_rows].mean()

    return {
        "expected_choice_loss": expected_choice_loss,
        "non_expected_primitive_loss": non_expected_primitive_loss,
        "expected_active_loss": expected_active_loss,
        "non_expected_active_loss": non_expected_active_loss,
        "non_expected_tape_loss": non_expected_tape_loss,
        "non_expected_transform_loss": non_expected_transform_loss,
    }


def train(args) -> None:
    torch.manual_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    out_dir = ensure_dir(Path(args.out_dir))
    latest_report = Path(args.latest_report)

    task = SyntheticKnownProgramTask(task=args.task, slots=args.slots, dim=args.dim)
    model = ActionMatrixModel(dim=args.dim, slots=args.slots, layers=args.layers, classes=2, top_k=args.top_k, sim_rank=args.sim_rank, input_norm=args.input_norm).to(device)
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
                chosen_candidates = trace["layers"][0]["candidate_ids"]
                pred_gain = trace["layers"][0]["predicted_gain_for_loss"]

                sim_target = -0.2 * torch.ones_like(pred_gain)
                edges_per_sample = model.slots * model.slots
                edge_index = batch.expected_src * model.slots + batch.expected_tgt
                row_ids = torch.arange(batch.x.shape[0], device=chosen_candidates.device) * edges_per_sample + edge_index
                expected_mask = chosen_candidates[row_ids] == expected_id
                sim_target[row_ids] = torch.where(
                    expected_mask,
                    torch.ones_like(sim_target[row_ids]),
                    -0.2 * torch.ones_like(sim_target[row_ids]),
                )

                sim_loss = F.mse_loss(pred_gain.float(), sim_target.float())
                struct_losses = proof_slice_structure_losses(trace, batch, model, expected_id)
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
            f"cand={ev.get('expected_candidate_present', 0):.3f} choice_mass={ev.get('expected_edge_choice_mass', 0):.3f} oracle={100*ev.get('oracle_acc', 0):.1f}% sim_delta={sdelta:+.4f}",
            flush=True,
        )

    program = inspect_action_program(model, task, args.eval_batch_size, device, tau)
    write_program_report(out_dir, program, args.epochs)

    conclusion = "vertical slice completed"
    if last_eval.get("skip_mass", 0) > 0.85:
        conclusion = "WARNING: skip-all risk"
    if last_eval.get("sim_disabled_delta", 0) <= 0:
        conclusion = "WARNING: simulator may be decorative"

    summary = {
        "task": args.task,
        "epochs": args.epochs,
        "best_acc": best,
        "last_acc": last_eval.get("val_acc"),
        **last_eval,
        "conclusion": conclusion,
        "seconds": time.time() - start,
    }
    write_json(out_dir / "final_report.json", summary)
    write_json(out_dir / "credit_ablation_epoch_final.json", credit.values)
    write_latest_report(out_dir / "REPORT_TO_CHATGPT.txt", str(out_dir), summary)
    write_latest_report(latest_report, str(out_dir), summary)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="diff", choices=["diff", "merge", "product", "memory", "semantic_rescue"])
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
    p.add_argument("--lambda-collapse", type=float, default=0.01)
    p.add_argument("--min-transform-mass", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="agent_reports/vertical_slice")
    p.add_argument("--latest-report", default="LATEST_RUN_REPORT.md")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
