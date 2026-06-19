from __future__ import annotations

import argparse
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
                choice_loss = expected_edge_choice_loss(trace, batch, model, expected_id)
                mode = trace["layers"][0]["mode_for_loss"]
                min_transform = F.relu(args.min_transform_mass - mode[:, 0].mean())
                loss = ce + args.lambda_sim * sim_loss + args.lambda_choice * choice_loss + args.lambda_collapse * min_transform

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
    p.add_argument("--lambda-collapse", type=float, default=0.01)
    p.add_argument("--min-transform-mass", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="agent_reports/vertical_slice")
    p.add_argument("--latest-report", default="LATEST_RUN_REPORT.md")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
