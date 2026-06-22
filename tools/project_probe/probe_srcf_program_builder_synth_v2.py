from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from arch_builder.srcf_program_builder_v2 import (
    SRCFProgramBuilderV2,
    SRCFProgramConfigV2,
    SRCFProgramLossWeightsV2,
    srcf_program_loss_v2,
)
from tools.project_probe.probe_srcf_graph_synthetic_program import make_pair_batch, metric_pack, set_seed


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = logits.new_tensor([1.0, 1.0, 0.75, 0.75])
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w)[mask].mean()


@torch.no_grad()
def evaluate(model, args, rng, device, hard=False) -> Dict[str, float]:
    model.eval()
    vals: Dict[str, list[float]] = {}
    names = ["edge", "path2", "comm", "basin"]
    for _ in range(args.eval_batches):
        b, _ = make_pair_batch(args, rng, device, hard=hard)
        score = torch.sigmoid(model(b.rel)["logits"])
        for i, name in enumerate(names):
            m = metric_pack(b.target[..., i], score[..., i], b.mask)
            for mk, mv in m.items():
                vals.setdefault(f"{name}_{mk}", []).append(mv)
    model.train()
    return {k: float(np.nanmean(v)) for k, v in vals.items()}


@torch.no_grad()
def node_ablation_credit(model, args, rng, device) -> Dict[int, float]:
    model.eval()
    out = {i: [] for i in range(model.cfg.max_nodes)}
    for _ in range(args.ablate_batches):
        b, _ = make_pair_batch(args, rng, device, hard=False)
        base = masked_bce(model(b.rel)["logits"], b.target, b.mask)
        for i in range(model.cfg.max_nodes):
            ab = masked_bce(model(b.rel, disabled_nodes={i})["logits"], b.target, b.mask)
            out[i].append(float((ab - base).detach().cpu()))
    model.train()
    return {i: float(np.mean(v)) for i, v in out.items()}


def fmt(res, task):
    return f"{task}=auc:{res[f'{task}_auc']:.3f} ap:{res[f'{task}_ap']:.3f} acc:{res[f'{task}_acc']:.3f}"


def print_program(trace):
    print("ASSEMBLED_PROGRAM")
    for n in trace["assembled_program"]:
        ops = ",".join([f"op{x['op']}:{x['prob']:.2f}" for x in n["top_ops"][:2]])
        src = ",".join([f"{x['source']}:{x['prob']:.2f}" for x in n["top_sources"][:2]])
        print(f"  node={n['node']} alive={n['alive_prob']:.3f} repeat={n['repeat']} readout={n['readout_prob']:.3f} impact={n['ablation_task_loss_delta']:.5f} ops=[{ops}] sources=[{src}]")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--communities", type=int, default=4)
    p.add_argument("--p-in", type=float, default=0.18)
    p.add_argument("--p-out", type=float, default=0.015)
    p.add_argument("--motifs-per-comm", type=int, default=3)
    p.add_argument("--hub-degree", type=int, default=3)
    p.add_argument("--extra-bridge-p", type=float, default=0.35)
    p.add_argument("--rel-mode", choices=["raw", "full"], default="raw")
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--max-nodes", type=int, default=5)
    p.add_argument("--ops", type=int, default=6)
    p.add_argument("--max-repeat", type=int, default=2)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=3)
    p.add_argument("--ablate-batches", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=30)
    p.add_argument("--drop-p", type=float, default=0.30)
    p.add_argument("--add-p", type=float, default=0.08)
    p.add_argument("--hard-drop-p", type=float, default=0.55)
    p.add_argument("--hard-add-p", type=float, default=0.18)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--closure-w", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/srcf_program_builder_v2")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    rel_dim = 1 if args.rel_mode == "raw" else 8
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SRCFProgramConfigV2(rel_dim=rel_dim, out_dim=4, state_dim=args.state_dim, hidden=args.hidden, max_nodes=args.max_nodes, op_count=args.ops, max_repeat=args.max_repeat)
    model = SRCFProgramBuilderV2(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = SRCFProgramLossWeightsV2()

    print("SRCF_PROGRAM_BUILDER_SYNTH_V2_PROBE")
    print(f"device={device} params={sum(p.numel() for p in model.parameters() if p.requires_grad):,} out_dir={out_dir}")
    history = []
    last = {}
    for step in range(1, args.steps + 1):
        b, peer = make_pair_batch(args, rng, device, hard=False)
        out = model(b.rel)
        out_peer = model(peer.rel)
        task = masked_bce(out["logits"], b.target, b.mask)
        ploss, pm = srcf_program_loss_v2(out, weights, peer_output=out_peer)
        loss = task + args.closure_w * ploss
        if not torch.isfinite(loss):
            raise RuntimeError(f"bad loss {loss}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise RuntimeError(f"bad grad {grad}")
        opt.step()
        last = pm
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            ein = evaluate(model, args, rng, device, hard=False)
            eood = evaluate(model, args, rng, device, hard=True)
            row = {"step": step, "loss": float(loss.detach().cpu()), "task": float(task.detach().cpu()), **pm, **{f"in_{k}": v for k, v in ein.items()}, **{f"ood_{k}": v for k, v in eood.items()}}
            history.append(row)
            print(f"step={step:04d} loss={row['loss']:.4f} task={row['task']:.4f} program={pm['srcf_program_loss']:.4f} nodes={pm['program_node_cost']:.3f} edge_cost={pm['program_edge_cost']:.3f}")
            for t in ["edge", "path2", "comm", "basin"]:
                print(f"  IN  {fmt(ein, t)}")
                print(f"  OOD {fmt(eood, t)}")
            print(f"  diag: fixed={pm['program_fixed']:.4f} rec={pm['program_recovery']:.4f} move={pm['program_move']:.4f} var={pm['program_state_var']:.4f}")

    impacts = node_ablation_credit(model, args, rng, device)
    trace_path = out_dir / "program_trace.json"
    metrics_path = out_dir / "metrics.json"
    model.save_program_trace(trace_path, impacts)
    metrics_path.write_text(json.dumps({"history": history, "last_metrics": last, "node_impacts": impacts}, indent=2), encoding="utf-8")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    print_program(trace)
    print(f"PROGRAM_TRACE_SAVED={trace_path}")
    print(f"METRICS_SAVED={metrics_path}")
    for v in last.values():
        if not math.isfinite(float(v)):
            raise RuntimeError("bad metric")
    print("SRCF_PROGRAM_BUILDER_SYNTH_V2_PASS")


if __name__ == "__main__":
    main()
