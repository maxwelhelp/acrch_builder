#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_builder.primitive_matrix import PrimitiveMatrix5x5
from arch_builder.executor import ActionExecutor
from arch_builder.simulator import LowRankSimulator


class FullInputSimulator(nn.Module):
    def __init__(self, dim: int, num_primitives: int, rank: int = 16, embed_dim: int = 32):
        super().__init__()
        self.prim_emb = nn.Embedding(num_primitives, embed_dim)
        self.to_rank = nn.Linear(3 * dim + embed_dim, rank)
        self.from_rank = nn.Linear(rank, dim)

    def forward(self, src, tgt, mem, ids):
        n, k = ids.shape
        d = src.shape[-1]
        emb = self.prim_emb(ids)
        src_e = src.unsqueeze(1).expand(n, k, d)
        tgt_e = tgt.unsqueeze(1).expand(n, k, d)
        mem_e = mem.unsqueeze(1).expand(n, k, d)
        x = torch.cat([src_e, tgt_e, mem_e, emb], dim=-1)
        return self.from_rank(torch.tanh(self.to_rank(x)))


def grad_norm(module):
    total = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().cpu())
            count += 1
    return total ** 0.5, count


def stable_rel(pred, actual):
    # Global relative RMSE, not per-sample division by near-zero vectors.
    return float(
        ((pred.float() - actual.float()).pow(2).mean().sqrt() /
         actual.float().pow(2).mean().sqrt().clamp_min(1e-6)).detach().cpu()
    )


def train_one(args, mode: str, rank: int):
    pm = PrimitiveMatrix5x5(embed_dim=32).to(args.device)
    executor = ActionExecutor(dim=args.dim, primitive_matrix=pm).to(args.device)

    for p in pm.parameters():
        p.requires_grad_(False)
    for p in executor.parameters():
        p.requires_grad_(False)

    if mode == "source_only":
        sim = LowRankSimulator(args.dim, pm.num_primitives, rank=rank, embed_dim=32).to(args.device)
    elif mode == "full_input":
        sim = FullInputSimulator(args.dim, pm.num_primitives, rank=rank, embed_dim=32).to(args.device)
    else:
        raise ValueError(mode)

    opt = torch.optim.AdamW(sim.parameters(), lr=args.lr, weight_decay=1e-4)
    first = last = None
    last_g = 0.0
    hist = []

    for step in range(1, args.steps + 1):
        src = torch.randn(args.batch, args.dim, device=args.device)
        tgt = torch.randn(args.batch, args.dim, device=args.device)
        mem = torch.randn(args.batch, args.dim, device=args.device)
        ids = torch.randint(0, pm.num_primitives, (args.batch, args.top_k), device=args.device)

        with torch.no_grad():
            actual = executor(src, tgt, mem, ids)

        if mode == "source_only":
            pred, _ = sim(src, ids)
        else:
            pred = sim(src, tgt, mem, ids)

        loss = F.mse_loss(pred.float(), actual.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g, gc = grad_norm(sim)
        opt.step()

        if first is None:
            first = float(loss.detach().cpu())
        last = float(loss.detach().cpu())
        last_g = g

        if step == 1 or step % max(1, args.steps // 5) == 0:
            with torch.no_grad():
                pred_f = pred.float().reshape(-1, args.dim)
                actual_f = actual.float().reshape(-1, args.dim)
                cos = F.cosine_similarity(pred_f, actual_f, dim=-1).mean()
                rel = stable_rel(pred, actual)
            hist.append({
                "step": step,
                "mse": float(loss.detach().cpu()),
                "cos": float(cos.detach().cpu()),
                "global_rel_rmse": rel,
                "grad_norm": float(g),
                "grad_param_count": int(gc),
            })

    return {
        "mode": mode,
        "rank": rank,
        "first_mse": first,
        "last_mse": last,
        "improvement": first / max(last, 1e-12),
        "last_grad_norm": last_g,
        "history": hist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ranks", default="8,16,32,64")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    results = []
    for rank in ranks:
        for mode in ["source_only", "full_input"]:
            print(f"[run] mode={mode} rank={rank}", flush=True)
            results.append(train_one(args, mode, rank))

    out = {
        "device": args.device,
        "config": vars(args),
        "results": results,
    }

    by = {(r["mode"], r["rank"]): r for r in results}
    checks = {}
    for rank in ranks:
        s = by[("source_only", rank)]
        f = by[("full_input", rank)]
        checks[f"rank{rank}_full_beats_source"] = f["last_mse"] < s["last_mse"] * 0.85
        checks[f"rank{rank}_full_learns_2x"] = f["improvement"] >= 2.0

    out["checks"] = checks
    out["status"] = "PASS" if any(checks.values()) else "FAIL"
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
