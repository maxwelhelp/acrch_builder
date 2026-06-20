#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import torch
import torch.nn.functional as F

from arch_builder.primitive_matrix import PrimitiveMatrix5x5
from arch_builder.executor import ActionExecutor
from arch_builder.simulator import LowRankSimulator


def grad_norm(module):
    total = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().cpu())
            count += 1
    return total ** 0.5, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)

    pm = PrimitiveMatrix5x5(embed_dim=32).to(args.device)
    executor = ActionExecutor(dim=args.dim, primitive_matrix=pm).to(args.device)
    sim = LowRankSimulator(
        dim=args.dim,
        num_primitives=pm.num_primitives,
        rank=args.rank,
        embed_dim=pm.emb.shape[-1],
    ).to(args.device)

    # Executor/primitive bank frozen for probe.
    for p in pm.parameters():
        p.requires_grad_(False)
    for p in executor.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(sim.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    first_loss = None
    last_loss = None
    last_grad = None

    for step in range(1, args.steps + 1):
        src = torch.randn(args.batch, args.dim, device=args.device)
        tgt = torch.randn(args.batch, args.dim, device=args.device)
        mem = torch.randn(args.batch, args.dim, device=args.device)
        ids = torch.randint(0, pm.num_primitives, (args.batch, args.top_k), device=args.device)

        with torch.no_grad():
            actual = executor(src, tgt, mem, ids)

        pred, gain = sim(src, ids)
        loss = F.mse_loss(pred.float(), actual.float())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm, gcount = grad_norm(sim)
        opt.step()

        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        last_loss = float(loss.detach().cpu())
        last_grad = gnorm

        if step % max(1, args.steps // 10) == 0 or step == 1:
            with torch.no_grad():
                cos = F.cosine_similarity(pred.float().flatten(0, 1), actual.float().flatten(0, 1), dim=-1).mean()
                rel = (
                    (pred.float() - actual.float()).norm(dim=-1)
                    / actual.float().norm(dim=-1).clamp_min(1e-6)
                ).mean()
            history.append({
                "step": step,
                "mse": float(loss.detach().cpu()),
                "cos": float(cos.detach().cpu()),
                "rel_error": float(rel.detach().cpu()),
                "grad_norm": float(gnorm),
                "grad_param_count": int(gcount),
            })

    report = {
        "device": args.device,
        "dim": args.dim,
        "rank": args.rank,
        "batch": args.batch,
        "top_k": args.top_k,
        "steps": args.steps,
        "first_mse": first_loss,
        "last_mse": last_loss,
        "improvement": first_loss / max(last_loss, 1e-12),
        "last_grad_norm": last_grad,
        "history": history,
        "checks": {
            "loss_decreases_2x": (first_loss / max(last_loss, 1e-12)) >= 2.0,
            "gradient_nonzero": last_grad is not None and last_grad > 1e-8,
        },
    }
    report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
