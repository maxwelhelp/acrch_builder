#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


class ProjectedSimulator(nn.Module):
    def __init__(self, dim: int, out_dim: int, num_primitives: int, rank: int = 16, embed_dim: int = 32):
        super().__init__()
        self.prim_emb = nn.Embedding(num_primitives, embed_dim)
        self.to_rank = nn.Linear(dim + embed_dim, rank)
        self.from_rank = nn.Linear(rank, out_dim)

    def forward(self, src, ids):
        n, k = ids.shape
        d = src.shape[-1]
        emb = self.prim_emb(ids)
        src_e = src.unsqueeze(1).expand(n, k, d)
        x = torch.cat([src_e, emb], dim=-1)
        return self.from_rank(torch.tanh(self.to_rank(x)))


class ScalarCritic(nn.Module):
    def __init__(self, dim: int, num_primitives: int, hidden: int = 128, embed_dim: int = 32):
        super().__init__()
        self.prim_emb = nn.Embedding(num_primitives, embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim + embed_dim),
            nn.Linear(dim + embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, src, ids):
        n, k = ids.shape
        d = src.shape[-1]
        emb = self.prim_emb(ids)
        src_e = src.unsqueeze(1).expand(n, k, d)
        x = torch.cat([src_e, emb], dim=-1)
        return self.net(x).squeeze(-1)


def make_R(dim, proj_dim, device):
    R = torch.randn(dim, proj_dim, device=device) / math.sqrt(dim)
    return R


def grad_norm(module):
    total = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().cpu())
            count += 1
    return total ** 0.5, count


def top1_stats(pred_score, true_score):
    # [B,K]
    pred_idx = pred_score.argmax(dim=-1)
    true_idx = true_score.argmax(dim=-1)
    acc = (pred_idx == true_idx).float().mean()

    pred_true = true_score.gather(1, pred_idx[:, None]).squeeze(1)
    oracle = true_score.max(dim=-1).values
    random_score = true_score.mean(dim=-1)

    regret = (oracle - pred_true).mean()
    oracle_gain_vs_random = (oracle - random_score).mean()
    captured = 1.0 - regret / oracle_gain_vs_random.clamp_min(1e-6)

    return {
        "top1_acc": float(acc.detach().cpu()),
        "regret": float(regret.detach().cpu()),
        "captured_oracle_gain": float(captured.detach().cpu()),
    }


def run(args):
    pm = PrimitiveMatrix5x5(embed_dim=32).to(args.device)
    executor = ActionExecutor(dim=args.dim, primitive_matrix=pm).to(args.device)

    for p in pm.parameters():
        p.requires_grad_(False)
    for p in executor.parameters():
        p.requires_grad_(False)

    R16 = make_R(args.dim, 16, args.device)
    R32 = make_R(args.dim, 32, args.device)
    # task/head direction: what downstream head cares about
    head = torch.randn(args.dim, device=args.device)
    head = head / head.norm().clamp_min(1e-6)

    full_sim = LowRankSimulator(args.dim, pm.num_primitives, rank=args.rank, embed_dim=32).to(args.device)
    jl16_sim = ProjectedSimulator(args.dim, 16, pm.num_primitives, rank=args.rank, embed_dim=32).to(args.device)
    jl32_sim = ProjectedSimulator(args.dim, 32, pm.num_primitives, rank=args.rank, embed_dim=32).to(args.device)
    scalar = ScalarCritic(args.dim, pm.num_primitives, hidden=args.hidden, embed_dim=32).to(args.device)

    opt = torch.optim.AdamW(
        list(full_sim.parameters()) +
        list(jl16_sim.parameters()) +
        list(jl32_sim.parameters()) +
        list(scalar.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    history = []

    for step in range(1, args.steps + 1):
        src = torch.randn(args.batch, args.dim, device=args.device)
        tgt = torch.randn(args.batch, args.dim, device=args.device)
        mem = torch.randn(args.batch, args.dim, device=args.device)
        ids = torch.randint(0, pm.num_primitives, (args.batch, args.top_k), device=args.device)

        with torch.no_grad():
            actual = executor(src, tgt, mem, ids).float()
            target16 = actual @ R16
            target32 = actual @ R32
            true_score = (actual * head).sum(dim=-1)

        full_pred, _ = full_sim(src, ids)
        full_pred = full_pred.float()
        jl16_pred = jl16_sim(src, ids).float()
        jl32_pred = jl32_sim(src, ids).float()
        scalar_pred = scalar(src, ids).float()

        # full-D simulation
        loss_full = F.mse_loss(full_pred, actual)

        # compressed task-space simulation
        loss_jl16 = F.mse_loss(jl16_pred, target16)
        loss_jl32 = F.mse_loss(jl32_pred, target32)

        # direct head/task critic
        loss_scalar = F.mse_loss(scalar_pred, true_score)

        loss = loss_full + loss_jl16 + loss_jl32 + loss_scalar

        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_full, _ = grad_norm(full_sim)
        g_jl16, _ = grad_norm(jl16_sim)
        g_jl32, _ = grad_norm(jl32_sim)
        g_scalar, _ = grad_norm(scalar)
        opt.step()

        if step == 1 or step % max(1, args.steps // 5) == 0:
            with torch.no_grad():
                # convert simulations to predicted task score
                full_score = (full_pred * head).sum(dim=-1)
                # projected score uses projected head
                h16 = head @ R16
                h32 = head @ R32
                jl16_score = (jl16_pred * h16).sum(dim=-1)
                jl32_score = (jl32_pred * h32).sum(dim=-1)

                random_pred = torch.randn_like(true_score)

                entry = {
                    "step": step,
                    "loss_fullD": float(loss_full.detach().cpu()),
                    "loss_jl16": float(loss_jl16.detach().cpu()),
                    "loss_jl32": float(loss_jl32.detach().cpu()),
                    "loss_scalar": float(loss_scalar.detach().cpu()),
                    "grad_full": g_full,
                    "grad_jl16": g_jl16,
                    "grad_jl32": g_jl32,
                    "grad_scalar": g_scalar,
                    "random_choice": top1_stats(random_pred, true_score),
                    "fullD_choice": top1_stats(full_score, true_score),
                    "jl16_choice": top1_stats(jl16_score, true_score),
                    "jl32_choice": top1_stats(jl32_score, true_score),
                    "scalar_choice": top1_stats(scalar_pred, true_score),
                    "oracle_choice": {
                        "top1_acc": 1.0,
                        "regret": 0.0,
                        "captured_oracle_gain": 1.0,
                    },
                }
                history.append(entry)

    last = history[-1]
    checks = {
        "scalar_beats_random_top1": last["scalar_choice"]["top1_acc"] > last["random_choice"]["top1_acc"] + 0.10,
        "scalar_captures_gain": last["scalar_choice"]["captured_oracle_gain"] > 0.25,
        "jl16_beats_random_top1": last["jl16_choice"]["top1_acc"] > last["random_choice"]["top1_acc"] + 0.05,
        "jl32_beats_random_top1": last["jl32_choice"]["top1_acc"] > last["random_choice"]["top1_acc"] + 0.05,
        "fullD_beats_random_top1": last["fullD_choice"]["top1_acc"] > last["random_choice"]["top1_acc"] + 0.05,
    }

    return {
        "device": args.device,
        "config": vars(args),
        "primitive_count": pm.num_primitives,
        "history": history,
        "last": last,
        "checks": checks,
        "status": "PASS" if any(checks.values()) else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)

    out = run(args)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
