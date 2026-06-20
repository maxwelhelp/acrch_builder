#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


def corr_score(features_nb: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
    x = features_nb.float()
    y = y_b.float()
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean()
    num = (x * y[None, :]).mean(dim=1).abs()
    den = x.std(dim=1, unbiased=False).clamp_min(1e-8) * y.std(unbiased=False).clamp_min(1e-8)
    return num / den


def make_batch(batch: int, slots: int, dim: int, task: str, device: str):
    x = torch.randn(batch, slots, dim, device=device)
    if task == "first_diff":
        signal = (x[:, 0] - x[:, 1]).mean(dim=-1)
    elif task == "chain_product":
        signal = ((x[:, 0] - x[:, 1]) * (x[:, 2] - x[:, 3])).mean(dim=-1)
    else:
        raise ValueError(task)
    y = (signal > 0).long()
    y_pm = y.float() * 2.0 - 1.0
    return x, y, y_pm


def primitive_bank(src: torch.Tensor, tgt: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    eps = 1e-4
    return [
        ("identity_src", src),
        ("identity_tgt", tgt),
        ("diff", src - tgt),
        ("revdiff", tgt - src),
        ("sum", src + tgt),
        ("mean", 0.5 * (src + tgt)),
        ("product", src * tgt),
        ("neg_src", -src),
        ("neg_tgt", -tgt),
        ("absdiff", (src - tgt).abs()),
        ("sqdiff", (src - tgt).pow(2)),
        ("max", torch.maximum(src, tgt)),
        ("min", torch.minimum(src, tgt)),
        ("gate_src_tgt", torch.sigmoid(src) * tgt),
        ("gate_tgt_src", torch.sigmoid(tgt) * src),
        ("tanh_sum", torch.tanh(src + tgt)),
        ("relu_diff", F.relu(src - tgt)),
        ("relu_revdiff", F.relu(tgt - src)),
        ("norm_diff", (src - tgt) / (src.abs() + tgt.abs() + eps)),
        ("src_sq", src.pow(2)),
        ("tgt_sq", tgt.pow(2)),
        ("src_times_abs_tgt", src * tgt.abs()),
        ("tgt_times_abs_src", tgt * src.abs()),
        ("smooth_src", 0.75 * src + 0.25 * tgt),
        ("smooth_tgt", 0.25 * src + 0.75 * tgt),
    ]


def build_candidates(x: torch.Tensor):
    b, slots, dim = x.shape
    effects = []
    meta = []
    for src_i in range(slots):
        for tgt_i in range(slots):
            src = x[:, src_i]
            tgt = x[:, tgt_i]
            for name, out in primitive_bank(src, tgt):
                effects.append(out)
                meta.append({"src": src_i, "tgt": tgt_i, "primitive": name})
    return torch.stack(effects, dim=0), meta


def train_gate(features_nb: torch.Tensor, y: torch.Tensor, steps: int, lr: float):
    n, b = features_nb.shape
    logits = torch.zeros(n, device=features_nb.device, requires_grad=True)
    w = torch.ones((), device=features_nb.device, requires_grad=True)
    bias = torch.zeros((), device=features_nb.device, requires_grad=True)
    opt = torch.optim.Adam([logits, w, bias], lr=lr)

    # one-step gradient diagnosis at init
    probs = logits.softmax(dim=0)
    mixed = probs @ features_nb
    pred = mixed * w + bias
    loss0 = F.binary_cross_entropy_with_logits(pred, y.float())
    loss0.backward()
    init_grad = logits.grad.detach().clone()
    logits.grad.zero_()
    w.grad.zero_()
    bias.grad.zero_()

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        probs = logits.softmax(dim=0)
        mixed = probs @ features_nb
        pred = mixed * w + bias
        loss = F.binary_cross_entropy_with_logits(pred, y.float())
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = logits.softmax(dim=0)
        mixed = probs @ features_nb
        pred = mixed * w + bias
        acc = ((pred > 0) == (y > 0)).float().mean()
        top = probs.topk(min(10, n))
    return {
        "acc": float(acc.cpu()),
        "init_grad_abs_mean": float(init_grad.abs().mean().cpu()),
        "top_idx": top.indices.detach().cpu().tolist(),
        "top_mass": top.values.detach().cpu().tolist(),
        "grad": init_grad.detach(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["first_diff", "chain_product"], default="chain_product")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--proj-dim", type=int, default=16)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--joint-topk", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)

    x, y, y_pm = make_batch(args.batch_size, args.slots, args.dim, args.task, args.device)
    effects, meta = build_candidates(x)
    n, b, d = effects.shape

    # projection scanner: compress primitive effect to proj_dim and score vs task signal
    R = torch.randn(d, args.proj_dim, device=args.device) / math.sqrt(d)
    proj = torch.einsum("nbd,dp->nbp", effects, R)
    flat = proj.permute(0, 2, 1).reshape(n * args.proj_dim, b)
    flat_scores = corr_score(flat, y_pm).view(n, args.proj_dim)
    single_score, best_proj_dim = flat_scores.max(dim=1)
    scalar = proj[torch.arange(n, device=args.device), :, best_proj_dim]  # [N, B]

    def find_candidate(src, tgt, prim):
        for i, m in enumerate(meta):
            if m["src"] == src and m["tgt"] == tgt and m["primitive"] == prim:
                return i
        raise KeyError((src, tgt, prim))

    true_diff01 = find_candidate(0, 1, "diff")
    true_diff23 = find_candidate(2, 3, "diff")
    true_first = true_diff01

    single_rank = int((single_score > single_score[true_first]).sum().item() + 1)
    top_single = single_score.topk(12).indices.detach().cpu().tolist()

    single_train = train_gate(scalar, y_pm, args.steps, lr=0.05)

    # joint scan: feature pair product, vectorized per anchor candidate
    true_joint_score = corr_score((scalar[true_diff01] * scalar[true_diff23])[None, :], y_pm)[0]
    true_joint_rank = 1
    top_pairs = []
    for a in range(n - 1):
        feats = scalar[a][None, :] * scalar[a + 1 :]
        scores = corr_score(feats, y_pm)
        true_joint_rank += int((scores > true_joint_score).sum().item())
        k = min(4, scores.numel())
        vals, idxs = scores.topk(k)
        for val, rel_idx in zip(vals.detach().cpu().tolist(), idxs.detach().cpu().tolist()):
            top_pairs.append((float(val), a, a + 1 + int(rel_idx)))
    top_pairs.sort(reverse=True, key=lambda t: t[0])
    top_pairs = top_pairs[: args.joint_topk]

    true_pair = (min(true_diff01, true_diff23), max(true_diff01, true_diff23))
    pair_set = {(min(a, b), max(a, b)) for _, a, b in top_pairs}
    true_pair_forced_into_joint_train = False
    if args.task == "chain_product" and true_pair not in pair_set:
        top_pairs.append((float(true_joint_score.cpu()), true_pair[0], true_pair[1]))
        true_pair_forced_into_joint_train = True

    joint_feats = torch.stack([scalar[a] * scalar[b] for _, a, b in top_pairs], dim=0)
    joint_train = train_gate(joint_feats, y_pm, args.steps, lr=0.05)

    def meta_str(i):
        m = meta[i]
        return f"{m['src']}->{m['tgt']}:{m['primitive']}"

    report = {
        "task": args.task,
        "batch_size": args.batch_size,
        "slots": args.slots,
        "dim": args.dim,
        "primitive_count": 25,
        "candidate_count": n,
        "projection_dim": args.proj_dim,
        "single_scan": {
            "true_diff01_score": float(single_score[true_diff01].cpu()),
            "true_diff01_rank": single_rank,
            "top_candidates": [
                {"rank": r + 1, "candidate": meta_str(i), "score": float(single_score[i].cpu())}
                for r, i in enumerate(top_single)
            ],
            "single_gate_train_acc": single_train["acc"],
            "true_diff01_init_grad": float(single_train["grad"][true_diff01].cpu()),
            "true_diff23_init_grad": float(single_train["grad"][true_diff23].cpu()),
        },
        "joint_scan": {
            "true_pair": f"{meta_str(true_diff01)} * {meta_str(true_diff23)}",
            "true_joint_score": float(true_joint_score.cpu()),
            "true_joint_rank": int(true_joint_rank),
            "true_pair_forced_into_joint_train": bool(true_pair_forced_into_joint_train),
            "joint_gate_train_acc": joint_train["acc"],
            "top_pairs": [
                {"rank": r + 1, "pair": f"{meta_str(a)} * {meta_str(b)}", "score": float(score)}
                for r, (score, a, b) in enumerate(top_pairs[:12])
            ],
        },
    }

    if args.task == "first_diff":
        report["verdict"] = {
            "single_projection_signal_ok": single_rank <= 10,
            "single_gate_learns": single_train["acc"] > 0.80,
            "joint_required": False,
        }
    else:
        report["verdict"] = {
            "single_projection_expected_to_be_weak": single_train["acc"] < 0.70,
            "joint_projection_signal_ok": int(true_joint_rank) <= 50,
            "joint_gate_learns": joint_train["acc"] > 0.80,
            "joint_required": True,
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
