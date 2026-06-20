#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F


def signed_corr(features_mb: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
    x = features_mb.float()
    y = y_b.float()
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean()
    return (x * y[None]).mean(dim=1) / (
        x.std(dim=1, unbiased=False).clamp_min(1e-8)
        * y.std(unbiased=False).clamp_min(1e-8)
    )


def make_R(dim: int, proj_dim: int, device: str) -> torch.Tensor:
    R = torch.randn(dim, proj_dim, device=device) / math.sqrt(dim)
    R[:, 0] = 1.0 / math.sqrt(dim)
    return R


def make_lowrank_projector(dim: int, rank: int, device: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A = torch.randn(dim, max(1, rank), device=device, generator=g)
    Q, _ = torch.linalg.qr(A.float())
    return Q[:, :rank].to(device)


def primitive_bank(src: torch.Tensor, tgt: torch.Tensor):
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
    for s in range(slots):
        for t in range(slots):
            for name, out in primitive_bank(x[:, s], x[:, t]):
                effects.append(out)
                meta.append((s, t, name))
    return torch.stack(effects, dim=0), meta


def find(meta, s: int, t: int, name: str) -> int:
    for i, m in enumerate(meta):
        if m == (s, t, name):
            return i
    raise KeyError((s, t, name))


def meta_str(meta, idx: int) -> str:
    s, t, n = meta[idx]
    return f"{s}->{t}:{n}"


def make_batch(batch: int, slots: int, dim: int, task: str, device: str):
    x = torch.randn(batch, slots, dim, device=device)

    if task == "diff":
        signal = (x[:, 0] - x[:, 1]).mean(dim=-1)
        true = (0, 1, "diff")
    elif task == "product":
        signal = (x[:, 0] * x[:, 1]).mean(dim=-1)
        true = (0, 1, "product")
    elif task == "contrast":
        signal = (torch.tanh(x[:, 0] - x[:, 1]) * (x[:, 0] + x[:, 1])).mean(dim=-1)
        true = (0, 1, "tanh_sum")  # not exact, intentionally imperfect
    else:
        raise ValueError(task)

    y = (signal > 0).long()
    y_pm = y.float() * 2.0 - 1.0
    return x, y, y_pm, true


def make_sim(actual_nbd: torch.Tensor, sim_rank: int, seed: int) -> torch.Tensor:
    n, b, d = actual_nbd.shape
    Q = make_lowrank_projector(d, sim_rank, str(actual_nbd.device), seed)
    # low-rank simulator preview: sees only rank-limited effect
    return (actual_nbd @ Q) @ Q.T


def project_bank(bank_nbd: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    # [N,B,D] -> [N,B,P]
    return torch.einsum("nbd,dp->nbp", bank_nbd, R)


def select_features_from_combo(combo_nbp: torch.Tensor, y_pm: torch.Tensor, top_m: int):
    # combo: [N,B,P]
    n, b, p = combo_nbp.shape
    flat = combo_nbp.permute(0, 2, 1).reshape(n * p, b)
    scores = signed_corr(flat, y_pm).abs()
    vals, idx = scores.topk(min(top_m, scores.numel()))
    cand_idx = idx // p
    proj_idx = idx % p
    feats = combo_nbp[cand_idx, :, proj_idx]  # [M,B]
    return feats, vals, cand_idx, proj_idx, scores.view(n, p).amax(dim=1)


def train_linear(train_mb: torch.Tensor, y_train: torch.Tensor, eval_mb: torch.Tensor, y_eval: torch.Tensor, steps: int):
    m, b = train_mb.shape
    w = torch.zeros(m, device=train_mb.device, requires_grad=True)
    bias = torch.zeros((), device=train_mb.device, requires_grad=True)
    opt = torch.optim.Adam([w, bias], lr=0.05)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = (w[:, None] * train_mb).sum(dim=0) + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_train.float())
        loss.backward()
        opt.step()

    with torch.no_grad():
        tr_logits = (w[:, None] * train_mb).sum(dim=0) + bias
        ev_logits = (w[:, None] * eval_mb).sum(dim=0) + bias
        tr_acc = ((tr_logits > 0) == (y_train > 0)).float().mean()
        ev_acc = ((ev_logits > 0) == (y_eval > 0)).float().mean()
    return float(tr_acc.cpu()), float(ev_acc.cpu())


def eval_mode(
    name: str,
    train_combo: torch.Tensor,
    eval_combo: torch.Tensor,
    y_train: torch.Tensor,
    y_eval: torch.Tensor,
    ypm_train: torch.Tensor,
    meta,
    true_idx: int,
    top_m: int,
    steps: int,
):
    tr_feats, top_vals, cand_idx, proj_idx, cand_scores = select_features_from_combo(train_combo, ypm_train, top_m)
    ev_feats = eval_combo[cand_idx, :, proj_idx]
    tr_acc, ev_acc = train_linear(tr_feats, y_train, ev_feats, y_eval, steps)

    true_score = float(cand_scores[true_idx].detach().cpu())
    true_rank = int((cand_scores > cand_scores[true_idx]).sum().item() + 1)

    top = []
    for r, ci in enumerate(cand_idx[:10].detach().cpu().tolist(), 1):
        top.append({
            "rank": r,
            "candidate": meta_str(meta, int(ci)),
            "score": float(cand_scores[int(ci)].detach().cpu()),
        })

    return {
        "name": name,
        "train_acc": tr_acc,
        "eval_acc": ev_acc,
        "true_rank": true_rank,
        "true_score": true_score,
        "top_candidates": top,
    }


def run_task(args, task: str):
    x_tr, y_tr, ypm_tr, true_meta = make_batch(args.batch_size, args.slots, args.dim, task, args.device)
    x_ev, y_ev, ypm_ev, _ = make_batch(args.eval_batch_size, args.slots, args.dim, task, args.device)

    actual_tr, meta = build_candidates(x_tr)
    actual_ev, meta2 = build_candidates(x_ev)
    assert meta == meta2

    true_idx = find(meta, *true_meta)

    sim_tr = make_sim(actual_tr, args.sim_rank, args.seed + 100)
    sim_ev = make_sim(actual_ev, args.sim_rank, args.seed + 100)

    delta_tr = actual_tr.detach() - sim_tr
    delta_ev = actual_ev.detach() - sim_ev
    abs_delta_tr = delta_tr.abs()
    abs_delta_ev = delta_ev.abs()

    # shuffled control: destroys per-sample delta meaning
    perm_tr = torch.randperm(delta_tr.shape[1], device=args.device)
    perm_ev = torch.randperm(delta_ev.shape[1], device=args.device)
    shuf_delta_tr = delta_tr[:, perm_tr]
    shuf_delta_ev = delta_ev[:, perm_ev]

    R = make_R(args.dim, args.proj_dim, args.device)

    sim_proj_tr = project_bank(sim_tr, R)
    sim_proj_ev = project_bank(sim_ev, R)

    actual_proj_tr = project_bank(actual_tr, R)
    actual_proj_ev = project_bank(actual_ev, R)

    delta_proj_tr = project_bank(delta_tr, R)
    delta_proj_ev = project_bank(delta_ev, R)

    abs_delta_proj_tr = project_bank(abs_delta_tr, R)
    abs_delta_proj_ev = project_bank(abs_delta_ev, R)

    shuf_delta_proj_tr = project_bank(shuf_delta_tr, R)
    shuf_delta_proj_ev = project_bank(shuf_delta_ev, R)

    zero = torch.zeros_like(delta_proj_tr)
    zero_ev = torch.zeros_like(delta_proj_ev)

    modes = {
        "sim_only": (
            sim_proj_tr,
            sim_proj_ev,
        ),
        "self_delta": (
            torch.cat([sim_proj_tr, delta_proj_tr, abs_delta_proj_tr], dim=-1),
            torch.cat([sim_proj_ev, delta_proj_ev, abs_delta_proj_ev], dim=-1),
        ),
        "zero_delta": (
            torch.cat([sim_proj_tr, zero, zero], dim=-1),
            torch.cat([sim_proj_ev, zero_ev, zero_ev], dim=-1),
        ),
        "shuffle_delta": (
            torch.cat([sim_proj_tr, shuf_delta_proj_tr, shuf_delta_proj_tr.abs()], dim=-1),
            torch.cat([sim_proj_ev, shuf_delta_proj_ev, shuf_delta_proj_ev.abs()], dim=-1),
        ),
        "actual_oracle": (
            actual_proj_tr,
            actual_proj_ev,
        ),
    }

    reports = {}
    for name, (tr_combo, ev_combo) in modes.items():
        reports[name] = eval_mode(
            name,
            tr_combo,
            ev_combo,
            y_tr,
            y_ev,
            ypm_tr,
            meta,
            true_idx,
            args.top_m,
            args.steps,
        )

    benefit = reports["self_delta"]["eval_acc"] - reports["sim_only"]["eval_acc"]
    shuffle_gap = reports["self_delta"]["eval_acc"] - reports["shuffle_delta"]["eval_acc"]
    zero_gap = reports["self_delta"]["eval_acc"] - reports["zero_delta"]["eval_acc"]

    return {
        "task": task,
        "true_candidate": meta_str(meta, true_idx),
        "candidate_count": len(meta),
        "sim_rank": args.sim_rank,
        "proj_dim": args.proj_dim,
        "modes": reports,
        "benefit": {
            "self_delta_minus_sim_eval_acc": benefit,
            "self_delta_minus_shuffle_eval_acc": shuffle_gap,
            "self_delta_minus_zero_eval_acc": zero_gap,
            "useful": benefit > args.min_gain and shuffle_gap > 0.02 and zero_gap > 0.02,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--sim-rank", type=int, default=8)
    ap.add_argument("--proj-dim", type=int, default=32)
    ap.add_argument("--top-m", type=int, default=32)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--min-gain", type=float, default=0.03)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)

    t0 = time.time()
    report = {
        "device": args.device,
        "seed": args.seed,
        "config": vars(args),
        "tasks": {},
    }
    for task in ["diff", "product", "contrast"]:
        report["tasks"][task] = run_task(args, task)

    useful_count = sum(1 for t in report["tasks"].values() if t["benefit"]["useful"])
    report["verdict"] = {
        "self_delta_useful_tasks": useful_count,
        "self_delta_probe_pass": useful_count >= 1,
        "seconds": time.time() - t0,
        "meaning": "PASS means self-delta carries usable candidate-choice signal in at least one controlled task.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
