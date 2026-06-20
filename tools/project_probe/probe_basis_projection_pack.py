#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, time
import torch
import torch.nn.functional as F


def signed_corr(features_mb, y_b):
    x = features_mb.float()
    y = y_b.float()
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean()
    return (x * y[None]).mean(dim=1) / (
        x.std(dim=1, unbiased=False).clamp_min(1e-8)
        * y.std(unbiased=False).clamp_min(1e-8)
    )


def make_q(dim, device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A = torch.randn(dim, dim, device=device, generator=g)
    Q, _ = torch.linalg.qr(A.float())
    return Q.to(device)


def train_linear(features_mb, y, steps=120, lr=0.05):
    m, b = features_mb.shape
    w = torch.zeros(m, device=features_mb.device, requires_grad=True)
    bias = torch.zeros((), device=features_mb.device, requires_grad=True)
    opt = torch.optim.Adam([w, bias], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = (w[:, None] * features_mb).sum(dim=0) + bias
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = (w[:, None] * features_mb).sum(dim=0) + bias
        acc = ((logits > 0) == (y > 0)).float().mean()
    return float(acc.cpu())


def primitive_bank(src, tgt):
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


def build_candidates(x):
    b, slots, dim = x.shape
    effects, meta = [], []
    for s in range(slots):
        for t in range(slots):
            for name, out in primitive_bank(x[:, s], x[:, t]):
                effects.append(out)
                meta.append((s, t, name))
    return torch.stack(effects, dim=0), meta


def find(meta, s, t, name):
    for i, m in enumerate(meta):
        if m == (s, t, name):
            return i
    raise KeyError((s, t, name))


def meta_str(meta, i):
    s, t, n = meta[i]
    return f"{s}->{t}:{n}"


def make_R(dim, proj_dim, device):
    R = torch.randn(dim, proj_dim, device=device) / math.sqrt(dim)
    R[:, 0] = 1.0 / math.sqrt(dim)
    return R


def single_scan(effects, y_pm, proj_dim):
    n, b, d = effects.shape
    R = make_R(d, proj_dim, effects.device)
    proj = torch.einsum("nbd,dp->nbp", effects, R)
    flat = proj.permute(0, 2, 1).reshape(n * proj_dim, b)
    signed = signed_corr(flat, y_pm).view(n, proj_dim)
    abs_score, best_dim = signed.abs().max(dim=1)
    sign = torch.sign(signed[torch.arange(n, device=effects.device), best_dim])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    feature = proj[torch.arange(n, device=effects.device), :, best_dim] * sign[:, None]
    return abs_score, signed[torch.arange(n, device=effects.device), best_dim], feature


@torch.no_grad()
def pair_exact(effects, y_pm, row_chunk=4):
    n, b, d = effects.shape
    scores = torch.empty(n, n, device=effects.device)
    t0 = time.time()
    for i0 in range(0, n, row_chunk):
        a = effects[i0:i0 + row_chunk]
        feat = (a[:, None] * effects[None]).mean(dim=-1).reshape(-1, b)
        sc = signed_corr(feat, y_pm).abs().view(a.shape[0], n)
        scores[i0:i0 + a.shape[0]] = sc
    return scores, time.time() - t0


@torch.no_grad()
def pair_jl(effects, y_pm, proj_dim=16, row_chunk=4):
    n, b, d = effects.shape
    R = make_R(d, proj_dim, effects.device)
    proj = torch.einsum("nbd,dp->nbp", effects, R)
    scores = torch.empty(n, n, device=effects.device)
    t0 = time.time()
    for i0 in range(0, n, row_chunk):
        a = proj[i0:i0 + row_chunk]
        feat = (a[:, None] * proj[None]).mean(dim=-1).reshape(-1, b)
        sc = signed_corr(feat, y_pm).abs().view(a.shape[0], n)
        scores[i0:i0 + a.shape[0]] = sc
    return scores, time.time() - t0


def rank_score(scores, idx):
    s = scores[idx]
    return int((scores > s).sum().item() + 1), float(s.detach().cpu())


def pair_rank_score(scores, a, b):
    s = scores[a, b]
    return int((scores > s).sum().item() + 1), float(s.detach().cpu())


def top_pairs(scores, meta, k=8):
    n = scores.shape[0]
    vals, idx = scores.flatten().topk(k)
    out = []
    for r, (v, flat_i) in enumerate(zip(vals.cpu().tolist(), idx.cpu().tolist()), 1):
        a, b = flat_i // n, flat_i % n
        out.append({"rank": r, "pair": f"{meta_str(meta, a)} * {meta_str(meta, b)}", "score": float(v)})
    return out


def make_batch(batch, slots, dim, task, shift, device):
    latent = torch.randn(batch, slots, dim, device=device)

    if task == "first_diff":
        signal = (latent[:, 0] - latent[:, 1]).mean(dim=-1)
    elif task == "chain_product":
        signal = ((latent[:, 0] - latent[:, 1]) * (latent[:, 2] - latent[:, 3])).mean(dim=-1)
    else:
        raise ValueError(task)

    y = (signal > 0).long()
    y_pm = y.float() * 2.0 - 1.0

    Q0 = make_q(dim, device, 101)
    Q1 = make_q(dim, device, 202)

    if shift == "none":
        obs = latent
        unrot = obs
    elif shift == "global":
        obs = latent @ Q0.T
        unrot = obs @ Q0
    elif shift == "per_sample":
        mask = torch.randint(0, 2, (batch,), device=device)
        obs = latent.clone()
        obs[mask == 0] = latent[mask == 0] @ Q0.T
        obs[mask == 1] = latent[mask == 1] @ Q1.T
        unrot = obs.clone()
        unrot[mask == 0] = obs[mask == 0] @ Q0
        unrot[mask == 1] = obs[mask == 1] @ Q1
    else:
        raise ValueError(shift)

    return obs, unrot, y, y_pm


def eval_case(task, shift, args):
    obs, unrot, y, y_pm = make_batch(args.batch_size, args.slots, args.dim, task, shift, args.device)

    out = {}
    for mode, x in [("raw", obs), ("basis_unrotated", unrot)]:
        effects, meta = build_candidates(x)
        diff01 = find(meta, 0, 1, "diff")
        diff23 = find(meta, 2, 3, "diff")

        mode_report = {"single": {}, "pair": {}}

        for pd in args.single_proj_dims:
            single_abs, single_signed, single_feat = single_scan(effects, y_pm, pd)
            r, s = rank_score(single_abs, diff01)
            top_idx = single_abs.topk(min(args.topk_train, effects.shape[0])).indices
            acc = train_linear(single_feat[top_idx], y, steps=args.steps)
            mode_report["single"][str(pd)] = {
                "true_diff_rank": r,
                "true_diff_abs_score": s,
                "true_diff_signed_score": float(single_signed[diff01].cpu()),
                "reverse_diff_signed_score": float(single_signed[find(meta, 0, 1, "revdiff")].cpu()),
                "top_linear_acc": acc,
            }

        exact_scores, exact_sec = pair_exact(effects, y_pm, args.row_chunk)
        er, es = pair_rank_score(exact_scores, diff01, diff23)
        mode_report["pair"]["exact"] = {
            "true_pair_rank": er,
            "true_pair_score": es,
            "seconds": exact_sec,
            "top_pairs": top_pairs(exact_scores, meta, 8),
        }

        for jd in args.joint_proj_dims:
            jl_scores, jl_sec = pair_jl(effects, y_pm, jd, args.row_chunk)
            jr, js = pair_rank_score(jl_scores, diff01, diff23)
            mode_report["pair"][f"jl{jd}"] = {
                "true_pair_rank": jr,
                "true_pair_score": js,
                "seconds": jl_sec,
                "speedup_vs_exact": exact_sec / max(jl_sec, 1e-9),
                "top_pairs": top_pairs(jl_scores, meta, 4),
            }

        out[mode] = mode_report

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--single-proj-dims", default="16,32")
    p.add_argument("--joint-proj-dims", default="8,16,32")
    p.add_argument("--row-chunk", type=int, default=4)
    p.add_argument("--topk-train", type=int, default=32)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)
    args.single_proj_dims = [int(x) for x in args.single_proj_dims.split(",")]
    args.joint_proj_dims = [int(x) for x in args.joint_proj_dims.split(",")]

    report = {"device": args.device, "seed": args.seed, "cases": {}}
    for task in ["first_diff", "chain_product"]:
        for shift in ["none", "global", "per_sample"]:
            report["cases"][f"{task}/{shift}"] = eval_case(task, shift, args)

    # compact verdicts
    c = report["cases"]
    report["verdict"] = {
        "single_raw_works_when_no_basis_shift":
            c["first_diff/none"]["raw"]["single"]["32"]["true_diff_rank"] <= 4
            and c["first_diff/none"]["raw"]["single"]["32"]["top_linear_acc"] > 0.90,

        "basis_unrotate_helps_under_global_shift":
            c["first_diff/global"]["basis_unrotated"]["single"]["32"]["top_linear_acc"]
            > c["first_diff/global"]["raw"]["single"]["32"]["top_linear_acc"] + 0.05,

        "basis_unrotate_helps_under_per_sample_shift":
            c["first_diff/per_sample"]["basis_unrotated"]["single"]["32"]["top_linear_acc"]
            > c["first_diff/per_sample"]["raw"]["single"]["32"]["top_linear_acc"] + 0.05,

        "jl16_finds_chain_pair_no_shift":
            c["chain_product/none"]["raw"]["pair"]["jl16"]["true_pair_rank"] <= 16,

        "basis_plus_jl16_finds_chain_pair_per_sample_shift":
            c["chain_product/per_sample"]["basis_unrotated"]["pair"]["jl16"]["true_pair_rank"] <= 16,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
