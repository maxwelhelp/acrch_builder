#!/usr/bin/env python3
"""Standalone acceptance probe for production projection scanner sources.

Synthetic labels are used only to verify that the generic proposal operators
preserve signed and bilinear signal.  This file is not imported by training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arch_builder.projection_scanner import (
    make_jl_projection,
    pair_jl_bilinear_scores,
    signed_corr,
    single_signed_projection_scores,
)


def make_batch(batch: int, slots: int, dim: int, task: str, device: str):
    x = torch.randn(batch, slots, dim, device=device)
    if task == "first_diff":
        signal = (x[:, 0] - x[:, 1]).mean(dim=-1)
    elif task == "chain_product":
        signal = ((x[:, 0] - x[:, 1]) * (x[:, 2] - x[:, 3])).mean(dim=-1)
    else:
        raise ValueError(task)
    y = (signal > 0).long()
    return x, y, y.float() * 2.0 - 1.0


def primitive_bank(src: torch.Tensor, tgt: torch.Tensor):
    eps = 1e-4
    return [
        ("identity_src", src), ("identity_tgt", tgt),
        ("diff", src - tgt), ("revdiff", tgt - src),
        ("sum", src + tgt), ("mean", 0.5 * (src + tgt)),
        ("product", src * tgt), ("neg_src", -src), ("neg_tgt", -tgt),
        ("absdiff", (src - tgt).abs()), ("sqdiff", (src - tgt).pow(2)),
        ("max", torch.maximum(src, tgt)), ("min", torch.minimum(src, tgt)),
        ("gate_src_tgt", torch.sigmoid(src) * tgt),
        ("gate_tgt_src", torch.sigmoid(tgt) * src),
        ("tanh_sum", torch.tanh(src + tgt)),
        ("relu_diff", F.relu(src - tgt)), ("relu_revdiff", F.relu(tgt - src)),
        ("norm_diff", (src - tgt) / (src.abs() + tgt.abs() + eps)),
        ("src_sq", src.pow(2)), ("tgt_sq", tgt.pow(2)),
        ("src_times_abs_tgt", src * tgt.abs()),
        ("tgt_times_abs_src", tgt * src.abs()),
        ("smooth_src", 0.75 * src + 0.25 * tgt),
        ("smooth_tgt", 0.25 * src + 0.75 * tgt),
    ]


def build_candidates(x: torch.Tensor):
    effects, meta = [], []
    for src_i in range(x.shape[1]):
        for tgt_i in range(x.shape[1]):
            for name, effect in primitive_bank(x[:, src_i], x[:, tgt_i]):
                effects.append(effect)
                meta.append({"src": src_i, "tgt": tgt_i, "primitive": name})
    return torch.stack(effects), meta


def find_candidate(meta, src: int, tgt: int, primitive: str) -> int:
    return next(
        i for i, item in enumerate(meta)
        if item == {"src": src, "tgt": tgt, "primitive": primitive}
    )


def meta_str(meta, index: int) -> str:
    item = meta[index]
    return f"{item['src']}->{item['tgt']}:{item['primitive']}"


def rank_of(scores: torch.Tensor, index: int) -> int:
    return int((scores > scores[index]).sum().item() + 1)


def pair_rank(scores: torch.Tensor, left: int, right: int) -> int:
    return int((scores > scores[left, right]).sum().item() + 1)


def train_linear(features_mb: torch.Tensor, y: torch.Tensor, steps: int) -> float:
    weight = torch.zeros(features_mb.shape[0], device=features_mb.device, requires_grad=True)
    bias = torch.zeros((), device=features_mb.device, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = (weight[:, None] * features_mb).sum(dim=0) + bias
        F.binary_cross_entropy_with_logits(logits, y.float()).backward()
        optimizer.step()
    with torch.no_grad():
        logits = (weight[:, None] * features_mb).sum(dim=0) + bias
        return float(((logits > 0) == (y > 0)).float().mean().cpu())


@torch.no_grad()
def exact_pair_scores(effects: torch.Tensor, target: torch.Tensor, row_chunk: int):
    n, batch, _ = effects.shape
    scores = torch.empty(n, n, device=effects.device)
    if effects.is_cuda:
        torch.cuda.synchronize(effects.device)
    started = time.perf_counter()
    for i0 in range(0, n, row_chunk):
        left = effects[i0:i0 + row_chunk]
        pair = (left[:, None] * effects[None]).mean(dim=-1)
        scores[i0:i0 + left.shape[0]] = signed_corr(pair.reshape(-1, batch), target).abs().view(left.shape[0], n)
    if effects.is_cuda:
        torch.cuda.synchronize(effects.device)
    return scores, time.perf_counter() - started


def top_pairs(scores: torch.Tensor, meta, count: int = 8):
    values, indices = scores.flatten().topk(count)
    n = scores.shape[0]
    return [
        {
            "rank": rank,
            "pair": f"{meta_str(meta, flat // n)} * {meta_str(meta, flat % n)}",
            "score": float(value),
        }
        for rank, (value, flat) in enumerate(
            zip(values.cpu().tolist(), indices.cpu().tolist()), 1
        )
    ]


def write_markdown(path: Path, report: dict) -> None:
    checks = report["checks"]
    lines = [
        "# Projection joint scanner probe", "",
        f"- status: `{report['status']}`",
        f"- device: `{report['device']}`",
        "- measured_delta_loss_is_source_of_truth: `true`",
        "- expected_actions_used_for_training: `false`", "",
        "## Checks", "",
    ]
    lines += [
        f"- {name}: `{'SKIP' if value is None else ('PASS' if value else 'FAIL')}`"
        for name, value in checks.items()
    ]
    lines += ["", "## Key measurements", ""]
    first = report["first_diff"]
    chain = report["chain_product"]
    lines += [
        f"- first_diff true rank: `{first['true_diff_rank']}`",
        f"- first_diff top32 linear acc: `{first['top32_linear_acc']:.4f}`",
        f"- exact true pair rank: `{chain['exact_joint']['true_pair_rank']}`",
        f"- JL16 true pair rank: `{chain['jl_joint']['16']['true_pair_rank']}`",
        f"- JL16 speedup: `{chain['jl16_speedup_vs_exact']:.3f}x`",
        "",
        "JL is a proposer only. Bounded measured counterfactual delta-loss remains the training truth.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--single-proj-dim", type=int, default=32)
    parser.add_argument("--joint-proj-dims", default="4,8,16,32")
    parser.add_argument("--row-chunk", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="reports/agent_inspector/projection_joint_scanner_probe.json")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)

    # Single-source acceptance uses the production sign-preserving function.
    x, y, target = make_batch(args.batch_size, args.slots, args.dim, "first_diff", args.device)
    effects, meta = build_candidates(x)
    diff01 = find_candidate(meta, 0, 1, "diff")
    revdiff01 = find_candidate(meta, 0, 1, "revdiff")
    single_projection = make_jl_projection(args.dim, args.single_proj_dim, args.seed).to(args.device)
    single = single_signed_projection_scores(effects, target, single_projection)
    top_indices = single["rank_score"].topk(min(32, effects.shape[0])).indices
    single_acc = train_linear(single["signed_feature"][top_indices], y, args.steps)
    first_report = {
        "candidate_count": effects.shape[0],
        "true_diff_rank": rank_of(single["rank_score"], diff01),
        "true_diff_signed_score": float(single["signed_score"][diff01].cpu()),
        "reverse_diff_signed_score": float(single["signed_score"][revdiff01].cpu()),
        "diff_revdiff_sign_preserved": bool(
            (single["signed_score"][diff01] * single["signed_score"][revdiff01] < 0).item()
        ),
        "top32_linear_acc": single_acc,
    }

    # Pair acceptance uses the production JL proposer and an exact debug oracle.
    x, _, target = make_batch(args.batch_size, args.slots, args.dim, "chain_product", args.device)
    effects, meta = build_candidates(x)
    diff01 = find_candidate(meta, 0, 1, "diff")
    diff23 = find_candidate(meta, 2, 3, "diff")
    exact, exact_seconds = exact_pair_scores(effects, target, args.row_chunk)
    jl_reports = {}
    for proj_dim in [int(v) for v in args.joint_proj_dims.split(",") if v.strip()]:
        projection = make_jl_projection(args.dim, proj_dim, args.seed + 100 + proj_dim).to(args.device)
        result = pair_jl_bilinear_scores(
            effects, target, projection,
            candidate_indices=torch.arange(effects.shape[0], device=effects.device),
            row_chunk=args.row_chunk,
        )
        jl_reports[str(proj_dim)] = {
            "true_pair_rank": pair_rank(result["rank_score"], diff01, diff23),
            "true_pair_signed_score": float(result["signed_score"][diff01, diff23].cpu()),
            "seconds": float(result["seconds"]),
            "pairs_tested": int(result["pairs_tested"]),
            "top_pairs": top_pairs(result["rank_score"], meta),
        }
    if "16" not in jl_reports:
        raise ValueError("--joint-proj-dims must include 16 for acceptance")
    speedup = exact_seconds / max(jl_reports["16"]["seconds"], 1e-9)
    chain_report = {
        "candidate_count": effects.shape[0],
        "exact_joint": {
            "true_pair_rank": pair_rank(exact, diff01, diff23),
            "seconds": exact_seconds,
            "top_pairs": top_pairs(exact, meta),
        },
        "jl_joint": jl_reports,
        "jl16_speedup_vs_exact": speedup,
    }

    checks = {
        "first_diff_true_rank_le_4": first_report["true_diff_rank"] <= 4,
        "first_diff_direction_sign_preserved": first_report["diff_revdiff_sign_preserved"],
        "first_diff_top32_linear_acc_gt_0_90": first_report["top32_linear_acc"] > 0.90,
        "chain_exact_true_pair_rank_le_8": chain_report["exact_joint"]["true_pair_rank"] <= 8,
        "chain_jl16_true_pair_rank_le_16": jl_reports["16"]["true_pair_rank"] <= 16,
        # CPU timings are too backend-dependent to be an acceptance statement.
        "chain_jl16_speedup_ge_1_5": speedup >= 1.5 if args.device == "cuda" else None,
    }
    required_checks = [value for value in checks.values() if value is not None]
    core_pass = all(required_checks)
    status = ("PASS" if core_pass else "FAIL") if args.device == "cuda" else (
        "SMOKE_PASS" if core_pass else "SMOKE_FAIL"
    )
    report = {
        "status": status,
        "device": args.device,
        "seed": args.seed,
        "checks": checks,
        "first_diff": first_report,
        "chain_product": chain_report,
        "scanner_metrics": {
            "single_signed_projection_usage": 1.0,
            "single_signed_projection_candidate_count": float(first_report["candidate_count"]),
            "single_signed_projection_top_score": float(single["rank_score"].max().cpu()),
            "single_signed_projection_signed_score_mean": float(single["signed_score"].mean().cpu()),
            "pair_jl16_usage": 1.0,
            "pair_jl16_candidate_count": float(effects.shape[0] ** 2),
            "pair_jl16_top_score": float(jl_reports["16"]["top_pairs"][0]["score"]),
            "pair_jl16_seconds": jl_reports["16"]["seconds"],
            "pair_jl16_pairs_tested": float(jl_reports["16"]["pairs_tested"]),
            "flat_shortcut_candidate_usage": 1.0,
            "compositional_pair_candidate_usage": 1.0,
            "measured_delta_loss_is_source_of_truth": True,
            "expected_actions_used_for_training": False,
        },
        "fallback": "keep pair proposal broad and verify by bounded counterfactual measured delta-loss",
        "synthetic_probe_is_real_discovery_acceptance": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(out.with_suffix(".md"), report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"PASS", "SMOKE_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
