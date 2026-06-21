from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arch_builder.vnext_controller import batched_mmr_select, candidate_similarity


def offdiag_mean(sim: torch.Tensor) -> float:
    k = sim.shape[-1]
    mask = (1.0 - torch.eye(k, dtype=sim.dtype, device=sim.device)).bool().unsqueeze(0)
    return float(sim.masked_select(mask.expand_as(sim)).mean().item())


def selected_offdiag_mean(sim: torch.Tensor, selected: torch.Tensor) -> float:
    k = sim.shape[-1]
    pair = selected.float()[:, :, None] * selected.float()[:, None, :]
    offdiag = pair * (1.0 - torch.eye(k, dtype=pair.dtype, device=pair.device).unsqueeze(0))
    denom = offdiag.sum(dim=(1, 2)).clamp_min(1.0)
    value = ((sim * offdiag).sum(dim=(1, 2)) / denom).mean()
    return float(value.item())


def main() -> int:
    torch.manual_seed(7)
    n, k, emb = 4, 8, 6
    budget = 3
    pm_emb = torch.randn(16, emb)
    top_ids = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ])

    # Deliberately make the top utility candidates behavior-collapsed while
    # preserving other high-enough candidates with diverse behavior.
    utility = torch.tensor([
        [1.00, 0.98, 0.96, 0.80, 0.78, 0.76, 0.20, 0.10],
        [1.00, 0.98, 0.96, 0.80, 0.78, 0.76, 0.20, 0.10],
        [1.00, 0.98, 0.96, 0.80, 0.78, 0.76, 0.20, 0.10],
        [1.00, 0.98, 0.96, 0.80, 0.78, 0.76, 0.20, 0.10],
    ])
    base = torch.tensor([1.0, 0.0, 0.0, 0.0])
    behavior = torch.stack([
        base,
        base + 0.01 * torch.tensor([0.0, 1.0, 0.0, 0.0]),
        base + 0.01 * torch.tensor([0.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    ]).to(dtype=torch.float32).unsqueeze(0).expand(n, -1, -1).contiguous()

    sim = candidate_similarity(top_ids, pm_emb, behavior, mode="effect")
    topk_idx = utility.topk(k=budget, dim=-1).indices
    topk_mask = torch.zeros_like(utility, dtype=torch.bool).scatter_(1, topk_idx, True)
    topk_sim = selected_offdiag_mean(sim, topk_mask)

    mask, metrics = batched_mmr_select(
        utility=utility,
        top_ids=top_ids,
        pm_emb=pm_emb,
        budget=budget,
        beta=0.70,
        behavior_feature=behavior,
        mode="effect",
        identity_weight=0.0,
    )
    selected_sim = selected_offdiag_mean(sim, mask)
    selected_utility = (utility * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1)
    topk_utility = (utility * topk_mask.float()).sum(dim=-1) / topk_mask.float().sum(dim=-1)
    utility_drop = float((topk_utility - selected_utility).mean().item())

    assert mask.shape == utility.shape
    assert torch.all(mask.sum(dim=-1) == budget), mask.sum(dim=-1)
    assert selected_sim < topk_sim - 0.10, (selected_sim, topk_sim)
    assert utility_drop <= 0.25, utility_drop
    assert float(metrics["mmr_post_similarity_selected"]) == float(metrics["mmr_selected_similarity"])
    # Diagonal self-similarity is 1.0, but the off-diagonal metric must not be 1.0.
    assert float(metrics["behavior_feature_pair_sim_mean"]) < 0.90

    out = {
        "status": "PASS",
        "mmr_selected_count": float(metrics["mmr_selected_count"]),
        "mmr_pre_similarity_topk": float(metrics["mmr_pre_similarity_topk"]),
        "mmr_post_similarity_selected": float(metrics["mmr_post_similarity_selected"]),
        "manual_topk_similarity": topk_sim,
        "manual_selected_similarity": selected_sim,
        "utility_drop_vs_topk": utility_drop,
        "global_offdiag_similarity": offdiag_mean(sim),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
