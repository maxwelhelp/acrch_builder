from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def normalize_scores(x: torch.Tensor) -> torch.Tensor:
    """Per-row min-max normalization for controller utilities."""
    x = x.float()
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    return (x - x_min) / (x_max - x_min).clamp_min(1e-6)


def _offdiag_values(sim: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if sim.ndim != 3:
        raise ValueError(f"similarity must be [N,K,K], got {tuple(sim.shape)}")
    k = sim.shape[-1]
    if k < 2:
        return sim.new_zeros((0,))
    offdiag = (1.0 - torch.eye(k, dtype=sim.dtype, device=sim.device)).bool().unsqueeze(0)
    if mask is not None:
        offdiag = offdiag & mask.bool()
    return sim.masked_select(offdiag.expand_as(sim))


def _safe_mean(values: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return fallback
    return values.mean()


def _safe_std(values: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return fallback
    return values.std(unbiased=False)


def _offdiag_stats(sim: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
    zero = sim.new_zeros(())
    values = _offdiag_values(sim)
    if values.numel() == 0:
        return {
            f"{prefix}_mean": zero,
            f"{prefix}_std": zero,
            f"{prefix}_min": zero,
            f"{prefix}_max": zero,
        }
    return {
        f"{prefix}_mean": values.mean(),
        f"{prefix}_std": values.std(unbiased=False),
        f"{prefix}_min": values.min(),
        f"{prefix}_max": values.max(),
    }


def candidate_similarity_parts(
    top_ids: torch.Tensor,
    pm_emb: torch.Tensor,
    behavior_feature: Optional[torch.Tensor] = None,
    mode: str = "hybrid",
    identity_weight: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return selected MMR similarity and diagnostic similarity parts."""
    mode = str(mode).lower()
    norm_emb = F.normalize(pm_emb.float(), p=2, dim=-1, eps=1e-6)
    cand_emb = norm_emb[top_ids]
    identity_sim = torch.bmm(cand_emb, cand_emb.transpose(1, 2)).clamp(-1.0, 1.0)

    if behavior_feature is None:
        effect_sim = identity_sim
    else:
        behavior = F.normalize(behavior_feature.float(), p=2, dim=-1, eps=1e-6)
        effect_sim = torch.bmm(behavior, behavior.transpose(1, 2)).clamp(-1.0, 1.0)

    if mode == "identity":
        sim = identity_sim
    elif mode == "effect":
        sim = effect_sim
    elif mode == "hybrid":
        w = float(identity_weight)
        sim = w * identity_sim + (1.0 - w) * effect_sim
    else:
        raise ValueError(f"unknown MMR mode: {mode!r}")
    return sim.clamp(-1.0, 1.0), {"identity": identity_sim, "effect": effect_sim}


def candidate_similarity(
    top_ids: torch.Tensor,
    pm_emb: torch.Tensor,
    behavior_feature: Optional[torch.Tensor] = None,
    mode: str = "hybrid",
    identity_weight: float = 0.2,
) -> torch.Tensor:
    """Return [N,K,K] candidate similarity for MMR."""
    sim, _ = candidate_similarity_parts(
        top_ids=top_ids,
        pm_emb=pm_emb,
        behavior_feature=behavior_feature,
        mode=mode,
        identity_weight=identity_weight,
    )
    return sim


def _selected_offdiag_mean(sim: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
    k = sim.shape[-1]
    pair_mask = selected_mask.float()[:, :, None] * selected_mask.float()[:, None, :]
    eye = torch.eye(k, dtype=pair_mask.dtype, device=sim.device).unsqueeze(0)
    offdiag_mask = pair_mask * (1.0 - eye)
    denom = offdiag_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return ((sim * offdiag_mask).sum(dim=(1, 2)) / denom).mean()


def batched_mmr_select(
    utility: torch.Tensor,
    top_ids: torch.Tensor,
    pm_emb: torch.Tensor,
    budget: int,
    beta: float,
    behavior_feature: Optional[torch.Tensor] = None,
    mode: str = "hybrid",
    uncertainty: Optional[torch.Tensor] = None,
    uncertainty_weight: float = 0.0,
    identity_weight: float = 0.2,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Select a diverse candidate subset without executing all candidates.

    Diagnostics intentionally report off-diagonal similarity only. Diagonal
    self-similarity must never make a collapsed controller look healthy.
    """
    if utility.ndim != 2:
        raise ValueError(f"utility must be [N,K], got {tuple(utility.shape)}")
    device = utility.device
    n, k = utility.shape
    budget = max(1, min(int(budget), int(k)))

    score = utility.float()
    if uncertainty is not None and float(uncertainty_weight) != 0.0:
        score = score + float(uncertainty_weight) * uncertainty.float()
    score_norm = normalize_scores(score)
    sim, parts = candidate_similarity_parts(
        top_ids=top_ids,
        pm_emb=pm_emb,
        behavior_feature=behavior_feature,
        mode=mode,
        identity_weight=identity_weight,
    )
    sim = sim.to(device=device)
    identity_sim = parts["identity"].to(device=device)
    effect_sim = parts["effect"].to(device=device)

    selected_mask = torch.zeros((n, k), dtype=torch.bool, device=device)
    max_sim = torch.zeros((n, k), dtype=score_norm.dtype, device=device)
    beta = float(beta)

    for step in range(budget):
        if step == 0:
            mmr_score = score_norm
        else:
            mmr_score = (1.0 - beta) * score_norm - beta * max_sim
        mmr_score = mmr_score.masked_fill(selected_mask, float("-inf"))
        idx = mmr_score.argmax(dim=-1, keepdim=True)
        selected_mask.scatter_(1, idx, True)
        new_sim = sim.gather(2, idx.unsqueeze(-1).expand(-1, k, -1)).squeeze(-1)
        max_sim = torch.maximum(max_sim, new_sim)

    with torch.no_grad():
        topk_idx = score_norm.topk(k=budget, dim=-1).indices
        topk_mask = torch.zeros_like(selected_mask).scatter_(1, topk_idx, True)
        selected_count = selected_mask.float().sum(dim=-1).mean()
        selected_sim = _selected_offdiag_mean(sim, selected_mask)
        topk_sim = _selected_offdiag_mean(sim, topk_mask)
        selected_utility = (score * selected_mask.float()).sum(dim=-1) / selected_mask.float().sum(dim=-1).clamp_min(1.0)
        topk_utility = (score * topk_mask.float()).sum(dim=-1) / topk_mask.float().sum(dim=-1).clamp_min(1.0)
        metrics = {
            "mmr_selected_count": selected_count.detach(),
            "mmr_selected_similarity": selected_sim.detach(),
            "mmr_post_similarity_selected": selected_sim.detach(),
            "mmr_pre_similarity_topk": topk_sim.detach(),
            "mmr_selected_utility_mean": selected_utility.mean().detach(),
            "mmr_utility_drop_vs_topk": (topk_utility - selected_utility).mean().detach(),
            "mmr_beta": torch.tensor(beta, device=device),
            "mmr_identity_weight": torch.tensor(float(identity_weight), device=device),
            "behavior_pair_sim_before": _selected_offdiag_mean(effect_sim, topk_mask).detach(),
            "behavior_pair_sim_after": _selected_offdiag_mean(effect_sim, selected_mask).detach(),
            **{k: v.detach() for k, v in _offdiag_stats(effect_sim, "behavior_feature_pair_sim").items()},
            **{k: v.detach() for k, v in _offdiag_stats(identity_sim, "identity_feature_pair_sim").items()},
        }
    return selected_mask, metrics
