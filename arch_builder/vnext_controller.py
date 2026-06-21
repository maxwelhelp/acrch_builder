from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def normalize_scores(x: torch.Tensor) -> torch.Tensor:
    """Per-row min-max normalization for controller utilities."""
    x = x.float()
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    return (x - x_min) / (x_max - x_min).clamp_min(1e-6)


def candidate_similarity(
    top_ids: torch.Tensor,
    pm_emb: torch.Tensor,
    behavior_feature: Optional[torch.Tensor] = None,
    mode: str = "hybrid",
    identity_weight: float = 0.2,
) -> torch.Tensor:
    """Return [N,K,K] candidate similarity for MMR.

    Contract:
    - identity mode uses primitive embeddings only.
    - effect mode uses learned critic behavior features only.
    - hybrid uses identity + behavior. This is the default vNext path because
      it supports Lazy Executor: no true executor effects are needed for all
      candidates.
    """
    mode = str(mode).lower()
    norm_emb = F.normalize(pm_emb.float(), p=2, dim=-1, eps=1e-6)
    cand_emb = norm_emb[top_ids]
    identity_sim = torch.bmm(cand_emb, cand_emb.transpose(1, 2)).clamp(-1.0, 1.0)

    if behavior_feature is None or mode == "identity":
        return identity_sim

    behavior = F.normalize(behavior_feature.float(), p=2, dim=-1, eps=1e-6)
    effect_sim = torch.bmm(behavior, behavior.transpose(1, 2)).clamp(-1.0, 1.0)

    if mode == "effect":
        return effect_sim
    if mode == "hybrid":
        w = float(identity_weight)
        return w * identity_sim + (1.0 - w) * effect_sim
    raise ValueError(f"unknown MMR mode: {mode!r}")


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

    The score is scalar utility plus optional uncertainty bonus. Diversity uses
    cheap learned behavior features from UtilityCritic, not true primitive
    outputs. This keeps the MMR Controller compatible with Lazy Executor.
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
    sim = candidate_similarity(
        top_ids=top_ids,
        pm_emb=pm_emb,
        behavior_feature=behavior_feature,
        mode=mode,
        identity_weight=identity_weight,
    ).to(device=device)

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
        selected_count = selected_mask.float().sum(dim=-1).mean()
        selected_sim = (sim * selected_mask[:, :, None].float() * selected_mask[:, None, :].float())
        denom = (selected_mask.float().sum(dim=-1).clamp_min(1.0) ** 2).mean().clamp_min(1.0)
        metrics = {
            "mmr_selected_count": selected_count.detach(),
            "mmr_selected_similarity": (selected_sim.sum(dim=(1, 2)).mean() / denom).detach(),
            "mmr_beta": torch.tensor(beta, device=device),
        }
    return selected_mask, metrics
