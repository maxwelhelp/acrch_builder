from __future__ import annotations

import math
import time
from typing import Dict, Optional

import torch
import torch.nn as nn


def signed_corr(features_mb: torch.Tensor, target_b: torch.Tensor) -> torch.Tensor:
    """Signed Pearson correlation for M candidate features over B examples."""
    if features_mb.ndim != 2 or target_b.ndim != 1:
        raise ValueError("expected features [M,B] and target [B]")
    if features_mb.shape[1] != target_b.shape[0]:
        raise ValueError("feature batch and target batch differ")
    x = features_mb.float()
    y = target_b.float()
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean()
    return (x * y.unsqueeze(0)).mean(dim=1) / (
        x.std(dim=1, unbiased=False).clamp_min(1e-8)
        * y.std(unbiased=False).clamp_min(1e-8)
    )


def make_jl_projection(dim: int, proj_dim: int, seed: int = 0) -> torch.Tensor:
    """Stable JL matrix with an explicit DC/mean channel."""
    if dim <= 0 or proj_dim <= 0:
        raise ValueError("projection dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(dim, proj_dim, generator=generator) / math.sqrt(dim)
    projection[:, 0] = 1.0 / math.sqrt(dim)
    return projection


def single_signed_projection_scores(
    effects_nbd: torch.Tensor,
    target_b: torch.Tensor,
    projection_dp: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Rank individual effects while retaining signed controller features."""
    if effects_nbd.ndim != 3:
        raise ValueError("effects must be [N,B,D]")
    n, b, d = effects_nbd.shape
    if projection_dp.shape[0] != d:
        raise ValueError("projection input dimension differs from effect dimension")
    projection = projection_dp.to(device=effects_nbd.device, dtype=effects_nbd.dtype)
    projected = torch.einsum("nbd,dp->nbp", effects_nbd, projection)
    flat = projected.permute(0, 2, 1).reshape(n * projection.shape[1], b)
    signed_all = signed_corr(flat, target_b).view(n, projection.shape[1])
    rank_score, best_dim = signed_all.abs().max(dim=1)
    row = torch.arange(n, device=effects_nbd.device)
    signed_score = signed_all[row, best_dim]
    direction = signed_score.sign()
    direction = torch.where(direction == 0, torch.ones_like(direction), direction)
    signed_feature = projected[row, :, best_dim] * direction[:, None]
    return {
        "rank_score": rank_score,
        "signed_score": signed_score,
        "signed_feature": signed_feature,
        "best_proj_dim": best_dim,
        "signed_scores_all": signed_all,
    }


@torch.no_grad()
def pair_jl_bilinear_scores(
    effects_nbd: torch.Tensor,
    target_b: torch.Tensor,
    projection_dp: torch.Tensor,
    candidate_indices: Optional[torch.Tensor] = None,
    row_chunk: int = 8,
) -> Dict[str, torch.Tensor | float | int]:
    """Bounded bilinear pair proposer over a preselected candidate pool."""
    if effects_nbd.ndim != 3:
        raise ValueError("effects must be [N,B,D]")
    if candidate_indices is None:
        candidate_indices = torch.arange(effects_nbd.shape[0], device=effects_nbd.device)
    candidate_indices = candidate_indices.to(device=effects_nbd.device, dtype=torch.long)
    if effects_nbd.is_cuda:
        torch.cuda.synchronize(effects_nbd.device)
    started = time.perf_counter()
    selected = effects_nbd[candidate_indices]
    projection = projection_dp.to(device=effects_nbd.device, dtype=effects_nbd.dtype)
    projected = torch.einsum("mbd,dp->mbp", selected, projection)
    m, b, _ = projected.shape
    signed_out = torch.empty(m, m, device=effects_nbd.device, dtype=torch.float32)
    for i0 in range(0, m, max(1, row_chunk)):
        left = projected[i0 : i0 + row_chunk]
        pair_feature = (left[:, None] * projected[None]).mean(dim=-1)
        signed = signed_corr(pair_feature.reshape(-1, b), target_b)
        signed_out[i0 : i0 + left.shape[0]] = signed.view(left.shape[0], m)
    if effects_nbd.is_cuda:
        torch.cuda.synchronize(effects_nbd.device)
    seconds = time.perf_counter() - started
    return {
        "rank_score": signed_out.abs(),
        "signed_score": signed_out,
        "candidate_indices": candidate_indices,
        "seconds": seconds,
        "pairs_tested": int(m * m),
    }


class ProjectionScannerSources(nn.Module):
    """Reusable single-signed and bounded JL16 pair proposal sources.

    The target is supplied by the caller and must be a task-gradient/credit
    signal in real discovery. This module never reads labels for primitives,
    expected actions, or layer roles.
    """

    def __init__(
        self,
        effect_dim: int,
        single_proj_dim: int = 32,
        pair_proj_dim: int = 16,
        seed: int = 6416,
    ) -> None:
        super().__init__()
        self.effect_dim = effect_dim
        self.single_proj_dim = single_proj_dim
        self.pair_proj_dim = pair_proj_dim
        self.register_buffer(
            "single_projection",
            make_jl_projection(effect_dim, single_proj_dim, seed=seed),
            persistent=False,
        )
        self.register_buffer(
            "pair_projection",
            make_jl_projection(effect_dim, pair_proj_dim, seed=seed + 1),
            persistent=False,
        )

    def propose(
        self,
        effects_nbd: torch.Tensor,
        target_b: torch.Tensor,
        single_top_k: int = 32,
        pair_pool_indices: Optional[torch.Tensor] = None,
        pair_top_k: int = 32,
        row_chunk: int = 8,
        enable_single: bool = True,
        enable_pair: bool = True,
    ) -> Dict[str, object]:
        if not enable_single and not enable_pair:
            raise ValueError("at least one projection source must be enabled")
        single = single_signed_projection_scores(
            effects_nbd, target_b, self.single_projection
        )
        single_k = min(single_top_k, effects_nbd.shape[0])
        single_indices = single["rank_score"].topk(single_k).indices
        pair = None
        pair_left = pair_right = torch.empty(0, dtype=torch.long, device=effects_nbd.device)
        pair_signed = torch.empty(0, dtype=torch.float32, device=effects_nbd.device)
        pair_k = 0
        pair_top_score = pair_seconds = pair_tested = 0.0
        if enable_pair:
            pool = single_indices if pair_pool_indices is None else pair_pool_indices
            pair = pair_jl_bilinear_scores(
                effects_nbd,
                target_b,
                self.pair_projection,
                candidate_indices=pool,
                row_chunk=row_chunk,
            )
            flat_score = pair["rank_score"].flatten()
            pair_k = min(pair_top_k, flat_score.numel())
            top_flat = flat_score.topk(pair_k).indices
            pool_size = int(pair["rank_score"].shape[0])
            pool_indices = pair["candidate_indices"]
            pair_left = pool_indices[top_flat // pool_size]
            pair_right = pool_indices[top_flat % pool_size]
            pair_signed = pair["signed_score"].flatten()[top_flat]
            pair_top_score = float(flat_score.max().detach().cpu())
            pair_seconds = float(pair["seconds"])
            pair_tested = float(pair["pairs_tested"])
        metrics = {
            "single_signed_projection_usage": float(enable_single),
            "single_signed_projection_candidate_count": float(single_k if enable_single else 0),
            "single_signed_projection_top_score": float(single["rank_score"].max().detach().cpu()),
            "single_signed_projection_signed_score_mean": float(single["signed_score"].mean().detach().cpu()),
            "pair_jl16_usage": float(pair_k > 0),
            "pair_jl16_candidate_count": float(pair_k),
            "pair_jl16_top_score": pair_top_score,
            "pair_jl16_seconds": pair_seconds,
            "pair_jl16_pairs_tested": pair_tested,
            "flat_shortcut_candidate_usage": float(enable_single and single_k > 0),
            "compositional_pair_candidate_usage": float(pair_k > 0),
            "measured_delta_loss_is_source_of_truth": True,
            "expected_actions_used_for_training": False,
        }
        return {
            "single": single,
            "single_indices": single_indices,
            "pair": pair,
            "pair_left": pair_left,
            "pair_right": pair_right,
            "pair_signed_score": pair_signed,
            "metrics": metrics,
        }
