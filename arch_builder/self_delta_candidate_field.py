from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfDeltaCandidateField(nn.Module):
    """
    Diagnostic candidate scorer.

    Compares low-rank simulator predicted primitive effect with detached
    executor actual primitive effect.

    This module must not replace scanner/simulator/controller.
    It only produces diagnostic candidate scores unless explicitly enabled
    later by a separate flag.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        prim_embed_dim: int,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.norm_sim = nn.LayerNorm(dim)
        self.norm_delta = nn.LayerNorm(dim)
        self.ctx_proj = nn.Linear(context_dim, hidden)
        self.prim_proj = nn.Linear(prim_embed_dim, hidden)
        self.sim_proj = nn.Linear(dim, hidden)
        self.delta_proj = nn.Linear(dim * 2, hidden)
        self.src_tgt_proj = nn.Linear(dim * 3, hidden)

        self.score = nn.Sequential(
            nn.LayerNorm(hidden * 5),
            nn.Linear(hidden * 5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # Start tiny, so diagnostic module does not create huge random logits.
        last = self.score[-1]
        nn.init.normal_(last.weight, std=1e-4)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        flat_context: torch.Tensor,      # [N, context_dim]
        flat_src: torch.Tensor,          # [N, D]
        flat_tgt: torch.Tensor,          # [N, D]
        top_prim_emb: torch.Tensor,      # [N, K, E]
        sim: torch.Tensor,               # [N, K, D]
        actual: torch.Tensor,            # [N, K, D]
        detach_actual: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        actual_ref = actual.detach() if detach_actual else actual

        delta = actual_ref - sim
        abs_delta = delta.abs()

        n, k, d = sim.shape

        ctx_f = self.ctx_proj(flat_context).unsqueeze(1).expand(n, k, -1)
        prim_f = self.prim_proj(top_prim_emb)
        sim_f = self.sim_proj(self.norm_sim(sim))

        delta_f = self.delta_proj(
            torch.cat(
                [
                    self.norm_delta(delta),
                    self.norm_delta(abs_delta),
                ],
                dim=-1,
            )
        )

        src_tgt = torch.cat(
            [
                flat_src.unsqueeze(1).expand(n, k, -1),
                flat_tgt.unsqueeze(1).expand(n, k, -1),
                (flat_src - flat_tgt).unsqueeze(1).expand(n, k, -1),
            ],
            dim=-1,
        )
        st_f = self.src_tgt_proj(src_tgt)

        feat = torch.cat([ctx_f, prim_f, sim_f, delta_f, st_f], dim=-1)
        score = self.score(feat).squeeze(-1)

        metrics = {
            "self_delta_mean": delta.pow(2).mean().detach(),
            "self_delta_abs": delta.abs().mean().detach(),
            "self_delta_score_mean": score.mean().detach(),
            "self_delta_score_std": score.std(unbiased=False).detach(),
            "self_delta_sim_norm": sim.float().norm(dim=-1).mean().detach(),
            "self_delta_actual_norm": actual_ref.float().norm(dim=-1).mean().detach(),
            "self_delta_rel_error": (
                delta.float().pow(2).sum(dim=-1).sqrt()
                / actual_ref.float().pow(2).sum(dim=-1).sqrt().clamp_min(1e-2)
            ).mean().detach(),
            "self_delta_sim_actual_cos": F.cosine_similarity(sim.float(), actual_ref.float(), dim=-1).mean().detach(),
        }
        return score, metrics
