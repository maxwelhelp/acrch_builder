from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UtilityCritic(nn.Module):
    """Full-context utility critic with optional behavior features."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        prim_embed_dim: int,
        hidden: int = 128,
        behavior_dim: int = 32,
    ) -> None:
        super().__init__()
        in_dim = context_dim + prim_embed_dim + dim
        self.behavior_dim = int(behavior_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.utility_head = nn.Linear(hidden, 1)
        self.log_uncertainty_head = nn.Linear(hidden, 1)
        self.behavior_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.behavior_dim),
        )

    def score_with_features(
        self,
        flat_context: torch.Tensor,
        prim_emb: torch.Tensor,
        head_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, k, _ = prim_emb.shape
        ctx_exp = flat_context.unsqueeze(1).expand(n, k, -1)
        head_exp = head_vector.unsqueeze(1).expand(n, k, -1)
        x = torch.cat([ctx_exp, prim_emb, head_exp], dim=-1)
        h = self.trunk(x)
        utility = self.utility_head(h).squeeze(-1)
        uncertainty = F.softplus(self.log_uncertainty_head(h).squeeze(-1))
        raw_behavior = self.behavior_head(h)
        # Center behavior features across the candidates of each cell (dim=1)
        mean_behavior = raw_behavior.mean(dim=1, keepdim=True)
        centered_behavior = raw_behavior - mean_behavior
        behavior = F.normalize(centered_behavior, p=2, dim=-1, eps=1e-6)
        return utility, uncertainty, behavior

    def forward(
        self,
        flat_context: torch.Tensor,
        prim_emb: torch.Tensor,
        head_vector: torch.Tensor,
        return_behavior: bool = False,
    ):
        utility, uncertainty, behavior = self.score_with_features(
            flat_context=flat_context,
            prim_emb=prim_emb,
            head_vector=head_vector,
        )
        if return_behavior:
            return utility, uncertainty, behavior
        return utility, uncertainty
