from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class UtilityCritic(nn.Module):
    """vNext Utility Critic that evaluates candidates using full uncompressed context."""

    def __init__(self, dim: int, context_dim: int, prim_embed_dim: int, hidden: int = 128) -> None:
        super().__init__()
        in_dim = context_dim + prim_embed_dim + dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),  # [utility, log_uncertainty]
        )

    def forward(
        self,
        flat_context: torch.Tensor,
        prim_emb: torch.Tensor,
        head_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            flat_context: [N, context_dim]
            prim_emb: [N, top_k, prim_embed_dim]
            head_vector: [N, dim]
        Returns:
            utility: [N, top_k]
            uncertainty: [N, top_k]
        """
        n, k, e = prim_emb.shape
        
        # Expand context and head vector across top_k candidates
        ctx_exp = flat_context.unsqueeze(1).expand(n, k, -1)
        head_exp = head_vector.unsqueeze(1).expand(n, k, -1)
        
        # Concat inputs
        x = torch.cat([ctx_exp, prim_emb, head_exp], dim=-1)
        
        out = self.net(x)
        utility = out[..., 0]
        uncertainty = F.softplus(out[..., 1])
        return utility, uncertainty
