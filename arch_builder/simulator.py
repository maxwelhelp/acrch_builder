from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankSimulator(nn.Module):
    """Low-rank candidate preview and predicted gain head."""

    def __init__(self, dim: int, num_primitives: int, rank: int = 16, embed_dim: int = 32) -> None:
        super().__init__()
        self.rank = rank
        self.prim_emb = nn.Embedding(num_primitives, embed_dim)
        self.to_rank = nn.Linear(dim + embed_dim, rank)
        self.from_rank = nn.Linear(rank, dim)
        self.gain = nn.Sequential(
            nn.LayerNorm(dim + embed_dim),
            nn.Linear(dim + embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, state: torch.Tensor, candidate_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # state: [N,D], candidate_ids: [N,K]
        n, k = candidate_ids.shape
        d = state.shape[-1]
        emb = self.prim_emb(candidate_ids)
        st = state.unsqueeze(1).expand(n, k, d)
        x = torch.cat([st, emb], dim=-1)
        sim = self.from_rank(torch.tanh(self.to_rank(x)))
        predicted_gain = self.gain(torch.cat([sim, emb], dim=-1)).squeeze(-1)
        return sim, predicted_gain
