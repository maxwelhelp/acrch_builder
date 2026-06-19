from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .primitive_matrix import PrimitiveMatrix5x5


class HybridScanner(nn.Module):
    """Grid + semantic + usage + random candidate proposal scanner."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        prim_embed_dim: int,
        local_k: int = 9,
        semantic_k: int = 4,
        usage_k: int = 5,
        random_k: int = 1,
    ) -> None:
        super().__init__()
        self.local_k = local_k
        self.semantic_k = semantic_k
        self.usage_k = usage_k
        self.random_k = random_k
        self.source_type = nn.Embedding(4, prim_embed_dim)
        self.context_proj = nn.Linear(context_dim, prim_embed_dim)
        self.before_proj = nn.Linear(prim_embed_dim, prim_embed_dim)
        self.candidate_proj = nn.Linear(prim_embed_dim, prim_embed_dim)
        self.memory_proj = nn.Linear(dim, prim_embed_dim)
        self.score = nn.Sequential(
            nn.LayerNorm(prim_embed_dim * 4),
            nn.Linear(prim_embed_dim * 4, prim_embed_dim),
            nn.SiLU(),
            nn.Linear(prim_embed_dim, 1),
        )
        self.anchor = nn.Linear(context_dim, 25)

    def forward(
        self,
        context: torch.Tensor,
        memory: torch.Tensor,
        primitive_matrix: PrimitiveMatrix5x5,
        prev_action_emb: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        n = context.shape[0]
        device = context.device

        anchor_logits = self.anchor(context)
        anchor_ids = anchor_logits.argmax(dim=-1)

        local = primitive_matrix.local_window(anchor_ids, radius=1).reshape(n, -1)[:, : self.local_k]
        semantic = primitive_matrix.semantic_topk(anchor_ids, k=self.semantic_k).reshape(n, -1)
        usage = primitive_matrix.usage_topk(anchor_ids, k=self.usage_k).reshape(n, -1)
        random = torch.randint(0, primitive_matrix.num_primitives, (n, self.random_k), device=device)

        candidate_ids = torch.cat([local, semantic, usage, random], dim=-1)
        source_ids = torch.cat(
            [
                torch.zeros_like(local),
                torch.ones_like(semantic),
                torch.full_like(usage, 2),
                torch.full_like(random, 3),
            ],
            dim=-1,
        )

        cand_emb = primitive_matrix.emb[candidate_ids] + self.source_type(source_ids)
        ctx = self.context_proj(context).unsqueeze(1).expand_as(cand_emb)
        mem = self.memory_proj(memory).unsqueeze(1).expand_as(cand_emb)
        if prev_action_emb is None:
            prev = torch.zeros_like(cand_emb)
        else:
            prev = self.before_proj(prev_action_emb).unsqueeze(1).expand_as(cand_emb)

        feat = torch.cat([ctx, mem, self.candidate_proj(cand_emb), prev], dim=-1)
        proposal_logits = self.score(feat).squeeze(-1)

        with torch.no_grad():
            sem_not_grid = []
            for i in range(n):
                grid = set(local[i].tolist())
                sem = semantic[i].tolist()
                sem_not_grid.append(sum(1 for x in sem if x not in grid) / max(1, len(sem)))
            metrics = {
                "semantic_grid_mismatch": float(sum(sem_not_grid) / max(1, len(sem_not_grid))),
            }

        return candidate_ids, proposal_logits, source_ids, metrics
