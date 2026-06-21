from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .primitive_matrix import PrimitiveMatrix5x5
from .projection_scanner import ProjectionScannerSources


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
        single_proj_dim: int = 32,
        pair_jl_dim: int = 16,
        enable_scanner_feedback_memory: bool = False,
        enable_category_scanner: bool = False,
        num_primitives: int = 25,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.local_k = local_k
        self.semantic_k = semantic_k
        self.usage_k = usage_k
        self.random_k = random_k
        # Four production proposal sources plus an explicit debug full-scan
        # source. The latter is used only when top_k covers the whole matrix.
        self.source_type = nn.Embedding(5, prim_embed_dim)
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
        self.anchor = nn.Linear(context_dim, num_primitives)
        self.projection_sources = ProjectionScannerSources(
            effect_dim=dim,
            single_proj_dim=single_proj_dim,
            pair_proj_dim=pair_jl_dim,
        )
        self.feedback_source_type = None
        if enable_scanner_feedback_memory:
            cpu_rng = torch.random.get_rng_state()
            self.feedback_source_type = nn.Embedding(1, prim_embed_dim)
            torch.random.set_rng_state(cpu_rng)

        self.category_source_type = None
        if enable_category_scanner:
            cpu_rng = torch.random.get_rng_state()
            self.category_source_type = nn.Embedding(1, prim_embed_dim)
            torch.random.set_rng_state(cpu_rng)


    def projection_proposals(
        self,
        effects_nbd: torch.Tensor,
        credit_signal_b: torch.Tensor,
        **kwargs,
    ) -> Dict[str, object]:
        """Bounded proposal API; measured credit remains source of truth."""
        return self.projection_sources.propose(
            effects_nbd,
            credit_signal_b,
            **kwargs,
        )

    def forward(
        self,
        context: torch.Tensor,
        memory: torch.Tensor,
        primitive_matrix: PrimitiveMatrix5x5,
        prev_action_emb: torch.Tensor | None = None,
        ensure_all_candidates: bool = False,
        collect_metrics: bool = True,
        cell_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        n = context.shape[0]
        device = context.device

        anchor_logits = self.anchor(context)
        anchor_ids = anchor_logits.argmax(dim=-1)

        local = primitive_matrix.local_window(anchor_ids, radius=1).reshape(n, -1)[:, : self.local_k]
        semantic = primitive_matrix.semantic_topk(anchor_ids, k=self.semantic_k).reshape(n, -1)
        usage = primitive_matrix.usage_topk(anchor_ids, k=self.usage_k).reshape(n, -1)
        random = torch.randint(0, primitive_matrix.num_primitives, (n, self.random_k), device=device)

        if ensure_all_candidates:
            candidate_ids = torch.arange(
                primitive_matrix.num_primitives, device=device
            ).view(1, -1).expand(n, -1)
            source_ids = torch.full_like(candidate_ids, 4)
            cand_emb = primitive_matrix.emb[candidate_ids] + self.source_type(source_ids)
        else:
            cand_list = [local, semantic, usage, random]
            src_list = [
                torch.zeros_like(local),
                torch.ones_like(semantic),
                torch.full_like(usage, 2),
                torch.full_like(random, 3),
            ]
            emb_list = [
                primitive_matrix.emb[local] + self.source_type(torch.zeros_like(local)),
                primitive_matrix.emb[semantic] + self.source_type(torch.ones_like(semantic)),
                primitive_matrix.emb[usage] + self.source_type(torch.full_like(usage, 2)),
                primitive_matrix.emb[random] + self.source_type(torch.full_like(random, 3)),
            ]
            
            if self.feedback_source_type is not None:
                feedback = primitive_matrix.feedback_topk(anchor_ids, k=3, layer_idx=self.layer_idx, cell_ids=cell_ids)
                cand_list.append(feedback)
                src_list.append(torch.full_like(feedback, 4))
                emb_list.append(primitive_matrix.emb[feedback] + self.feedback_source_type(torch.zeros_like(feedback)))
                
            if self.category_source_type is not None:
                category = primitive_matrix.category_best(anchor_ids, layer_idx=self.layer_idx, cell_ids=cell_ids)
                cand_list.append(category)
                src_list.append(torch.full_like(category, 7))
                emb_list.append(primitive_matrix.emb[category] + self.category_source_type(torch.zeros_like(category)))
                
            candidate_ids = torch.cat(cand_list, dim=-1)
            source_ids = torch.cat(src_list, dim=-1)
            cand_emb = torch.cat(emb_list, dim=1)

        ctx = self.context_proj(context).unsqueeze(1).expand_as(cand_emb)
        mem = self.memory_proj(memory).unsqueeze(1).expand_as(cand_emb)
        if prev_action_emb is None:
            prev = torch.zeros_like(cand_emb)
        else:
            prev = self.before_proj(prev_action_emb).unsqueeze(1).expand_as(cand_emb)

        feat = torch.cat([ctx, mem, self.candidate_proj(cand_emb), prev], dim=-1)
        proposal_logits = self.score(feat).squeeze(-1)

        metrics: Dict[str, float] = {}
        if collect_metrics:
            with torch.no_grad():
                semantic_outside_grid = ~(
                    semantic.unsqueeze(-1) == local.unsqueeze(1)
                ).any(dim=-1)
                semantic_grid_mismatch = semantic_outside_grid.float().mean()
                emb = torch.nn.functional.normalize(primitive_matrix.emb, dim=-1)
                similarity = emb[anchor_ids] @ emb.t()
                semantic_prob = torch.softmax(similarity, dim=-1)
                semantic_entropy = (-(semantic_prob + 1e-8) * (semantic_prob + 1e-8).log()).sum(dim=-1)
                metrics = {
                    "semantic_grid_mismatch": float(semantic_grid_mismatch.cpu()),
                    "semantic_neighbor_entropy": float(semantic_entropy.mean().cpu()),
                    "scanner_full_scan": float(ensure_all_candidates),
                    "single_signed_projection_usage": 0.0,
                    "pair_jl16_usage": 0.0,
                    "flat_shortcut_candidate_usage": 0.0,
                    "compositional_pair_candidate_usage": 0.0,
                    "measured_delta_loss_is_source_of_truth": True,
                    "expected_actions_used_for_training": False,
                }

        return candidate_ids, proposal_logits, source_ids, metrics
