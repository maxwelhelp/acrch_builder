"""Relation Consistency — triangle co-occurrence memory for primitive compatibility.

Tracks which primitives work well together (co-occurrence) and which conflict,
then computes a triangle relation score for candidate ranking:

    T_support[cell_i, prim_j] = Σ_k gain_ema[l, i, k] × co_pos[k, j] × (1 - conflict_norm[k, j])

This captures the path: cell_i → primitive_k (already useful) → primitive_j (compatible with k).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class RelationMemory(nn.Module):
    """Tracks primitive co-occurrence statistics from measured credit.

    Maintains four [num_primitives, num_primitives] matrices:
    - co_occurrence_pos: primitives that appeared together with positive gain
    - co_occurrence_neg: primitives that appeared together with negative gain
    - conflict: primitives that conflicted (appeared together with negative gain)
    - pair_count: total co-occurrence observation count

    Memory cost: 4 × P² × 4 bytes. For P=40: 25 KB total.
    """

    def __init__(self, num_primitives: int) -> None:
        super().__init__()
        self.num_primitives = num_primitives
        self.register_buffer(
            "co_occurrence_pos",
            torch.zeros(num_primitives, num_primitives),
            persistent=False,
        )
        self.register_buffer(
            "co_occurrence_neg",
            torch.zeros(num_primitives, num_primitives),
            persistent=False,
        )
        self.register_buffer(
            "conflict",
            torch.zeros(num_primitives, num_primitives),
            persistent=False,
        )
        self.register_buffer(
            "pair_count",
            torch.zeros(num_primitives, num_primitives),
            persistent=False,
        )
        # Diagnostic counters
        self.register_buffer("relation_update_called", torch.zeros(()), persistent=False)
        self.register_buffer("relation_update_pairs", torch.zeros(()), persistent=False)
        self.register_buffer("relation_update_positive", torch.zeros(()), persistent=False)
        self.register_buffer("relation_update_negative", torch.zeros(()), persistent=False)

    @torch.no_grad()
    def update(
        self,
        chosen_ids: torch.Tensor,
        measured_gain: float,
        momentum: float = 0.98,
    ) -> None:
        """Update co-occurrence memory from a set of chosen primitives and their measured gain.

        Uses vectorized outer-product: mask[k] × mask[j] gives the pair indicator,
        then updates the appropriate matrix based on the sign of measured_gain.
        Diagonal is always kept at zero.

        Args:
            chosen_ids: primitive indices that were active together in the same
                forward pass or credit measurement group.
            measured_gain: the measured counterfactual gain for this group.
            momentum: EMA momentum for the co-occurrence matrices.
        """
        ids = chosen_ids.detach().long().unique()
        ids = ids[(ids >= 0) & (ids < self.num_primitives)]
        if ids.numel() <= 1:
            return

        self.relation_update_called.add_(1.0)

        device = self.co_occurrence_pos.device
        mask = torch.zeros(
            self.num_primitives,
            device=device,
            dtype=self.co_occurrence_pos.dtype,
        )
        mask[ids] = 1.0

        pair = mask[:, None] * mask[None, :]
        pair.fill_diagonal_(0.0)

        n_pairs = float(pair.sum().item())
        self.relation_update_pairs.add_(n_pairs)

        gain = float(measured_gain)
        abs_gain = abs(gain)

        self.pair_count.add_(pair)

        if gain > 0:
            self.relation_update_positive.add_(1.0)
            self.co_occurrence_pos.mul_(momentum).add_(
                pair * abs_gain * (1.0 - momentum)
            )
        elif gain < 0:
            self.relation_update_negative.add_(1.0)
            self.co_occurrence_neg.mul_(momentum).add_(
                pair * abs_gain * (1.0 - momentum)
            )
            self.conflict.mul_(momentum).add_(
                pair * abs_gain * (1.0 - momentum)
            )

    def metrics(self) -> Dict[str, float]:
        """Return diagnostic metrics for logging."""
        observed = self.pair_count > 0
        n_observed = float(observed.sum().item())
        return {
            "relation_update_called": float(self.relation_update_called.item()),
            "relation_update_pairs": float(self.relation_update_pairs.item()),
            "relation_update_positive": float(self.relation_update_positive.item()),
            "relation_update_negative": float(self.relation_update_negative.item()),
            "relation_pair_count_total": float(self.pair_count.sum().item()),
            "relation_observed_pairs": n_observed,
            "relation_co_pos_mean": float(self.co_occurrence_pos[observed].mean().item()) if n_observed > 0 else 0.0,
            "relation_co_neg_mean": float(self.co_occurrence_neg[observed].mean().item()) if n_observed > 0 else 0.0,
            "relation_conflict_mean": float(self.conflict[observed].mean().item()) if n_observed > 0 else 0.0,
        }


class TriangleRelationScorer:
    """Computes triangle relation scores from RelationMemory + PrimitiveMatrix feedback.

    Pure diagnostic scorer — no learnable parameters, all @torch.no_grad().
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    @torch.no_grad()
    def compute(
        self,
        *,
        candidate_ids: torch.Tensor,
        layer_idx: int,
        cell_ids: torch.Tensor,
        pm: "PrimitiveMatrix5x5",  # noqa: F821
        program_entropy_norm: Optional[float] = None,
    ) -> Dict[str, object]:
        """Compute triangle support/contradiction/novelty for candidates.

        Args:
            candidate_ids: [N, pool] primitive indices for each cell.
            layer_idx: which layer's feedback to use.
            cell_ids: [N] cell indices.
            pm: PrimitiveMatrix5x5 with feedback buffers and relation_memory.
            program_entropy_norm: normalized program entropy (0-1), controls
                novelty vs stability gating.

        Returns:
            Dict with 'support', 'contradiction', 'novelty', 'score', 'metrics'.
        """
        if pm.relation_memory is None:
            empty = torch.zeros_like(candidate_ids, dtype=torch.float32)
            return {
                "support": empty,
                "contradiction": empty,
                "novelty": empty,
                "score": empty,
                "metrics": {},
            }

        pid = candidate_ids.clamp(0, pm.num_primitives - 1)
        N, pool = pid.shape

        gain = pm.feedback_gain_ema[layer_idx]       # [cells, prim]
        regret = pm.feedback_regret_ema[layer_idx]    # [cells, prim]
        count = pm.feedback_count[layer_idx]          # [cells, prim]

        local_gain = torch.relu(gain[cell_ids])       # [N, prim]
        local_regret = torch.relu(regret[cell_ids])   # [N, prim]

        rm = pm.relation_memory
        co_pos = rm.co_occurrence_pos                  # [prim, prim]
        co_neg = rm.co_occurrence_neg                  # [prim, prim]
        conflict = rm.conflict                         # [prim, prim]

        # Normalized conflict
        conflict_norm = conflict / (conflict + co_pos + self.eps)

        # Triangle support: Σ_k gain[cell,k] × co_pos[k,j] × (1 - conflict_norm[k,j])
        support_all = local_gain @ (co_pos * (1.0 - conflict_norm))  # [N, prim]

        # Triangle contradiction: Σ_k gain[cell,k] × conflict[k,j] + regret[cell,k] × co_neg[k,j]
        contra_all = (local_gain @ conflict) + (local_regret @ co_neg)  # [N, prim]

        # Gather for candidate pool
        support = support_all.gather(1, pid)          # [N, pool]
        contradiction = contra_all.gather(1, pid)     # [N, pool]

        # Novelty
        local_count = count[cell_ids].gather(1, pid)  # [N, pool]
        pair_seen = rm.pair_count.sum(dim=0).gather(
            0, pid.reshape(-1)
        ).reshape(N, pool)
        novelty = 1.0 / torch.sqrt(1.0 + local_count + 0.01 * pair_seen)

        # Entropy gating
        if program_entropy_norm is None:
            novelty_gate = torch.ones_like(novelty) * 0.1
        else:
            pe = torch.tensor(
                program_entropy_norm,
                device=pid.device,
                dtype=support.dtype,
            )
            novelty_gate = torch.sigmoid((0.45 - pe) * 8.0)

        # Triangle score
        triangle_score = support - contradiction + 0.25 * novelty * novelty_gate

        metrics = {
            "relation_tri_support_mean": float(support.mean().cpu()),
            "relation_tri_contradiction_mean": float(contradiction.mean().cpu()),
            "relation_tri_novelty_mean": float(novelty.mean().cpu()),
            "relation_tri_score_mean": float(triangle_score.mean().cpu()),
            "relation_program_entropy_norm": float(program_entropy_norm or 0.0),
        }

        return {
            "support": support,
            "contradiction": contradiction,
            "novelty": novelty,
            "score": triangle_score,
            "metrics": metrics,
        }
