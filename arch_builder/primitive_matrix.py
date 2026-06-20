from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


PRIMITIVE_GRID: List[List[str]] = [
    ["identity", "gated_keep", "diff", "contrast", "smooth"],
    ["low_rank", "channel", "ctx_matrix", "product", "gated_add"],
    ["merge", "split", "route", "edge_gate", "write_gate"],
    ["memory_read", "memory_write", "forget", "recall", "memory_gate"],
    ["output_write", "output_mix", "skip", "replace", "disable"],
]


@dataclass(frozen=True)
class PrimitiveInfo:
    name: str
    row: int
    col: int
    family: str
    arity: str
    effect: str
    cost: str
    rank_capable: bool
    sign_capable: bool


def _info(name: str, row: int, col: int) -> PrimitiveInfo:
    if row == 0:
        family = "local"
    elif row == 1:
        family = "learned"
    elif row == 2:
        family = "routing"
    elif row == 3:
        family = "memory"
    else:
        family = "control"

    arity = "memory" if family == "memory" else ("binary" if name in {"diff", "merge", "product", "ctx_matrix", "gated_add"} else "unary")
    effect_map = {
        "identity": "preserve", "gated_keep": "preserve", "diff": "transform", "contrast": "transform", "smooth": "transform",
        "low_rank": "transform", "channel": "transform", "ctx_matrix": "transform", "product": "transform", "gated_add": "merge",
        "merge": "merge", "split": "split", "route": "write", "edge_gate": "write", "write_gate": "write",
        "memory_read": "read", "memory_write": "write", "forget": "disable", "recall": "read", "memory_gate": "write",
        "output_write": "write", "output_mix": "merge", "skip": "skip", "replace": "replace", "disable": "disable",
    }
    effect = effect_map.get(name, "transform")
    cost = "cheap" if row in {0, 2, 4} else ("medium" if row == 3 else "expensive")
    rank_capable = name in {"low_rank", "channel", "ctx_matrix", "product", "gated_add"}
    sign_capable = name not in {"disable", "replace"}
    return PrimitiveInfo(name, row, col, family, arity, effect, cost, rank_capable, sign_capable)


class PrimitiveMatrix5x5(nn.Module):
    """Topological primitive/action library with functional descriptor init."""

    def __init__(self, embed_dim: int = 32, noise_std: float = 0.02) -> None:
        super().__init__()
        self.grid = PRIMITIVE_GRID
        infos: List[PrimitiveInfo] = []
        for r, row in enumerate(self.grid):
            for c, name in enumerate(row):
                infos.append(_info(name, r, c))
        self.infos = infos
        self.names = [x.name for x in infos]
        self.name_to_id = {n: i for i, n in enumerate(self.names)}

        desc = self._descriptor_matrix(infos)
        proj = torch.randn(desc.shape[1], embed_dim) / max(1.0, desc.shape[1] ** 0.5)
        init = F.normalize(desc @ proj, dim=-1) + noise_std * torch.randn(len(infos), embed_dim)
        self.emb = nn.Parameter(init)

        self.register_buffer("descriptor", desc, persistent=False)
        self.register_buffer("usage_score", torch.zeros(len(self.names)), persistent=False)
        self.register_buffer("usage_observations", torch.zeros(len(self.names)), persistent=False)
        local_lookup = []
        for idx in range(len(self.names)):
            r, c = self.infos[idx].row, self.infos[idx].col
            vals = [
                rr * 5 + cc
                for rr in range(max(0, r - 1), min(5, r + 2))
                for cc in range(max(0, c - 1), min(5, c + 2))
            ]
            vals.extend([idx] * (9 - len(vals)))
            local_lookup.append(vals[:9])
        self.register_buffer(
            "local_radius1_lookup",
            torch.tensor(local_lookup, dtype=torch.long),
            persistent=False,
        )

    @property
    def num_primitives(self) -> int:
        return len(self.names)

    def _descriptor_matrix(self, infos: List[PrimitiveInfo]) -> torch.Tensor:
        families = ["local", "learned", "routing", "memory", "control"]
        arities = ["unary", "binary", "memory"]
        effects = ["preserve", "transform", "merge", "split", "write", "read", "skip", "replace", "disable"]
        costs = ["cheap", "medium", "expensive"]
        rows = []
        for x in infos:
            v: List[float] = []
            v += [1.0 if x.family == y else 0.0 for y in families]
            v += [1.0 if x.arity == y else 0.0 for y in arities]
            v += [1.0 if x.effect == y else 0.0 for y in effects]
            v += [1.0 if x.cost == y else 0.0 for y in costs]
            v += [float(x.rank_capable), float(x.sign_capable), x.row / 4.0, x.col / 4.0]
            rows.append(v)
        return torch.tensor(rows, dtype=torch.float32)

    def local_window(self, ids: torch.Tensor, radius: int = 1) -> torch.Tensor:
        if radius != 1:
            raise ValueError("only the precomputed radius=1 topology is supported")
        lookup = self.local_radius1_lookup.to(ids.device)
        return lookup[ids]

    def semantic_topk(self, ids: torch.Tensor, k: int = 4) -> torch.Tensor:
        emb = F.normalize(self.emb, dim=-1)
        sim = emb @ emb.t()
        top = sim.topk(k=min(k, self.num_primitives), dim=-1).indices
        return top[ids.reshape(-1)].view(*ids.shape, -1)

    def usage_topk(self, ids: torch.Tensor, k: int = 5) -> torch.Tensor:
        k = min(k, self.num_primitives)
        if self.usage_observations.sum() <= 0:
            # Cold start has no semantic/task prior: sample uniformly without replacement.
            top = torch.randperm(self.num_primitives, device=ids.device)[:k]
        else:
            seen = self.usage_observations > 0
            ranking = self.usage_score.masked_fill(~seen, float("-inf"))
            top = ranking.topk(k=min(k, int(seen.sum().item()))).indices.to(ids.device)
            if top.numel() < k:
                unseen = (~seen).nonzero(as_tuple=False).flatten().to(ids.device)
                unseen = unseen[torch.randperm(unseen.numel(), device=ids.device)]
                top = torch.cat([top, unseen[: k - top.numel()]])
        return top.view(*([1] * ids.dim()), -1).expand(*ids.shape, -1)

    def update_usage_credit(
        self,
        chosen_ids: torch.Tensor,
        credit: torch.Tensor,
        momentum: float = 0.95,
    ) -> None:
        """Update delayed usage ranking from observed task reward.

        `credit` is aligned with chosen ids and is supplied only after the caller
        has measured an outcome. Candidate presence or selection alone is never
        treated as useful credit.
        """
        with torch.no_grad():
            ids = chosen_ids.detach().flatten().to(self.usage_score.device)
            values = credit.detach().flatten().to(self.usage_score.device, dtype=self.usage_score.dtype)
            if values.numel() == 1 and ids.numel() != 1:
                values = values.expand_as(ids)
            if ids.numel() != values.numel():
                raise ValueError(f"credit count {values.numel()} does not match chosen ids {ids.numel()}")
            counts = torch.bincount(ids, minlength=self.num_primitives).to(self.usage_score.dtype)
            sums = torch.zeros_like(self.usage_score).scatter_add_(0, ids, values)
            observed = counts > 0
            quality = sums / counts.clamp_min(1.0)
            self.usage_score[observed] = (
                momentum * self.usage_score[observed] + (1.0 - momentum) * quality[observed]
            )
            self.usage_observations.add_(counts)

    def metrics(self) -> Dict[str, float]:
        emb = F.normalize(self.emb.detach(), dim=-1)
        cov_rank = torch.linalg.matrix_rank(emb).item()
        pair = emb @ emb.t()
        off = pair[~torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)]
        usage_prob = F.softmax(self.usage_score.masked_fill(self.usage_observations <= 0, -1e9), dim=0)
        usage_entropy = 0.0 if self.usage_observations.sum() <= 0 else float(
            (-(usage_prob + 1e-8) * (usage_prob + 1e-8).log()).sum().cpu()
        )
        return {
            "primitive_embedding_rank": float(cov_rank),
            "primitive_pair_cos_mean": float(off.mean().cpu()),
            "primitive_pair_cos_max": float(off.max().cpu()),
            "usage_entropy": usage_entropy,
            "usage_credit_observations": float(self.usage_observations.sum().cpu()),
        }
