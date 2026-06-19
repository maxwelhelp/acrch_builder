from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    """Topological primitive/action library with functional descriptor init.

    It exposes physical grid windows, semantic top-k by learned embeddings, and
    primitive names/indices for reports. The grid is only a prior; HybridScanner
    will add semantic, usage and random candidates.
    """

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
        self.rows = torch.tensor([x.row for x in infos], dtype=torch.long)
        self.cols = torch.tensor([x.col for x in infos], dtype=torch.long)

        desc = self._descriptor_matrix(infos)
        proj = torch.randn(desc.shape[1], embed_dim) / max(1.0, desc.shape[1] ** 0.5)
        init = desc @ proj
        init = F.normalize(init, dim=-1)
        init = init + noise_std * torch.randn_like(init)
        self.emb = nn.Parameter(init)

        self.register_buffer("descriptor", desc, persistent=False)
        self.register_buffer("usage_score", torch.zeros(len(self.names)), persistent=False)

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
        """Return grid-neighbor ids for each primitive id. Shape: [N, K]."""
        flat = ids.reshape(-1).detach().cpu()
        out: List[List[int]] = []
        for idx in flat.tolist():
            r, c = self.infos[idx].row, self.infos[idx].col
            vals: List[int] = []
            for rr in range(max(0, r - radius), min(5, r + radius + 1)):
                for cc in range(max(0, c - radius), min(5, c + radius + 1)):
                    vals.append(rr * 5 + cc)
            while len(vals) < (2 * radius + 1) ** 2:
                vals.append(idx)
            out.append(vals[: (2 * radius + 1) ** 2])
        return torch.tensor(out, dtype=torch.long, device=ids.device).view(*ids.shape, -1)

    def semantic_topk(self, ids: torch.Tensor, k: int = 4) -> torch.Tensor:
        """Nearest primitives by embedding cosine, excluding no one for simplicity."""
        emb = F.normalize(self.emb, dim=-1)
        sim = emb @ emb.t()
        top = sim.topk(k=min(k, self.num_primitives), dim=-1).indices
        return top[ids.reshape(-1)].view(*ids.shape, -1)

    def usage_topk(self, ids: torch.Tensor, k: int = 2) -> torch.Tensor:
        top = self.usage_score.topk(k=min(k, self.num_primitives)).indices.to(ids.device)
        return top.view(*([1] * ids.dim()), -1).expand(*ids.shape, -1)

    def update_usage_ema(self, chosen_ids: torch.Tensor, momentum: float = 0.95) -> None:
        with torch.no_grad():
            hist = torch.bincount(chosen_ids.detach().flatten().cpu(), minlength=self.num_primitives).float()
            if hist.sum() > 0:
                hist = hist / hist.sum()
            self.usage_score.mul_(momentum).add_(hist.to(self.usage_score.device), alpha=1 - momentum)

    def metrics(self) -> Dict[str, float]:
        emb = F.normalize(self.emb.detach(), dim=-1)
        cov_rank = torch.linalg.matrix_rank(emb).item()
        pair = emb @ emb.t()
        off = pair[~torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)]
        return {
            "primitive_embedding_rank": float(cov_rank),
            "primitive_pair_cos_mean": float(off.mean().cpu()),
            "primitive_pair_cos_max": float(off.max().cpu()),
            "usage_entropy": float((-(self.usage_score + 1e-8) * (self.usage_score + 1e-8).log()).sum().cpu()),
        }
