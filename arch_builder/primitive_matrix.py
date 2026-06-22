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
    ["dct", "fft_filter", "wavelet", "spectral_mix", "spectral_gate"],
    ["qkv_gate", "cross_attend", "self_attend", "key_align", "value_mix"],
    ["svd_atom_k", "diag", "toeplitz", "block_mean", "mined_gate"],
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
    elif row == 4:
        family = "control"
    elif row == 5:
        family = "spectral"
    elif row == 6:
        family = "attention_like"
    else:
        family = "mined_local"

    if family == "memory" or name == "self_attend":
        arity = "memory"
    elif name in {"diff", "merge", "product", "ctx_matrix", "gated_add", "cross_attend", "key_align", "value_mix", "spectral_mix"}:
        arity = "binary"
    else:
        arity = "unary"

    effect_map = {
        "identity": "preserve", "gated_keep": "preserve", "diff": "transform", "contrast": "transform", "smooth": "transform",
        "low_rank": "transform", "channel": "transform", "ctx_matrix": "transform", "product": "transform", "gated_add": "merge",
        "merge": "merge", "split": "split", "route": "write", "edge_gate": "write", "write_gate": "write",
        "memory_read": "read", "memory_write": "write", "forget": "disable", "recall": "read", "memory_gate": "write",
        "output_write": "write", "output_mix": "merge", "skip": "skip", "replace": "replace", "disable": "disable",
        "dct": "transform", "fft_filter": "transform", "wavelet": "transform", "spectral_mix": "merge", "spectral_gate": "transform",
        "qkv_gate": "transform", "cross_attend": "transform", "self_attend": "transform", "key_align": "transform", "value_mix": "merge",
        "svd_atom_k": "transform", "diag": "transform", "toeplitz": "transform", "block_mean": "transform", "mined_gate": "transform",
    }
    effect = effect_map.get(name, "transform")
    cost = "cheap" if row in {0, 2, 4} else ("medium" if row in {3, 5} else "expensive")
    rank_capable = name in {"low_rank", "channel", "ctx_matrix", "product", "gated_add", "svd_atom_k"}
    sign_capable = name not in {"disable", "replace"}
    return PrimitiveInfo(name, row, col, family, arity, effect, cost, rank_capable, sign_capable)


class PrimitiveMatrix5x5(nn.Module):
    """Topological primitive/action library with functional descriptor init."""

    def __init__(
        self,
        embed_dim: int = 32,
        noise_std: float = 0.02,
        enable_vnext: bool = False,
        num_layers: int = 1,
        enable_scanner_feedback_memory: bool = False,
        slots: int = 4,
    ) -> None:
        super().__init__()
        self.grid = PRIMITIVE_GRID if enable_vnext else PRIMITIVE_GRID[:5]
        infos: List[PrimitiveInfo] = []
        for r, row in enumerate(self.grid):
            for c, name in enumerate(row):
                infos.append(_info(name, r, c))
        self.infos = infos
        self.names = [x.name for x in infos]
        self.name_to_id = {n: i for i, n in enumerate(self.names)}
        self.num_layers = num_layers
        self.slots = slots
        self.num_cells = slots * slots
        self.enable_scanner_feedback_memory = enable_scanner_feedback_memory

        desc = self._descriptor_matrix(infos)
        proj = torch.randn(desc.shape[1], embed_dim) / max(1.0, desc.shape[1] ** 0.5)
        init = F.normalize(desc @ proj, dim=-1) + noise_std * torch.randn(len(infos), embed_dim)
        self.emb = nn.Parameter(init)

        self.register_buffer("descriptor", desc, persistent=False)
        self.register_buffer("usage_score", torch.zeros(len(self.names)), persistent=False)
        self.register_buffer("usage_observations", torch.zeros(len(self.names)), persistent=False)
        self.register_buffer("feedback_gain_ema", torch.zeros(num_layers, self.num_cells, len(self.names)), persistent=False)
        self.register_buffer("feedback_regret_ema", torch.zeros(num_layers, self.num_cells, len(self.names)), persistent=False)
        self.register_buffer("feedback_count", torch.zeros(num_layers, self.num_cells, len(self.names)), persistent=False)
        self.register_buffer("feedback_age", torch.zeros(num_layers, self.num_cells, len(self.names)), persistent=False)

        local_lookup = []
        for idx in range(len(self.names)):
            r, c = self.infos[idx].row, self.infos[idx].col
            vals = [
                rr * 5 + cc
                for rr in range(max(0, r - 1), min(len(self.grid), r + 2))
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
        families = ["local", "learned", "routing", "memory", "control", "spectral", "attention_like", "mined_local"]
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
            v += [float(x.rank_capable), float(x.sign_capable), x.row / 7.0, x.col / 4.0]
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

    def feedback_topk(self, ids: torch.Tensor, k: int, layer_idx: int = 0, cell_ids: torch.Tensor | None = None) -> torch.Tensor:
        device = ids.device
        n = ids.shape[0]
        k = min(k, self.num_primitives)
        if cell_ids is None:
            cell_ids = torch.arange(n, device=device) % self.num_cells
            
        gain = self.feedback_gain_ema[layer_idx, cell_ids]
        regret = self.feedback_regret_ema[layer_idx, cell_ids]
        bias = torch.clamp(gain - 0.5 * regret, min=-2.0, max=2.0)
        
        count = self.feedback_count[layer_idx, cell_ids]
        seen = count > 0
        
        ranking = bias.masked_fill(~seen, float("-inf"))
        top = ranking.topk(k=k, dim=-1).indices
        
        no_measurements = (count.sum(dim=-1) <= 0)
        if no_measurements.any():
            rand_ids = torch.stack([torch.randperm(self.num_primitives, device=device)[:k] for _ in range(int(no_measurements.sum().item()))])
            top[no_measurements] = rand_ids
            
        for idx in range(n):
            row_seen_count = int(seen[idx].sum().item())
            if row_seen_count < k:
                unseen = (~seen[idx]).nonzero(as_tuple=False).flatten()
                perm = torch.randperm(unseen.numel(), device=device)
                fill_needed = k - row_seen_count
                top[idx, row_seen_count:] = unseen[perm[:fill_needed]]
                
        return top.view(*ids.shape, -1)

    def category_best(self, ids: torch.Tensor, layer_idx: int = 0, cell_ids: torch.Tensor | None = None) -> torch.Tensor:
        device = ids.device
        n = ids.shape[0]
        if cell_ids is None:
            cell_ids = torch.arange(n, device=device) % self.num_cells
            
        gain = self.feedback_gain_ema[layer_idx, cell_ids]
        regret = self.feedback_regret_ema[layer_idx, cell_ids]
        bias = torch.clamp(gain - 0.5 * regret, min=-2.0, max=2.0)
        
        no_measurements = (self.feedback_count[layer_idx, cell_ids].sum(dim=-1) <= 0)
        if no_measurements.any():
            bias[no_measurements] = self.usage_score.unsqueeze(0).expand(int(no_measurements.sum().item()), -1).to(bias.dtype)
        
        best_ids = []
        num_rows = len(self.grid)
        for r in range(num_rows):
            row_slice = bias[:, r * 5 : r * 5 + 5]
            best_idx_in_row = row_slice.argmax(dim=-1)
            best_ids.append(r * 5 + best_idx_in_row)
            
        best_tensor = torch.stack(best_ids, dim=-1)
        return best_tensor.view(*ids.shape, -1)

    def category_topk(self, ids: torch.Tensor, k: int, layer_idx: int = 0, cell_ids: torch.Tensor | None = None) -> torch.Tensor:
        device = ids.device
        n = ids.shape[0]
        if cell_ids is None:
            cell_ids = torch.arange(n, device=device) % self.num_cells
            
        gain = self.feedback_gain_ema[layer_idx, cell_ids]
        regret = self.feedback_regret_ema[layer_idx, cell_ids]
        bias = torch.clamp(gain - 0.5 * regret, min=-2.0, max=2.0)
        
        no_measurements = (self.feedback_count[layer_idx, cell_ids].sum(dim=-1) <= 0)
        if no_measurements.any():
            bias[no_measurements] = self.usage_score.unsqueeze(0).expand(int(no_measurements.sum().item()), -1).to(bias.dtype)
        
        k_val = min(k, 5)
        top_ids = []
        num_rows = len(self.grid)
        for r in range(num_rows):
            row_slice = bias[:, r * 5 : r * 5 + 5]
            best_indices = row_slice.topk(k=k_val, dim=-1).indices
            top_ids.append(r * 5 + best_indices)
            
        best_tensor = torch.cat(top_ids, dim=-1)
        return best_tensor.view(*ids.shape, -1)

    def update_usage_credit(
        self,
        chosen_ids: torch.Tensor,
        credit: torch.Tensor,
        layer_ids: torch.Tensor | None = None,
        cell_ids: torch.Tensor | None = None,
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

            # Scanner feedback memory updates (gain/regret EMAs) - only if explicitly enabled
            if self.enable_scanner_feedback_memory:
                self.feedback_age.add_(1.0)
                if layer_ids is None:
                    layers_t = torch.zeros_like(ids)
                else:
                    layers_t = layer_ids.detach().flatten().to(self.usage_score.device, dtype=torch.long)
                
                if cell_ids is None:
                    cells_t = torch.zeros_like(ids)
                else:
                    cells_t = cell_ids.detach().flatten().to(self.usage_score.device, dtype=torch.long)
                
                for l, c, p_id, val in zip(layers_t.tolist(), cells_t.tolist(), ids.tolist(), values.tolist()):
                    if l < 0 or l >= self.num_layers or c < 0 or c >= self.num_cells or p_id < 0 or p_id >= self.num_primitives:
                        continue
                    self.feedback_count[l, c, p_id] += 1
                    self.feedback_age[l, c, p_id] = 0.0
                    if val > 0:
                        self.feedback_gain_ema[l, c, p_id] = momentum * self.feedback_gain_ema[l, c, p_id] + (1.0 - momentum) * val
                    elif val < 0:
                        self.feedback_regret_ema[l, c, p_id] = momentum * self.feedback_regret_ema[l, c, p_id] - (1.0 - momentum) * val


    def metrics(self) -> Dict[str, float]:
        emb = F.normalize(self.emb.detach(), dim=-1)
        cov_rank = torch.linalg.matrix_rank(emb).item()
        pair = emb @ emb.t()
        off = pair[~torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)]
        usage_prob = F.softmax(self.usage_score.masked_fill(self.usage_observations <= 0, -1e9), dim=0)
        usage_entropy = 0.0 if self.usage_observations.sum() <= 0 else float(
            (-(usage_prob + 1e-8) * (usage_prob + 1e-8).log()).sum().cpu()
        )
        
        m = {
            "primitive_embedding_rank": float(cov_rank),
            "primitive_pair_cos_mean": float(off.mean().cpu()),
            "primitive_pair_cos_max": float(off.max().cpu()),
            "usage_entropy": usage_entropy,
            "usage_credit_observations": float(self.usage_observations.sum().cpu()),
        }
        
        if self.enable_scanner_feedback_memory:
            seen = self.feedback_count > 0
            if seen.any():
                gain = self.feedback_gain_ema
                regret = self.feedback_regret_ema
                bias = torch.clamp(gain - 0.5 * regret, min=-2.0, max=2.0)
                feedback_bias_abs = float(bias[seen].abs().mean().cpu())
                feedback_staleness = float(self.feedback_age[seen].mean().cpu())
                
                total_counts = self.feedback_count.sum(dim=(0, 1))
                if total_counts.sum() > 0:
                    prob = total_counts / total_counts.sum()
                    feedback_entropy = float((-(prob + 1e-8) * (prob + 1e-8).log()).sum().cpu())
                    feedback_top_share = float(prob.max().cpu())
                else:
                    feedback_entropy = 0.0
                    feedback_top_share = 0.0
            else:
                feedback_bias_abs = 0.0
                feedback_staleness = 0.0
                feedback_entropy = 0.0
                feedback_top_share = 0.0
                
            m.update({
                "feedback_bias_abs": feedback_bias_abs,
                "feedback_staleness": feedback_staleness,
                "feedback_count": float(self.feedback_count.sum().cpu()),
                "feedback_entropy": feedback_entropy,
                "feedback_top_share": feedback_top_share,
            })
            
        return m
