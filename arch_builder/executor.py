from __future__ import annotations

import torch
import torch.nn as nn

from .primitive_matrix import PrimitiveMatrix5x5


class ActionExecutor(nn.Module):
    """Full primitive execution for selected candidates."""

    def __init__(self, dim: int, primitive_matrix: PrimitiveMatrix5x5) -> None:
        super().__init__()
        self.dim = dim
        self.names = primitive_matrix.names
        self.name_to_id = primitive_matrix.name_to_id
        p = primitive_matrix.num_primitives
        r = max(4, dim // 4)
        self.low_a = nn.Parameter(torch.randn(p, dim, r) * 0.02)
        self.low_b = nn.Parameter(torch.randn(p, r, dim) * 0.02)
        self.channel = nn.Parameter(torch.eye(dim).unsqueeze(0).repeat(p, 1, 1) + 0.01 * torch.randn(p, dim, dim))
        self.ctx = nn.Sequential(nn.Linear(dim * 3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.gate = nn.Linear(dim * 2, dim)

    def _single(self, name: str, src: torch.Tensor, tgt: torch.Tensor, memory: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        if name in {"identity", "route", "edge_gate", "write_gate", "output_write"}:
            return src
        if name == "gated_keep":
            g = torch.sigmoid(self.gate(torch.cat([src, tgt], dim=-1)))
            return g * src + (1 - g) * tgt
        if name == "diff":
            return src - tgt
        if name == "contrast":
            return src - src.mean(dim=-1, keepdim=True)
        if name == "smooth":
            return 0.5 * src + 0.25 * torch.roll(src, 1, dims=-1) + 0.25 * torch.roll(src, -1, dims=-1)
        if name == "low_rank":
            a = self.low_a[pid]
            b = self.low_b[pid]
            return torch.bmm(torch.bmm(src.unsqueeze(1), a), b).squeeze(1)
        if name == "channel":
            w = self.channel[pid]
            return torch.bmm(src.unsqueeze(1), w).squeeze(1)
        if name == "ctx_matrix":
            return self.ctx(torch.cat([src, tgt, src - tgt], dim=-1))
        if name == "product":
            return src * tgt
        if name in {"gated_add", "merge", "output_mix"}:
            return 0.5 * (src + tgt)
        if name == "split":
            return src
        if name in {"memory_read", "recall"}:
            return memory
        if name == "memory_write":
            return 0.5 * (src + memory)
        if name == "forget":
            return torch.zeros_like(src)
        if name == "memory_gate":
            g = torch.sigmoid((src * memory).mean(dim=-1, keepdim=True))
            return g * memory + (1 - g) * src
        if name == "skip":
            return src
        if name == "replace":
            return tgt
        if name == "disable":
            return torch.zeros_like(src)
        return src

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, memory: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        n, k = candidate_ids.shape
        flat_ids = candidate_ids.reshape(-1)
        src_k = src.unsqueeze(1).expand(n, k, src.shape[-1]).reshape(n * k, -1)
        tgt_k = tgt.unsqueeze(1).expand(n, k, tgt.shape[-1]).reshape(n * k, -1)
        mem_k = memory.unsqueeze(1).expand(n, k, memory.shape[-1]).reshape(n * k, -1)

        # Avoid 25 ``mask.any()`` host synchronizations per layer. Each branch is
        # evaluated as a batched tensor and selected entirely on-device.
        out = src_k

        def put(name: str, value: torch.Tensor) -> None:
            nonlocal out
            mask = (flat_ids == self.name_to_id[name]).unsqueeze(-1)
            out = torch.where(mask, value.to(dtype=out.dtype), out)

        gate = torch.sigmoid(self.gate(torch.cat([src_k, tgt_k], dim=-1)))
        put("gated_keep", gate * src_k + (1.0 - gate) * tgt_k)
        put("diff", src_k - tgt_k)
        put("contrast", src_k - src_k.mean(dim=-1, keepdim=True))
        put("smooth", 0.5 * src_k + 0.25 * torch.roll(src_k, 1, -1) + 0.25 * torch.roll(src_k, -1, -1))

        low_pid = self.name_to_id["low_rank"]
        put("low_rank", (src_k @ self.low_a[low_pid]) @ self.low_b[low_pid])
        channel_pid = self.name_to_id["channel"]
        put("channel", src_k @ self.channel[channel_pid])
        put("ctx_matrix", self.ctx(torch.cat([src_k, tgt_k, src_k - tgt_k], dim=-1)))
        put("product", src_k * tgt_k)
        average = 0.5 * (src_k + tgt_k)
        for name in ("gated_add", "merge", "output_mix"):
            put(name, average)
        for name in ("memory_read", "recall"):
            put(name, mem_k)
        put("memory_write", 0.5 * (src_k + mem_k))
        put("forget", torch.zeros_like(src_k))
        memory_gate = torch.sigmoid((src_k * mem_k).mean(dim=-1, keepdim=True))
        put("memory_gate", memory_gate * mem_k + (1.0 - memory_gate) * src_k)
        put("replace", tgt_k)
        put("disable", torch.zeros_like(src_k))
        return out.view(n, k, -1)
