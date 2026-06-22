from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitive_matrix import PrimitiveMatrix5x5


class ActionExecutor(nn.Module):
    """Full primitive execution for selected candidates."""

    def __init__(self, dim: int, primitive_matrix: PrimitiveMatrix5x5, enable_vnext: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.enable_vnext = enable_vnext
        self.names = primitive_matrix.names
        self.name_to_id = primitive_matrix.name_to_id
        p = primitive_matrix.num_primitives
        r = max(4, dim // 4)
        
        # Original parameters
        self.low_a = nn.Parameter(torch.randn(p, dim, r) * 0.02)
        self.low_b = nn.Parameter(torch.randn(p, r, dim) * 0.02)
        self.channel = nn.Parameter(torch.eye(dim).unsqueeze(0).repeat(p, 1, 1) + 0.01 * torch.randn(p, dim, dim))
        self.ctx = nn.Sequential(nn.Linear(dim * 3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.gate = nn.Linear(dim * 2, dim)

        if enable_vnext:
            # Spectral: DCT Matrix
            import numpy as np
            dct_mat = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    dct_mat[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2.0 * dim))
            dct_mat[0, :] *= np.sqrt(1.0 / dim)
            dct_mat[1:, :] *= np.sqrt(2.0 / dim)
            self.register_buffer("dct_matrix", torch.tensor(dct_mat, dtype=torch.float32), persistent=False)

            # Spectral: FFT filter weight
            self.fft_filter_weight = nn.Parameter(torch.ones(p, dim // 2 + 1, dtype=torch.float32))

            # Attention-like projections
            self.q_proj = nn.Parameter(torch.randn(p, dim, dim) * 0.02)
            self.k_proj = nn.Parameter(torch.randn(p, dim, dim) * 0.02)
            self.v_proj = nn.Parameter(torch.randn(p, dim, dim) * 0.02)

            # Mined Local parameters
            self.mined_u = nn.Parameter(torch.randn(dim, 4) * 0.02)
            self.mined_v = nn.Parameter(torch.randn(4, dim) * 0.02)
            self.mined_diag = nn.Parameter(torch.ones(dim))
            self.mined_toeplitz_filter = nn.Parameter(torch.randn(1, 1, 5) * 0.02)
            self.mined_source_matrix = nn.Parameter(torch.randn(dim, dim) * 0.02)

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
            
        # Spectral Primitives
        if name == "dct":
            return src @ self.dct_matrix.t()
        if name == "fft_filter":
            src_fft = torch.fft.rfft(src.float(), dim=-1)
            filtered = src_fft * self.fft_filter_weight[pid]
            return torch.fft.irfft(filtered, n=self.dim, dim=-1).to(src.dtype)
        if name == "wavelet":
            even = src[..., 0::2]
            odd = src[..., 1::2]
            approx = (even + odd) / 1.41421356
            detail = (even - odd) / 1.41421356
            return torch.cat([approx, detail], dim=-1)
        if name == "spectral_mix":
            src_fft = torch.fft.rfft(src.float(), dim=-1)
            tgt_fft = torch.fft.rfft(tgt.float(), dim=-1)
            return torch.fft.irfft(src_fft * torch.sigmoid(tgt_fft.real), n=self.dim, dim=-1).to(src.dtype)
        if name == "spectral_gate":
            src_fft = torch.fft.rfft(src.float(), dim=-1)
            energy = src_fft.abs()
            gate = torch.sigmoid(energy - energy.mean(dim=-1, keepdim=True))
            return torch.fft.irfft(src_fft * gate, n=self.dim, dim=-1).to(src.dtype)

        # Attention-like Primitives
        if name in {"qkv_gate", "cross_attend", "self_attend", "key_align", "value_mix"}:
            q = src @ self.q_proj[pid]
            k = (tgt if name in {"cross_attend", "key_align", "value_mix"} else (memory if name == "self_attend" else src)) @ self.k_proj[pid]
            v = (tgt if name in {"cross_attend", "value_mix"} else src) @ self.v_proj[pid]
            if name == "qkv_gate":
                return torch.sigmoid((q * k).sum(dim=-1, keepdim=True)) * v
            if name in {"cross_attend", "self_attend"}:
                attn = torch.tanh((q * k).sum(dim=-1, keepdim=True) / (self.dim ** 0.5))
                return attn * v
            if name == "key_align":
                sim = F.cosine_similarity(q, k, dim=-1).unsqueeze(-1)
                return sim * src
            if name == "value_mix":
                weight = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / (self.dim ** 0.5))
                return weight * src + (1.0 - weight) * tgt

        # Mined Local Primitives
        if name == "svd_atom_k":
            return (src @ self.mined_u) @ self.mined_v
        if name == "diag":
            return src * self.mined_diag
        if name == "toeplitz":
            padded = F.pad(src.unsqueeze(1), (2, 2), mode="circular")
            return F.conv1d(padded, self.mined_toeplitz_filter).squeeze(1)
        if name == "block_mean":
            block_size = self.dim // 4
            return src.view(-1, 4, block_size).mean(dim=-1, keepdim=True).repeat(1, 1, block_size).view(-1, self.dim)
        if name == "mined_gate":
            return torch.sigmoid(src @ self.mined_source_matrix) * src

        return src

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, memory: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        n, k = candidate_ids.shape
        flat_ids = candidate_ids.reshape(-1)
        src_k = src.unsqueeze(1).expand(n, k, src.shape[-1]).reshape(n * k, -1)
        tgt_k = tgt.unsqueeze(1).expand(n, k, tgt.shape[-1]).reshape(n * k, -1)
        mem_k = memory.unsqueeze(1).expand(n, k, memory.shape[-1]).reshape(n * k, -1)

        out = src_k.clone()

        def run_prim(name: str, func) -> None:
            pid = self.name_to_id[name]
            mask = (flat_ids == pid)
            if mask.any():
                val_sub = func(src_k[mask], tgt_k[mask], mem_k[mask], pid)
                out[mask] = val_sub.to(dtype=out.dtype)

        # Original Primitives
        def run_gated_keep(s, t, m, pid):
            g = torch.sigmoid(self.gate(torch.cat([s, t], dim=-1)))
            return g * s + (1.0 - g) * t

        run_prim("gated_keep", run_gated_keep)
        run_prim("diff", lambda s, t, m, pid: s - t)
        run_prim("contrast", lambda s, t, m, pid: s - s.mean(dim=-1, keepdim=True))
        run_prim("smooth", lambda s, t, m, pid: 0.5 * s + 0.25 * torch.roll(s, 1, -1) + 0.25 * torch.roll(s, -1, -1))
        run_prim("low_rank", lambda s, t, m, pid: (s @ self.low_a[pid]) @ self.low_b[pid])
        run_prim("channel", lambda s, t, m, pid: s @ self.channel[pid])
        run_prim("ctx_matrix", lambda s, t, m, pid: self.ctx(torch.cat([s, t, s - t], dim=-1)))
        run_prim("product", lambda s, t, m, pid: s * t)
        
        average = lambda s, t, m, pid: 0.5 * (s + t)
        for name in ("gated_add", "merge", "output_mix"):
            run_prim(name, average)
            
        # Combine simple memory, zero, and replace primitive evaluations to avoid multiple run_prim calls
        mem_mask = (flat_ids == self.name_to_id["memory_read"]) | (flat_ids == self.name_to_id["recall"])
        if mem_mask.any():
            out[mem_mask] = mem_k[mem_mask]
            
        run_prim("memory_write", lambda s, t, m, pid: 0.5 * (s + m))
        
        zero_mask = (flat_ids == self.name_to_id["forget"]) | (flat_ids == self.name_to_id["disable"])
        if zero_mask.any():
            out[zero_mask] = 0.0
            
        def run_memory_gate(s, t, m, pid):
            memory_gate = torch.sigmoid((s * m).mean(dim=-1, keepdim=True))
            return memory_gate * m + (1.0 - memory_gate) * s
        run_prim("memory_gate", run_memory_gate)
        
        rep_mask = (flat_ids == self.name_to_id["replace"])
        if rep_mask.any():
            out[rep_mask] = tgt_k[rep_mask]

        if self.enable_vnext:
            # Spectral Primitives
            run_prim("dct", lambda s, t, m, pid: s @ self.dct_matrix.t())
            
            def run_fft_filter(s, t, m, pid):
                src_fft = torch.fft.rfft(s.float(), dim=-1)
                filtered = src_fft * self.fft_filter_weight[pid]
                return torch.fft.irfft(filtered, n=self.dim, dim=-1).to(s.dtype)
            run_prim("fft_filter", run_fft_filter)
            
            def run_wavelet(s, t, m, pid):
                even = s[..., 0::2]
                odd = s[..., 1::2]
                approx = (even + odd) / 1.41421356
                detail = (even - odd) / 1.41421356
                return torch.cat([approx, detail], dim=-1)
            run_prim("wavelet", run_wavelet)
            
            def run_spectral_mix(s, t, m, pid):
                src_fft = torch.fft.rfft(s.float(), dim=-1)
                tgt_fft = torch.fft.rfft(t.float(), dim=-1)
                return torch.fft.irfft(src_fft * torch.sigmoid(tgt_fft.real), n=self.dim, dim=-1).to(s.dtype)
            run_prim("spectral_mix", run_spectral_mix)
            
            def run_spectral_gate(s, t, m, pid):
                src_fft = torch.fft.rfft(s.float(), dim=-1)
                energy = src_fft.abs()
                sg_gate = torch.sigmoid(energy - energy.mean(dim=-1, keepdim=True))
                return torch.fft.irfft(src_fft * sg_gate, n=self.dim, dim=-1).to(s.dtype)
            run_prim("spectral_gate", run_spectral_gate)

            # Attention-like Primitives
            def run_qkv_gate(s, t, m, pid):
                q = s @ self.q_proj[pid]
                k = s @ self.k_proj[pid]
                v = s @ self.v_proj[pid]
                return torch.sigmoid((q * k).sum(dim=-1, keepdim=True)) * v
            run_prim("qkv_gate", run_qkv_gate)
            
            def run_cross_attend(s, t, m, pid):
                q = s @ self.q_proj[pid]
                k = t @ self.k_proj[pid]
                v = t @ self.v_proj[pid]
                attn_cross = torch.tanh((q * k).sum(dim=-1, keepdim=True) / (self.dim ** 0.5))
                return attn_cross * v
            run_prim("cross_attend", run_cross_attend)
            
            def run_self_attend(s, t, m, pid):
                q = s @ self.q_proj[pid]
                k = m @ self.k_proj[pid]
                v = s @ self.v_proj[pid]
                attn_self = torch.tanh((q * k).sum(dim=-1, keepdim=True) / (self.dim ** 0.5))
                return attn_self * v
            run_prim("self_attend", run_self_attend)
            
            def run_key_align(s, t, m, pid):
                q = s @ self.q_proj[pid]
                k = t @ self.k_proj[pid]
                align_sim = F.cosine_similarity(q, k, dim=-1).unsqueeze(-1)
                return align_sim * s
            run_prim("key_align", run_key_align)
            
            def run_value_mix(s, t, m, pid):
                q = s @ self.q_proj[pid]
                k = t @ self.k_proj[pid]
                weight_vm = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / (self.dim ** 0.5))
                return weight_vm * s + (1.0 - weight_vm) * t
            run_prim("value_mix", run_value_mix)

            # Mined Local Primitives
            run_prim("svd_atom_k", lambda s, t, m, pid: (s @ self.mined_u) @ self.mined_v)
            run_prim("diag", lambda s, t, m, pid: s * self.mined_diag)
            
            def run_toeplitz(s, t, m, pid):
                padded_k = F.pad(s.unsqueeze(1), (2, 2), mode="circular")
                return F.conv1d(padded_k, self.mined_toeplitz_filter).squeeze(1)
            run_prim("toeplitz", run_toeplitz)
            
            def run_block_mean(s, t, m, pid):
                block_size = self.dim // 4
                return s.view(-1, 4, block_size).mean(dim=-1, keepdim=True).repeat(1, 1, block_size).view(-1, self.dim)
            run_prim("block_mean", run_block_mean)
            
            run_prim("mined_gate", lambda s, t, m, pid: torch.sigmoid(s @ self.mined_source_matrix) * s)

        return out.view(n, k, -1)
