from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SRCFProgramConfig:
    """Explicit differentiable architecture/program builder.

    This is not a fixed closure layer. It has a bounded program budget, then learns
    which nodes exist, which previous states each node reads, which latent op each
    node uses, how many repeats it applies, and which nodes feed the head.

    Contract:
      frontend builds R [B,N,N,C]
      builder assembles ProgramGraph under task loss + closure pressure
      head reads assembled H*
      export_program_trace() writes the assembled program
    """

    rel_dim: int
    out_dim: int
    state_dim: int = 48
    hidden: int = 96
    max_nodes: int = 6
    op_count: int = 8
    op_emb_dim: int = 16
    max_repeat: int = 3
    dropout: float = 0.0
    noise_std: float = 0.03
    arch_temperature: float = 1.0
    use_triangle: bool = True
    use_rel_skip: bool = True


@dataclass
class SRCFProgramLossWeights:
    fixed: float = 0.04
    recovery: float = 0.04
    contract: float = 0.05
    far_keep: float = 0.02
    state_var: float = 0.02
    move_band: float = 0.02
    node_cost: float = 0.003
    edge_cost: float = 0.001
    entropy_cost: float = 0.000
    far_margin: float = 0.20
    state_var_floor: float = 0.003
    move_min: float = 0.003
    move_max: float = 2.5


def _pairwise_distances(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_zeros((0,))
    dist = torch.cdist(z.float(), z.float(), p=2)
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return dist[mask]


class SRCFProgramBuilder(nn.Module):
    """Self-assembling differentiable program graph over relation states.

    Program elements:
      source 0      = encoded input state H0
      node m        = soft node that reads H0 or previous nodes only
      edge_logits   = differentiable dataflow graph
      alive_logits  = differentiable birth/prune gate
      op_logits     = latent operation choice per node
      repeat_logits = how many times to reapply node transition
      readout       = which states feed the task head

    The ops are intentionally latent, not named recipes. Interpretation happens
    after training through program_trace + ablation impact.
    """

    def __init__(self, cfg: SRCFProgramConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.state_dim
        h = cfg.hidden
        m = cfg.max_nodes
        k = cfg.op_count
        e = cfg.op_emb_dim

        self.enc = nn.Sequential(nn.Linear(cfg.rel_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.op_emb = nn.Parameter(torch.randn(k, e) * 0.02)

        # Architecture parameters. These are the actual assembled program graph.
        self.node_alive_logits = nn.Parameter(torch.full((m,), -0.35))
        self.op_logits = nn.Parameter(torch.zeros(m, k))
        self.edge_logits = nn.Parameter(torch.zeros(m, m + 1))
        self.repeat_logits = nn.Parameter(torch.zeros(m, cfg.max_repeat))
        self.readout_logits = nn.Parameter(torch.zeros(m + 1))

        pieces = 4 + int(cfg.use_triangle)
        ctx_dim = d * pieces
        if cfg.use_rel_skip:
            ctx_dim += cfg.rel_dim
        self.ctx_norm = nn.LayerNorm(ctx_dim)
        self.op_net = nn.Sequential(
            nn.LayerNorm(ctx_dim + e),
            nn.Linear(ctx_dim + e, h),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, d),
        )
        self.gate_net = nn.Sequential(nn.Linear(ctx_dim, h), nn.SiLU(), nn.Linear(h, d), nn.Sigmoid())
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.SiLU(), nn.Linear(h, cfg.out_dim))

    @staticmethod
    def triangle_update(h: torch.Tensor) -> torch.Tensor:
        b, n, _, d = h.shape
        hc = h.permute(0, 3, 1, 2).contiguous().reshape(b * d, n, n)
        tri = torch.bmm(hc, hc) / max(float(n), 1.0)
        return tri.reshape(b, d, n, n).permute(0, 2, 3, 1).contiguous()

    def _source_mask(self, node_idx: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((self.cfg.max_nodes + 1,), -1e9, device=device)
        # Source 0 is H0. Sources 1..node_idx are previous nodes.
        mask[: node_idx + 1] = 0.0
        return mask

    def architecture_probs(self) -> Dict[str, torch.Tensor]:
        tau = max(float(self.cfg.arch_temperature), 1e-4)
        alive = torch.sigmoid(self.node_alive_logits)
        op = torch.softmax(self.op_logits / tau, dim=-1)
        repeat = torch.softmax(self.repeat_logits / tau, dim=-1)
        readout = torch.softmax(self.readout_logits / tau, dim=-1)
        edge_rows = []
        for i in range(self.cfg.max_nodes):
            masked = self.edge_logits[i] + self._source_mask(i, self.edge_logits.device)
            edge_rows.append(torch.softmax(masked / tau, dim=-1))
        edge = torch.stack(edge_rows, dim=0)
        return {"alive": alive, "op": op, "repeat": repeat, "readout": readout, "edge": edge}

    def build_context(self, h: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        row = h.mean(dim=2, keepdim=True).expand_as(h)
        col = h.mean(dim=1, keepdim=True).expand_as(h)
        glob = h.mean(dim=(1, 2), keepdim=True).expand_as(h)
        pieces = [h, row, col, glob]
        if self.cfg.use_triangle:
            pieces.append(self.triangle_update(h))
        if self.cfg.use_rel_skip:
            pieces.append(rel)
        return self.ctx_norm(torch.cat(pieces, dim=-1))

    def apply_op_once(self, h: torch.Tensor, rel: torch.Tensor, node_idx: int, op_prob: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        ctx = self.build_context(h, rel)
        b, n, _, ctx_dim = ctx.shape
        k = self.cfg.op_count
        ctx_a = ctx.unsqueeze(3).expand(b, n, n, k, ctx_dim)
        emb = self.op_emb.view(1, 1, 1, k, self.cfg.op_emb_dim).expand(b, n, n, k, self.cfg.op_emb_dim)
        deltas = self.op_net(torch.cat([ctx_a, emb], dim=-1))
        delta = (op_prob.view(1, 1, 1, k, 1) * deltas).sum(dim=3)
        gate = self.gate_net(ctx)
        return self.norm(h + alive * gate * delta)

    def apply_program(
        self,
        rel: torch.Tensor,
        h0_override: Optional[torch.Tensor] = None,
        disabled_nodes: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if rel.ndim != 4:
            raise ValueError(f"rel must be [B,N,N,C], got {tuple(rel.shape)}")
        probs = self.architecture_probs()
        h0 = self.enc(rel) if h0_override is None else h0_override
        states = [h0]
        disabled_nodes = disabled_nodes or set()

        node_outputs = []
        for i in range(self.cfg.max_nodes):
            sources = torch.stack(states, dim=0)  # [S,B,N,N,D], S=i+1
            weights = probs["edge"][i, : i + 1].view(i + 1, 1, 1, 1, 1)
            h_in = (weights * sources).sum(dim=0)
            alive = probs["alive"][i]
            if i in disabled_nodes:
                alive = alive * 0.0
            op_prob = probs["op"][i]
            repeat_prob = probs["repeat"][i]

            repeat_states = []
            h_rep = h_in
            for _ in range(self.cfg.max_repeat):
                h_rep = self.apply_op_once(h_rep, rel, i, op_prob, alive)
                repeat_states.append(h_rep)
            repeat_stack = torch.stack(repeat_states, dim=0)
            h_node = (repeat_prob.view(self.cfg.max_repeat, 1, 1, 1, 1) * repeat_stack).sum(dim=0)
            states.append(h_node)
            node_outputs.append(h_node)

        all_states = torch.stack(states, dim=0)
        readout = probs["readout"].view(self.cfg.max_nodes + 1, 1, 1, 1, 1)
        h_final = (readout * all_states).sum(dim=0)
        logits = self.head(h_final)
        diag = self.program_diagnostics(h0, h_final, probs)
        return logits, h_final, diag

    def program_diagnostics(self, h0: torch.Tensor, h: torch.Tensor, probs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        desc0 = h0.mean(dim=(1, 2))
        desc = h.mean(dim=(1, 2))
        edge_cost = 0.0
        edge_entropy = 0.0
        for i in range(self.cfg.max_nodes):
            p = probs["edge"][i, : i + 1].clamp_min(1e-8)
            edge_cost = edge_cost + probs["alive"][i] * (1.0 - p[0])
            edge_entropy = edge_entropy + (-(p * p.log()).sum())
        op_entropy = (-(probs["op"].clamp_min(1e-8) * probs["op"].clamp_min(1e-8).log()).sum(dim=-1)).mean()
        return {
            "descriptor": desc,
            "descriptor_start": desc0,
            "move": (desc - desc0).pow(2).mean(dim=-1).sqrt().mean(),
            "state_var": desc.var(dim=0, unbiased=False).mean() if desc.shape[0] > 1 else desc.var(unbiased=False),
            "node_cost": probs["alive"].mean(),
            "edge_cost": edge_cost / max(float(self.cfg.max_nodes), 1.0),
            "edge_entropy": edge_entropy / max(float(self.cfg.max_nodes), 1.0),
            "op_entropy": op_entropy,
        }

    def forward(
        self,
        rel: torch.Tensor,
        disabled_nodes: Optional[Set[int]] = None,
        return_h: bool = False,
    ) -> Dict[str, object]:
        logits, h, diag = self.apply_program(rel, disabled_nodes=disabled_nodes)
        # Fixed/recovery are true program-level closure checks: execute the same assembled program again.
        _, h_fixed, _ = self.apply_program(rel, h0_override=h.detach())
        if self.training and self.cfg.noise_std > 0:
            h_noisy = h + torch.randn_like(h) * float(self.cfg.noise_std)
        else:
            h_noisy = h
        _, h_rec, _ = self.apply_program(rel, h0_override=h_noisy)
        diag["fixed"] = F.mse_loss(h_fixed, h.detach())
        diag["recovery"] = F.mse_loss(h_rec, h.detach())
        out: Dict[str, object] = {"logits": logits, "h": h, "descriptor": diag["descriptor"], "diagnostics": diag}
        if return_h:
            out["h_final"] = h
        return out

    def program_trace(self, node_impacts: Optional[Dict[int, float]] = None) -> Dict[str, object]:
        probs = self.architecture_probs()
        alive = probs["alive"].detach().cpu()
        op = probs["op"].detach().cpu()
        edge = probs["edge"].detach().cpu()
        repeat = probs["repeat"].detach().cpu()
        readout = probs["readout"].detach().cpu()
        nodes = []
        for i in range(self.cfg.max_nodes):
            top_ops = torch.topk(op[i], k=min(3, self.cfg.op_count))
            valid_sources = edge[i, : i + 1]
            top_src = torch.topk(valid_sources, k=min(3, valid_sources.numel()))
            rep = int(torch.argmax(repeat[i]).item()) + 1
            nodes.append(
                {
                    "node": i,
                    "alive_prob": float(alive[i]),
                    "selected": bool(alive[i] > 0.35 or readout[i + 1] > 0.10),
                    "repeat": rep,
                    "top_ops": [{"op": int(idx), "prob": float(val)} for val, idx in zip(top_ops.values, top_ops.indices)],
                    "top_sources": [
                        {"source": "input" if int(idx) == 0 else f"node_{int(idx) - 1}", "prob": float(val)}
                        for val, idx in zip(top_src.values, top_src.indices)
                    ],
                    "readout_prob": float(readout[i + 1]),
                    "ablation_task_loss_delta": None if node_impacts is None else float(node_impacts.get(i, 0.0)),
                }
            )
        return {
            "type": "SRCFProgramBuilderTrace",
            "note": "Explicit differentiable architecture graph. Ops are latent; semantics are inferred by ablation impact.",
            "config": {
                "max_nodes": self.cfg.max_nodes,
                "op_count": self.cfg.op_count,
                "max_repeat": self.cfg.max_repeat,
                "use_triangle": self.cfg.use_triangle,
            },
            "readout": [float(x) for x in readout.tolist()],
            "nodes": nodes,
            "assembled_program": [n for n in nodes if n["selected"]],
        }

    def save_program_trace(self, path: str | Path, node_impacts: Optional[Dict[int, float]] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.program_trace(node_impacts), indent=2, ensure_ascii=False), encoding="utf-8")


def srcf_program_loss(
    output: Dict[str, object],
    weights: SRCFProgramLossWeights = SRCFProgramLossWeights(),
    peer_output: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    d = output["diagnostics"]
    desc = output["descriptor"]
    loss = desc.new_zeros(())
    fixed = d["fixed"]
    recovery = d["recovery"]
    move = d["move"]
    state_var = d["state_var"]
    loss = loss + weights.fixed * fixed
    loss = loss + weights.recovery * recovery
    loss = loss + weights.node_cost * d["node_cost"]
    loss = loss + weights.edge_cost * d["edge_cost"]
    loss = loss + weights.entropy_cost * (d["op_entropy"] + d["edge_entropy"])
    loss = loss + weights.state_var * F.relu(desc.new_tensor(weights.state_var_floor) - state_var)
    loss = loss + weights.move_band * (
        F.relu(desc.new_tensor(weights.move_min) - move).pow(2)
        + F.relu(move - desc.new_tensor(weights.move_max)).pow(2)
    )
    far_dist = _pairwise_distances(F.normalize(desc.float(), dim=-1, eps=1e-6))
    if far_dist.numel() > 0:
        far_keep = F.relu(desc.new_tensor(weights.far_margin) - far_dist.to(desc.device)).mean()
        loss = loss + weights.far_keep * far_keep
    else:
        far_keep = desc.new_zeros(())
    if peer_output is not None:
        contract = F.mse_loss(desc, peer_output["descriptor"])
        loss = loss + weights.contract * contract
    else:
        contract = desc.new_zeros(())
    metrics = {
        "srcf_program_loss": float(loss.detach().cpu()),
        "program_fixed": float(fixed.detach().cpu()),
        "program_recovery": float(recovery.detach().cpu()),
        "program_contract": float(contract.detach().cpu()),
        "program_far_keep": float(far_keep.detach().cpu()),
        "program_move": float(move.detach().cpu()),
        "program_state_var": float(state_var.detach().cpu()),
        "program_node_cost": float(d["node_cost"].detach().cpu()),
        "program_edge_cost": float(d["edge_cost"].detach().cpu()),
        "program_op_entropy": float(d["op_entropy"].detach().cpu()),
    }
    return loss, metrics
