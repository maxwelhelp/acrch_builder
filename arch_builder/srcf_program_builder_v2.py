from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SRCFProgramConfigV2:
    rel_dim: int
    out_dim: int
    state_dim: int = 32
    hidden: int = 64
    max_nodes: int = 5
    op_count: int = 6
    op_emb_dim: int = 12
    max_repeat: int = 2
    noise_std: float = 0.02
    arch_temperature: float = 1.0
    use_triangle: bool = True
    use_rel_skip: bool = True


@dataclass
class SRCFProgramLossWeightsV2:
    fixed: float = 0.03
    recovery: float = 0.03
    contract: float = 0.04
    far_keep: float = 0.015
    state_var: float = 0.015
    move_band: float = 0.015
    node_cost: float = 0.002
    edge_cost: float = 0.001
    far_margin: float = 0.20
    state_var_floor: float = 0.002
    move_min: float = 0.002
    move_max: float = 2.5


def _pairwise_distances(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_zeros((0,))
    dist = torch.cdist(z.float(), z.float(), p=2)
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return dist[mask]


class SRCFProgramBuilderV2(nn.Module):
    """Memory efficient explicit program builder.

    It learns an explicit DAG program:
      input -> node_0 -> node_1 ... -> readout

    Each node learns:
      alive gate, source edges, op mixture, repeat count.

    Difference from v1: it does NOT materialize all op outputs. It mixes op
    embeddings first, then runs one transition. This keeps GPU memory small.
    """

    def __init__(self, cfg: SRCFProgramConfigV2) -> None:
        super().__init__()
        self.cfg = cfg
        d, h, m, k, e = cfg.state_dim, cfg.hidden, cfg.max_nodes, cfg.op_count, cfg.op_emb_dim
        self.enc = nn.Sequential(nn.Linear(cfg.rel_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.op_emb = nn.Parameter(torch.randn(k, e) * 0.02)
        self.node_alive_logits = nn.Parameter(torch.full((m,), -0.25))
        self.op_logits = nn.Parameter(torch.zeros(m, k))
        self.edge_logits = nn.Parameter(torch.zeros(m, m + 1))
        self.repeat_logits = nn.Parameter(torch.zeros(m, cfg.max_repeat))
        self.readout_logits = nn.Parameter(torch.zeros(m + 1))

        pieces = 4 + int(cfg.use_triangle)
        ctx_dim = d * pieces + (cfg.rel_dim if cfg.use_rel_skip else 0)
        self.ctx_norm = nn.LayerNorm(ctx_dim)
        self.op_net = nn.Sequential(nn.LayerNorm(ctx_dim + e), nn.Linear(ctx_dim + e, h), nn.SiLU(), nn.Linear(h, d))
        self.gate_net = nn.Sequential(nn.Linear(ctx_dim, h), nn.SiLU(), nn.Linear(h, d), nn.Sigmoid())
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.SiLU(), nn.Linear(h, cfg.out_dim))

    @staticmethod
    def triangle(h: torch.Tensor) -> torch.Tensor:
        b, n, _, d = h.shape
        hc = h.permute(0, 3, 1, 2).contiguous().reshape(b * d, n, n)
        y = torch.bmm(hc, hc) / max(float(n), 1.0)
        return y.reshape(b, d, n, n).permute(0, 2, 3, 1).contiguous()

    def source_mask(self, node_idx: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((self.cfg.max_nodes + 1,), -1e9, device=device)
        mask[: node_idx + 1] = 0.0
        return mask

    def arch_probs(self) -> Dict[str, torch.Tensor]:
        tau = max(float(self.cfg.arch_temperature), 1e-4)
        edge = []
        for i in range(self.cfg.max_nodes):
            edge.append(torch.softmax((self.edge_logits[i] + self.source_mask(i, self.edge_logits.device)) / tau, dim=-1))
        return {
            "alive": torch.sigmoid(self.node_alive_logits),
            "op": torch.softmax(self.op_logits / tau, dim=-1),
            "edge": torch.stack(edge, dim=0),
            "repeat": torch.softmax(self.repeat_logits / tau, dim=-1),
            "readout": torch.softmax(self.readout_logits / tau, dim=-1),
        }

    def context(self, h: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        row = h.mean(dim=2, keepdim=True).expand_as(h)
        col = h.mean(dim=1, keepdim=True).expand_as(h)
        glob = h.mean(dim=(1, 2), keepdim=True).expand_as(h)
        parts = [h, row, col, glob]
        if self.cfg.use_triangle:
            parts.append(self.triangle(h))
        if self.cfg.use_rel_skip:
            parts.append(rel)
        return self.ctx_norm(torch.cat(parts, dim=-1))

    def op_once(self, h: torch.Tensor, rel: torch.Tensor, op_prob: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        ctx = self.context(h, rel)
        op_e = op_prob @ self.op_emb
        op_e = op_e.view(1, 1, 1, -1).expand(ctx.shape[0], ctx.shape[1], ctx.shape[2], -1)
        delta = self.op_net(torch.cat([ctx, op_e], dim=-1))
        gate = self.gate_net(ctx)
        return self.norm(h + alive * gate * delta)

    def apply_program(self, rel: torch.Tensor, h0_override: Optional[torch.Tensor] = None, disabled_nodes: Optional[Set[int]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        probs = self.arch_probs()
        h0 = self.enc(rel) if h0_override is None else h0_override
        states = [h0]
        disabled_nodes = disabled_nodes or set()
        for i in range(self.cfg.max_nodes):
            src = torch.stack(states, dim=0)
            ew = probs["edge"][i, : i + 1].view(i + 1, 1, 1, 1, 1)
            h = (ew * src).sum(dim=0)
            alive = probs["alive"][i] * (0.0 if i in disabled_nodes else 1.0)
            reps = []
            h_rep = h
            for _ in range(self.cfg.max_repeat):
                h_rep = self.op_once(h_rep, rel, probs["op"][i], alive)
                reps.append(h_rep)
            h_node = (probs["repeat"][i].view(self.cfg.max_repeat, 1, 1, 1, 1) * torch.stack(reps, dim=0)).sum(dim=0)
            states.append(h_node)
        all_states = torch.stack(states, dim=0)
        h_final = (probs["readout"].view(self.cfg.max_nodes + 1, 1, 1, 1, 1) * all_states).sum(dim=0)
        logits = self.head(h_final)
        diag = self.diagnostics(h0, h_final, probs)
        return logits, h_final, diag

    def diagnostics(self, h0: torch.Tensor, h: torch.Tensor, probs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        d0 = h0.mean(dim=(1, 2))
        d = h.mean(dim=(1, 2))
        edge_cost = h.new_zeros(())
        for i in range(self.cfg.max_nodes):
            edge_cost = edge_cost + probs["alive"][i] * (1.0 - probs["edge"][i, 0])
        op_entropy = (-(probs["op"].clamp_min(1e-8) * probs["op"].clamp_min(1e-8).log()).sum(dim=-1)).mean()
        return {
            "descriptor": d,
            "move": (d - d0).pow(2).mean(dim=-1).sqrt().mean(),
            "state_var": d.var(dim=0, unbiased=False).mean() if d.shape[0] > 1 else d.var(unbiased=False),
            "node_cost": probs["alive"].mean(),
            "edge_cost": edge_cost / max(float(self.cfg.max_nodes), 1.0),
            "op_entropy": op_entropy,
        }

    def forward(self, rel: torch.Tensor, disabled_nodes: Optional[Set[int]] = None) -> Dict[str, object]:
        logits, h, diag = self.apply_program(rel, disabled_nodes=disabled_nodes)
        with torch.enable_grad():
            _, hf, _ = self.apply_program(rel, h0_override=h.detach())
            noisy = h + torch.randn_like(h) * float(self.cfg.noise_std) if self.training and self.cfg.noise_std > 0 else h
            _, hr, _ = self.apply_program(rel, h0_override=noisy)
        diag["fixed"] = F.mse_loss(hf, h.detach())
        diag["recovery"] = F.mse_loss(hr, h.detach())
        return {"logits": logits, "h": h, "descriptor": diag["descriptor"], "diagnostics": diag}

    def program_trace(self, node_impacts: Optional[Dict[int, float]] = None) -> Dict[str, object]:
        p = self.arch_probs()
        alive, op, edge, repeat, readout = [p[k].detach().cpu() for k in ["alive", "op", "edge", "repeat", "readout"]]
        nodes = []
        for i in range(self.cfg.max_nodes):
            top_ops = torch.topk(op[i], k=min(3, self.cfg.op_count))
            top_src = torch.topk(edge[i, : i + 1], k=min(3, i + 1))
            node = {
                "node": i,
                "alive_prob": float(alive[i]),
                "selected": bool(alive[i] > 0.35 or readout[i + 1] > 0.10),
                "repeat": int(torch.argmax(repeat[i]).item()) + 1,
                "top_ops": [{"op": int(idx), "prob": float(val)} for val, idx in zip(top_ops.values, top_ops.indices)],
                "top_sources": [{"source": "input" if int(idx) == 0 else f"node_{int(idx)-1}", "prob": float(val)} for val, idx in zip(top_src.values, top_src.indices)],
                "readout_prob": float(readout[i + 1]),
                "ablation_task_loss_delta": None if node_impacts is None else float(node_impacts.get(i, 0.0)),
            }
            nodes.append(node)
        return {
            "type": "SRCFProgramBuilderV2Trace",
            "note": "Explicit differentiable DAG. Ops are latent; node importance comes from ablation_task_loss_delta.",
            "config": {"max_nodes": self.cfg.max_nodes, "op_count": self.cfg.op_count, "max_repeat": self.cfg.max_repeat, "use_triangle": self.cfg.use_triangle},
            "readout": [float(x) for x in readout.tolist()],
            "nodes": nodes,
            "assembled_program": [n for n in nodes if n["selected"]],
        }

    def save_program_trace(self, path: str | Path, node_impacts: Optional[Dict[int, float]] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.program_trace(node_impacts), indent=2, ensure_ascii=False), encoding="utf-8")


def srcf_program_loss_v2(output: Dict[str, object], weights: SRCFProgramLossWeightsV2 = SRCFProgramLossWeightsV2(), peer_output: Optional[Dict[str, object]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    d = output["diagnostics"]
    desc = output["descriptor"]
    fixed, rec, move, var = d["fixed"], d["recovery"], d["move"], d["state_var"]
    loss = desc.new_zeros(())
    loss = loss + weights.fixed * fixed + weights.recovery * rec
    loss = loss + weights.node_cost * d["node_cost"] + weights.edge_cost * d["edge_cost"]
    loss = loss + weights.state_var * F.relu(desc.new_tensor(weights.state_var_floor) - var)
    loss = loss + weights.move_band * (F.relu(desc.new_tensor(weights.move_min) - move).pow(2) + F.relu(move - desc.new_tensor(weights.move_max)).pow(2))
    far = _pairwise_distances(F.normalize(desc.float(), dim=-1, eps=1e-6))
    far_keep = F.relu(desc.new_tensor(weights.far_margin) - far.to(desc.device)).mean() if far.numel() else desc.new_zeros(())
    loss = loss + weights.far_keep * far_keep
    if peer_output is not None:
        contract = F.mse_loss(desc, peer_output["descriptor"])
        loss = loss + weights.contract * contract
    else:
        contract = desc.new_zeros(())
    metrics = {
        "srcf_program_loss": float(loss.detach().cpu()),
        "program_fixed": float(fixed.detach().cpu()),
        "program_recovery": float(rec.detach().cpu()),
        "program_contract": float(contract.detach().cpu()),
        "program_far_keep": float(far_keep.detach().cpu()),
        "program_move": float(move.detach().cpu()),
        "program_state_var": float(var.detach().cpu()),
        "program_node_cost": float(d["node_cost"].detach().cpu()),
        "program_edge_cost": float(d["edge_cost"].detach().cpu()),
        "program_op_entropy": float(d["op_entropy"].detach().cpu()),
    }
    return loss, metrics
