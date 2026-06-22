from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SRCFLightConfig:
    """Configuration for the lightweight SRCF closure core.

    The core is intentionally task-agnostic: change only the frontend that creates
    slots and the head that reads slots/memory. The center of the model is a
    learned self-closing transition over relations, not a task-specific recipe.
    """

    dim: int = 64
    slots: int = 6
    relation_dim: int = 32
    action_count: int = 8
    action_emb_dim: int = 16
    hidden: int = 128
    layers: int = 3
    micro_steps: int = 2
    dropout: float = 0.0
    noise_std: float = 0.03
    action_temperature: float = 1.0


@dataclass
class SRCFLossWeights:
    """Weights for the self-supervised closure pressure.

    These are not task recipes. They are generic health constraints: close near
    states, recover from perturbation, become stable, move enough to avoid an
    identity solution, and keep different samples distinguishable.
    """

    fixed: float = 0.10
    recovery: float = 0.10
    contract: float = 0.10
    far_keep: float = 0.05
    state_var: float = 0.05
    move_band: float = 0.05
    action_entropy: float = 0.00
    edge_sparsity: float = 0.00
    far_margin: float = 0.35
    state_var_floor: float = 0.02
    move_min: float = 0.02
    move_max: float = 1.50


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean() if x.numel() else x.new_zeros(())


def _pairwise_distances(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_zeros((0,))
    dist = torch.cdist(z.float(), z.float(), p=2)
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return dist[mask]


class LearnedRelationBuilder(nn.Module):
    """Build a learned relation tensor R[i,j,c] from slot states.

    This module does not encode task-specific roles. It exposes generic pair
    interactions and lets training discover which relations matter for the head.
    """

    def __init__(self, dim: int, relation_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, relation_dim),
        )

    def forward(self, slots: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        b, s, d = slots.shape
        src = slots.unsqueeze(2).expand(b, s, s, d)
        tgt = slots.unsqueeze(1).expand(b, s, s, d)
        mem = memory[:, None, None, :].expand(b, s, s, d)
        pair = torch.cat([src, tgt, src - tgt, src * tgt, mem], dim=-1)
        return self.net(pair)


class SelfOrganizingTransition(nn.Module):
    """One learned closure transition over slots and memory.

    Instead of a named primitive bank or hand-written role recipes, this layer has
    a small set of learned latent moves. A step is an action graph over slots:
    edge weights choose which relations communicate, and latent move weights
    choose how each relation transforms information.
    """

    def __init__(self, cfg: SRCFLightConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        r = cfg.relation_dim
        a = cfg.action_count
        e = cfg.action_emb_dim
        h = cfg.hidden

        self.relation_builder = LearnedRelationBuilder(d, r, h)
        self.action_emb = nn.Parameter(torch.randn(a, e) * 0.02)
        self.edge_head = nn.Sequential(
            nn.LayerNorm(r),
            nn.Linear(r, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(r),
            nn.Linear(r, h),
            nn.SiLU(),
            nn.Linear(h, a),
        )
        self.move_net = nn.Sequential(
            nn.LayerNorm(d * 5 + e),
            nn.Linear(d * 5 + e, h),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, d),
        )
        self.slot_gate = nn.Sequential(
            nn.LayerNorm(d * 3),
            nn.Linear(d * 3, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        self.mem_update = nn.Sequential(
            nn.LayerNorm(d * 3),
            nn.Linear(d * 3, h),
            nn.SiLU(),
            nn.Linear(h, d),
        )
        self.mem_gate = nn.Sequential(
            nn.LayerNorm(d * 3),
            nn.Linear(d * 3, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        self.slot_norm = nn.LayerNorm(d)
        self.mem_norm = nn.LayerNorm(d)

    def transition_once(
        self,
        slots: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b, s, d = slots.shape
        r = self.relation_builder(slots, memory)
        edge = torch.sigmoid(self.edge_head(r))  # [B,S,S,1]
        logits = self.action_head(r) / max(float(self.cfg.action_temperature), 1e-4)
        action_prob = torch.softmax(logits, dim=-1)  # [B,S,S,A]

        src = slots.unsqueeze(2).expand(b, s, s, d)
        tgt = slots.unsqueeze(1).expand(b, s, s, d)
        mem = memory[:, None, None, :].expand(b, s, s, d)
        base = torch.cat([src, tgt, src - tgt, src * tgt, mem], dim=-1)
        base_a = base.unsqueeze(3).expand(b, s, s, self.cfg.action_count, d * 5)
        emb = self.action_emb.view(1, 1, 1, self.cfg.action_count, self.cfg.action_emb_dim).expand(
            b, s, s, self.cfg.action_count, self.cfg.action_emb_dim
        )
        move_in = torch.cat([base_a, emb], dim=-1)
        moves = self.move_net(move_in)  # [B,S,S,A,D]
        edge_msg = (action_prob.unsqueeze(-1) * moves).sum(dim=3)  # [B,S,S,D]

        weighted = edge * edge_msg
        incoming_mass = edge.sum(dim=1).clamp_min(1e-6)  # target slot mass [B,S,1]
        incoming = weighted.sum(dim=1) / incoming_mass

        slot_gate_in = torch.cat([slots, incoming, memory[:, None, :].expand(b, s, d)], dim=-1)
        slot_gate = torch.sigmoid(self.slot_gate(slot_gate_in))
        next_slots = self.slot_norm(slots + slot_gate * incoming)

        slot_mean = next_slots.mean(dim=1)
        prev_mean = slots.mean(dim=1)
        mem_in = torch.cat([memory, slot_mean, prev_mean], dim=-1)
        mem_delta = self.mem_update(mem_in)
        mem_gate = torch.sigmoid(self.mem_gate(mem_in))
        next_memory = self.mem_norm(memory + mem_gate * mem_delta)

        action_entropy = (-(action_prob.clamp_min(1e-8) * action_prob.clamp_min(1e-8).log()).sum(dim=-1)).mean()
        edge_mass = edge.mean()
        metrics = {
            "relation": r,
            "edge": edge,
            "action_prob": action_prob,
            "action_entropy": action_entropy,
            "edge_mass": edge_mass,
            "slot_gate_mean": slot_gate.mean(),
            "memory_gate_mean": mem_gate.mean(),
        }
        return next_slots, next_memory, metrics

    def forward(
        self,
        slots: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        start_slots = slots
        start_memory = memory
        step_metrics = []
        for _ in range(max(1, int(self.cfg.micro_steps))):
            slots, memory, metrics = self.transition_once(slots, memory)
            step_metrics.append(metrics)

        fixed_slots, fixed_memory, _ = self.transition_once(slots, memory)
        if self.training and self.cfg.noise_std > 0:
            noisy_slots = slots + torch.randn_like(slots) * float(self.cfg.noise_std)
            noisy_memory = memory + torch.randn_like(memory) * float(self.cfg.noise_std)
        else:
            noisy_slots = slots
            noisy_memory = memory
        recovered_slots, recovered_memory, _ = self.transition_once(noisy_slots, noisy_memory)

        descriptor = torch.cat([slots.mean(dim=1), memory], dim=-1)
        start_descriptor = torch.cat([start_slots.mean(dim=1), start_memory], dim=-1)
        move = (descriptor - start_descriptor).pow(2).mean(dim=-1).sqrt().mean()
        fixed = F.mse_loss(fixed_slots, slots) + F.mse_loss(fixed_memory, memory)
        recovery = F.mse_loss(recovered_slots, slots.detach()) + F.mse_loss(recovered_memory, memory.detach())
        state_var = descriptor.var(dim=0, unbiased=False).mean() if descriptor.shape[0] > 1 else descriptor.var(unbiased=False)
        step_delta = (slots - start_slots).pow(2).mean(dim=-1).sqrt().mean()

        diag = {
            "descriptor": descriptor,
            "move": move,
            "fixed": fixed,
            "recovery": recovery,
            "state_var": state_var,
            "step_delta": step_delta,
            "action_entropy": torch.stack([m["action_entropy"] for m in step_metrics]).mean(),
            "edge_mass": torch.stack([m["edge_mass"] for m in step_metrics]).mean(),
            "slot_gate_mean": torch.stack([m["slot_gate_mean"] for m in step_metrics]).mean(),
            "memory_gate_mean": torch.stack([m["memory_gate_mean"] for m in step_metrics]).mean(),
        }
        return slots, memory, diag


class SRCFLightCore(nn.Module):
    """Task-agnostic self-closing core.

    Use this directly when you already have a frontend that produces slots. The
    core only assumes [B,S,D] slots and optional [B,D] memory.
    """

    def __init__(self, cfg: SRCFLightConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([SelfOrganizingTransition(cfg) for _ in range(cfg.layers)])
        self.init_memory = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, cfg.dim),
            nn.SiLU(),
            nn.Linear(cfg.dim, cfg.dim),
        )

    def forward(self, slots: torch.Tensor, memory: Optional[torch.Tensor] = None) -> Dict[str, object]:
        if slots.ndim != 3:
            raise ValueError(f"slots must be [B,S,D], got {tuple(slots.shape)}")
        if slots.shape[1] != self.cfg.slots or slots.shape[2] != self.cfg.dim:
            raise ValueError(
                f"expected slots [B,{self.cfg.slots},{self.cfg.dim}], got {tuple(slots.shape)}"
            )
        if memory is None:
            memory = self.init_memory(slots.mean(dim=1))
        layer_diags = []
        for layer in self.layers:
            slots, memory, diag = layer(slots, memory)
            layer_diags.append(diag)

        descriptor = torch.cat([slots.mean(dim=1), memory], dim=-1)
        agg: Dict[str, torch.Tensor] = {"descriptor": descriptor}
        keys = [k for k in layer_diags[0] if k != "descriptor"] if layer_diags else []
        for key in keys:
            agg[key] = torch.stack([d[key] for d in layer_diags]).mean()
        return {
            "slots": slots,
            "memory": memory,
            "descriptor": descriptor,
            "diagnostics": agg,
            "layer_diagnostics": layer_diags,
        }


class SRCFLightModel(nn.Module):
    """Simple wrapper: replace `frontend` and `head` for any task.

    This class is only a convenient default. The self-organizing architecture is
    in SRCFLightCore; the contract is: frontend -> slots, core -> descriptor,
    head -> task output.
    """

    def __init__(self, input_dim: int, output_dim: int, cfg: SRCFLightConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.slots * cfg.dim),
        )
        self.memory_frontend = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.dim),
        )
        self.core = SRCFLightCore(cfg)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.dim * 2),
            nn.Linear(cfg.dim * 2, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        slots = self.frontend(x).view(x.shape[0], self.cfg.slots, self.cfg.dim)
        memory = self.memory_frontend(x)
        out = self.core(slots, memory)
        logits = self.head(out["descriptor"])
        out["logits"] = logits
        return out


def srcf_closure_loss(
    output: Dict[str, object],
    weights: SRCFLossWeights = SRCFLossWeights(),
    peer_output: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Generic closure loss for any task head.

    `peer_output` should be the same sample under a small augmentation/noise. If
    absent, the contract term is skipped. This keeps the core usable with any
    input modality and any head.
    """

    diagnostics = output["diagnostics"]
    descriptor = output["descriptor"]
    device = descriptor.device
    loss = descriptor.new_zeros(())

    fixed = diagnostics["fixed"]
    recovery = diagnostics["recovery"]
    move = diagnostics["move"]
    state_var = diagnostics["state_var"]
    action_entropy = diagnostics["action_entropy"]
    edge_mass = diagnostics["edge_mass"]

    loss = loss + weights.fixed * fixed
    loss = loss + weights.recovery * recovery
    loss = loss + weights.state_var * F.relu(descriptor.new_tensor(weights.state_var_floor) - state_var)
    loss = loss + weights.move_band * (
        F.relu(descriptor.new_tensor(weights.move_min) - move).pow(2)
        + F.relu(move - descriptor.new_tensor(weights.move_max)).pow(2)
    )

    far_dist = _pairwise_distances(F.normalize(descriptor.float(), dim=-1, eps=1e-6))
    if far_dist.numel() > 0:
        far_keep_loss = F.relu(descriptor.new_tensor(weights.far_margin) - far_dist.to(device)).mean()
        loss = loss + weights.far_keep * far_keep_loss
    else:
        far_keep_loss = descriptor.new_zeros(())

    if peer_output is not None:
        peer_descriptor = peer_output["descriptor"]
        contract = F.mse_loss(descriptor, peer_descriptor)
        loss = loss + weights.contract * contract
    else:
        contract = descriptor.new_zeros(())

    if weights.action_entropy != 0.0:
        # Negative sign means larger entropy is rewarded. Keep weight at 0 by default
        # if you do not want this exploration pressure.
        loss = loss - weights.action_entropy * action_entropy
    if weights.edge_sparsity != 0.0:
        loss = loss + weights.edge_sparsity * edge_mass

    metrics = {
        "srcf_loss": float(loss.detach().cpu()),
        "srcf_fixed": float(fixed.detach().cpu()),
        "srcf_recovery": float(recovery.detach().cpu()),
        "srcf_contract": float(contract.detach().cpu()),
        "srcf_far_keep_loss": float(far_keep_loss.detach().cpu()),
        "srcf_move": float(move.detach().cpu()),
        "srcf_state_var": float(state_var.detach().cpu()),
        "srcf_action_entropy": float(action_entropy.detach().cpu()),
        "srcf_edge_mass": float(edge_mass.detach().cpu()),
    }
    return loss, metrics
