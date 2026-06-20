from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitive_matrix import PrimitiveMatrix5x5
from .hybrid_scanner import HybridScanner
from .simulator import LowRankSimulator
from .executor import ActionExecutor


def _ln_logits(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, x.shape[-1:])


def _primitive_distribution(candidate_ids: torch.Tensor, choice: torch.Tensor, num_primitives: int) -> torch.Tensor:
    out = torch.zeros(candidate_ids.shape[0], num_primitives, device=choice.device, dtype=choice.dtype)
    out.scatter_add_(1, candidate_ids, choice)
    return out


class ActionMatrixLayer(nn.Module):
    """One sequential ActionMatrix layer.

    Inside the layer edges are still soft all-to-all for gradient flow, but
    train-time losses/reporting now enforce sparse useful structure.
    """

    def __init__(
        self,
        dim: int,
        slots: int,
        primitive_matrix: PrimitiveMatrix5x5,
        top_k: int = 25,
        sim_rank: int = 16,
        state_norm: str = "none",
    ) -> None:
        super().__init__()
        if state_norm not in {"none", "layernorm"}:
            raise ValueError(f"state_norm must be none or layernorm, got {state_norm}")
        self.dim = dim
        self.slots = slots
        self.top_k = top_k
        self.pm = primitive_matrix
        self.state_norm_mode = state_norm

        context_dim = dim * 5
        emb_dim = primitive_matrix.emb.shape[-1]
        self.scanner = HybridScanner(dim=dim, context_dim=context_dim, prim_embed_dim=emb_dim)
        self.simulator = LowRankSimulator(dim=dim, num_primitives=primitive_matrix.num_primitives, rank=sim_rank, embed_dim=emb_dim)
        self.executor = ActionExecutor(dim=dim, primitive_matrix=primitive_matrix)

        self.context_logits = nn.Linear(context_dim, top_k)
        self.sim_logits = nn.Linear(dim, 1)
        self.prev_output_proj = nn.Linear(dim, context_dim, bias=False)
        self.prev_action_proj = nn.Linear(primitive_matrix.num_primitives, context_dim, bias=False)
        self.prev_active_proj = nn.Linear(1, context_dim, bias=False)
        self.listen_gate = nn.Linear(context_dim, 1)
        self.mode_head = nn.Linear(context_dim, 3)  # transform / skip / disable
        self.edge_gate = nn.Linear(context_dim, 1)
        self.write_gate = nn.Linear(context_dim, 1)
        self.phase_gate = nn.Linear(context_dim, 1)
        self.edge_op = nn.Linear(context_dim, 1)
        self.split_head = nn.Linear(dim, 3)
        self.child_gate = nn.Linear(dim, 1)
        self.merge_gate = nn.Linear(dim, 1)
        self.slot_alive_head = nn.Linear(dim, 1)
        self.output_gate = nn.Linear(dim, 1)
        self.cell_output_gate = nn.Linear(context_dim, 1)
        self.norm = nn.LayerNorm(dim) if state_norm == "layernorm" else nn.Identity()

        # Pair priors are learnable anchors, not hardcoded expected edges.
        self.edge_pair_bias = nn.Parameter(torch.zeros(slots, slots))
        self.write_pair_bias = nn.Parameter(torch.zeros(slots, slots))
        self.phase_pair_bias = nn.Parameter(torch.zeros(slots, slots))
        self.cell_output_pair_bias = nn.Parameter(torch.zeros(slots, slots))

        self._init_gate_priors()

    def _init_gate_priors(self) -> None:
        with torch.no_grad():
            self.mode_head.bias.zero_()
            self.mode_head.bias[0] = 0.6
            self.mode_head.bias[1] = -0.2
            self.mode_head.bias[2] = -0.6
            self.edge_gate.bias.fill_(0.5)
            self.write_gate.bias.fill_(0.5)
            self.phase_gate.bias.fill_(0.5)
            self.edge_op.bias.fill_(1.0)
            self.cell_output_gate.bias.fill_(0.2)
            self.listen_gate.bias.fill_(0.3)

    def _flat_pair_bias(self, param: torch.Tensor, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return param.to(dtype=dtype, device=device).reshape(1, self.slots * self.slots, 1).expand(batch, -1, -1).reshape(batch * self.slots * self.slots, 1)

    def apply_state_update(
        self,
        state: torch.Tensor,
        cell_value_grid: torch.Tensor,
        cell_write_mass_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize incoming cell values and apply a bounded soft-OR write."""
        write_mass = cell_write_mass_grid.clamp(0.0, 1.0)
        mass_sum = write_mass.sum(dim=1)
        write_value = (write_mass * cell_value_grid).sum(dim=1) / mass_sum.clamp_min(1e-8)
        target_write_gate = 1.0 - (1.0 - write_mass).prod(dim=1)
        next_state = (1.0 - target_write_gate) * state + target_write_gate * write_value
        return self.norm(next_state), write_value, target_write_gate

    def forward(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        slot_address: torch.Tensor,
        prev_context: Optional[Dict[str, torch.Tensor]] = None,
        tau: float = 1.0,
        disable_sim: bool = False,
        disable_gain: bool = False,
        disable_sim_result: bool = False,
        curriculum_mode: str = "teacher",
    ):
        b, s, d = state.shape
        src = state.unsqueeze(2).expand(b, s, s, d)
        tgt = state.unsqueeze(1).expand(b, s, s, d)
        mem = memory[:, None, None, :].expand(b, s, s, d)
        source_address = slot_address.view(1, s, 1, d).expand(b, s, s, d)
        target_address = slot_address.view(1, 1, s, d).expand(b, s, s, d)
        control_src = src + source_address
        control_tgt = tgt + target_address
        context = torch.cat([control_src, control_tgt, src - tgt, src * tgt, mem], dim=-1)

        listen_gate = torch.zeros(b, 1, device=state.device, dtype=state.dtype)
        prev_context_mix = None
        if prev_context is not None:
            mix_parts = []
            prev_output = prev_context.get("output")
            prev_action = prev_context.get("action_dist")
            prev_active = prev_context.get("active_mass")
            if prev_output is not None:
                mix_parts.append(self.prev_output_proj(prev_output))
            if prev_action is not None:
                mix_parts.append(self.prev_action_proj(prev_action))
            if prev_active is not None:
                mix_parts.append(self.prev_active_proj(prev_active))
            if mix_parts:
                prev_context_mix = sum(mix_parts)
                listen_gate = torch.sigmoid(self.listen_gate(prev_context_mix))
                context = context + (listen_gate[:, None, None, :] * prev_context_mix[:, None, None, :])

        flat_context = context.reshape(b * s * s, -1)
        flat_src = src.reshape(b * s * s, d)
        flat_tgt = tgt.reshape(b * s * s, d)
        flat_mem = mem.reshape(b * s * s, d)

        cand_ids, proposal_logits, source_ids, scan_metrics = self.scanner(flat_context, flat_mem, self.pm)
        k = min(self.top_k, proposal_logits.shape[-1])
        top_vals, top_pos = proposal_logits.topk(k=k, dim=-1)
        top_ids = cand_ids.gather(1, top_pos)
        top_source_ids = source_ids.gather(1, top_pos)

        sim, predicted_gain = self.simulator(flat_src, top_ids)
        sim_component = (
            torch.zeros_like(predicted_gain)
            if disable_sim or disable_sim_result
            else self.sim_logits(sim).squeeze(-1)
        )
        gain_component = (
            torch.zeros_like(predicted_gain)
            if disable_sim or disable_gain
            else predicted_gain
        )

        context_component = self.context_logits(flat_context)[:, :k]
        choice_logits = (
            _ln_logits(context_component)
            + _ln_logits(gain_component)
            + _ln_logits(sim_component)
            + _ln_logits(top_vals)
        )
        choice = F.gumbel_softmax(choice_logits, tau=tau, hard=False, dim=-1) if self.training else F.softmax(choice_logits, dim=-1)

        source_names = ("grid", "semantic", "usage", "random")
        with torch.no_grad():
            for source_id, source_name in enumerate(source_names):
                source_mass = (choice * (top_source_ids == source_id).to(choice.dtype)).sum(dim=-1).mean()
                scan_metrics[f"{source_name}_candidate_usage"] = float(source_mass.cpu())
            scan_metrics["scanner_source_mass_sum"] = float(
                sum(scan_metrics[f"{name}_candidate_usage"] for name in source_names)
            )

        primitive_out = self.executor(flat_src, flat_tgt, flat_mem, top_ids)
        transformed = (choice.unsqueeze(-1) * primitive_out).sum(dim=1)

        pair_edge = self._flat_pair_bias(self.edge_pair_bias, b, flat_context.dtype, flat_context.device)
        pair_write = self._flat_pair_bias(self.write_pair_bias, b, flat_context.dtype, flat_context.device)
        pair_phase = self._flat_pair_bias(self.phase_pair_bias, b, flat_context.dtype, flat_context.device)
        pair_cell_out = self._flat_pair_bias(self.cell_output_pair_bias, b, flat_context.dtype, flat_context.device)

        mode = F.softmax(self.mode_head(flat_context), dim=-1)
        edge = torch.sigmoid(self.edge_gate(flat_context) + pair_edge)
        write = torch.sigmoid(self.write_gate(flat_context) + pair_write)
        phase = torch.sigmoid(self.phase_gate(flat_context) + pair_phase)

        edge_scale = 0.25 + torch.sigmoid(self.edge_op(flat_context))
        active = edge * write * phase

        enabled_mode_mass = mode[:, 0:1] + mode[:, 1:2]
        cell_value = (
            mode[:, 0:1] * transformed + mode[:, 1:2] * flat_src
        ) / enabled_mode_mass.clamp_min(1e-8)
        cell_write_mass = active * enabled_mode_mass
        scaled_cell_value = edge_scale * cell_value
        cell_value_grid = scaled_cell_value.view(b, s, s, d)
        cell_write_mass_grid = cell_write_mass.view(b, s, s, 1)

        next_state, write_value, target_write_gate = self.apply_state_update(
            state,
            cell_value_grid,
            cell_write_mass_grid,
        )

        cell_out_gate = torch.sigmoid(self.cell_output_gate(flat_context) + pair_cell_out)
        cell_tape_weight = cell_out_gate * active * mode[:, 0:1]
        cell_tape = (cell_tape_weight * transformed).view(b, s, s, d)
        denom = cell_tape_weight.view(b, s, s, 1).sum(dim=(1, 2)).clamp_min(1e-5)
        output_tape_state = cell_tape.sum(dim=(1, 2)) / denom

        slot_alive_logits = self.slot_alive_head(next_state).squeeze(-1)
        split_count = F.softmax(self.split_head(next_state), dim=-1)
        child_gate = torch.sigmoid(self.child_gate(next_state)).squeeze(-1)
        merge_gate = torch.sigmoid(self.merge_gate(next_state)).squeeze(-1)
        slot_output_gate = torch.sigmoid(self.output_gate(next_state)).squeeze(-1)
        slot_output_state = (
            (slot_output_gate * torch.sigmoid(slot_alive_logits)).unsqueeze(-1) * next_state
        ).sum(dim=1) / (slot_output_gate * torch.sigmoid(slot_alive_logits)).sum(dim=1, keepdim=True).clamp_min(1e-5)

        collector_mass = slot_output_gate * merge_gate
        output_state = output_tape_state + 0.10 * slot_output_state + 0.02 * (collector_mass.unsqueeze(-1) * next_state).sum(dim=1)

        chosen = top_ids.gather(1, choice.argmax(dim=-1, keepdim=True)).squeeze(1)

        trace: Dict[str, object] = {
            "candidate_ids": top_ids.detach(),
            "candidate_source_ids": top_source_ids.detach(),
            "choice": choice.detach(),
            # Losses need the live distribution; reports intentionally use the
            # detached `choice` field above.
            "choice_for_loss": choice,
            "chosen": chosen.detach(),
            "predicted_gain": predicted_gain.detach(),
            "predicted_gain_for_loss": predicted_gain,
            "mode": mode.detach(),
            "mode_for_loss": mode,
            "edge": edge.detach(),
            "edge_for_loss": edge,
            "write": write.detach(),
            "write_for_loss": write,
            "phase": phase.detach(),
            "phase_for_loss": phase,
            "edge_scale": edge_scale.detach(),
            "active": active.detach(),
            "output_gate": slot_output_gate.detach(),
            "cell_output_gate": cell_out_gate.detach(),
            "slot_alive": torch.sigmoid(slot_alive_logits).detach(),
            "slot_alive_for_loss": torch.sigmoid(slot_alive_logits),
            "split_count": split_count.detach(),
            "split_count_for_loss": split_count,
            "child_gate": child_gate.detach(),
            "child_gate_for_loss": child_gate,
            "merge_gate": merge_gate.detach(),
            "merge_gate_for_loss": merge_gate,
            "collector_mass": collector_mass.detach(),
            "collector_mass_for_loss": collector_mass,
            "listen_gate": listen_gate.detach(),
            "prev_context_mix": prev_context_mix.detach() if prev_context_mix is not None else None,
            "cell_tape_weight": cell_tape_weight.detach(),
            "cell_tape_weight_for_loss": cell_tape_weight,
            "cell_write_mass": cell_write_mass.detach(),
            "target_write_gate": target_write_gate.detach(),
            "write_value": write_value.detach(),
            "state_norm_mode": self.state_norm_mode,
            "slot_address_used_by_controller": True,
            "slot_address_used_by_executor": False,
            "edge_pair_bias": self.edge_pair_bias.detach(),
            "write_pair_bias": self.write_pair_bias.detach(),
            "phase_pair_bias": self.phase_pair_bias.detach(),
            "cell_output_pair_bias": self.cell_output_pair_bias.detach(),
            "scan_metrics": scan_metrics,
            "curriculum_mode": curriculum_mode,
        }

        new_memory = 0.95 * memory + 0.05 * next_state.mean(dim=1)
        return next_state, output_state, new_memory, trace


class ActionMatrixModel(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        slots: int = 4,
        layers: int = 1,
        classes: int = 2,
        top_k: int = 25,
        sim_rank: int = 16,
        slot_embed_scale: float = 0.5,
        input_norm: str = "none",
        state_norm: str = "none",
        final_read: str = "last",
    ) -> None:
        super().__init__()
        if input_norm not in {"none", "layernorm"}:
            raise ValueError(f"input_norm must be none or layernorm, got {input_norm}")
        if final_read not in {"last", "mean", "learned"}:
            raise ValueError(f"final_read must be last, mean, or learned, got {final_read}")
        self.dim = dim
        self.slots = slots
        self.num_layers = layers
        self.input_norm_mode = input_norm
        self.state_norm_mode = state_norm
        self.final_read = final_read
        self.slot_embed = nn.Parameter(torch.randn(slots, dim) * slot_embed_scale)
        self.pm = PrimitiveMatrix5x5(embed_dim=32)
        self.layers = nn.ModuleList([
            ActionMatrixLayer(
                dim,
                slots,
                self.pm,
                top_k=top_k,
                sim_rank=sim_rank,
                state_norm=state_norm,
            )
            for _ in range(layers)
        ])
        self.input_norm = nn.LayerNorm(dim) if input_norm == "layernorm" else nn.Identity()
        self.layer_read_logits = nn.Parameter(torch.zeros(layers))
        self.classifier = nn.Linear(dim, classes)

    def _merge_outputs(self, outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stack = torch.stack(outputs, dim=0)  # [L,B,D]
        if self.final_read == "last":
            weights = torch.zeros(len(outputs), device=stack.device, dtype=stack.dtype)
            weights[-1] = 1.0
            return stack[-1], weights
        if self.final_read == "mean":
            weights = torch.ones(len(outputs), device=stack.device, dtype=stack.dtype) / len(outputs)
            return stack.mean(dim=0), weights
        weights = F.softmax(self.layer_read_logits[: len(outputs)].to(dtype=stack.dtype), dim=0)
        return (weights[:, None, None] * stack).sum(dim=0), weights

    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        disable_sim: bool = False,
        disable_gain: bool = False,
        disable_sim_result: bool = False,
        disable_slot_address: bool = False,
        ablate_layer_output: Optional[int] = None,
        ablate_state_after: Optional[int] = None,
        curriculum_mode: str = "teacher",
    ):
        b = x.shape[0]
        s = self.slots
        state = self.input_norm(x)
        memory = state.mean(dim=1)
        slot_address = self.slot_embed.to(dtype=state.dtype, device=state.device)
        if disable_slot_address:
            slot_address = torch.zeros_like(slot_address)

        outputs = []
        traces = []
        prev_context: Optional[Dict[str, torch.Tensor]] = None
        for idx, layer in enumerate(self.layers):
            layer_prev_context = prev_context
            if curriculum_mode == "audit" and prev_context is not None:
                layer_prev_context = {
                    key: prev_context[key]
                    for key in ("active_mass", "write_mass")
                    if key in prev_context
                }
            elif curriculum_mode == "deploy":
                layer_prev_context = None
            state, out, memory, tr = layer(
                state,
                memory,
                slot_address,
                prev_context=layer_prev_context,
                tau=tau,
                disable_sim=disable_sim,
                disable_gain=disable_gain,
                disable_sim_result=disable_sim_result,
                curriculum_mode=curriculum_mode,
            )
            if ablate_state_after is not None and idx == ablate_state_after:
                state = torch.zeros_like(state)
                memory = torch.zeros_like(memory)
            if ablate_layer_output is not None and idx == ablate_layer_output:
                out = torch.zeros_like(out)
            outputs.append(out)
            traces.append(tr)

            cand = tr["candidate_ids"]
            choice = tr["choice"]
            action_dist = _primitive_distribution(cand, choice, self.pm.num_primitives)
            action_dist = action_dist.view(b, s * s, -1).mean(dim=1)
            active = tr["active"].view(b, s * s)
            write_mass = tr["cell_write_mass"].view(b, s * s)
            prev_context = {
                "output": out,
                "action_dist": action_dist,
                "active_mass": active.mean(dim=1, keepdim=True),
                "write_mass": write_mass.mean(dim=1, keepdim=True),
            }

        final, read_weights = self._merge_outputs(outputs)
        logits = self.classifier(final)
        return logits, {
            "layers": traces,
            "primitive_metrics": self.pm.metrics(),
            "final_read_mode": self.final_read,
            "final_read_weights": read_weights.detach(),
            "state_norm_mode": self.state_norm_mode,
            "slot_address_used_by_controller": not disable_slot_address,
            "slot_address_used_by_executor": False,
            "curriculum_mode": curriculum_mode,
        }
