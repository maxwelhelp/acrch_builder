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


class ActionMatrixLayer(nn.Module):
    """One sequential ActionMatrix layer.

    Inside the layer edges are still soft all-to-all for gradient flow, but
    train-time losses/reporting now enforce sparse useful structure.
    """

    def __init__(self, dim: int, slots: int, primitive_matrix: PrimitiveMatrix5x5, top_k: int = 25, sim_rank: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.top_k = top_k
        self.pm = primitive_matrix

        context_dim = dim * 5
        emb_dim = primitive_matrix.emb.shape[-1]
        self.scanner = HybridScanner(dim=dim, context_dim=context_dim, prim_embed_dim=emb_dim)
        self.simulator = LowRankSimulator(dim=dim, num_primitives=primitive_matrix.num_primitives, rank=sim_rank, embed_dim=emb_dim)
        self.executor = ActionExecutor(dim=dim, primitive_matrix=primitive_matrix)

        self.context_logits = nn.Linear(context_dim, top_k)
        self.sim_logits = nn.Linear(dim, 1)
        self.mode_head = nn.Linear(context_dim, 3)  # transform / skip / disable
        self.edge_gate = nn.Linear(context_dim, 1)
        self.write_gate = nn.Linear(context_dim, 1)
        self.phase_gate = nn.Linear(context_dim, 1)
        self.edge_op = nn.Linear(context_dim, 1)
        self.output_gate = nn.Linear(dim, 1)
        self.cell_output_gate = nn.Linear(context_dim, 1)
        self.norm = nn.LayerNorm(dim)

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

    def _flat_pair_bias(self, param: torch.Tensor, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return param.to(dtype=dtype, device=device).reshape(1, self.slots * self.slots, 1).expand(batch, -1, -1).reshape(batch * self.slots * self.slots, 1)

    def forward(self, state: torch.Tensor, memory: torch.Tensor, tau: float = 1.0, disable_sim: bool = False):
        b, s, d = state.shape
        src = state.unsqueeze(2).expand(b, s, s, d)
        tgt = state.unsqueeze(1).expand(b, s, s, d)
        mem = memory[:, None, None, :].expand(b, s, s, d)
        context = torch.cat([src, tgt, src - tgt, src * tgt, mem], dim=-1)

        flat_context = context.reshape(b * s * s, -1)
        flat_src = src.reshape(b * s * s, d)
        flat_tgt = tgt.reshape(b * s * s, d)
        flat_mem = mem.reshape(b * s * s, d)

        cand_ids, proposal_logits, scan_metrics = self.scanner(flat_context, flat_mem, self.pm)
        k = min(self.top_k, proposal_logits.shape[-1])
        top_vals, top_pos = proposal_logits.topk(k=k, dim=-1)
        top_ids = cand_ids.gather(1, top_pos)

        sim, predicted_gain = self.simulator(flat_src, top_ids)
        sim_component = torch.zeros_like(predicted_gain) if disable_sim else self.sim_logits(sim).squeeze(-1)
        gain_component = torch.zeros_like(predicted_gain) if disable_sim else predicted_gain

        context_component = self.context_logits(flat_context)[:, :k]
        choice_logits = (
            _ln_logits(context_component)
            + _ln_logits(gain_component)
            + _ln_logits(sim_component)
            + _ln_logits(top_vals)
        )
        choice = F.gumbel_softmax(choice_logits, tau=tau, hard=False, dim=-1) if self.training else F.softmax(choice_logits, dim=-1)

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

        cell_out = mode[:, 0:1] * transformed + mode[:, 1:2] * flat_src + mode[:, 2:3] * 0.0
        msg = active * edge_scale * cell_out
        msg_grid = msg.view(b, s, s, d)

        incoming = msg_grid.sum(dim=1) / max(1, s)
        next_state = self.norm(state + incoming)

        cell_out_gate = torch.sigmoid(self.cell_output_gate(flat_context) + pair_cell_out)
        cell_tape_weight = cell_out_gate * active * mode[:, 0:1]
        cell_tape = (cell_tape_weight * transformed).view(b, s, s, d)
        denom = cell_tape_weight.view(b, s, s, 1).sum(dim=(1, 2)).clamp_min(1e-5)
        output_tape_state = cell_tape.sum(dim=(1, 2)) / denom

        slot_output_gate = torch.sigmoid(self.output_gate(next_state)).squeeze(-1)
        slot_output_state = (
            slot_output_gate.unsqueeze(-1) * next_state
        ).sum(dim=1) / slot_output_gate.sum(dim=1, keepdim=True).clamp_min(1e-5)

        output_state = output_tape_state + 0.10 * slot_output_state

        chosen = top_ids.gather(1, choice.argmax(dim=-1, keepdim=True)).squeeze(1)
        with torch.no_grad():
            self.pm.update_usage_ema(chosen)

        trace: Dict[str, object] = {
            "candidate_ids": top_ids.detach(),
            "choice": choice.detach(),
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
            "cell_tape_weight": cell_tape_weight.detach(),
            "cell_tape_weight_for_loss": cell_tape_weight,
            "edge_pair_bias": self.edge_pair_bias.detach(),
            "write_pair_bias": self.write_pair_bias.detach(),
            "phase_pair_bias": self.phase_pair_bias.detach(),
            "cell_output_pair_bias": self.cell_output_pair_bias.detach(),
            "scan_metrics": scan_metrics,
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
        self.final_read = final_read
        self.slot_embed = nn.Parameter(torch.randn(slots, dim) * slot_embed_scale)
        self.pm = PrimitiveMatrix5x5(embed_dim=32)
        self.layers = nn.ModuleList([ActionMatrixLayer(dim, slots, self.pm, top_k=top_k, sim_rank=sim_rank) for _ in range(layers)])
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
        ablate_layer_output: Optional[int] = None,
        ablate_state_after: Optional[int] = None,
    ):
        state = self.input_norm(x)
        state = state + self.slot_embed.unsqueeze(0).to(dtype=state.dtype, device=state.device)
        memory = state.mean(dim=1)

        outputs = []
        traces = []
        for idx, layer in enumerate(self.layers):
            state, out, memory, tr = layer(state, memory, tau=tau, disable_sim=disable_sim)
            if ablate_state_after is not None and idx == ablate_state_after:
                state = torch.zeros_like(state)
                memory = torch.zeros_like(memory)
            if ablate_layer_output is not None and idx == ablate_layer_output:
                out = torch.zeros_like(out)
            outputs.append(out)
            traces.append(tr)

        final, read_weights = self._merge_outputs(outputs)
        logits = self.classifier(final)
        return logits, {
            "layers": traces,
            "primitive_metrics": self.pm.metrics(),
            "final_read_mode": self.final_read,
            "final_read_weights": read_weights.detach(),
        }
