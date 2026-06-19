from __future__ import annotations

from typing import Dict

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
        self.mode_head = nn.Linear(context_dim, 3)
        self.edge_gate = nn.Linear(context_dim, 1)
        self.write_gate = nn.Linear(context_dim, 1)
        self.phase_gate = nn.Linear(context_dim, 1)
        self.edge_op = nn.Linear(context_dim, 1)
        self.output_gate = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)

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

        mode = F.softmax(self.mode_head(flat_context), dim=-1)
        edge = torch.sigmoid(self.edge_gate(flat_context))
        write = torch.sigmoid(self.write_gate(flat_context))
        phase = torch.sigmoid(self.phase_gate(flat_context))
        sign = torch.tanh(self.edge_op(flat_context))

        cell_out = mode[:, 0:1] * transformed + mode[:, 1:2] * flat_src + mode[:, 2:3] * 0.0
        msg = edge * write * phase * sign * cell_out
        msg = msg.view(b, s, s, d)

        incoming = msg.sum(dim=1) / max(1, s)
        next_state = self.norm(state + incoming)

        output_gate = torch.sigmoid(self.output_gate(next_state)).squeeze(-1)
        output_state = (output_gate.unsqueeze(-1) * next_state).sum(dim=1) / output_gate.sum(dim=1, keepdim=True).clamp_min(1e-5)

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
            "write": write.detach(),
            "phase": phase.detach(),
            "output_gate": output_gate.detach(),
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
    ) -> None:
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.slot_embed = nn.Parameter(torch.randn(slots, dim) * slot_embed_scale)
        self.pm = PrimitiveMatrix5x5(embed_dim=32)
        self.layers = nn.ModuleList([ActionMatrixLayer(dim, slots, self.pm, top_k=top_k, sim_rank=sim_rank) for _ in range(layers)])
        self.input_norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))

    def forward(self, x: torch.Tensor, tau: float = 1.0, disable_sim: bool = False):
        state = self.input_norm(x)
        state = state + self.slot_embed.unsqueeze(0).to(dtype=state.dtype, device=state.device)
        memory = state.mean(dim=1)

        outputs = []
        traces = []
        for layer in self.layers:
            state, out, memory, tr = layer(state, memory, tau=tau, disable_sim=disable_sim)
            outputs.append(out)
            traces.append(tr)

        final = torch.stack(outputs, dim=0).mean(dim=0)
        logits = self.classifier(final)
        return logits, {"layers": traces, "primitive_metrics": self.pm.metrics()}
