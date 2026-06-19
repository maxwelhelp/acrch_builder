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
    """One ActionMatrix layer.

    v5 fixes the first proof-slice bottleneck:
    - slot identity exists in ActionMatrixModel;
    - selected primitive now has a direct cell-output tape, not only a weak
      residual into target slots;
    - edge scale is positive/non-zero at initialization instead of tanh≈0;
    - transform mode has a small prior, skip/disable are not free defaults.
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

        self._init_gate_priors()

    def _init_gate_priors(self) -> None:
        # Avoid the first-slice dead start where all messages are multiplied by
        # tanh(0)≈0 or by low transform mass.
        with torch.no_grad():
            self.mode_head.bias.zero_()
            self.mode_head.bias[0] = 0.6    # transform
            self.mode_head.bias[1] = -0.2   # skip
            self.mode_head.bias[2] = -0.6   # disable
            self.edge_gate.bias.fill_(0.5)
            self.write_gate.bias.fill_(0.5)
            self.phase_gate.bias.fill_(0.5)
            self.edge_op.bias.fill_(1.0)
            self.cell_output_gate.bias.fill_(0.2)

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

        # Positive scale avoids the tanh-zero dead start. Signed edge ops can be
        # reintroduced after the proof slice is alive.
        edge_scale = 0.25 + torch.sigmoid(self.edge_op(flat_context))
        active = edge * write * phase

        cell_out = mode[:, 0:1] * transformed + mode[:, 1:2] * flat_src + mode[:, 2:3] * 0.0
        msg = active * edge_scale * cell_out
        msg_grid = msg.view(b, s, s, d)

        incoming = msg_grid.sum(dim=1) / max(1, s)
        next_state = self.norm(state + incoming)

        # Direct output tape from cells: the classifier can now see the executed
        # ActionMatrix result, not only a diluted residual after target-slot sum.
        cell_out_gate = torch.sigmoid(self.cell_output_gate(flat_context))
        cell_tape_weight = cell_out_gate * active * mode[:, 0:1]
        cell_tape = (cell_tape_weight * transformed).view(b, s, s, d)
        # denom must stay [B, 1] so it broadcasts over feature dim D.
        # Squeezing to [B] breaks broadcasting against [B, D].
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
            "write": write.detach(),
            "phase": phase.detach(),
            "edge_scale": edge_scale.detach(),
            "active": active.detach(),
            "output_gate": slot_output_gate.detach(),
            "cell_output_gate": cell_out_gate.detach(),
            "cell_tape_weight": cell_tape_weight.detach(),
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
        # Do not remove mean information here: diff task label is encoded in mean(x0-x1).
        self.classifier = nn.Linear(dim, classes)

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
