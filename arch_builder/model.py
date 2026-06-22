from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitive_matrix import PrimitiveMatrix5x5
from .hybrid_scanner import HybridScanner
from .simulator import LowRankSimulator
from .executor import ActionExecutor
from .self_delta_candidate_field import SelfDeltaCandidateField
from .utility_critic import UtilityCritic
from .vnext_controller import batched_mmr_select as vnext_mmr_select

def _ln_logits(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, x.shape[-1:])


def _primitive_distribution(candidate_ids: torch.Tensor, choice: torch.Tensor, num_primitives: int) -> torch.Tensor:
    out = torch.zeros(candidate_ids.shape[0], num_primitives, device=choice.device, dtype=choice.dtype)
    out.scatter_add_(1, candidate_ids, choice)
    return out


def _behavior_decorrelation_loss(behavior: Optional[torch.Tensor]) -> torch.Tensor:
    if behavior is None:
        return torch.zeros(())
    if behavior.ndim != 3 or behavior.shape[1] < 2:
        return behavior.new_zeros(())
    feat = F.normalize(behavior.float(), p=2, dim=-1, eps=1e-6)
    sim = torch.bmm(feat, feat.transpose(1, 2))
    k = sim.shape[-1]
    offdiag = 1.0 - torch.eye(k, dtype=sim.dtype, device=sim.device).unsqueeze(0)
    return (sim.pow(2) * offdiag).sum(dim=(1, 2)).div(offdiag.sum().clamp_min(1.0)).mean()


def batched_mmr_select(
    ucb: torch.Tensor,
    top_ids: torch.Tensor,
    pm_emb: torch.Tensor,
    budget: int,
    beta: float,
) -> torch.Tensor:
    device = ucb.device
    N, K = ucb.shape
    ucb_min = ucb.min(dim=-1, keepdim=True).values
    ucb_max = ucb.max(dim=-1, keepdim=True).values
    ucb_span = (ucb_max - ucb_min).clamp_min(1e-6)
    ucb_norm = (ucb - ucb_min) / ucb_span

    norm_emb = F.normalize(pm_emb, p=2, dim=-1)
    cand_emb = norm_emb[top_ids]
    sim_matrix = torch.bmm(cand_emb, cand_emb.transpose(1, 2))

    selected_mask = torch.zeros((N, K), dtype=torch.bool, device=device)
    max_sim = torch.zeros((N, K), device=device)

    for step in range(min(budget, K)):
        if step == 0:
            scores = (1.0 - beta) * ucb_norm
        else:
            scores = (1.0 - beta) * ucb_norm - beta * max_sim

        scores = scores.masked_fill(selected_mask, float("-inf"))
        best_indices = scores.argmax(dim=-1, keepdim=True)
        selected_mask.scatter_(1, best_indices, True)

        new_sims = sim_matrix.gather(2, best_indices.unsqueeze(-1).expand(-1, K, -1)).squeeze(-1)
        if step == 0:
            max_sim = new_sims
        else:
            max_sim = torch.max(max_sim, new_sims)

    return selected_mask



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
        layer_idx: int = 0,
        enable_single_signed_projection: bool = False,
        single_proj_dim: int = 32,
        enable_pair_jl_bilinear: bool = False,
        pair_jl_dim: int = 16,
        pair_candidate_budget: int = 64,
        projection_logit_cap: float = 0.0,
        enable_self_delta_probe: bool = False,
        enable_self_delta_choice: bool = False,
        self_delta_max_scale: float = 0.25,
        enable_vnext: bool = False,
        enable_utility_critic_probe: bool = False,
        enable_utility_critic_choice: bool = False,
        utility_pool_size: int = 16,
        utility_budget: int = 3,
        utility_mmr_beta: float = 0.35,
        utility_mmr_mode: str = "hybrid",
        utility_choice_warmup_steps: int = 50,
        mmr_controller_warmup_steps: int = 50,
        utility_choice_scale: float = 0.05,
        utility_choice_scale_max: float = 0.20,
        utility_mmr_identity_weight: float = 0.50,
        enable_scanner_feedback_memory: bool = False,
        enable_mmr_controller: bool = False,
        enable_lazy_executor: bool = False,
        enable_category_scanner: bool = False,
        enable_auto_mined_atoms: bool = False,
        utility_exploration_start_weight: float = 0.35,
        utility_exploration_end_weight: float = 0.35,
        utility_exploration_warmup_steps: int = 0,
        utility_budget_start: int = 3,
        utility_budget_end: int = 3,
        utility_budget_warmup_steps: int = 0,
        utility_category_k: int = 1,
    ) -> None:
        super().__init__()
        if state_norm not in {"none", "layernorm"}:
            raise ValueError(f"state_norm must be none or layernorm, got {state_norm}")
        self.dim = dim
        self.slots = slots
        self.top_k = top_k
        self.pm = primitive_matrix
        self.state_norm_mode = state_norm
        self.enable_single_signed_projection = enable_single_signed_projection
        self.enable_pair_jl_bilinear = enable_pair_jl_bilinear
        self.pair_candidate_budget = pair_candidate_budget
        self.projection_logit_cap = float(projection_logit_cap)
        self.enable_self_delta_probe = bool(enable_self_delta_probe)
        self.enable_self_delta_choice = bool(enable_self_delta_choice)
        self.self_delta_max_scale = float(self_delta_max_scale)
        self.enable_vnext = bool(enable_vnext)
        self.enable_utility_critic_probe = bool(enable_utility_critic_probe)
        self.enable_utility_critic_choice = bool(enable_utility_critic_choice)
        self.utility_pool_size = int(utility_pool_size)
        self.utility_budget = int(utility_budget)
        self.utility_mmr_beta = float(utility_mmr_beta)
        self.utility_mmr_mode = str(utility_mmr_mode)
        self.utility_choice_warmup_steps = int(utility_choice_warmup_steps)
        self.mmr_controller_warmup_steps = int(mmr_controller_warmup_steps)
        self.utility_choice_scale = float(utility_choice_scale)
        self.utility_choice_scale_max = float(utility_choice_scale_max)
        self.utility_mmr_identity_weight = float(utility_mmr_identity_weight)
        self.vnext_global_step = 0
        self.enable_scanner_feedback_memory = bool(enable_scanner_feedback_memory)
        self.enable_mmr_controller = bool(enable_mmr_controller)
        self.enable_lazy_executor = bool(enable_lazy_executor)
        self.enable_category_scanner = bool(enable_category_scanner)
        self.enable_auto_mined_atoms = bool(enable_auto_mined_atoms)
        self.utility_exploration_start_weight = float(utility_exploration_start_weight)
        self.utility_exploration_end_weight = float(utility_exploration_end_weight)
        self.utility_exploration_warmup_steps = int(utility_exploration_warmup_steps)
        self.utility_budget_start = int(utility_budget_start)
        self.utility_budget_end = int(utility_budget_end)
        self.utility_budget_warmup_steps = int(utility_budget_warmup_steps)
        self.utility_category_k = int(utility_category_k)

        context_dim = dim * 5
        emb_dim = primitive_matrix.emb.shape[-1]
        self.scanner = HybridScanner(
            dim=dim,
            context_dim=context_dim,
            prim_embed_dim=emb_dim,
            single_proj_dim=single_proj_dim,
            pair_jl_dim=pair_jl_dim,
            enable_scanner_feedback_memory=self.enable_scanner_feedback_memory,
            enable_category_scanner=self.enable_category_scanner,
            num_primitives=primitive_matrix.num_primitives,
            layer_idx=layer_idx,
            category_k=self.utility_category_k,
        )
        self.simulator = LowRankSimulator(dim=dim, num_primitives=primitive_matrix.num_primitives, rank=sim_rank, embed_dim=emb_dim)
        self.executor = ActionExecutor(dim=dim, primitive_matrix=primitive_matrix, enable_vnext=enable_vnext)

        self.context_logits = nn.Linear(context_dim, top_k)
        self.sim_logits = nn.Linear(dim, 1)
        self.prev_action_proj = nn.Linear(primitive_matrix.num_primitives, context_dim, bias=False)
        self.prev_active_proj = nn.Linear(1, context_dim, bias=False)
        self.prev_write_proj = nn.Linear(1, context_dim, bias=False)
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
        # Stable cell-to-primitive prior. Unlike context_logits this is indexed
        # by primitive identity, not by a candidate's changing list position.
        self.primitive_pair_bias = nn.Parameter(
            torch.zeros(slots, slots, primitive_matrix.num_primitives)
        )

        self._init_gate_priors()

        self.self_delta_field = None
        if self.enable_self_delta_probe or self.enable_self_delta_choice:
            cpu_rng = torch.random.get_rng_state()
            self.self_delta_field = SelfDeltaCandidateField(
                dim=dim,
                context_dim=context_dim,
                prim_embed_dim=emb_dim,
                hidden=max(64, dim * 2),
            )
            torch.random.set_rng_state(cpu_rng)
        self.self_delta_logit_scale = nn.Parameter(torch.tensor(-6.0))

        self.utility_critic = None
        self.utility_logit_scale = None
        if self.enable_vnext or self.enable_utility_critic_probe or self.enable_utility_critic_choice:
            cpu_rng = torch.random.get_rng_state()
            self.utility_critic = UtilityCritic(
                dim=dim,
                context_dim=context_dim,
                prim_embed_dim=emb_dim,
                hidden=128,
            )
            self.utility_logit_scale = nn.Parameter(torch.tensor(-6.0))
            torch.random.set_rng_state(cpu_rng)

        self.slot_output_weight = None
        self.collector_weight = None
        if self.enable_vnext:
            cpu_rng = torch.random.get_rng_state()
            self.slot_output_weight = nn.Parameter(torch.tensor(0.10))
            self.collector_weight = nn.Parameter(torch.tensor(0.02))
            self.routing_head = nn.Sequential(
                nn.Linear(5 * dim, dim // 2),
                nn.SiLU(),
                nn.Linear(dim // 2, 3),
            )
            nn.init.zeros_(self.routing_head[-1].weight)
            nn.init.constant_(self.routing_head[-1].bias, 0.0)
            torch.random.set_rng_state(cpu_rng)

        self.grad_credit_queue = []


    def set_vnext_step(self, step: int) -> None:
        self.vnext_global_step = int(step)

    def _warmup_progress(self, steps: int) -> float:
        if not self.training or int(steps) <= 0:
            return 1.0
        return max(0.0, min(1.0, float(self.vnext_global_step) / float(max(1, int(steps)))))

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

    def _projection_candidate_column(
        self,
        indices: torch.Tensor,
        signed_scores: torch.Tensor,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collapse bounded global proposals to one signed proposal per cell."""
        cells = self.slots * self.slots
        primitives = self.pm.num_primitives
        total = cells * primitives
        if indices.numel() == 0:
            ids = torch.zeros(batch * cells, 1, dtype=torch.long, device=indices.device)
            neg = torch.full((batch * cells, 1), float("-inf"), device=indices.device)
            return ids, neg, neg
        matches = F.one_hot(indices.to(torch.long), num_classes=total).to(torch.bool)
        signed_scores = signed_scores.float()
        ranked = torch.where(
            matches,
            signed_scores.abs()[:, None],
            torch.full((), float("-inf"), device=indices.device),
        )
        best_row = ranked.argmax(dim=0)
        best_abs = ranked.max(dim=0).values
        best_signed = signed_scores[best_row]
        abs_grid = best_abs.view(cells, primitives)
        primitive = abs_grid.argmax(dim=-1)
        rank_score = abs_grid.gather(1, primitive[:, None]).squeeze(1)
        signed_grid = best_signed.view(cells, primitives)
        choice_score = signed_grid.gather(1, primitive[:, None]).squeeze(1)
        ids = primitive[None].expand(batch, -1).reshape(batch * cells, 1)
        rank = rank_score[None].expand(batch, -1).reshape(batch * cells, 1)
        choice = choice_score[None].expand(batch, -1).reshape(batch * cells, 1)
        return ids, rank, choice

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
        choice_sampling: str = "auto",
        primitive_ablation_ids: Optional[torch.Tensor] = None,
        primitive_override_ids: Optional[torch.Tensor] = None,
        collect_scan_metrics: bool = True,
        projection_target_b: Optional[torch.Tensor] = None,
        disable_self_delta: bool = False,
        zero_self_delta: bool = False,
        shuffle_self_delta: bool = False,
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
            prev_action = prev_context.get("action_dist")
            prev_active = prev_context.get("active_mass")
            prev_write = prev_context.get("write_mass")
            if prev_action is not None:
                mix_parts.append(self.prev_action_proj(prev_action))
            if prev_active is not None:
                mix_parts.append(self.prev_active_proj(prev_active))
            if prev_write is not None:
                mix_parts.append(self.prev_write_proj(prev_write))
            if mix_parts:
                prev_context_mix = sum(mix_parts)
                listen_gate = torch.sigmoid(self.listen_gate(prev_context_mix))
                context = context + (listen_gate[:, None, None, :] * prev_context_mix[:, None, None, :])

        flat_context = context.reshape(b * s * s, -1)
        flat_src = src.reshape(b * s * s, d)
        flat_tgt = tgt.reshape(b * s * s, d)
        flat_mem = mem.reshape(b * s * s, d)

        prev_action_emb = None
        if prev_context is not None and prev_context.get("action_dist") is not None:
            prev_action_emb = prev_context["action_dist"] @ self.pm.emb
            prev_action_emb = (
                prev_action_emb[:, None, :]
                .expand(b, s * s, -1)
                .reshape(b * s * s, -1)
            )
        full_scan = self.top_k >= self.pm.num_primitives
        cell_ids = torch.arange(s * s, device=state.device).repeat(b)
        cand_ids, proposal_logits, source_ids, scan_metrics = self.scanner(
            flat_context,
            flat_mem,
            self.pm,
            prev_action_emb=prev_action_emb,
            ensure_all_candidates=full_scan,
            collect_metrics=collect_scan_metrics,
            cell_ids=cell_ids,
        )
        proposal_rank_logits = proposal_logits
        proposal_choice_logits = proposal_logits
        all_primitive_effects = None
        projection_enabled = (
            projection_target_b is not None
            and (self.enable_single_signed_projection or self.enable_pair_jl_bilinear)
        )
        if projection_enabled:
            primitive_count = self.pm.num_primitives
            all_ids = torch.arange(primitive_count, device=state.device).view(1, -1)
            all_ids = all_ids.expand(b * s * s, -1)
            all_primitive_effects = self.executor(flat_src, flat_tgt, flat_mem, all_ids)
            effects = (
                all_primitive_effects.view(b, s * s, primitive_count, d)
                .permute(1, 2, 0, 3)
                .reshape(s * s * primitive_count, b, d)
            )
            projected = self.scanner.projection_proposals(
                effects.detach(),
                projection_target_b.detach(),
                single_top_k=min(32, effects.shape[0]),
                pair_top_k=self.pair_candidate_budget,
                enable_single=self.enable_single_signed_projection,
                enable_pair=self.enable_pair_jl_bilinear,
            )
            scan_metrics.update(projected["metrics"])
            extra_ids, extra_rank, extra_choice, extra_sources = [], [], [], []
            if self.enable_single_signed_projection:
                single_indices = projected["single_indices"]
                single_signed = projected["single"]["signed_score"][single_indices]
                ids, rank, choice_score = self._projection_candidate_column(
                    single_indices, single_signed, b
                )
                extra_ids.append(ids)
                extra_rank.append(rank.to(proposal_logits.dtype))
                extra_choice.append(choice_score.to(proposal_logits.dtype))
                extra_sources.append(torch.full_like(ids, 5))
            if self.enable_pair_jl_bilinear:
                endpoints = torch.cat([projected["pair_left"], projected["pair_right"]])
                endpoint_scores = torch.cat([
                    projected["pair_signed_score"], projected["pair_signed_score"]
                ])
                ids, rank, choice_score = self._projection_candidate_column(
                    endpoints, endpoint_scores, b
                )
                extra_ids.append(ids)
                extra_rank.append(rank.to(proposal_logits.dtype))
                extra_choice.append(choice_score.to(proposal_logits.dtype))
                extra_sources.append(torch.full_like(ids, 6))
            cand_ids = torch.cat([cand_ids, *extra_ids], dim=-1)
            source_ids = torch.cat([source_ids, *extra_sources], dim=-1)
            proposal_rank_logits = torch.cat([proposal_rank_logits, *extra_rank], dim=-1)
            proposal_choice_logits = torch.cat([proposal_choice_logits, *extra_choice], dim=-1)
        k = min(self.top_k, proposal_rank_logits.shape[-1])
        if not full_scan and k >= 4:
            # Dynamic exploration quota across all 8 sources to prevent censoring
            active_sources = []
            for source_id in range(8):
                if (source_ids == source_id).any():
                    active_sources.append(source_id)
            
            # Sort active_sources based on defined priority:
            # 0 (grid), 1 (semantic), 5 (single_proj), 6 (pair_jl), 4 (feedback), 7 (category), 3 (random), 2 (usage)
            priority = {0: 0, 1: 1, 5: 2, 6: 3, 4: 4, 7: 5, 3: 6, 2: 7}
            active_sources = sorted(active_sources, key=lambda x: priority.get(x, 99))
            
            quota_limit = min(len(active_sources), k)
            active_sources = active_sources[:quota_limit]
            
            quota_pos_list = []
            for source_id in active_sources:
                # Exclude masked -inf logits from the source count to support fallbacks
                mask = (source_ids == source_id) & (proposal_rank_logits > -1e18)
                row_has_source = mask.any(dim=-1, keepdim=True)
                logits_masked = proposal_rank_logits.masked_fill(~mask, float("-inf"))
                best_idx = logits_masked.argmax(dim=-1, keepdim=True)
                global_best = proposal_rank_logits.argmax(dim=-1, keepdim=True)
                row_best = torch.where(row_has_source, best_idx, global_best)
                quota_pos_list.append(row_best.squeeze(-1))
            
            if quota_pos_list:
                quota_pos = torch.stack(quota_pos_list, dim=-1)
            else:
                quota_pos = torch.zeros((b * s * s, 0), dtype=torch.long, device=state.device)
            
            selected = torch.zeros_like(proposal_rank_logits, dtype=torch.bool)
            selected.scatter_(1, quota_pos, True)
            remaining = proposal_rank_logits.masked_fill(selected, float("-inf")).topk(k=k, dim=-1).indices
            combined = torch.cat([quota_pos, remaining], dim=-1)
            top_pos = combined[:, :k]
            top_vals = proposal_choice_logits.gather(1, top_pos)
        else:
            top_pos = proposal_rank_logits.topk(k=k, dim=-1).indices
            top_vals = proposal_choice_logits.gather(1, top_pos)
        top_ids = cand_ids.gather(1, top_pos)
        top_source_ids = source_ids.gather(1, top_pos)

        # Collect detailed single signed projection diagnostics
        if collect_scan_metrics:
            with torch.no_grad():
                has_proj = (source_ids == 5)
                scan_metrics["single_projection_generated_count"] = float(
                    projected["single_indices"].numel() if (projection_enabled and "projected" in locals()) else 0.0
                )
                scan_metrics["single_projection_raw_candidate_count"] = scan_metrics["single_projection_generated_count"]
                scan_metrics["single_projection_pre_topk_count"] = float(has_proj.sum().item())
                
                in_top = (top_source_ids == 5)
                scan_metrics["single_projection_topk_count"] = float(in_top.sum().item())
                scan_metrics["single_projection_in_utility_pool_count"] = scan_metrics["single_projection_topk_count"]
                
                if in_top.any():
                    top_rank_logits = proposal_rank_logits.gather(1, top_pos)
                    top_choice_logits = proposal_choice_logits.gather(1, top_pos)
                    proj_top_ranks = top_rank_logits[in_top]
                    proj_top_choices = top_choice_logits[in_top]
                    
                    scan_metrics["single_projection_rank_min"] = float(proj_top_ranks.min().item())
                    scan_metrics["single_projection_rank_mean"] = float(proj_top_ranks.mean().item())
                    scan_metrics["single_projection_score_mean"] = float(proj_top_choices.mean().item())
                    scan_metrics["single_projection_score_max"] = float(proj_top_choices.max().item())
                else:
                    scan_metrics["single_projection_rank_min"] = 0.0
                    scan_metrics["single_projection_rank_mean"] = 0.0
                    scan_metrics["single_projection_score_mean"] = 0.0
                    scan_metrics["single_projection_score_max"] = 0.0

        sim, predicted_gain = self.simulator(flat_src, top_ids)

        # Dynamic schedules for budget and exploration
        if self.training and self.utility_budget_warmup_steps > 0:
            progress = max(0.0, min(1.0, float(self.vnext_global_step) / float(self.utility_budget_warmup_steps)))
            current_budget = int(round(self.utility_budget_start + progress * (self.utility_budget_end - self.utility_budget_start)))
        else:
            current_budget = self.utility_budget
        current_budget = max(1, current_budget)

        if self.training and self.utility_exploration_warmup_steps > 0:
            progress = max(0.0, min(1.0, float(self.vnext_global_step) / float(self.utility_exploration_warmup_steps)))
            current_exploration_weight = self.utility_exploration_start_weight + progress * (self.utility_exploration_end_weight - self.utility_exploration_start_weight)
        else:
            current_exploration_weight = self.utility_exploration_end_weight

        utility_score = None
        utility_uncertainty = None
        utility_behavior = None
        utility_metrics = {}
        if self.utility_critic is not None:
            # Inference-safe head/context vector: target slot address.
            # This is not label leakage. Future variants may add readout/query context.
            flat_target_address = target_address.reshape(b * s * s, d)
            top_prim_emb = self.pm.emb.to(device=flat_context.device, dtype=flat_context.dtype)[top_ids]
            utility_score, utility_uncertainty, utility_behavior = self.utility_critic(
                flat_context,
                top_prim_emb,
                flat_target_address,
                return_behavior=True,
            )
            with torch.no_grad():
                utility_metrics = {
                    "utility_score_mean": float(utility_score.mean().cpu()),
                    "utility_score_std": float(utility_score.std().cpu()),
                    "utility_behavior_norm": float(utility_behavior.norm(dim=-1).mean().cpu()),
                    "utility_critic_enabled": 1.0,
                    "utility_choice_enabled": float(self.enable_utility_critic_choice),
                    "utility_pool_size": float(self.utility_pool_size),
                    "utility_budget": float(current_budget),
                    "utility_budget_current": float(current_budget),
                    "utility_exploration_weight_current": float(current_exploration_weight),
                    "utility_mmr_beta": float(self.utility_mmr_beta),
                    "utility_mmr_mode": 1.0 if self.utility_mmr_mode == "hybrid" else 0.0,
                    "utility_choice_warmup_steps": float(self.utility_choice_warmup_steps),
                    "mmr_controller_warmup_steps": float(self.mmr_controller_warmup_steps),
                    "utility_choice_scale": float(self.utility_choice_scale),
                    "utility_choice_scale_max": float(self.utility_choice_scale_max),
                    "utility_mmr_identity_weight": float(self.utility_mmr_identity_weight),
                }

        self_delta_active = (
            self.self_delta_field is not None
            and (self.enable_self_delta_probe or self.enable_self_delta_choice)
        )
        lazy_active = self.enable_lazy_executor and not (self_delta_active and self.enable_self_delta_choice)

        if not lazy_active:
            if all_primitive_effects is None:
                primitive_out = self.executor(flat_src, flat_tgt, flat_mem, top_ids)
            else:
                primitive_out = all_primitive_effects.gather(
                    1, top_ids.unsqueeze(-1).expand(-1, -1, d)
                )
        else:
            primitive_out = None

        self_delta_component = torch.zeros_like(predicted_gain)
        self_delta_metrics = {}
        self_delta_scale = torch.zeros((), device=flat_context.device, dtype=flat_context.dtype)

        if self_delta_active and not disable_self_delta:
            top_prim_emb = self.pm.emb.to(
                device=flat_context.device,
                dtype=flat_context.dtype,
            )[top_ids]

            if not lazy_active:
                self_delta_component, self_delta_metrics = self.self_delta_field(
                    flat_context=flat_context,
                    flat_src=flat_src,
                    flat_tgt=flat_tgt,
                    top_prim_emb=top_prim_emb,
                    sim=sim,
                    actual=primitive_out,
                    detach_actual=True,
                )

                if zero_self_delta:
                    self_delta_component = torch.zeros_like(self_delta_component)


            if shuffle_self_delta and self_delta_component.shape[0] > 1:
                perm = torch.randperm(
                    self_delta_component.shape[0],
                    device=self_delta_component.device,
                )
                self_delta_component = self_delta_component[perm]

            self_delta_scale = (
                self.self_delta_max_scale
                * torch.sigmoid(self.self_delta_logit_scale)
            ).to(dtype=flat_context.dtype)

        utility_choice_available = (
            self.enable_utility_critic_choice
            and utility_score is not None
            and not disable_sim
            and not disable_sim_result
        )
        utility_choice_progress = self._warmup_progress(self.utility_choice_warmup_steps)
        mmr_controller_progress = self._warmup_progress(self.mmr_controller_warmup_steps)
        utility_replaces_legacy = bool(utility_choice_available and utility_choice_progress >= 1.0)
        sim_component = (
            torch.zeros_like(predicted_gain)
            if disable_sim or disable_sim_result or utility_replaces_legacy
            else self.sim_logits(sim).squeeze(-1)
        )
        gain_component = (
            torch.zeros_like(predicted_gain)
            if disable_sim or disable_gain or utility_replaces_legacy
            else predicted_gain
        )


        context_component = self.context_logits(flat_context)[:, :k]
        primitive_pair = self.primitive_pair_bias.to(
            dtype=flat_context.dtype, device=flat_context.device
        ).reshape(1, s * s, self.pm.num_primitives).expand(b, -1, -1)
        primitive_pair = primitive_pair.reshape(b * s * s, self.pm.num_primitives)
        primitive_pair_component = primitive_pair.gather(1, top_ids)
        proposal_component = _ln_logits(top_vals)
        projection_mask = top_source_ids >= 5
        if projection_enabled and self.projection_logit_cap > 0:
            # Only limit positive source dominance. A symmetric clamp would
            # lift strongly negative signed evidence and increase usage.
            capped = proposal_component.clamp(max=self.projection_logit_cap)
            with torch.no_grad():
                selected_projection = projection_mask
                clipped = selected_projection & (proposal_component > self.projection_logit_cap)
                scan_metrics["projection_logit_cap"] = self.projection_logit_cap
                scan_metrics["projection_logit_clipped_fraction"] = float(
                    clipped.float().sum().div(selected_projection.float().sum().clamp_min(1.0)).cpu()
                )
            proposal_component = torch.where(projection_mask, capped, proposal_component)
        choice_logits = (
            _ln_logits(context_component)
            + _ln_logits(gain_component)
            + _ln_logits(sim_component)
            + proposal_component
            + primitive_pair_component
        )
        if self.enable_self_delta_choice and not disable_self_delta:
            choice_logits = choice_logits + self_delta_scale * _ln_logits(self_delta_component)
        if utility_choice_available:
            scale_min = float(self.utility_choice_scale)
            scale_max = float(self.utility_choice_scale_max)
            schedule_scale = scale_min + utility_choice_progress * (scale_max - scale_min)
            utility_scale = (schedule_scale * torch.sigmoid(self.utility_logit_scale)).to(dtype=choice_logits.dtype)
            utility_norm = F.layer_norm(utility_score, (utility_score.shape[-1],))
            choice_logits = choice_logits + utility_scale * utility_norm
            utility_metrics["utility_choice_warmup_progress"] = float(utility_choice_progress)
            utility_metrics["utility_choice_scale_value"] = float(schedule_scale)

        choice_logits_pre_mmr = choice_logits
        candidate_log_prob_for_loss = F.log_softmax(choice_logits_pre_mmr, dim=-1)
        mmr_mask = torch.ones_like(choice_logits, dtype=torch.bool)
        mmr_active = False
        if self.enable_mmr_controller and utility_score is not None and not disable_sim and not disable_sim_result:
            # vNext contract:
            # - commit during forward uses critic/policy score;
            # - measured_gain trains later;
            # - MMR similarity uses cheap learned behavior_feature, so Lazy Executor
            #   does not need true executor effects for every candidate.
            mmr_mask, mmr_metrics = vnext_mmr_select(
                utility=utility_score,
                top_ids=top_ids,
                pm_emb=self.pm.emb,
                budget=current_budget,
                beta=mmr_controller_progress * self.utility_mmr_beta,
                behavior_feature=utility_behavior,
                mode=self.utility_mmr_mode,
                uncertainty=utility_uncertainty,
                uncertainty_weight=current_exploration_weight,
                identity_weight=self.utility_mmr_identity_weight,
            )
            mmr_active = bool((not self.training) or mmr_controller_progress >= 1.0)
            with torch.no_grad():
                for _k, _v in mmr_metrics.items():
                    utility_metrics[_k] = float(_v.detach().cpu())
                utility_metrics["mmr_controller_active"] = float(mmr_active)
                utility_metrics["mmr_controller_warmup_progress"] = float(mmr_controller_progress)
            if mmr_active:
                choice_logits = choice_logits.masked_fill(~mmr_mask, float("-inf"))

        behavior_div_loss = _behavior_decorrelation_loss(utility_behavior)
        if utility_behavior is not None:
            utility_metrics["behavior_div_loss"] = float(behavior_div_loss.detach().cpu())
            utility_metrics["behavior_decorr_loss"] = float(behavior_div_loss.detach().cpu())

        if choice_sampling not in {"auto", "gumbel", "softmax", "uniform"}:
            raise ValueError(f"unknown choice sampling: {choice_sampling!r}")
        use_gumbel = self.training and choice_sampling in {"auto", "gumbel"}
        if choice_sampling == "uniform":
            choice = torch.full_like(choice_logits, 1.0 / choice_logits.shape[-1])
        else:
            choice = (
                F.gumbel_softmax(choice_logits, tau=tau, hard=False, dim=-1)
                if use_gumbel
                else F.softmax(choice_logits / max(float(tau), 1e-4), dim=-1)
            )
        if primitive_ablation_ids is not None:
            ablate = primitive_ablation_ids.to(device=top_ids.device).reshape(b * s * s, 1)
            keep = (top_ids != ablate).to(choice.dtype)
            choice = choice * keep
            choice = choice / choice.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if primitive_override_ids is not None:
            override = primitive_override_ids.to(device=top_ids.device).reshape(b * s * s, 1)
            forced = (top_ids == override).to(choice.dtype)
            available = forced.sum(dim=-1, keepdim=True) > 0
            forced = forced / forced.sum(dim=-1, keepdim=True).clamp_min(1.0)
            choice = torch.where(available, forced, choice)

        if self.utility_critic is not None:
            with torch.no_grad():
                source_names_for_pres = ("grid", "semantic", "usage", "random", "feedback", "single_signed_projection", "pair_jl16", "category")
                for source_id, source_name in enumerate(source_names_for_pres):
                    pool_pres = (top_source_ids == source_id).any(dim=-1).float().mean()
                    utility_metrics[f"source_pool_presence_{source_name}"] = float(pool_pres.cpu())
                    
                    if "mmr_mask" in locals() and mmr_mask is not None:
                        mmr_pres = ((top_source_ids == source_id) & mmr_mask).any(dim=-1).float().mean()
                        utility_metrics[f"source_after_mmr_presence_{source_name}"] = float(mmr_pres.cpu())
                    else:
                        utility_metrics[f"source_after_mmr_presence_{source_name}"] = 0.0
                        
                    choice_pres = (choice * (top_source_ids == source_id).to(choice.dtype)).sum(dim=-1).mean()
                    utility_metrics[f"source_after_choice_presence_{source_name}"] = float(choice_pres.cpu())

        source_names = ("grid", "semantic", "usage", "random", "feedback", "single_signed_projection", "pair_jl16", "category")
        if collect_scan_metrics:
            with torch.no_grad():
                for source_id, source_name in enumerate(source_names):
                    source_mass = (choice * (top_source_ids == source_id).to(choice.dtype)).sum(dim=-1).mean()
                    scan_metrics[f"{source_name}_candidate_usage"] = float(source_mass.cpu())
                    source_coverage = (top_source_ids == source_id).any(dim=-1).float().mean()
                    scan_metrics[f"{source_name}_candidate_coverage"] = float(source_coverage.cpu())
                scan_metrics["scanner_source_mass_sum"] = float(
                    sum(scan_metrics[f"{name}_candidate_usage"] for name in source_names)
                )
                scan_metrics["single_signed_projection_usage"] = scan_metrics[
                    "single_signed_projection_candidate_usage"
                ]
                scan_metrics["pair_jl16_usage"] = scan_metrics[
                    "pair_jl16_candidate_usage"
                ]
        elif projection_enabled:
            with torch.no_grad():
                scan_metrics["single_signed_projection_usage"] = float(
                    (choice * (top_source_ids == 5).to(choice.dtype)).sum(dim=-1).mean().cpu()
                )
                scan_metrics["pair_jl16_usage"] = float(
                    (choice * (top_source_ids == 6).to(choice.dtype)).sum(dim=-1).mean().cpu()
                )

        # primitive_out already computed before choice logits for optional self-delta diagnostics.
        if lazy_active:
            if self.enable_mmr_controller and utility_score is not None and not disable_sim and not disable_sim_result and mmr_active:
                selected_mask = mmr_mask
            else:
                _, top_b_idx = choice.topk(k=current_budget, dim=-1)
                selected_mask = torch.zeros_like(choice, dtype=torch.bool).scatter_(1, top_b_idx, True)

            selected_pos = selected_mask.nonzero(as_tuple=False)[:, 1].reshape(b * s * s, current_budget)
            lazy_top_ids = top_ids.gather(1, selected_pos)

            if all_primitive_effects is None:
                lazy_primitive_out = self.executor(flat_src, flat_tgt, flat_mem, lazy_top_ids)
            else:
                lazy_primitive_out = all_primitive_effects.gather(
                    1, lazy_top_ids.unsqueeze(-1).expand(-1, -1, d)
                )

            lazy_choice = choice.gather(1, selected_pos)
            lazy_choice = lazy_choice / lazy_choice.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            transformed = (lazy_choice.unsqueeze(-1) * lazy_primitive_out).sum(dim=1)

            if self_delta_active and not disable_self_delta:
                lazy_top_prim_emb = top_prim_emb.gather(1, selected_pos.unsqueeze(-1).expand(-1, -1, top_prim_emb.shape[-1]))
                lazy_sim = sim.gather(1, selected_pos.unsqueeze(-1).expand(-1, -1, sim.shape[-1]))
                self_delta_score, self_delta_metrics = self.self_delta_field(
                    flat_context=flat_context,
                    flat_src=flat_src,
                    flat_tgt=flat_tgt,
                    top_prim_emb=lazy_top_prim_emb,
                    sim=lazy_sim,
                    actual=lazy_primitive_out,
                    detach_actual=True,
                )
                self_delta_component = torch.zeros_like(predicted_gain)
                self_delta_component.scatter_(1, selected_pos, self_delta_score)
                if zero_self_delta:
                    self_delta_component = torch.zeros_like(self_delta_component)
        else:
            transformed = (choice.unsqueeze(-1) * primitive_out).sum(dim=1)

        # Online Gradient Trace Credit Hook
        if self.training and self.utility_critic is not None:
            ctx_det = flat_context.detach().cpu()
            head_det = flat_target_address.detach().cpu()
            top_prim_emb_val = self.pm.emb.to(device=flat_context.device, dtype=flat_context.dtype)[top_ids]
            if lazy_active:
                emb_det = top_prim_emb_val.gather(1, selected_pos.unsqueeze(-1).expand(-1, -1, top_prim_emb_val.shape[-1])).detach().cpu()
                cand_det = top_ids.gather(1, selected_pos).detach().cpu()
                eff_det = lazy_primitive_out.detach()
                
                def make_hook(c_det, e_det, h_det, cand_det, eff_det):
                    def backward_hook(grad):
                        if grad is not None:
                            grad_credit = - (grad * eff_det).sum(dim=-1)
                            self.grad_credit_queue.append((c_det, e_det, h_det, cand_det, grad_credit.detach().cpu()))
                            if len(self.grad_credit_queue) > 10:
                                self.grad_credit_queue.pop(0)
                        return grad
                    return backward_hook
                
                lazy_primitive_out.register_hook(make_hook(ctx_det, emb_det, head_det, cand_det, eff_det))
            else:
                emb_det = top_prim_emb_val.detach().cpu()
                cand_det = top_ids.detach().cpu()
                eff_det = primitive_out.detach()
                
                def make_hook(c_det, e_det, h_det, cand_det, eff_det):
                    def backward_hook(grad):
                        if grad is not None:
                            grad_credit = - (grad * eff_det).sum(dim=-1)
                            self.grad_credit_queue.append((c_det, e_det, h_det, cand_det, grad_credit.detach().cpu()))
                            if len(self.grad_credit_queue) > 10:
                                self.grad_credit_queue.pop(0)
                        return grad
                    return backward_hook
                
                primitive_out.register_hook(make_hook(ctx_det, emb_det, head_det, cand_det, eff_det))

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
        slot_alive_logits = self.slot_alive_head(next_state).squeeze(-1)
        split_count = F.softmax(self.split_head(next_state), dim=-1)
        child_gate = torch.sigmoid(self.child_gate(next_state)).squeeze(-1)
        merge_gate = torch.sigmoid(self.merge_gate(next_state)).squeeze(-1)
        slot_output_gate = torch.sigmoid(self.output_gate(next_state)).squeeze(-1)
        collector_mass = slot_output_gate * merge_gate

        if self.enable_vnext:
            routing_logits = self.routing_head(flat_context)
            w_tape, w_slot, w_coll = torch.sigmoid(routing_logits).chunk(3, dim=-1)
            w_tape = w_tape * 2.0
            w_slot = w_slot * 0.20
            w_coll = w_coll * 0.04

            cell_tape_weight = cell_out_gate * active * mode[:, 0:1] * w_tape
            cell_tape = (cell_tape_weight * transformed).view(b, s, s, d)
            denom = cell_tape_weight.view(b, s, s, 1).sum(dim=(1, 2)).clamp_min(1e-5)
            output_tape_state = cell_tape.sum(dim=(1, 2)) / denom

            w_slot_j = w_slot.view(b, s, s, 1).sum(dim=1)
            slot_output_state = (
                (slot_output_gate * torch.sigmoid(slot_alive_logits)).unsqueeze(-1) * w_slot_j * next_state
            ).sum(dim=1) / ((slot_output_gate * torch.sigmoid(slot_alive_logits)).unsqueeze(-1) * w_slot_j).sum(dim=1, keepdim=True).squeeze(-1).clamp_min(1e-5)

            w_coll_j = w_coll.view(b, s, s, 1).sum(dim=1)
            collector_state = (w_coll_j * collector_mass.unsqueeze(-1) * next_state).sum(dim=1)

            output_state = output_tape_state + slot_output_state + collector_state
        else:
            cell_tape_weight = cell_out_gate * active * mode[:, 0:1]
            cell_tape = (cell_tape_weight * transformed).view(b, s, s, d)
            denom = cell_tape_weight.view(b, s, s, 1).sum(dim=(1, 2)).clamp_min(1e-5)
            output_tape_state = cell_tape.sum(dim=(1, 2)) / denom

            slot_output_state = (
                (slot_output_gate * torch.sigmoid(slot_alive_logits)).unsqueeze(-1) * next_state
            ).sum(dim=1) / (slot_output_gate * torch.sigmoid(slot_alive_logits)).sum(dim=1, keepdim=True).clamp_min(1e-5)

            output_state = output_tape_state + 0.10 * slot_output_state + 0.02 * (collector_mass.unsqueeze(-1) * next_state).sum(dim=1)


        chosen = top_ids.gather(1, choice.argmax(dim=-1, keepdim=True)).squeeze(1)

        trace: Dict[str, object] = {
            "candidate_ids": top_ids.detach(),
            "candidate_source_ids": top_source_ids.detach(),
            "choice": choice.detach(),
            # Losses need the live distribution; reports intentionally use the
            # detached `choice` field above.
            "choice_for_loss": choice,
            "candidate_log_prob_for_loss": candidate_log_prob_for_loss,
            "selected_mask_for_loss": mmr_mask.detach(),
            "behavior_div_loss_for_loss": behavior_div_loss,
            "chosen": chosen.detach(),
            "predicted_gain": predicted_gain.detach(),
            "predicted_gain_for_loss": predicted_gain,
            "scanner_anchor_logits_for_loss": self.scanner.anchor(flat_context),
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
            "primitive_pair_bias": self.primitive_pair_bias.detach(),
            "scan_metrics": scan_metrics,
            "self_delta_component": self_delta_component.detach(),
            "self_delta_scale": self_delta_scale.detach(),
            "self_delta_enabled": torch.tensor(
                float(self_delta_active),
                device=flat_context.device,
            ).detach(),
            "self_delta_choice_enabled": torch.tensor(
                float(self.enable_self_delta_choice),
                device=flat_context.device,
            ).detach(),
            "self_delta_metrics": {
                k: v.detach() for k, v in self_delta_metrics.items()
            },
            "pm_emb": self.pm.emb.detach(),
            "curriculum_mode": curriculum_mode,
            "choice_sampling": "uniform" if choice_sampling == "uniform" else ("gumbel" if use_gumbel else "softmax"),
        }
        if utility_score is not None:
            trace["utility_for_loss"] = utility_score
            trace["uncertainty_for_loss"] = utility_uncertainty
            trace["utility_metrics"] = utility_metrics

        new_memory = 0.95 * memory + 0.05 * next_state.mean(dim=1)
        return next_state, output_state, new_memory, trace

    def get_and_clear_grad_credits(self):
        ready = list(self.grad_credit_queue)
        self.grad_credit_queue.clear()
        return ready


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
        enable_single_signed_projection: bool = False,
        single_proj_dim: int = 32,
        enable_pair_jl_bilinear: bool = False,
        pair_jl_dim: int = 16,
        pair_candidate_budget: int = 64,
        projection_logit_cap: float = 0.0,
        enable_self_delta_probe: bool = False,
        enable_self_delta_choice: bool = False,
        self_delta_max_scale: float = 0.25,
        enable_vnext: bool = False,
        enable_utility_critic_probe: bool = False,
        enable_utility_critic_choice: bool = False,
        utility_pool_size: int = 16,
        utility_budget: int = 3,
        utility_mmr_beta: float = 0.35,
        utility_mmr_mode: str = "hybrid",
        utility_choice_warmup_steps: int = 50,
        mmr_controller_warmup_steps: int = 50,
        utility_choice_scale: float = 0.05,
        utility_choice_scale_max: float = 0.20,
        utility_mmr_identity_weight: float = 0.50,
        enable_scanner_feedback_memory: bool = False,
        enable_mmr_controller: bool = False,
        enable_lazy_executor: bool = False,
        enable_category_scanner: bool = False,
        enable_auto_mined_atoms: bool = False,
        utility_exploration_start_weight: float = 0.35,
        utility_exploration_end_weight: float = 0.35,
        utility_exploration_warmup_steps: int = 0,
        utility_budget_start: int = 3,
        utility_budget_end: int = 3,
        utility_budget_warmup_steps: int = 0,
        utility_category_k: int = 1,
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
        self.pm = PrimitiveMatrix5x5(
            embed_dim=32,
            enable_vnext=enable_vnext,
            num_layers=layers,
            enable_scanner_feedback_memory=enable_scanner_feedback_memory,
            slots=slots,
        )
        self.layers = nn.ModuleList([
            ActionMatrixLayer(
                dim,
                slots,
                self.pm,
                top_k=top_k,
                sim_rank=sim_rank,
                state_norm=state_norm,
                layer_idx=i,
                enable_single_signed_projection=enable_single_signed_projection,
                single_proj_dim=single_proj_dim,
                enable_pair_jl_bilinear=enable_pair_jl_bilinear,
                pair_jl_dim=pair_jl_dim,
                pair_candidate_budget=pair_candidate_budget,
                projection_logit_cap=projection_logit_cap,
                enable_self_delta_probe=enable_self_delta_probe,
                enable_self_delta_choice=enable_self_delta_choice,
                self_delta_max_scale=self_delta_max_scale,
                enable_vnext=enable_vnext,
                enable_utility_critic_probe=enable_utility_critic_probe,
                enable_utility_critic_choice=enable_utility_critic_choice,
                utility_pool_size=utility_pool_size,
                utility_budget=utility_budget,
                utility_mmr_beta=utility_mmr_beta,
                utility_mmr_mode=utility_mmr_mode,
                utility_choice_warmup_steps=utility_choice_warmup_steps,
                mmr_controller_warmup_steps=mmr_controller_warmup_steps,
                utility_choice_scale=utility_choice_scale,
                utility_choice_scale_max=utility_choice_scale_max,
                utility_mmr_identity_weight=utility_mmr_identity_weight,
                enable_scanner_feedback_memory=enable_scanner_feedback_memory,
                enable_mmr_controller=enable_mmr_controller,
                enable_lazy_executor=enable_lazy_executor,
                enable_category_scanner=enable_category_scanner,
                enable_auto_mined_atoms=enable_auto_mined_atoms,
                utility_exploration_start_weight=utility_exploration_start_weight,
                utility_exploration_end_weight=utility_exploration_end_weight,
                utility_exploration_warmup_steps=utility_exploration_warmup_steps,
                utility_budget_start=utility_budget_start,
                utility_budget_end=utility_budget_end,
                utility_budget_warmup_steps=utility_budget_warmup_steps,
                utility_category_k=utility_category_k,
            )
            for i in range(layers)
        ])
        self.input_norm = nn.LayerNorm(dim) if input_norm == "layernorm" else nn.Identity()
        self.layer_read_logits = nn.Parameter(torch.zeros(layers))
        self.classifier = nn.Linear(dim, classes)
        self.vnext_global_step = 0

    def set_vnext_step(self, step: int) -> None:
        self.vnext_global_step = int(step)
        for layer in self.layers:
            if hasattr(layer, "set_vnext_step"):
                layer.set_vnext_step(step)

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

    def _projection_task_signal(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic pseudo-label CE-gradient magnitude, without task labels."""
        pooled = state.mean(dim=1)
        if isinstance(self.classifier, nn.Sequential):
            logits = pooled
            for module in self.classifier:
                if not isinstance(module, nn.Dropout):
                    logits = module(logits)
        else:
            logits = self.classifier(pooled)
        confidence = logits.float().softmax(dim=-1).max(dim=-1).values
        return (1.0 - confidence).detach()

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
        choice_sampling: str = "auto",
        primitive_ablation_ids: Optional[list[Optional[torch.Tensor]]] = None,
        primitive_override_ids: Optional[list[Optional[torch.Tensor]]] = None,
        collect_scan_metrics: bool = True,
        disable_self_delta: bool = False,
        zero_self_delta: bool = False,
        shuffle_self_delta: bool = False,
    ):
        b = x.shape[0]
        s = self.slots
        state = self.input_norm(x)
        memory = state.mean(dim=1)
        slot_address = self.slot_embed.to(dtype=state.dtype, device=state.device)
        if disable_slot_address:
            slot_address = torch.zeros_like(slot_address)

        projection_target_b = None
        if any(
            layer.enable_single_signed_projection or layer.enable_pair_jl_bilinear
            for layer in self.layers
        ):
            with torch.no_grad():
                projection_target_b = self._projection_task_signal(state)

        outputs = []
        traces = []
        prev_context: Optional[Dict[str, torch.Tensor]] = None
        for idx, layer in enumerate(self.layers):
            # Previous-layer actions/activity/writes are internal recurrent state,
            # not an external oracle hint. They must remain available in deploy.
            layer_prev_context = prev_context
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
                choice_sampling=choice_sampling,
                primitive_ablation_ids=(
                    primitive_ablation_ids[idx]
                    if primitive_ablation_ids is not None and idx < len(primitive_ablation_ids)
                    else None
                ),
                primitive_override_ids=(
                    primitive_override_ids[idx]
                    if primitive_override_ids is not None and idx < len(primitive_override_ids)
                    else None
                ),
                collect_scan_metrics=collect_scan_metrics,
                projection_target_b=projection_target_b,
                disable_self_delta=disable_self_delta,
                zero_self_delta=zero_self_delta,
                shuffle_self_delta=shuffle_self_delta,
            )
            if ablate_state_after is not None and idx == ablate_state_after:
                state = torch.zeros_like(state)
                memory = torch.zeros_like(memory)
            if ablate_layer_output is not None and idx == ablate_layer_output:
                out = torch.zeros_like(out)
            outputs.append(out)
            traces.append(tr)

            cand = tr["candidate_ids"]
            choice = tr["choice_for_loss"]
            action_dist = _primitive_distribution(cand, choice, self.pm.num_primitives)
            action_dist = action_dist.view(b, s * s, -1).mean(dim=1)
            active = (
                tr["edge_for_loss"] * tr["write_for_loss"] * tr["phase_for_loss"]
            ).view(b, s * s)
            write_mass = (
                tr["edge_for_loss"]
                * tr["write_for_loss"]
                * tr["phase_for_loss"]
                * (tr["mode_for_loss"][:, 0:1] + tr["mode_for_loss"][:, 1:2])
            ).view(b, s * s)
            prev_context = {
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
            "choice_sampling": choice_sampling,
        }
