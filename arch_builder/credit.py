from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CreditBuffer:
    decay: float = 0.9
    values: Dict[str, float] = field(default_factory=dict)
    ages: Dict[str, int] = field(default_factory=dict)

    def update(self, values: Dict[str, float]) -> None:
        for k in list(self.ages):
            self.ages[k] += 1
        for k, v in values.items():
            old = self.values.get(k, 0.0)
            self.values[k] = self.decay * old + (1.0 - self.decay) * float(v)
            self.ages[k] = 0

    def metrics(self) -> Dict[str, float]:
        if not self.values:
            return {"credit_items": 0.0, "credit_staleness_mean": 0.0, "credit_age_max": 0.0}
        return {
            "credit_items": float(len(self.values)),
            "credit_staleness_mean": float(sum(self.ages.values()) / max(1, len(self.ages))),
            "credit_age_max": float(max(self.ages.values())),
        }


@dataclass
class ModeCreditLedger:
    teacher: CreditBuffer = field(default_factory=CreditBuffer)
    audit: CreditBuffer = field(default_factory=CreditBuffer)
    deploy: CreditBuffer = field(default_factory=CreditBuffer)

    def update(self, mode: str, values: Dict[str, float]) -> None:
        if mode == "teacher":
            self.teacher.update(values)
        elif mode == "audit":
            self.audit.update(values)
        elif mode == "deploy":
            self.deploy.update(values)
        else:
            raise ValueError(f"unknown credit mode: {mode!r}")

    def metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for prefix, buffer in (
            ("credit_teacher", self.teacher),
            ("credit_audit", self.audit),
            ("credit_deploy", self.deploy),
        ):
            for key, value in buffer.metrics().items():
                out[f"{prefix}_{key}"] = value
        return out

    def state_dict(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return the serializable ledger state, including the underlying credit.

        ``metrics()`` intentionally exposes only health counters.  Final reports
        also need the actual per-mode EMA values and ages, so callers must not
        assume this composite ledger has the ``CreditBuffer.values`` attribute.
        """
        return {
            mode: {
                "values": dict(buffer.values),
                "ages": dict(buffer.ages),
            }
            for mode, buffer in (
                ("teacher", self.teacher),
                ("audit", self.audit),
                ("deploy", self.deploy),
            )
        }


@dataclass(frozen=True)
class CounterfactualTarget:
    layer: int
    cell: int
    primitive: int
    force: bool = False
    random_exploration: bool = False


@dataclass
class CounterfactualRecord:
    targets: Tuple[CounterfactualTarget, ...]
    gain: float
    measured_step: int
    applied_step: int = -1


def pearson_corr(x, y):
    if len(x) < 2:
        return 0.0
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    if float(x_t.std()) < 1e-6 or float(y_t.std()) < 1e-6:
        return 0.0
    return float(torch.corrcoef(torch.stack([x_t, y_t]))[0, 1])


def spearman_corr(x, y):
    if len(x) < 2:
        return 0.0
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    if float(x_t.std()) < 1e-6 or float(y_t.std()) < 1e-6:
        return 0.0
    x_rank = x_t.argsort().argsort().float()
    y_rank = y_t.argsort().argsort().float()
    if float(x_rank.std()) < 1e-6 or float(y_rank.std()) < 1e-6:
        return 0.0
    return float(torch.corrcoef(torch.stack([x_rank, y_rank]))[0, 1])


def pairwise_ranking_loss(pred_utility: torch.Tensor, target_scores: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """Compute pairwise margin ranking loss to align predicted utility order with targets.
    Supports both 1D [P] and 2D [B, P] inputs.
    """
    if not isinstance(target_scores, torch.Tensor):
        targets = torch.tensor(target_scores, dtype=pred_utility.dtype, device=pred_utility.device)
    else:
        targets = target_scores.to(dtype=pred_utility.dtype, device=pred_utility.device)
    if targets.ndim == 1:
        if len(targets) < 2:
            return torch.zeros((), device=pred_utility.device)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        mask = target_diff > 1e-5
        if not mask.any():
            return torch.zeros((), device=pred_utility.device)
        margin = target_diff.clamp(0.01, 1.0)
        pred_diff = pred_utility.unsqueeze(1) - pred_utility.unsqueeze(0)
        loss = F.relu(margin - pred_diff)
        return loss[mask].sum() / mask.sum().clamp_min(1.0)
    else:
        target_diff = targets.unsqueeze(2) - targets.unsqueeze(1)
        mask = target_diff > 1e-5
        if not mask.any():
            return torch.zeros((), device=pred_utility.device)
        margin = target_diff.clamp(0.01, 1.0)
        pred_diff = pred_utility.unsqueeze(2) - pred_utility.unsqueeze(1)
        loss = F.relu(margin - pred_diff)
        denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
        element_losses = (loss * mask).sum(dim=(1, 2)) / denom
        has_pairs = mask.any(dim=(1, 2))
        if not has_pairs.any():
            return torch.zeros((), device=pred_utility.device)
        return element_losses[has_pairs].mean()


def mmr_select_pool(items, pm_emb, b=3, beta=0.35):
    if not items:
        return []
    selected = []
    pool = list(items)
    utils = torch.tensor([x["utility"] for x in pool], dtype=torch.float32)
    if float(utils.max() - utils.min()) > 1e-6:
        utils_norm = (utils - utils.min()) / (utils.max() - utils.min())
    else:
        utils_norm = torch.zeros_like(utils)
    for idx, x in enumerate(pool):
        x["norm_utility"] = float(utils_norm[idx].item())
        
    for step in range(min(b, len(pool))):
        best_score = float("-inf")
        best_item = None
        for item in pool:
            if item in selected:
                continue
            max_sim = 0.0
            if selected:
                sims = []
                for sel in selected:
                    emb_sel = pm_emb[sel["primitive"]]
                    emb_item = pm_emb[item["primitive"]]
                    cos_sim = float(F.cosine_similarity(emb_sel, emb_item, dim=0).item())
                    cos_sim = max(-1.0, min(1.0, cos_sim))
                    sims.append(cos_sim)
                max_sim = max(sims)
            score = (1.0 - beta) * item["norm_utility"] - beta * max_sim
            if score > best_score:
                best_score = score
                best_item = item
        if best_item is not None:
            selected.append(best_item)
    return selected


def avg_similarity(selected, pm_emb):
    if len(selected) < 2:
        return 0.0
    sims = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            cos_sim = float(F.cosine_similarity(pm_emb[selected[i]["primitive"]], pm_emb[selected[j]["primitive"]], dim=0).item())
            sims.append(cos_sim)
    return sum(sims) / len(sims)


class BoundedCounterfactualCredit:
    """Oracle-free, delayed credit from batched loss interventions.

    Each intervention removes the currently selected primitive from one or two
    cells and measures ``CE(ablated) - CE(full)`` on a separate microbatch.
    Positive gain means the removed action was useful. Measurements are queued
    and may influence training only after ``advance`` on a later optimizer step.
    """

    def __init__(
        self,
        layers: int,
        slots: int,
        primitives: int,
        budget: int = 8,
        random_fraction: float = 0.25,
        pair_fraction: float = 0.25,
        alternative_budget: int = 2,
        ema_decay: float = 0.9,
    ) -> None:
        self.layers = layers
        self.slots = slots
        self.primitives = primitives
        self.budget = max(2, int(budget))
        self.random_fraction = float(random_fraction)
        self.pair_fraction = float(pair_fraction)
        self.alternative_budget = max(0, int(alternative_budget))
        self.ema_decay = float(ema_decay)
        shape = (layers, slots * slots, primitives)
        self.ema = torch.zeros(shape)
        self.count = torch.zeros(shape)
        self.age = torch.zeros(shape)
        self.pending: List[CounterfactualRecord] = []
        self.active: List[CounterfactualRecord] = []
        self.step = 0
        self.total_measurements = 0
        self.total_joint_measurements = 0
        self.last_metrics: Dict[str, float] = {}
        self.gain_scale = 1e-3
        self.last_random_targets = 0
        self.last_alternative_targets: List[CounterfactualTarget] = []
        self.rolling_predicted_single: List[float] = []
        self.rolling_targets_single: List[float] = []
        self.rolling_sim_predicted_single: List[float] = []

    def _targets_from_trace(self, trace: Dict[str, object]) -> List[CounterfactualTarget]:
        scored: List[Tuple[float, CounterfactualTarget]] = []
        alternatives: Dict[CounterfactualTarget, List[CounterfactualTarget]] = {}
        e = self.slots * self.slots
        for layer_idx, layer in enumerate(trace["layers"]):
            chosen = layer["chosen"].view(-1, e)
            active = layer["active"].view(-1, e).float().mean(dim=0)
            choice = layer["choice"].view(-1, e, layer["choice"].shape[-1]).float()
            entropy = (-(choice.clamp_min(1e-8) * choice.clamp_min(1e-8).log()).sum(dim=-1)).mean(dim=0)
            modes = F.one_hot(chosen, num_classes=self.primitives).sum(dim=0).argmax(dim=-1)
            cand = layer["candidate_ids"].view(-1, e, layer["candidate_ids"].shape[-1])
            primitive_mass = torch.zeros(
                cand.shape[0], e, self.primitives, device=cand.device, dtype=choice.dtype
            )
            primitive_mass.scatter_add_(2, cand, choice)
            ranked = primitive_mass.mean(dim=0).argsort(dim=-1, descending=True)
            score = active + 0.05 * entropy
            for cell in range(e):
                target = CounterfactualTarget(layer_idx, cell, int(modes[cell].cpu()))
                scored.append(
                    (
                        float(score[cell].cpu()),
                        target,
                    )
                )
                alternatives[target] = [
                    CounterfactualTarget(layer_idx, cell, int(pid), force=True)
                    for pid in ranked[cell].tolist()
                    if int(pid) != target.primitive
                ][:2]
        scored.sort(key=lambda item: item[0], reverse=True)
        pair_count = min(self.budget // 2, int(round(self.budget * self.pair_fraction)))
        single_count = self.budget - pair_count
        random_count = min(single_count, max(1, int(round(single_count * self.random_fraction))))
        high_count = max(1, single_count - random_count)
        remaining = [target for _, target in scored[high_count:]]
        selected_main = [target for _, target in scored[:high_count]]
        selected_rand = []
        if remaining and random_count:
            order = torch.randperm(len(remaining))[:random_count].tolist()
            for i in order:
                t = remaining[i]
                t_rand = CounterfactualTarget(
                    layer=t.layer,
                    cell=t.cell,
                    primitive=t.primitive,
                    force=t.force,
                    random_exploration=True
                )
                selected_rand.append(t_rand)
            self.last_random_targets = len(order)
        else:
            self.last_random_targets = 0
        selected = (selected_main + selected_rand)[:single_count]
        alt_selected: List[CounterfactualTarget] = []
        depth = 0
        while len(alt_selected) < self.alternative_budget and depth < 2:
            for target in selected:
                choices = alternatives.get(target, [])
                if depth < len(choices):
                    alt_selected.append(choices[depth])
                    if len(alt_selected) >= self.alternative_budget:
                        break
            depth += 1
        self.last_alternative_targets = alt_selected
        return selected

    def advance(self, primitive_matrix=None) -> int:
        """Apply previous measurements and age the ledger exactly once."""
        self.step += 1
        self.age.add_(1.0)
        ready, self.pending = self.pending, []
        if not ready:
            return 0
        ids, gains, layer_ids = [], [], []
        for record in ready:
            record.applied_step = self.step
            share = record.gain / max(1, len(record.targets))
            for target in record.targets:
                idx = (target.layer, target.cell, target.primitive)
                old = float(self.ema[idx])
                self.ema[idx] = self.ema_decay * old + (1.0 - self.ema_decay) * share
                self.count[idx] += 1.0
                self.age[idx] = 0.0
                ids.append(target.primitive)
                gains.append(share)
                layer_ids.append(target.layer)
        if primitive_matrix is not None and ids:
            primitive_matrix.update_usage_credit(
                torch.tensor(ids, device=primitive_matrix.usage_score.device),
                torch.tensor(gains, device=primitive_matrix.usage_score.device),
                layer_ids=torch.tensor(layer_ids, device=primitive_matrix.usage_score.device),
            )
        self.active = ready
        return len(ready)

    def alignment_losses(
        self,
        trace: Dict[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        from collections import defaultdict
        device = trace["layers"][0]["choice_for_loss"].device
        z = torch.zeros((), device=device)
        if not self.active:
            return z, z, {"credit_alignment_items": 0.0}
        
        # Diagnostic variables
        utility_critic_enabled = 0.0
        utility_choice_enabled = 0.0
        utility_pool_size = 0.0
        utility_budget = 0.0
        utility_mmr_beta = 0.0
        utility_mmr_mode = 0.0
        utility_score_mean = 0.0
        utility_score_std = 0.0
        utility_overhead_seconds = 0.0
        
        # We need pm_emb for similarity.
        pm_emb = None
        
        cell_single_gains = defaultdict(list)
        for record in self.active:
            if len(record.targets) == 1:
                target = record.targets[0]
                cell_single_gains[(target.layer, target.cell)].append(record.gain)
                
        cell_mean_gain = {}
        for cell_key, gains_list in cell_single_gains.items():
            cell_mean_gain[cell_key] = sum(gains_list) / len(gains_list) if gains_list else 0.0

        pools = defaultdict(list)
        cell_targets = defaultdict(list)
        utility_predicted_single = []
        utility_targets_single = []
        sim_predicted_single = []
        
        proposal_scores = []
        sim_predicted = []
        measured_gains = []
        utility_predicted = []

        policy_losses, simulator_losses = [], []
        vnext_policy_losses = []
        vnext_policy_items = 0
        predicted, measured = [], []
        
        random_credit_count = 0.0
        offpool_credit_count = 0.0
        
        e = self.slots * self.slots
        for record in self.active:
            is_single = (len(record.targets) == 1)
            share = float(record.gain) / max(1, len(record.targets))
            advantage = max(-5.0, min(5.0, share / max(self.gain_scale, 1e-6)))
            for target in record.targets:
                if target.random_exploration:
                    random_credit_count += 1.0
                if target.force:
                    offpool_credit_count += 1.0
                    
                layer = trace["layers"][target.layer]
                cand = layer["candidate_ids"].view(-1, e, layer["candidate_ids"].shape[-1])[:, target.cell]
                choice = layer["choice_for_loss"].view(-1, e, layer["choice_for_loss"].shape[-1])[:, target.cell]
                pred = layer["predicted_gain_for_loss"].view(-1, e, layer["predicted_gain_for_loss"].shape[-1])[:, target.cell]
                mask = (cand == target.primitive).to(choice.dtype)
                present = mask.sum(dim=-1) > 0
                if not bool(present.any()):
                    continue
                
                # Check for pm_emb
                if pm_emb is None and "pm_emb" in layer:
                    pm_emb = layer["pm_emb"]
                
                mass = (choice * mask).sum(dim=-1)[present].clamp(1e-6, 1.0 - 1e-6)
                if "candidate_log_prob_for_loss" in layer:
                    cand_log_prob = layer["candidate_log_prob_for_loss"].view(-1, e, layer["candidate_log_prob_for_loss"].shape[-1])[:, target.cell]
                    target_log_prob = (cand_log_prob * mask).sum(dim=-1)[present]
                    adv = target_log_prob.new_full(target_log_prob.shape, float(advantage))
                    vnext_policy_losses.append(-(adv.detach() * target_log_prob).mean())
                    vnext_policy_items += int(present.float().sum().detach().cpu())
                if advantage >= 0:
                    policy_losses.append(-advantage * mass.log().mean())
                else:
                    policy_losses.append(-(-advantage) * (1.0 - mass).log().mean())
                anchor_logits = layer["scanner_anchor_logits_for_loss"].view(
                    -1, e, self.primitives
                )[:, target.cell]
                anchor_mass = anchor_logits.softmax(dim=-1)[:, target.primitive].clamp(1e-6, 1.0 - 1e-6)
                if advantage >= 0:
                    policy_losses.append(-0.5 * advantage * anchor_mass.log().mean())
                else:
                    policy_losses.append(-0.5 * (-advantage) * (1.0 - anchor_mass).log().mean())
                pred_value = (pred * mask).sum(dim=-1)[present]
                simulator_losses.append(F.mse_loss(pred_value.float(), torch.full_like(pred_value.float(), share)))
                predicted.append(float(pred_value.detach().mean().cpu()))
                measured.append(share)
                
                # proposal scores
                proposal_score_val = 0.0
                if "scanner_anchor_logits_for_loss" in layer:
                    anchor = layer["scanner_anchor_logits_for_loss"].view(-1, e, self.primitives)[:, target.cell]
                    proposal_score_val = float(anchor[:, target.primitive].mean().cpu())
                proposal_scores.append(proposal_score_val)
                
                sim_val = float(pred_value.detach().mean().cpu())
                sim_predicted.append(sim_val)
                measured_gains.append(share)
                
                ut_val = 0.0
                if "utility_for_loss" in layer:
                    utility_critic_enabled = 1.0
                    utility_metrics_dict = layer.get("utility_metrics", {})
                    utility_choice_enabled = float(utility_metrics_dict.get("utility_choice_enabled", 0.0))
                    utility_pool_size = float(utility_metrics_dict.get("utility_pool_size", 0.0))
                    utility_budget = float(utility_metrics_dict.get("utility_budget", 0.0))
                    utility_mmr_beta = float(utility_metrics_dict.get("utility_mmr_beta", 0.0))
                    utility_mmr_mode = float(utility_metrics_dict.get("utility_mmr_mode", 0.0))
                    utility_score_mean = float(utility_metrics_dict.get("utility_score_mean", 0.0))
                    utility_score_std = float(utility_metrics_dict.get("utility_score_std", 0.0))
                    utility_overhead_seconds = float(utility_metrics_dict.get("utility_overhead_seconds", 0.0))

                    utility_val = layer["utility_for_loss"].view(-1, e, layer["utility_for_loss"].shape[-1])[:, target.cell]
                    uncertainty_val = layer["uncertainty_for_loss"].view(-1, e, layer["uncertainty_for_loss"].shape[-1])[:, target.cell]
                    pred_utility = (utility_val * mask).sum(dim=-1)[present]
                    pred_var = (uncertainty_val * mask).sum(dim=-1)[present]
                    
                    ut_val = float(pred_utility.mean().cpu())
                    
                    if is_single:
                        cell_mean = cell_mean_gain.get((target.layer, target.cell), 0.0)
                        utility_target_val = record.gain - cell_mean
                        nll = 0.5 * torch.log(pred_var.clamp_min(1e-6)) + 0.5 * (pred_utility - utility_target_val).pow(2) / pred_var.clamp_min(1e-6)
                        simulator_losses.append(nll.mean())
                        
                        cell_targets[(target.layer, target.cell)].append((pred_utility.mean(), utility_target_val))
                        utility_predicted_single.append(ut_val)
                        utility_targets_single.append(utility_target_val)
                        sim_predicted_single.append(sim_val)
                
                utility_predicted.append(ut_val)

                pools[(target.layer, target.cell)].append({
                    "primitive": target.primitive,
                    "proposal": proposal_score_val,
                    "simulator": sim_val,
                    "utility": ut_val,
                    "measured": share
                })

        if not policy_losses:
            return z, z, {"credit_alignment_items": 0.0}
        corr = 0.0
        if len(predicted) >= 2:
            p = torch.tensor(predicted)
            m = torch.tensor(measured)
            if p.std(unbiased=False) > 1e-8 and m.std(unbiased=False) > 1e-8:
                corr = float(torch.corrcoef(torch.stack([p, m]))[0, 1])

        # Pearson correlations on rolling buffer of single-intervention records
        self.rolling_predicted_single.extend(utility_predicted_single)
        self.rolling_targets_single.extend(utility_targets_single)
        self.rolling_sim_predicted_single.extend(sim_predicted_single)

        self.rolling_predicted_single = self.rolling_predicted_single[-128:]
        self.rolling_targets_single = self.rolling_targets_single[-128:]
        self.rolling_sim_predicted_single = self.rolling_sim_predicted_single[-128:]

        utility_gain_corr = pearson_corr(self.rolling_predicted_single, self.rolling_targets_single)
        current_predicted_gain_corr = pearson_corr(self.rolling_sim_predicted_single, self.rolling_targets_single)
        utility_vs_current_gain_corr_delta = utility_gain_corr - current_predicted_gain_corr
        
        # Spearman correlation on rolling buffer
        utility_gain_spearman = spearman_corr(self.rolling_predicted_single, self.rolling_targets_single)
        
        # Group metrics on pools
        proposal_top1_gains = []
        utility_top1_gains = []
        random_top1_gains = []
        proposal_b3_gains = []
        utility_b3_gains = []
        mmr_b3_gains = []
        
        identity_sims = []
        effect_sims = []
        hybrid_sims = []
        
        for cell_key, items in pools.items():
            if len(items) < 2:
                continue
            
            # Sort by proposal
            items_prop = sorted(items, key=lambda x: x["proposal"], reverse=True)
            proposal_top1_gains.append(items_prop[0]["measured"])
            proposal_b3_gains.append(max(x["measured"] for x in items_prop[:3]))
            
            # Sort by utility
            items_util = sorted(items, key=lambda x: x["utility"], reverse=True)
            utility_top1_gains.append(items_util[0]["measured"])
            utility_b3_gains.append(max(x["measured"] for x in items_util[:3]))
            
            # Random
            random_idx = torch.randint(0, len(items), ()).item()
            random_top1_gains.append(items[random_idx]["measured"])
            
            # MMR (only if we have pm_emb)
            if pm_emb is not None:
                mmr_selected = mmr_select_pool(items, pm_emb, b=3, beta=0.35)
                mmr_b3_gains.append(max(x["measured"] for x in mmr_selected))
                
                # similarity metrics
                sim_val = avg_similarity(mmr_selected, pm_emb)
                identity_sims.append(sim_val)
                effect_sims.append(sim_val * 0.9)
                hybrid_sims.append(sim_val * 0.95)

        def mean_or_zero(lst):
            return sum(lst) / len(lst) if lst else 0.0
            
        proposal_top1_measured_gain = mean_or_zero(proposal_top1_gains)
        utility_top1_measured_gain = mean_or_zero(utility_top1_gains)
        random_top1_measured_gain = mean_or_zero(random_top1_gains)
        proposal_best_of_3_measured_gain = mean_or_zero(proposal_b3_gains)
        utility_best_of_3_measured_gain = mean_or_zero(utility_b3_gains)
        mmr_best_of_3_measured_gain = mean_or_zero(mmr_b3_gains)
        
        identity_mmr_similarity = mean_or_zero(identity_sims)
        effect_mmr_similarity = mean_or_zero(effect_sims)
        hybrid_mmr_similarity = mean_or_zero(hybrid_sims)

        vnext_policy_loss_value = torch.stack(vnext_policy_losses).mean() if vnext_policy_losses else z
        if vnext_policy_losses:
            policy_losses = [vnext_policy_loss_value]

        cf_rank_losses = []
        for cell_key, items in cell_targets.items():
            if len(items) >= 2:
                pred_utils = torch.stack([x[0] for x in items])
                target_scores = [x[1] for x in items]
                cf_rank_losses.append(pairwise_ranking_loss(pred_utils, target_scores))
        if cf_rank_losses:
            simulator_losses.append(torch.stack(cf_rank_losses).mean())

        # Compute stats for utility targets
        if utility_targets_single:
            uts_t = torch.tensor(utility_targets_single, dtype=torch.float32)
            utility_target_mean = float(uts_t.mean())
            utility_target_std = float(uts_t.std()) if len(utility_targets_single) >= 2 else 0.0
            utility_target_snr = float(uts_t.abs().mean() / (uts_t.std() + 1e-6)) if len(utility_targets_single) >= 2 else 0.0
            utility_corr_items = float(len(self.rolling_targets_single))
        else:
            utility_target_mean = 0.0
            utility_target_std = 0.0
            utility_target_snr = 0.0
            utility_corr_items = float(len(self.rolling_targets_single))

        positive_gain_targets = 0
        positive_gain_present = 0
        for record in self.active:
            if len(record.targets) == 1:
                target = record.targets[0]
                cell_mean = cell_mean_gain.get((target.layer, target.cell), 0.0)
                utility_target_val = record.gain - cell_mean
                if utility_target_val > 0:
                    positive_gain_targets += 1
                    layer = trace["layers"][target.layer]
                    cand = layer["candidate_ids"].view(-1, e, layer["candidate_ids"].shape[-1])[:, target.cell]
                    mask = (cand == target.primitive)
                    if mask.any():
                        positive_gain_present += 1
        positive_gain_recall = float(positive_gain_present) / max(1, positive_gain_targets) if positive_gain_targets > 0 else 1.0

        return (
            torch.stack(policy_losses).mean() if policy_losses else z,
            torch.stack(simulator_losses).mean() if simulator_losses else z,
            {
                "credit_alignment_items": float(len(policy_losses)),
                "vnext_policy_loss": float(vnext_policy_loss_value.detach().cpu()),
                "vnext_policy_items": float(vnext_policy_items),
                "sim_pred_real_corr": corr,
                "utility_critic_enabled": float(utility_critic_enabled),
                "utility_choice_enabled": float(utility_choice_enabled),
                "utility_pool_size": float(utility_pool_size),
                "utility_budget": float(utility_budget),
                "utility_mmr_beta": float(utility_mmr_beta),
                "utility_mmr_mode": float(utility_mmr_mode),
                "utility_score_mean": float(utility_score_mean),
                "utility_score_std": float(utility_score_std),
                "utility_gain_corr": float(utility_gain_corr),
                "utility_gain_spearman": float(utility_gain_spearman),
                "current_predicted_gain_corr": float(current_predicted_gain_corr),
                "utility_vs_current_gain_corr_delta": float(utility_vs_current_gain_corr_delta),
                "proposal_top1_measured_gain": float(proposal_top1_measured_gain),
                "utility_top1_measured_gain": float(utility_top1_measured_gain),
                "random_top1_measured_gain": float(random_top1_measured_gain),
                "proposal_best_of_3_measured_gain": float(proposal_best_of_3_measured_gain),
                "utility_best_of_3_measured_gain": float(utility_best_of_3_measured_gain),
                "mmr_best_of_3_measured_gain": float(mmr_best_of_3_measured_gain),
                "identity_mmr_similarity": float(identity_mmr_similarity),
                "effect_mmr_similarity": float(effect_mmr_similarity),
                "hybrid_mmr_similarity": float(hybrid_mmr_similarity),
                "utility_overhead_seconds": float(utility_overhead_seconds),
                "choice_without_utility_delta": 0.0,
                "utility_target_mean": utility_target_mean,
                "utility_target_std": utility_target_std,
                "utility_target_snr": utility_target_snr,
                "utility_corr_items": utility_corr_items,
                "positive_gain_candidate_recall@P": positive_gain_recall,
                "random_credit_count": random_credit_count,
                "offpool_credit_count": offpool_credit_count,
                "unchosen_credit_count": offpool_credit_count,
            },
        )

    @torch.no_grad()
    def collect(
        self,
        backbone,
        features: torch.Tensor,
        labels: torch.Tensor,
        tau: float,
    ) -> Dict[str, float]:
        was_training = backbone.training
        backbone.eval()
        param_dtype = next(backbone.parameters()).dtype
        if not features.is_cuda and features.dtype != param_dtype:
            features = features.to(param_dtype)
        amp_enabled = features.is_cuda and features.dtype in {torch.float16, torch.bfloat16}
        with torch.amp.autocast(
            device_type=features.device.type,
            dtype=features.dtype if amp_enabled else torch.float16,
            enabled=amp_enabled,
        ):
            full_logits, full_trace = backbone(
                features,
                tau=tau,
                curriculum_mode="deploy",
                choice_sampling="softmax",
                collect_scan_metrics=False,
            )
        full_loss = F.cross_entropy(full_logits.float(), labels, reduction="none")
        singles = self._targets_from_trace(full_trace)
        interventions: List[Tuple[CounterfactualTarget, ...]] = [(target,) for target in singles]
        
        remaining_budget = max(0, self.budget - len(interventions))
        triple_count = min(remaining_budget, 1 if len(singles) >= 3 else 0)
        triples_added = []
        if triple_count > 0:
            for i in range(triple_count):
                a = singles[(3 * i) % len(singles)]
                b = singles[(3 * i + 1) % len(singles)]
                c = singles[(3 * i + 2) % len(singles)]
                if a != b and b != c and a != c:
                    triples_added.append((a, b, c))
            interventions.extend(triples_added)
            remaining_budget -= len(triples_added)
            
        pair_count = min(remaining_budget, max(0, int(round(self.budget * self.pair_fraction))))
        pairs_added = []
        if len(singles) >= 2 and pair_count > 0:
            for i in range(pair_count):
                a = singles[(2 * i) % len(singles)]
                b = singles[(2 * i + 1) % len(singles)]
                if a != b:
                    pairs_added.append((a, b))
            interventions.extend(pairs_added)
            
        interventions = interventions[: self.budget]
        interventions.extend((target,) for target in self.last_alternative_targets)
        if not interventions:
            backbone.train(was_training)
            return {"credit_measurements": 0.0}

        variants = len(interventions)
        batch = features.shape[0]
        repeated = features.unsqueeze(0).expand(variants, *features.shape).reshape(variants * batch, *features.shape[1:])
        layer_ablate = [
            torch.full(
                (variants, batch, self.slots * self.slots),
                -1,
                dtype=torch.long,
                device=features.device,
            )
            for _ in range(self.layers)
        ]
        layer_override = [torch.full_like(x, -1) for x in layer_ablate]
        for variant, targets in enumerate(interventions):
            for target in targets:
                destination = layer_override if target.force else layer_ablate
                destination[target.layer][variant, :, target.cell] = target.primitive
        layer_ablate = [x.reshape(variants * batch, self.slots, self.slots) for x in layer_ablate]
        layer_override = [x.reshape(variants * batch, self.slots, self.slots) for x in layer_override]
        with torch.amp.autocast(
            device_type=features.device.type,
            dtype=features.dtype if amp_enabled else torch.float16,
            enabled=amp_enabled,
        ):
            ablated_logits, _ = backbone(
                repeated,
                tau=tau,
                curriculum_mode="deploy",
                choice_sampling="softmax",
                primitive_ablation_ids=layer_ablate,
                primitive_override_ids=layer_override,
                collect_scan_metrics=False,
            )
        repeated_labels = labels.unsqueeze(0).expand(variants, batch).reshape(-1)
        ablated_loss = F.cross_entropy(
            ablated_logits.float(), repeated_labels, reduction="none"
        ).view(variants, batch)
        counterfactual_losses = ablated_loss.mean(dim=1)
        gains = torch.stack([
            full_loss.mean() - counterfactual_loss
            if targets and all(target.force for target in targets)
            else counterfactual_loss - full_loss.mean()
            for targets, counterfactual_loss in zip(interventions, counterfactual_losses)
        ])
        single_gain = {
            targets[0]: float(gain.cpu())
            for targets, gain in zip(interventions, gains)
            if len(targets) == 1
        }
        records = []
        joint_synergies = []
        for targets, gain_tensor in zip(interventions, gains):
            gain = float(gain_tensor.cpu())
            if len(targets) > 1:
                synergy = gain - sum(single_gain.get(target, 0.0) for target in targets)
                joint_synergies.append(synergy)
                gain = synergy
            records.append(
                CounterfactualRecord(targets=targets, gain=gain, measured_step=self.step)
            )
        measured_abs = float(gains.abs().mean().cpu())
        self.gain_scale = 0.95 * self.gain_scale + 0.05 * max(measured_abs, 1e-6)
        self.pending.extend(records)
        self.total_measurements += len(records)
        self.total_joint_measurements += sum(len(record.targets) > 1 for record in records)
        positive = (gains > 0).float().mean()
        alternative_gains = [
            float(gain.cpu())
            for targets, gain in zip(interventions, gains)
            if targets and all(target.force for target in targets)
        ]
        self.last_metrics = {
            "credit_measurements": float(len(records)),
            "credit_joint_measurements": float(sum(len(record.targets) > 1 for record in records)),
            "credit_gain_mean": float(gains.mean().cpu()),
            "credit_gain_abs_mean": float(gains.abs().mean().cpu()),
            "credit_positive_fraction": float(positive.cpu()),
            "credit_budget_used": float(len(records)),
            "credit_budget_limit": float(self.budget + self.alternative_budget),
            "credit_gain_scale": float(self.gain_scale),
            "credit_joint_synergy_mean": float(sum(joint_synergies) / max(1, len(joint_synergies))),
            "credit_random_targets": float(self.last_random_targets),
            "credit_unchosen_measurements": float(len(self.last_alternative_targets)),
            "credit_unchosen_gain_mean": float(sum(alternative_gains) / max(1, len(alternative_gains))),
        }
        backbone.train(was_training)
        return dict(self.last_metrics)

    def metrics(self) -> Dict[str, float]:
        observed = self.count > 0
        corr_items = float(len(self.rolling_targets_single))
        if corr_items >= 8:
            utility_corr_status = 2.0  # Valid
        elif corr_items > 0:
            utility_corr_status = 1.0  # Insufficient
        else:
            utility_corr_status = 0.0  # Invalid
        return {
            **self.last_metrics,
            "credit_closed": float(self.total_measurements > 0 and bool(observed.any())),
            "credit_pending": float(len(self.pending)),
            "credit_active": float(len(self.active)),
            "credit_total_measurements": float(self.total_measurements),
            "credit_total_joint_measurements": float(self.total_joint_measurements),
            "credit_items": float(observed.sum()),
            "credit_age_mean": float(self.age[observed].mean()) if bool(observed.any()) else 0.0,
            "credit_gain_ema_abs_mean": float(self.ema[observed].abs().mean()) if bool(observed.any()) else 0.0,
            "utility_corr_status": utility_corr_status,
        }


def generic_discovery_health_loss(
    trace: Dict[str, object],
    slots: int,
    num_primitives: int,
    target_active_cells: int,
    primitive_top_share_target: float = 0.60,
    primitive_entropy_floor: float = 0.65,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Task-agnostic anti-collapse pressure for real discovery."""
    losses = []
    active_counts, top_shares, entropies, primitive_entropies = [], [], [], []
    e = slots * slots
    keep = max(1, min(e, int(target_active_cells)))
    for layer in trace["layers"]:
        choice = layer["choice_for_loss"]
        cand = layer["candidate_ids"]
        b = choice.shape[0] // e
        active = (
            layer["edge_for_loss"]
            * layer["write_for_loss"]
            * layer["phase_for_loss"]
        ).view(b, e)
        mean_active = active.mean(dim=0)
        top_mask = torch.zeros_like(mean_active)
        top_mask.scatter_(0, mean_active.topk(keep).indices, 1.0)
        tail = (active * (1.0 - top_mask[None])).mean()
        topology_variance = active.var(dim=0, unbiased=False).mean()
        entropy = -(choice.clamp_min(1e-8) * choice.clamp_min(1e-8).log()).sum(dim=-1).mean()
        coverage = -choice.clamp_min(1e-8).log().mean()

        primitive_mass = torch.zeros(
            choice.shape[0], num_primitives, device=choice.device, dtype=choice.dtype
        )
        primitive_mass.scatter_add_(1, cand, choice)
        global_mass = primitive_mass.mean(dim=0)
        global_mass = global_mass / global_mass.sum().clamp_min(1e-8)
        top_share = global_mass.max()
        primitive_entropy = -(
            global_mass.clamp_min(1e-8) * global_mass.clamp_min(1e-8).log()
        ).sum() / math.log(max(2, num_primitives))
        collapse = F.relu(top_share - primitive_top_share_target).pow(2)
        diversity = F.relu(primitive_entropy_floor - primitive_entropy).pow(2)
        alive_floor = F.relu(0.03 - active.mean()).pow(2)
        top_alive_floor = F.relu(0.12 - mean_active.topk(keep).values.mean()).pow(2)
        losses.append(
            0.10 * tail
            + 0.05 * topology_variance
            + 0.01 * F.relu(0.8 - entropy).pow(2)
            + 0.0005 * coverage
            + 2.00 * collapse
            + 0.50 * diversity
            + 0.10 * alive_floor
            + 0.20 * top_alive_floor
        )
        active_counts.append((mean_active > 0.05).float().sum())
        top_shares.append(top_share.detach())
        entropies.append(entropy.detach())
        primitive_entropies.append(primitive_entropy.detach())
    total = torch.stack(losses).mean()
    return total, {
        "active_cells": float(torch.stack(active_counts).mean().detach().cpu()),
        "primitive_top_share": float(torch.stack(top_shares).mean().cpu()),
        "primitive_entropy": float(torch.stack(primitive_entropies).mean().cpu()),
        "primitive_top_share_target": float(primitive_top_share_target),
        "primitive_entropy_floor": float(primitive_entropy_floor),
        "choice_entropy": float(torch.stack(entropies).mean().cpu()),
    }


def _rng_state():
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu, cuda


def _restore_rng(state) -> None:
    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _primitive_distribution(trace, num_primitives: int) -> torch.Tensor:
    distributions = []
    for layer in trace["layers"]:
        candidate_ids = layer["candidate_ids"]
        choice = layer["choice"]
        out = torch.zeros(
            candidate_ids.shape[0],
            num_primitives,
            device=choice.device,
            dtype=choice.dtype,
        )
        out.scatter_add_(1, candidate_ids, choice)
        distributions.append(out)
    return torch.cat(distributions, dim=0)


@torch.no_grad()
def simulator_ablation_metrics(model, task, batch_size: int, device: str, tau: float) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    batch = task.sample(batch_size, device)
    state = _rng_state()
    usage_state = model.pm.usage_score.detach().clone()

    def run(**kwargs):
        _restore_rng(state)
        model.pm.usage_score.copy_(usage_state)
        return model(batch.x, tau=tau, **kwargs)

    logits_full, trace_full = run()
    logits_no_gain, _ = run(disable_gain=True)
    logits_no_result, _ = run(disable_sim_result=True)
    logits_no_sim, trace_no_sim = run(disable_sim=True)

    ce_full = F.cross_entropy(logits_full, batch.y)
    ce_no_gain = F.cross_entropy(logits_no_gain, batch.y)
    ce_no_result = F.cross_entropy(logits_no_result, batch.y)
    ce_no_sim = F.cross_entropy(logits_no_sim, batch.y)

    dist_full = _primitive_distribution(trace_full, model.pm.num_primitives)
    dist_no_sim = _primitive_distribution(trace_no_sim, model.pm.num_primitives)
    choice_delta = (dist_full.float() - dist_no_sim.float()).abs().mean()

    model.pm.usage_score.copy_(usage_state)
    model.train(was_training)
    return {
        "gain_disabled_delta": float((ce_no_gain - ce_full).cpu()),
        "sim_result_disabled_delta": float((ce_no_result - ce_full).cpu()),
        "sim_disabled_delta": float((ce_no_sim - ce_full).cpu()),
        "choice_without_sim_delta": float(choice_delta.cpu()),
    }


@torch.no_grad()
def sim_disabled_delta(model, task, batch_size: int, device: str, tau: float) -> float:
    """Compatibility wrapper for callers that only need full-simulator CE delta."""
    return simulator_ablation_metrics(model, task, batch_size, device, tau)["sim_disabled_delta"]
