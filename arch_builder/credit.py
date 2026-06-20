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


@dataclass
class CounterfactualRecord:
    targets: Tuple[CounterfactualTarget, ...]
    gain: float
    measured_step: int
    applied_step: int = -1


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
        selected = [target for _, target in scored[:high_count]]
        remaining = [target for _, target in scored[high_count:]]
        if remaining and random_count:
            order = torch.randperm(len(remaining))[:random_count].tolist()
            selected.extend(remaining[i] for i in order)
            self.last_random_targets = len(order)
        else:
            self.last_random_targets = 0
        selected = selected[:single_count]
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
        ids, gains = [], []
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
        if primitive_matrix is not None and ids:
            primitive_matrix.update_usage_credit(
                torch.tensor(ids, device=primitive_matrix.usage_score.device),
                torch.tensor(gains, device=primitive_matrix.usage_score.device),
            )
        self.active = ready
        return len(ready)

    def alignment_losses(
        self,
        trace: Dict[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        device = trace["layers"][0]["choice_for_loss"].device
        z = torch.zeros((), device=device)
        if not self.active:
            return z, z, {"credit_alignment_items": 0.0}
        policy_losses, simulator_losses = [], []
        predicted, measured = [], []
        e = self.slots * self.slots
        for record in self.active:
            share = float(record.gain) / max(1, len(record.targets))
            advantage = max(-5.0, min(5.0, share / max(self.gain_scale, 1e-6)))
            for target in record.targets:
                layer = trace["layers"][target.layer]
                cand = layer["candidate_ids"].view(-1, e, layer["candidate_ids"].shape[-1])[:, target.cell]
                choice = layer["choice_for_loss"].view(-1, e, layer["choice_for_loss"].shape[-1])[:, target.cell]
                pred = layer["predicted_gain_for_loss"].view(-1, e, layer["predicted_gain_for_loss"].shape[-1])[:, target.cell]
                mask = (cand == target.primitive).to(choice.dtype)
                present = mask.sum(dim=-1) > 0
                if not bool(present.any()):
                    continue
                mass = (choice * mask).sum(dim=-1)[present].clamp(1e-6, 1.0 - 1e-6)
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
        if not policy_losses:
            return z, z, {"credit_alignment_items": 0.0}
        corr = 0.0
        if len(predicted) >= 2:
            p = torch.tensor(predicted)
            m = torch.tensor(measured)
            if p.std(unbiased=False) > 1e-8 and m.std(unbiased=False) > 1e-8:
                corr = float(torch.corrcoef(torch.stack([p, m]))[0, 1])
        return (
            torch.stack(policy_losses).mean(),
            torch.stack(simulator_losses).mean() if simulator_losses else z,
            {"credit_alignment_items": float(len(policy_losses)), "sim_pred_real_corr": corr},
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
        pair_count = min(self.budget - len(singles), max(0, int(round(self.budget * self.pair_fraction))))
        interventions: List[Tuple[CounterfactualTarget, ...]] = [(target,) for target in singles]
        if len(singles) >= 2:
            for i in range(pair_count):
                a = singles[(2 * i) % len(singles)]
                b = singles[(2 * i + 1) % len(singles)]
                if a != b:
                    interventions.append((a, b))
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
