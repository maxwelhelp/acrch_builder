from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

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
