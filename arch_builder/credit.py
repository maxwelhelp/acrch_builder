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


@torch.no_grad()
def sim_disabled_delta(model, task, batch_size: int, device: str, tau: float) -> float:
    batch = task.sample(batch_size, device)
    logits, _ = model(batch.x, tau=tau, disable_sim=False)
    logits_no, _ = model(batch.x, tau=tau, disable_sim=True)
    ce = F.cross_entropy(logits, batch.y)
    ce_no = F.cross_entropy(logits_no, batch.y)
    return float((ce_no - ce).detach().cpu())
