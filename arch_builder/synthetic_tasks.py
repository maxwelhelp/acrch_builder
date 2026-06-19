from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    expected_primitive: str


class SyntheticKnownProgramTask:
    """On-the-fly known-program tasks for vertical-slice proof.

    The task intentionally has a known useful primitive so reports can check
    whether the ActionMatrix recovers something interpretable.
    """

    def __init__(self, task: str = "diff", slots: int = 4, dim: int = 64, classes: int = 2) -> None:
        self.task = task
        self.slots = slots
        self.dim = dim
        self.classes = classes

    def sample(self, batch_size: int, device: str | torch.device) -> Batch:
        x = torch.randn(batch_size, self.slots, self.dim, device=device)
        if self.task == "diff":
            signal = (x[:, 0] - x[:, 1]).mean(dim=-1)
            expected = "diff"
        elif self.task == "merge":
            signal = (x[:, 0] + x[:, 1]).mean(dim=-1)
            expected = "merge"
        elif self.task == "product":
            signal = (x[:, 0] * x[:, 1]).mean(dim=-1)
            expected = "product"
        elif self.task == "memory":
            # Requires carrying a global average-like signal.
            signal = x.mean(dim=(1, 2))
            expected = "memory_write"
        elif self.task == "semantic_rescue":
            # Expected primitive is intentionally outside many local windows.
            signal = (x[:, 0] * x[:, 1]).mean(dim=-1) - (x[:, 2] - x[:, 3]).mean(dim=-1)
            expected = "product"
        else:
            raise ValueError(f"unknown task: {self.task}")
        y = (signal > 0).long()
        return Batch(x=x, y=y, expected_primitive=expected)
