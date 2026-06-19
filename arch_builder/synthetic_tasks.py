from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    expected_primitive: str
    expected_src: int
    expected_tgt: int


class SyntheticKnownProgramTask:
    """Known-program tasks for the vertical slice."""

    def __init__(self, task: str = "diff", slots: int = 4, dim: int = 64, classes: int = 2) -> None:
        self.task = task
        self.slots = slots
        self.dim = dim
        self.classes = classes

    def label_signal(self, x: torch.Tensor) -> tuple[torch.Tensor, str, int, int]:
        if self.task == "diff":
            signal = (x[:, 0] - x[:, 1]).mean(dim=-1)
            return signal, "diff", 0, 1
        if self.task == "merge":
            signal = (x[:, 0] + x[:, 1]).mean(dim=-1)
            return signal, "merge", 0, 1
        if self.task == "product":
            signal = (x[:, 0] * x[:, 1]).mean(dim=-1)
            return signal, "product", 0, 1
        if self.task == "memory":
            signal = x.mean(dim=(1, 2))
            return signal, "memory_write", 0, 1
        if self.task == "semantic_rescue":
            signal = (x[:, 0] * x[:, 1]).mean(dim=-1) - (x[:, 2] - x[:, 3]).mean(dim=-1)
            return signal, "product", 0, 1
        raise ValueError(f"unknown task: {self.task}")

    def sample(self, batch_size: int, device: str | torch.device) -> Batch:
        x = torch.randn(batch_size, self.slots, self.dim, device=device)
        signal, expected, src, tgt = self.label_signal(x)
        y = (signal > 0).long()
        return Batch(x=x, y=y, expected_primitive=expected, expected_src=src, expected_tgt=tgt)

    def oracle_accuracy(self, batch: Batch) -> float:
        signal, _, _, _ = self.label_signal(batch.x)
        pred = (signal > 0).long()
        return float((pred == batch.y).float().mean().detach().cpu())
