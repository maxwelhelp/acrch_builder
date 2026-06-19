from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    expected_primitive: str
    expected_src: int
    expected_tgt: int
    expected_actions: List[Dict[str, object]]


class SyntheticKnownProgramTask:
    """Known-program tasks for vertical-slice debugging.

    These tasks are microscopes, not final real-task training.
    """

    def __init__(self, task: str = "diff", slots: int = 4, dim: int = 64, classes: int = 2) -> None:
        self.task = task
        self.slots = slots
        self.dim = dim
        self.classes = classes

    def label_signal(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
        if self.task == "diff":
            signal = (x[:, 0] - x[:, 1]).mean(dim=-1)
            actions = [{"layer": 0, "src": 0, "tgt": 1, "primitive": "diff"}]
            return signal, actions

        if self.task == "two_diff":
            signal = (x[:, 0] - x[:, 1]).mean(dim=-1) + (x[:, 2] - x[:, 3]).mean(dim=-1)
            actions = [
                {"layer": 0, "src": 0, "tgt": 1, "primitive": "diff"},
                {"layer": 0, "src": 2, "tgt": 3, "primitive": "diff"},
            ]
            return signal, actions

        if self.task == "merge":
            signal = (x[:, 0] + x[:, 1]).mean(dim=-1)
            actions = [{"layer": 0, "src": 0, "tgt": 1, "primitive": "merge"}]
            return signal, actions

        if self.task == "product":
            signal = (x[:, 0] * x[:, 1]).mean(dim=-1)
            actions = [{"layer": 0, "src": 0, "tgt": 1, "primitive": "product"}]
            return signal, actions

        if self.task == "chain_diff_product":
            signal = ((x[:, 0] - x[:, 1]) * (x[:, 2] - x[:, 3])).mean(dim=-1)
            actions = [
                {"layer": 0, "src": 0, "tgt": 1, "primitive": "diff"},
                {"layer": 0, "src": 2, "tgt": 3, "primitive": "diff"},
                {"layer": 1, "src": 1, "tgt": 3, "primitive": "product"},
            ]
            return signal, actions

        if self.task == "semantic_rescue":
            signal = (x[:, 0] * x[:, 1]).mean(dim=-1) - (x[:, 2] - x[:, 3]).mean(dim=-1)
            actions = [{"layer": 0, "src": 0, "tgt": 1, "primitive": "product"}]
            return signal, actions

        raise ValueError(f"unknown task: {self.task}")

    def sample(self, batch_size: int, device: str | torch.device) -> Batch:
        x = torch.randn(batch_size, self.slots, self.dim, device=device)
        signal, actions = self.label_signal(x)
        y = (signal > 0).long()
        first = actions[0]
        return Batch(
            x=x,
            y=y,
            expected_primitive=str(first["primitive"]),
            expected_src=int(first["src"]),
            expected_tgt=int(first["tgt"]),
            expected_actions=actions,
        )

    def oracle_accuracy(self, batch: Batch) -> float:
        signal, _ = self.label_signal(batch.x)
        pred = (signal > 0).long()
        return float((pred == batch.y).float().mean().detach().cpu())
