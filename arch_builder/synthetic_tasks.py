from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .task_config import TaskConfig, legacy_task_path, load_task_config


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

    def __init__(self, task: str = "diff", slots: int | None = None, dim: int = 64,
                 classes: int = 2, config: TaskConfig | str | Path | None = None) -> None:
        if isinstance(config, TaskConfig):
            self.config = config
        else:
            self.config = load_task_config(config if config is not None else legacy_task_path(task))
        if slots is not None and slots != self.config.slots:
            raise ValueError(f"slots={slots} conflicts with task config slots={self.config.slots}")
        self.task = self.config.name
        self.slots = self.config.slots
        self.dim = dim
        self.classes = classes

    def label_signal(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
        if x.ndim != 3 or x.shape[1] != self.slots:
            raise ValueError(f"expected x shaped [batch, {self.slots}, dim], got {tuple(x.shape)}")
        return self.config.evaluate(x), [dict(action) for action in self.config.expected_actions]

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
