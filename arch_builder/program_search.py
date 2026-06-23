"""Active Program Search State — controls exploration/exploitation balance.

Monitors training dynamics (val accuracy, primitive diversity, choice entropy)
and automatically triggers exploration bursts when plateaus or collapses are
detected. Integrates with the training loop to modulate tau, random candidate
count, and health loss pressure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BurstEvent:
    """Record of a single exploration burst."""
    trigger_step: int
    trigger_reason: str  # "plateau", "collapse", "stale_primitives"
    duration: int
    val_acc_before: float
    val_acc_after: float = 0.0


class ProgramSearchState:
    """Tracks training dynamics and controls exploration/exploitation balance.

    The search loop works in two modes:
    - **Exploit**: normal training, crystallization increases gradually
    - **Explore (burst)**: triggered by plateau/collapse, increases tau and
      random candidates, decreases crystallization

    Usage::

        search = ProgramSearchState()
        for epoch in range(epochs):
            train(...)
            val_acc = evaluate(...)
            mods = search.update(
                val_acc=val_acc,
                primitive_top_share=metrics["primitive_top_share"],
                choice_entropy=metrics["choice_entropy"],
                feedback_count=metrics.get("feedback_count", 0),
                step=global_step,
            )
            # Apply mods["tau_multiplier"], mods["random_k_multiplier"], etc.
    """

    def __init__(
        self,
        patience: int = 50,
        burst_duration: int = 20,
        burst_tau_mult: float = 1.5,
        burst_random_mult: float = 3.0,
        min_improvement: float = 0.005,
        collapse_threshold: float = 0.70,
        stale_age_threshold: int = 100,
        safeguard_floor: float = 0.0,
    ) -> None:
        self.patience = patience
        self.burst_duration = burst_duration
        self.burst_tau_mult = burst_tau_mult
        self.burst_random_mult = burst_random_mult
        self.min_improvement = min_improvement
        self.collapse_threshold = collapse_threshold
        self.stale_age_threshold = stale_age_threshold
        self.safeguard_floor = safeguard_floor

        # State
        self.best_val_acc = 0.0
        self.steps_since_improvement = 0
        self.in_burst = False
        self.burst_steps_remaining = 0
        self.total_bursts = 0
        self.crystallization_score = 0.0  # 0=exploring, 1=fully crystallized
        self.current_burst: Optional[BurstEvent] = None

        # History
        self.val_history: List[float] = []
        self.diversity_history: List[float] = []
        self.burst_history: List[BurstEvent] = []
        self._last_modifiers: Dict[str, object] = {}

    def update(
        self,
        val_acc: float,
        primitive_top_share: float,
        choice_entropy: float,
        step: int = 0,
        feedback_count: float = 0.0,
    ) -> Dict[str, object]:
        """Called after each eval. Returns modifier dict for training loop."""
        self.val_history.append(val_acc)
        self.diversity_history.append(choice_entropy)

        # Track improvement
        if val_acc > self.best_val_acc + self.min_improvement:
            self.best_val_acc = val_acc
            self.steps_since_improvement = 0
            self.crystallization_score = min(1.0, self.crystallization_score + 0.05)
        else:
            self.steps_since_improvement += 1

        trigger_reason = None

        # Detect plateau
        if self.steps_since_improvement >= self.patience and not self.in_burst:
            trigger_reason = "plateau"

        # Detect collapse (single primitive dominates)
        if primitive_top_share > self.collapse_threshold and not self.in_burst:
            trigger_reason = "collapse"

        # Detect entropy death
        if choice_entropy < 0.1 and not self.in_burst and len(self.val_history) > 5:
            trigger_reason = "entropy_death"

        # Start burst if triggered
        if trigger_reason is not None:
            self.in_burst = True
            self.burst_steps_remaining = self.burst_duration
            self.total_bursts += 1
            self.crystallization_score = max(0.0, self.crystallization_score - 0.3)
            self.current_burst = BurstEvent(
                trigger_step=step,
                trigger_reason=trigger_reason,
                duration=self.burst_duration,
                val_acc_before=val_acc,
            )

        # Tick burst
        if self.in_burst:
            self.burst_steps_remaining -= 1
            # Safeguard: stop burst if accuracy drops below floor
            if self.safeguard_floor > 0 and val_acc < self.safeguard_floor:
                self.burst_steps_remaining = 0
            if self.burst_steps_remaining <= 0:
                self.in_burst = False
                self.steps_since_improvement = 0
                if self.current_burst is not None:
                    self.current_burst.val_acc_after = val_acc
                    self.burst_history.append(self.current_burst)
                    self.current_burst = None

        self._last_modifiers = self.get_modifiers()
        return self._last_modifiers

    def get_modifiers(self) -> Dict[str, object]:
        """Returns dict of multipliers/overrides for training loop."""
        if self.in_burst:
            return {
                "tau_multiplier": self.burst_tau_mult,
                "random_k_multiplier": self.burst_random_mult,
                "health_loss_multiplier": 2.0,
                "feedback_decay_override": 0.80,
                "exploration_mode": True,
                "in_burst": True,
                "burst_reason": self.current_burst.trigger_reason if self.current_burst else "unknown",
                "burst_steps_remaining": self.burst_steps_remaining,
                "crystallization_score": self.crystallization_score,
                "total_bursts": self.total_bursts,
            }
        return {
            "tau_multiplier": 1.0,
            "random_k_multiplier": 1.0,
            "health_loss_multiplier": 1.0,
            "feedback_decay_override": None,
            "exploration_mode": False,
            "in_burst": False,
            "burst_reason": None,
            "burst_steps_remaining": 0,
            "crystallization_score": self.crystallization_score,
            "total_bursts": self.total_bursts,
        }

    def metrics(self) -> Dict[str, float]:
        """Metrics for logging."""
        mods = self._last_modifiers or self.get_modifiers()
        return {
            "search_in_burst": float(mods.get("in_burst", False)),
            "search_total_bursts": float(self.total_bursts),
            "search_crystallization": float(self.crystallization_score),
            "search_steps_since_improvement": float(self.steps_since_improvement),
            "search_best_val_acc": float(self.best_val_acc),
            "search_burst_steps_remaining": float(mods.get("burst_steps_remaining", 0)),
            "search_tau_multiplier": float(mods.get("tau_multiplier", 1.0)),
        }
