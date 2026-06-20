from __future__ import annotations

import torch
import torch.nn.functional as F

from arch_builder.credit import BoundedCounterfactualCredit
from arch_builder.model import ActionMatrixModel


def main() -> None:
    torch.manual_seed(19)
    model = ActionMatrixModel(dim=16, slots=4, layers=2, classes=3, top_k=8, sim_rank=4)
    credit = BoundedCounterfactualCredit(layers=2, slots=4, primitives=25, budget=6, alternative_budget=2)
    heldout_x = torch.randn(8, 4, 16)
    heldout_y = torch.randint(0, 3, (8,))

    # Half features reproduce the real AMP boundary; collector must safely cast
    # on CPU and enter autocast on CUDA instead of mixing Half inputs/Float weights.
    collected = credit.collect(model, heldout_x.half(), heldout_y, tau=1.0)
    before = credit.alignment_losses(model(heldout_x, choice_sampling="softmax")[1])[2]
    applied = credit.advance(model.pm)

    model.train()
    logits, trace = model(
        torch.randn(8, 4, 16),
        choice_sampling="softmax",
        collect_scan_metrics=False,
    )
    policy, simulator, aligned = credit.alignment_losses(trace)
    loss = F.cross_entropy(logits, torch.randint(0, 3, (8,))) + policy + simulator
    loss.backward()
    controller_grad = model.layers[0].primitive_pair_bias.grad
    simulator_grads = [p.grad for layer in model.layers for p in layer.simulator.parameters()]
    anchor_grads = [layer.scanner.anchor.weight.grad for layer in model.layers]

    checks = {
        "budget_bounded": collected["credit_budget_used"] <= collected["credit_budget_limit"],
        "joint_interventions_present": collected["credit_joint_measurements"] > 0,
        "random_budget_present": collected["credit_random_targets"] > 0,
        "unchosen_candidates_measured": collected["credit_unchosen_measurements"] > 0,
        "strictly_delayed": before["credit_alignment_items"] == 0 and applied > 0,
        "alignment_active_after_advance": aligned["credit_alignment_items"] > 0,
        "controller_receives_credit_gradient": controller_grad is not None and float(controller_grad.abs().sum()) > 0,
        "simulator_receives_credit_gradient": any(g is not None and float(g.abs().sum()) > 0 for g in simulator_grads),
        "scanner_anchor_receives_credit_gradient": any(g is not None and float(g.abs().sum()) > 0 for g in anchor_grads),
        "usage_updated_from_measured_credit": float(model.pm.usage_observations.sum()) > 0,
    }
    print({"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "metrics": credit.metrics()})
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
