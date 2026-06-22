from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from arch_builder.srcf_light import (
    SRCFLightConfig,
    SRCFLightModel,
    SRCFLossWeights,
    srcf_closure_loss,
)


def make_batch(batch: int, input_dim: int, classes: int, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, input_dim)
    # Generic nonlinear teacher for smoke only. The model does not receive this
    # recipe; it only sees x and task loss through the head.
    logits = x @ teacher
    if input_dim >= 4:
        logits[:, 0] = logits[:, 0] + 0.35 * x[:, 0] * x[:, 1]
        logits[:, 1] = logits[:, 1] - 0.25 * x[:, 2] * x[:, 3]
    y = logits.argmax(dim=-1) % classes
    return x, y


def main() -> None:
    torch.manual_seed(7)
    random.seed(7)

    input_dim = 12
    classes = 3
    cfg = SRCFLightConfig(
        dim=24,
        slots=4,
        relation_dim=16,
        action_count=6,
        action_emb_dim=8,
        hidden=64,
        layers=2,
        micro_steps=2,
        noise_std=0.02,
        dropout=0.0,
    )
    model = SRCFLightModel(input_dim=input_dim, output_dim=classes, cfg=cfg)
    weights = SRCFLossWeights(
        fixed=0.03,
        recovery=0.03,
        contract=0.05,
        far_keep=0.02,
        state_var=0.02,
        move_band=0.02,
        action_entropy=0.0,
        edge_sparsity=0.0,
        far_margin=0.20,
        state_var_floor=0.005,
        move_min=0.005,
        move_max=2.0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    teacher = torch.randn(input_dim, classes)

    x0, y0 = make_batch(96, input_dim, classes, teacher)
    with torch.no_grad():
        out0 = model(x0)
        start_task = F.cross_entropy(out0["logits"], y0).item()

    last_metrics = {}
    for step in range(35):
        x, y = make_batch(96, input_dim, classes, teacher)
        x_aug = x + 0.03 * torch.randn_like(x)
        out = model(x)
        out_aug = model(x_aug)
        task_loss = F.cross_entropy(out["logits"], y)
        closure_loss, metrics = srcf_closure_loss(out, weights=weights, peer_output=out_aug)
        loss = task_loss + closure_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sq += float(p.grad.detach().pow(2).sum().cpu())
        if not math.isfinite(grad_norm_sq) or grad_norm_sq <= 0.0:
            raise RuntimeError(f"bad grad norm at step {step}: {grad_norm_sq}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_metrics = metrics
        last_metrics["task_loss"] = float(task_loss.detach().cpu())
        last_metrics["total_loss"] = float(loss.detach().cpu())

    with torch.no_grad():
        x1, y1 = make_batch(128, input_dim, classes, teacher)
        out1 = model(x1)
        final_task = F.cross_entropy(out1["logits"], y1).item()
        pred = out1["logits"].argmax(dim=-1)
        acc = (pred == y1).float().mean().item()

    required = [
        "srcf_fixed",
        "srcf_recovery",
        "srcf_contract",
        "srcf_move",
        "srcf_state_var",
        "srcf_action_entropy",
        "srcf_edge_mass",
    ]
    for key in required:
        val = float(last_metrics[key])
        if not math.isfinite(val):
            raise RuntimeError(f"metric {key} is not finite: {val}")

    print("SRCF_LIGHT_CLOSURE_CORE_PROBE")
    print(f"start_task_loss={start_task:.6f}")
    print(f"final_task_loss={final_task:.6f}")
    print(f"eval_acc={acc:.6f}")
    for key in required:
        print(f"{key}={float(last_metrics[key]):.6f}")
    print("SRCF_LIGHT_CLOSURE_CORE_PASS")


if __name__ == "__main__":
    main()
