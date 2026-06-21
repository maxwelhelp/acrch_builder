from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from arch_builder.credit import pairwise_ranking_loss
from arch_builder.utility_critic import UtilityCritic
from arch_builder.vnext_controller import batched_mmr_select


class TinyPolicy(nn.Module):
    def __init__(self, context_dim: int, k: int) -> None:
        super().__init__()
        self.scanner = nn.Linear(context_dim, k)
        self.controller = nn.Linear(context_dim, k)
        self.critic = UtilityCritic(
            dim=4,
            context_dim=context_dim,
            prim_embed_dim=5,
            hidden=32,
            behavior_dim=8,
        )

    def forward(self, ctx: torch.Tensor, prim_emb: torch.Tensor, head: torch.Tensor):
        util, var, behavior = self.critic(ctx, prim_emb, head, return_behavior=True)
        logits = self.scanner(ctx) + self.controller(ctx) + util
        return logits, util, var, behavior


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def main() -> int:
    torch.manual_seed(11)
    n, k, context_dim = 5, 6, 12
    model = TinyPolicy(context_dim=context_dim, k=k)
    ctx = torch.randn(n, context_dim)
    head = torch.randn(n, 4)
    primitive_table = torch.randn(12, 5)
    top_ids = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8],
        [4, 5, 6, 7, 8, 9],
    ])
    prim_emb = primitive_table[top_ids]
    measured_gains = torch.tensor([
        [0.10, 0.00, 0.20, -0.05, 0.03, 0.01],
        [0.00, 0.30, 0.10, -0.02, 0.04, 0.02],
        [-0.02, 0.05, 0.40, 0.01, 0.02, 0.00],
        [0.20, 0.10, 0.00, 0.35, -0.01, 0.03],
        [0.01, 0.02, 0.05, 0.00, 0.25, 0.03],
    ])

    logits, utility, variance, behavior = model(ctx, prim_emb, head)
    logits.retain_grad()
    mask, _ = batched_mmr_select(
        utility=utility,
        top_ids=top_ids,
        pm_emb=primitive_table,
        budget=2,
        beta=0.35,
        behavior_feature=behavior,
        mode="hybrid",
    )
    selected_idx = logits.masked_fill(~mask, float("-inf")).argmax(dim=-1)
    selected_log_prob = logits.log_softmax(dim=-1).gather(1, selected_idx[:, None]).squeeze(1)
    selected_gain = measured_gains.gather(1, selected_idx[:, None]).squeeze(1)
    advantage = selected_gain - measured_gains.mean(dim=-1)
    policy_loss = -(advantage.detach() * selected_log_prob).mean()
    rank_loss = torch.stack([
        pairwise_ranking_loss(utility[i], measured_gains[i]) for i in range(n)
    ]).mean()
    critic_nll = ((utility - measured_gains).pow(2) / variance.clamp_min(1e-6)).mean() + variance.log().mean() * 0.5
    loss = policy_loss + 0.25 * rank_loss + 0.10 * critic_nll
    loss.backward()

    grad_policy_logits = float(logits.grad.detach().abs().sum().cpu())
    grad_scanner = grad_norm(model.scanner)
    grad_controller = grad_norm(model.controller)
    grad_critic = grad_norm(model.critic)

    assert torch.isfinite(policy_loss), policy_loss
    assert torch.isfinite(rank_loss), rank_loss
    assert grad_policy_logits > 0.0
    assert grad_scanner > 0.0
    assert grad_controller > 0.0
    assert grad_critic > 0.0
    assert torch.all(mask.sum(dim=-1) == 2), mask.sum(dim=-1)

    print(json.dumps({
        "status": "PASS",
        "policy_loss": float(policy_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "grad_policy_logits": grad_policy_logits,
        "grad_scanner": grad_scanner,
        "grad_controller": grad_controller,
        "grad_utility_critic": grad_critic,
        "mmr_selected_count": float(mask.float().sum(dim=-1).mean().cpu()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
