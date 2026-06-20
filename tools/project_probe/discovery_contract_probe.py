from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from arch_builder.model import ActionMatrixModel
from arch_builder.synthetic_tasks import SyntheticKnownProgramTask
from arch_builder.train_vertical_slice import compute_training_objective, parser


def main() -> None:
    torch.manual_seed(7)
    args = parser().parse_args([])
    args.supervision_mode = "discovery"
    args.target_active_cells = 3.0

    task = SyntheticKnownProgramTask(config="configs/tasks/chain_diff_product.yml", dim=16)
    model = ActionMatrixModel(dim=16, slots=4, layers=2, top_k=8, sim_rank=4)
    model.eval()
    batch = task.sample(8, "cpu")
    logits, trace = model(batch.x)
    ce = F.cross_entropy(logits, batch.y)

    loss_a, _, _, _, weighted_a = compute_training_objective(trace, batch, model, ce, args)
    fake_actions = [
        {"layer": 0, "src": 3, "tgt": 3, "primitive": "disable"},
        {"layer": 1, "src": 2, "tgt": 2, "primitive": "replace"},
    ]
    altered = replace(
        batch,
        expected_primitive="disable",
        expected_src=3,
        expected_tgt=3,
        expected_actions=fake_actions,
    )
    loss_b, _, _, _, weighted_b = compute_training_objective(trace, altered, model, ce, args)

    oracle_keys = (
        "weighted_sim_loss",
        "weighted_expected_choice_loss",
        "weighted_expected_active_loss",
        "weighted_non_expected_primitive_loss",
        "weighted_non_expected_active_loss",
        "weighted_non_expected_tape_loss",
        "weighted_non_expected_transform_loss",
        "weighted_branch_split_loss",
        "weighted_branch_alive_loss",
        "weighted_branch_child_loss",
        "weighted_branch_merge_loss",
        "weighted_branch_collector_loss",
    )
    checks = {
        "expected_actions_do_not_change_discovery_loss": bool(torch.equal(loss_a, loss_b)),
        "oracle_weighted_terms_are_zero": all(
            float(weighted_a[key].detach()) == 0.0 and float(weighted_b[key].detach()) == 0.0
            for key in oracle_keys
        ),
        "discovery_terms_are_reported": all(
            key in weighted_a
            for key in (
                "weighted_discovery_active_floor_loss",
                "weighted_discovery_write_floor_loss",
                "weighted_discovery_choice_exploration_loss",
                "weighted_discovery_choice_coverage_loss",
                "weighted_discovery_active_tail_loss",
                "weighted_discovery_topology_consistency_loss",
            )
        ),
    }
    print({"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks})
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
