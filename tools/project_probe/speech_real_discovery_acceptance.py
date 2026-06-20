from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(paths):
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def _mean(rows, key):
    return statistics.mean(float(row.get(key, 0.0)) for row in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict multi-seed real discovery acceptance")
    ap.add_argument("--learned", nargs="+", required=True)
    ap.add_argument("--frozen", nargs="+", required=True)
    ap.add_argument("--random", nargs="+", required=True)
    ap.add_argument("--out", default="reports/agent_inspector/speech_real_discovery_acceptance.json")
    args = ap.parse_args()
    learned, frozen, random = _load(args.learned), _load(args.frozen), _load(args.random)

    def unique_seeds(rows):
        return len({row.get("seed") for row in rows})

    chance = 1.0 / len(learned[0].get("classes", []))
    sim_deltas = [float(row.get("ablations", {}).get("sim_disabled_delta", 0.0)) for row in learned]
    choice_deltas = [float(row.get("ablations", {}).get("choice_without_sim_delta", 0.0)) for row in learned]
    non_grid = [
        float(row.get("ablations", {}).get("semantic_candidate_usage", 0.0))
        + float(row.get("ablations", {}).get("usage_candidate_usage", 0.0))
        + float(row.get("ablations", {}).get("random_candidate_usage", 0.0))
        for row in learned
    ]
    mechanism_ok = all(
        not row.get("expected_actions_used_for_training", True)
        and not row.get("layer_role_priors_used", True)
        and bool(row.get("discovery_metrics", {}).get("credit_closed", 0.0))
        and row.get("discovery_metrics", {}).get("credit_total_joint_measurements", 0.0) > 0
        and row.get("discovery_metrics", {}).get("credit_unchosen_measurements", 0.0) > 0
        and row.get("discovery_metrics", {}).get("credit_random_targets", 0.0) > 0
        for row in learned
    )
    checks = {
        "three_unique_learned_seeds": unique_seeds(learned) >= 3,
        "three_unique_frozen_seeds": unique_seeds(frozen) >= 3,
        "three_unique_random_seeds": unique_seeds(random) >= 3,
        "learned_above_chance_margin": _mean(learned, "deploy_acc") >= chance + 0.02,
        "learned_beats_frozen_controller": _mean(learned, "deploy_acc") >= _mean(frozen, "deploy_acc") + 0.01,
        "learned_beats_random_controller": _mean(learned, "deploy_acc") >= _mean(random, "deploy_acc") + 0.01,
        "generic_credit_contract_all_seeds": mechanism_ok,
        "median_simulator_ablation_positive": statistics.median(sim_deltas) > 0.0,
        "median_simulator_changes_choice": statistics.median(choice_deltas) > 1e-5,
        "median_true_non_grid_usage_positive": statistics.median(non_grid) > 0.05,
        "no_primitive_collapse_all_seeds": all(
            row.get("discovery_metrics", {}).get("primitive_top_share", 1.0) < 0.65
            for row in learned
        ),
    }
    report = {
        "acceptance_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "chance": chance,
        "learned_mean_acc": _mean(learned, "deploy_acc"),
        "frozen_mean_acc": _mean(frozen, "deploy_acc"),
        "random_mean_acc": _mean(random, "deploy_acc"),
        "median_sim_disabled_delta": statistics.median(sim_deltas),
        "median_choice_without_sim_delta": statistics.median(choice_deltas),
        "median_true_non_grid_usage": statistics.median(non_grid),
        "learned_reports": args.learned,
        "frozen_reports": args.frozen,
        "random_reports": args.random,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["acceptance_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
