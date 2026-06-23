from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_csv(path: Path, row: Dict[str, object]) -> None:
    exists = path.exists()
    fields = list(row.keys())
    if exists:
        with path.open("r", encoding="utf-8") as f:
            old = next(csv.reader(f))
        fields = old
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def write_latest_report(path: Path, report_dir: str, summary: Dict[str, object]) -> None:
    lines: List[str] = ["# Latest vertical slice report", ""]
    lines.append(f"- report_dir: `{report_dir}`")
    for k in [
        "task", "task_config_path", "task_config_name", "task_config_digest", "expected_actions", "pass_thresholds", "pass_threshold_results", "pass_thresholds_met",
        "epochs", "curriculum_schedule", "curriculum_phase", "honesty_floor", "honesty_score",
        "state_norm_mode", "best_acc", "last_acc", "program_recovery_rate",
        "expected_edge_recovery", "expected_any_recovery", "expected_edge_active", "expected_candidate_present", "expected_edge_choice_mass",
        "layer_output_credit", "layer_listen_score", "layer_action_similarity",
        "slot_alive_mean", "slot_alive_count", "split_none_mass", "split_one_mass", "split_two_mass",
        "child_gate_mean", "merge_gate_mean", "collector_mass_mean",
        "gain_disabled_delta", "sim_result_disabled_delta", "sim_disabled_delta", "choice_without_sim_delta",
        "external_delete_delta", "internal_skip_delta", "state_ablation_delta",
        "semantic_grid_mismatch", "semantic_neighbor_entropy", "scanner_full_scan", "primitive_embedding_rank", "usage_entropy", "usage_credit_observations",
        "feedback_bias_abs", "feedback_staleness", "feedback_count", "feedback_entropy", "feedback_top_share", "feedback_candidate_usage", "category_candidate_usage", "feedback_candidate_coverage", "category_candidate_coverage",
        "feedback_update_called", "feedback_update_items", "feedback_update_positive", "feedback_update_negative", "feedback_update_skipped_disabled", "feedback_update_skipped_missing_address",
        "single_signed_projection_usage", "single_signed_projection_candidate_count", "single_signed_projection_top_score", "single_signed_projection_signed_score_mean",
        "pair_jl16_usage", "pair_jl16_candidate_count", "pair_jl16_top_score", "pair_jl16_seconds", "pair_jl16_pairs_tested",
        "projection_logit_cap", "projection_logit_clipped_fraction",
        "flat_shortcut_candidate_usage", "compositional_pair_candidate_usage",
        "measured_delta_loss_is_source_of_truth", "expected_actions_used_for_training",
        "grid_candidate_usage", "semantic_candidate_usage", "usage_candidate_usage", "random_candidate_usage", "global_candidate_usage", "scanner_source_mass_sum",
        "grid_candidate_coverage", "semantic_candidate_coverage", "usage_candidate_coverage", "random_candidate_coverage", "global_candidate_coverage",
        "grid_candidate_usage", "semantic_candidate_usage", "usage_candidate_usage", "random_candidate_usage", "scanner_source_mass_sum",
        "skip_mass", "transform_mass", "disable_mass",
        "oracle_acc", "edge_pair_bias_expected", "write_pair_bias_expected", "phase_pair_bias_expected", "cell_output_pair_bias_expected", "edge_pair_bias_std", "write_pair_bias_std", "choice_entropy", "edge_scale_mean", "cell_output_gate_mean", "cell_tape_weight_mean",
        "branch_split_loss", "branch_alive_loss", "branch_child_loss", "branch_merge_loss", "branch_collector_loss",
        "credit_teacher_credit_items", "credit_teacher_credit_staleness_mean", "credit_teacher_credit_age_max",
        "credit_audit_credit_items", "credit_audit_credit_staleness_mean", "credit_audit_credit_age_max",
        "credit_deploy_credit_items", "credit_deploy_credit_staleness_mean", "credit_deploy_credit_age_max",
    ]:
        if k in summary:
            lines.append(f"- {k}: `{summary[k]}`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append(str(summary.get("conclusion", "Run finished. Inspect metrics.csv, final_report.json, and PROGRAM_REPORT.md.")))
    lines.append("")
    lines.append("Program details: `PROGRAM_REPORT.md` and `program_epoch_XXX.json`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
