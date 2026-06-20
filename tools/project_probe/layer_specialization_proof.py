from __future__ import annotations

import json
from pathlib import Path

from arch_builder.train_vertical_slice import parser as train_parser, train


def _layer_action_stats(action_metrics: dict[str, float], layer: int) -> dict[str, float]:
    suffixes = ("_present", "_choice_mass", "_recovery", "_active", "_tape")
    values: dict[str, list[float]] = {suffix: [] for suffix in suffixes}
    prefix = f"action_L{layer}_"
    for key, value in action_metrics.items():
        if not key.startswith(prefix):
            continue
        for suffix in suffixes:
            if key.endswith(suffix):
                values[suffix].append(float(value))
                break
    return {
        f"layer{layer}{suffix}": (sum(vals) / len(vals) if vals else 0.0)
        for suffix, vals in values.items()
    }


def main() -> None:
    report_dir = Path("agent_reports/task07_layer_specialization_smoke")
    latest_report = Path("LATEST_RUN_REPORT.md")
    final_report_path = report_dir / "final_report.json"
    program_path = report_dir / "program_epoch_002.json"
    if not final_report_path.exists() or not program_path.exists():
        args = train_parser().parse_args(
            [
                "--task-config",
                "configs/tasks/chain_diff_merge.yml",
                "--epochs",
                "2",
                "--steps-per-epoch",
                "20",
                "--batch-size",
                "128",
                "--eval-steps",
                "5",
                "--eval-batch-size",
                "256",
                "--dim",
                "64",
                "--layers",
                "2",
                "--top-k",
                "25",
                "--sim-rank",
                "16",
                "--device",
                "cpu",
                "--amp",
                "none",
                "--out-dir",
                str(report_dir),
                "--latest-report",
                str(latest_report),
            ]
        )
        train(args)

    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    program = json.loads(program_path.read_text(encoding="utf-8"))
    action_metrics = final_report.get("program_action_metrics", {})
    layer0 = _layer_action_stats(action_metrics, 0)
    layer1 = _layer_action_stats(action_metrics, 1)

    checks = {
        "oracle_pass": float(final_report.get("oracle_acc", 0.0)) >= 1.0,
        "chain_readable": (
            layer0["layer0_recovery"] > 0.30
            and layer1["layer1_recovery"] > 0.30
            and layer0["layer0_choice_mass"] > 0.30
            and layer1["layer1_choice_mass"] > 0.30
        ),
        "layer_listen_alive": float(final_report.get("layer_listen_score", 0.0)) > 0.5,
        "adjacent_actions_non_identical": float(final_report.get("layer_action_similarity", 1.0)) < 0.98,
        "dependency_delta_positive": float(final_report.get("layer_dependency_delta", 0.0)) > 0.0,
        "external_delete_delta_positive": float(final_report.get("external_delete_delta", 0.0)) > 0.0,
        "internal_skip_delta_positive": float(final_report.get("internal_skip_delta", 0.0)) > 0.0,
        "layer0_credit_positive": layer0["layer0_recovery"] > 0.0 and layer0["layer0_choice_mass"] > 0.0,
        "layer1_credit_positive": layer1["layer1_recovery"] > 0.0 and layer1["layer1_choice_mass"] > 0.0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    out_dir = Path("reports/agent_inspector")
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "LAYER_SPECIALIZATION_PROOF.md"
    js = out_dir / "layer_specialization_proof.json"
    report = {
        "status": status,
        "task": "layer_specialization_chain_diff_merge",
        "report_dir": str(report_dir),
        "program_verdicts": final_report.get("program_verdicts", []),
        "oracle_acc": final_report.get("oracle_acc"),
        "program_recovery_rate": final_report.get("program_recovery_rate"),
        "layer_listen_score": final_report.get("layer_listen_score"),
        "layer_action_similarity": final_report.get("layer_action_similarity"),
        "layer_dependency_delta": final_report.get("layer_dependency_delta"),
        "external_delete_delta": final_report.get("external_delete_delta"),
        "internal_skip_delta": final_report.get("internal_skip_delta"),
        "state_ablation_delta": final_report.get("state_ablation_delta"),
        "program_verdicts": final_report.get("program_verdicts", []),
        "layer0": layer0,
        "layer1": layer1,
        "checks": checks,
        "program_action_metrics": action_metrics,
    }
    js.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Layer specialization proof",
        "",
        f"- status: `{status}`",
        f"- oracle_acc: `{report['oracle_acc']}`",
        f"- program_recovery_rate: `{report['program_recovery_rate']}`",
        f"- layer_listen_score: `{report['layer_listen_score']}`",
        f"- layer_action_similarity: `{report['layer_action_similarity']}`",
        f"- layer_dependency_delta: `{report['layer_dependency_delta']}`",
        f"- external_delete_delta: `{report['external_delete_delta']}`",
        f"- internal_skip_delta: `{report['internal_skip_delta']}`",
        f"- state_ablation_delta: `{report['state_ablation_delta']}`",
        f"- program_verdicts: `{report['program_verdicts']}`",
        "",
        "## Layer credit",
        "",
        f"- layer0: `{layer0}`",
        f"- layer1: `{layer1}`",
        "",
        "## Program verdicts",
        "",
        f"`{program.get('layers', [])}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in checks.items())
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[layer-specialization-proof] status={status}")
    print(f"[layer-specialization-proof] report={md}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
