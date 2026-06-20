from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from arch_builder.task_config import load_task_config
from arch_builder.train_vertical_slice import parser as train_parser, train


def _action_key(action: Dict[str, object]) -> str:
    primitive = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in str(action["primitive"]).lower()
    )
    return f"action_L{int(action['layer'])}_{int(action['src'])}_{int(action['tgt'])}_{primitive}"


def _action_stats(action_metrics: Dict[str, float], action: Dict[str, object]) -> Dict[str, float]:
    prefix = _action_key(action)
    return {
        "present": float(action_metrics.get(f"{prefix}_present", 0.0)),
        "choice_mass": float(action_metrics.get(f"{prefix}_choice_mass", 0.0)),
        "recovery": float(action_metrics.get(f"{prefix}_recovery", 0.0)),
        "active": float(action_metrics.get(f"{prefix}_active", 0.0)),
        "tape": float(action_metrics.get(f"{prefix}_tape", 0.0)),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def _fraction(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _run_task(
    task_name: str,
    config_path: str,
    report_dir: Path,
    latest_report: Path,
    epochs: int,
    extra_args: List[str] | None = None,
) -> Dict[str, object]:
    final_report_path = report_dir / "final_report.json"
    program_path = report_dir / f"program_epoch_{epochs:03d}.json"
    config_digest = load_task_config(config_path).digest
    rerun = True
    if final_report_path.exists() and program_path.exists():
        try:
            existing = json.loads(final_report_path.read_text(encoding="utf-8"))
            rerun = not (
                existing.get("task_config_digest") == config_digest
                and int(existing.get("epochs", -1)) == epochs
            )
        except Exception:
            rerun = True
    if rerun:
        base_args = [
                "--task-config",
                config_path,
                "--epochs",
                str(epochs),
                "--steps-per-epoch",
                "18",
                "--max-steps",
                "12",
                "--batch-size",
                "128",
                "--eval-steps",
                "4",
                "--eval-batch-size",
                "256",
                "--dim",
                "64",
                "--top-k",
                "25",
                "--sim-rank",
                "16",
                "--device",
                "cpu",
                "--amp",
                "none",
                "--lambda-branch",
                "0.08",
                "--out-dir",
                str(report_dir),
                "--latest-report",
                str(latest_report),
            ]
        if extra_args:
            base_args = base_args[:]
            for i, value in enumerate(extra_args):
                base_args.append(value)
        args = train_parser().parse_args(base_args)
        train(args)

    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    program = json.loads(program_path.read_text(encoding="utf-8"))
    action_metrics = final_report.get("program_action_metrics", {})
    expected_actions = list(final_report.get("expected_actions", []))
    layer0 = program["layers"][0] if program.get("layers") else {}

    expected_stats = [_action_stats(action_metrics, action) for action in expected_actions]
    expected_recovery = _mean(stat["recovery"] for stat in expected_stats)
    expected_choice_mass = _mean(stat["choice_mass"] for stat in expected_stats)
    expected_present = _mean(stat["present"] for stat in expected_stats)
    expected_active = _mean(stat["active"] for stat in expected_stats)

    non_expected_cells = [cell for cell in layer0.get("cells", []) if not cell.get("is_expected_edge", False)]
    non_expected_active = _mean(float(cell.get("active", 0.0)) for cell in non_expected_cells)
    non_expected_top_split = _fraction(sum(1 for cell in non_expected_cells if cell.get("top_primitive") == "split"), len(non_expected_cells))
    non_expected_top_skip = _fraction(sum(1 for cell in non_expected_cells if cell.get("top_primitive") == "skip"), len(non_expected_cells))
    non_expected_top_disable = _fraction(sum(1 for cell in non_expected_cells if cell.get("top_primitive") == "disable"), len(non_expected_cells))

    split_two_mass = float(final_report.get("split_two_mass", 0.0))
    split_one_mass = float(final_report.get("split_one_mass", 0.0))
    slot_alive_mean = float(final_report.get("slot_alive_mean", 0.0))
    slot_alive_count = float(final_report.get("slot_alive_count", 0.0))
    collector_mass_mean = float(final_report.get("collector_mass_mean", 0.0))

    checks = {
        "oracle_pass": float(final_report.get("oracle_acc", 0.0)) >= 1.0,
        "expected_actions_live": expected_present > 0.15 and expected_choice_mass > 0.15 and expected_recovery > 0.15,
        "slot_alive_bounded": 0.10 <= slot_alive_mean <= 0.95 and 0.5 <= slot_alive_count <= 3.5,
        "collector_live": collector_mass_mean > 0.0,
        "branch_metric_live": split_two_mass > 0.0 or split_one_mass > 0.0,
    }

    if task_name == "branch_split_merge":
        checks.update({
            "split_two_child_active": split_two_mass >= split_one_mass,
            "required_branch_credit": expected_recovery >= 0.15 and expected_choice_mass >= 0.15,
        })
    elif task_name == "branch_optional_branch":
        checks.update({
            "optional_branch_suppressed": non_expected_active < expected_active or non_expected_active < 0.12,
            "non_expected_split_suppressed": non_expected_top_split < 0.35,
        })
    elif task_name == "branch_skip_tradeoff":
        checks.update({
            "skip_branch_live": any(str(act["primitive"]) == "skip" for act in expected_actions) and expected_recovery >= 0.15,
            "harmful_skip_suppressed": non_expected_top_skip < 0.35 and non_expected_top_disable < 0.35,
        })

    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "task": task_name,
        "status": status,
        "report_dir": str(report_dir),
        "final_report": final_report,
        "program": program,
        "expected_actions": expected_actions,
        "expected_stats": expected_stats,
        "non_expected_active": non_expected_active,
        "non_expected_top_split": non_expected_top_split,
        "non_expected_top_skip": non_expected_top_skip,
        "non_expected_top_disable": non_expected_top_disable,
        "checks": checks,
    }


def main() -> None:
    out_dir = Path("reports/agent_inspector")
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_report = Path("LATEST_RUN_REPORT.md")

    task_specs = [
        ("branch_split_merge", "configs/tasks/branch_split_merge.yml", Path("agent_reports/task08_branch_split_merge_smoke"), 2, None),
        ("branch_optional_branch", "configs/tasks/branch_optional_branch.yml", Path("agent_reports/task08_branch_optional_branch_smoke"), 2, None),
        ("branch_skip_tradeoff", "configs/tasks/branch_skip_tradeoff.yml", Path("agent_reports/task08_branch_skip_tradeoff_smoke"), 4, [
            "--epochs", "4",
            "--steps-per-epoch", "12",
            "--max-steps", "12",
            "--batch-size", "64",
            "--eval-steps", "4",
            "--eval-batch-size", "128",
            "--top-k", "12",
            "--sim-rank", "8",
            "--lambda-branch", "0.12",
            "--lambda-non-expected-primitive", "0.5",
            "--lambda-primitive-usage-diversity", "0.15",
            "--lambda-cell-choice-diversity", "0.15",
            "--lambda-non-expected-active", "0.05",
            "--lambda-non-expected-tape", "0.05",
            "--lambda-non-expected-transform", "0.05",
            "--adapt-choice-floor", "0.25",
            "--adapt-top-share-floor", "0.35",
            "--adapt-choice-boost", "2.0",
        ]),
    ]

    results: List[Dict[str, object]] = []
    for task_name, config_path, report_dir, epochs, extra_args in task_specs:
        results.append(_run_task(task_name, config_path, report_dir, latest_report, epochs, extra_args=extra_args))

    overall_checks = {
        "all_tasks_pass": all(r["status"] == "PASS" for r in results),
        "split_merge_pass": results[0]["status"] == "PASS",
        "optional_branch_pass": results[1]["status"] == "PASS",
        "skip_tradeoff_pass": results[2]["status"] == "PASS",
    }
    status = "PASS" if all(overall_checks.values()) else "FAIL"
    report = {
        "status": status,
        "task": "branching_variable_output",
        "overall_checks": overall_checks,
        "results": results,
    }

    (out_dir / "branching_output_proof.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Branching and variable output proof",
        "",
        f"- status: `{status}`",
    ]
    for res in results:
        fr = res["final_report"]
        lines.extend([
            "",
            f"## {res['task']}",
            "",
            f"- status: `{res['status']}`",
            f"- oracle_acc: `{fr.get('oracle_acc')}`",
            f"- slot_alive_mean: `{fr.get('slot_alive_mean')}`",
            f"- slot_alive_count: `{fr.get('slot_alive_count')}`",
            f"- split_none_mass: `{fr.get('split_none_mass')}`",
            f"- split_one_mass: `{fr.get('split_one_mass')}`",
            f"- split_two_mass: `{fr.get('split_two_mass')}`",
            f"- collector_mass_mean: `{fr.get('collector_mass_mean')}`",
            f"- non_expected_active_mean(layer0): `{res['program'].get('layers', [{}])[0].get('non_expected_active_mean', 0.0) if res['program'].get('layers') else 0.0}`",
            f"- non_expected_top_split(layer0): `{res['program'].get('layers', [{}])[0].get('non_expected_top_split', 0.0) if res['program'].get('layers') else 0.0}`",
            f"- non_expected_top_skip(layer0): `{res['program'].get('layers', [{}])[0].get('non_expected_top_skip', 0.0) if res['program'].get('layers') else 0.0}`",
            f"- non_expected_top_disable(layer0): `{res['program'].get('layers', [{}])[0].get('non_expected_top_disable', 0.0) if res['program'].get('layers') else 0.0}`",
            f"- expected_actions: `{res['expected_actions']}`",
            "",
            "### Checks",
            "",
        ])
        lines.extend(f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in res["checks"].items())
    lines.extend([
        "",
        "## Overall checks",
        "",
    ])
    lines.extend(f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in overall_checks.items())
    (out_dir / "BRANCHING_OUTPUT_PROOF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[branching-output-proof] status={status}")
    print(f"[branching-output-proof] report={out_dir / 'BRANCHING_OUTPUT_PROOF.md'}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
