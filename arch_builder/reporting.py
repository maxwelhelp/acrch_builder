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
        "task", "epochs", "best_acc", "last_acc", "program_recovery_rate",
        "expected_edge_recovery", "expected_any_recovery", "expected_edge_active", "expected_candidate_present", "expected_edge_choice_mass",
        "sim_disabled_delta", "choice_without_sim_delta",
        "semantic_grid_mismatch", "skip_mass", "transform_mass", "disable_mass",
        "choice_entropy", "edge_scale_mean", "cell_output_gate_mean", "cell_tape_weight_mean",
    ]:
        if k in summary:
            lines.append(f"- {k}: `{summary[k]}`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append(str(summary.get("conclusion", "Run finished. Inspect metrics.csv and final_report.json.")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
