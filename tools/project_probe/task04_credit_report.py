import json
from pathlib import Path

runs = [
    "agent_reports/task03_chain_diff_merge_seed1/final_report.json",
    "agent_reports/task03_chain_diff_merge_seed2/final_report.json",
    "agent_reports/task03_chain_diff_merge_seed3/final_report.json",
    "agent_reports/task03_chain_diff_product_seed1_6ep/final_report.json",
    "agent_reports/task03_chain_diff_product_seed2_6ep/final_report.json",
    "agent_reports/task03_chain_diff_product_seed3_6ep/final_report.json",
]

rows = []
for p in runs:
    d = json.load(open(p))
    rows.append({
        "run": str(Path(p).parent),
        "task": d.get("task"),
        "best_acc": d.get("best_acc"),
        "recovery": d.get("expected_edge_recovery"),
        "choice_mass": d.get("expected_edge_choice_mass"),
        "sim_delta": d.get("sim_disabled_delta"),
        "gain_delta": d.get("gain_disabled_delta"),
        "sim_result_delta": d.get("sim_result_disabled_delta"),
        "choice_delta": d.get("choice_without_sim_delta"),
        "layer_dep": d.get("layer_dependency_delta"),
        "layer_ablate": d.get("layer_ablation_delta"),
        "active": d.get("program_active_cells"),
        "verdicts": d.get("program_verdicts"),
    })

def ok(x):
    return x is not None and x > 0

credit_ok = all(ok(r["layer_dep"]) and ok(r["layer_ablate"]) for r in rows)
sim_useful = sum(1 for r in rows if ok(r["sim_delta"])) >= 4
choice_useful = sum(1 for r in rows if ok(r["choice_delta"])) >= 4

status = "PASS" if credit_ok and choice_useful else "PARTIAL"

out = Path("reports/agent_tasks/TASK_04_RESULT.md")
with out.open("w") as f:
    f.write(f"# TASK 04 RESULT — {status}\n\n")
    f.write("Goal: check if learned credit paths are useful, not only decorative.\n\n")
    f.write("| Run | acc | recovery | choice | sim_delta | choice_delta | layer_dep | layer_ablate | active |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        f.write(
            f"| `{r['run']}` | {r['best_acc']:.4f} | {r['recovery']:.3f} | {r['choice_mass']:.4f} | "
            f"{(r['sim_delta'] or 0):.4f} | {(r['choice_delta'] or 0):.4f} | "
            f"{(r['layer_dep'] or 0):.4f} | {(r['layer_ablate'] or 0):.4f} | {r['active']} |\n"
        )
    f.write("\nConclusion:\n")
    f.write(f"- layer credit useful: {credit_ok}\n")
    f.write(f"- simulator useful in most runs: {sim_useful}\n")
    f.write(f"- choice/controller useful in most runs: {choice_useful}\n")
    f.write(f"\nVerdict: {status}\n")

print("TASK04", status)
print(out)
