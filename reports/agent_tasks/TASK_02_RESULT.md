# TASK 02 RESULT — PARTIAL

Task: multi-cell anti-collapse on `two_diff`.

| Run | OK | best_acc | cand | choice_mass | recovery | top_cells | active_cells | top_share | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `agent_reports/task02_two_diff_stronger_sparse_seed1` | True | 0.9230 | 1.000 | 0.9997 | 1.000 | 4 | 8 | 0.2615 | sparse_or_partly_sparse_program |
| `agent_reports/task02_two_diff_stronger_sparse_seed2` | False | 0.8187 | 1.000 | 0.9991 | 1.000 | 4 | 16 | 0.2696 | sparse_or_partly_sparse_program |
| `agent_reports/task02_two_diff_stronger_sparse_seed3` | True | 0.9309 | 1.000 | 0.9995 | 1.000 | 4 | 8 | 0.2532 | sparse_or_partly_sparse_program |

Acceptance:
- both expected actions present=1
- choice_mass>=0.70
- recovery>=0.80
- expected_top_cells<=4
- active_cells<16
- primitive_top_share<0.85
- program verdict != primitive_collapse
- validation accuracy > random

Verdict: PARTIAL.
Next: inspect failed rows before Task 03.
