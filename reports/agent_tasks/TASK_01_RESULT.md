# TASK 01 RESULT — PASS

Task: single primitive learned baselines.

Runs:
- diff seeds 1/2/3: PASS
- merge seeds 1/2/3: PASS
- product seeds 1/2/3: PASS

Acceptance:
- oracle accuracy = 1.0
- expected_candidate_present = 1.0
- expected_edge_choice_mass >= 0.70
- expected_edge_recovery >= 0.80
- validation accuracy reproducibly above random

Observed:
- diff best val: 0.8824–0.9434
- merge best val: 0.8949–0.9012
- product best val: 0.9004–0.9172
- recovery = 1.0 for all runs
- choice_mass ≈ 1.0 for all runs

Known remaining issue:
- primitive_collapse / dense active cells still appears in several reports.
- This is not solved in Task 01.
- This is owned by Task 02: multi-cell anti-collapse.

Verdict: PASS. Next allowed task: 02_MULTI_CELL_ANTI_COLLAPSE.
