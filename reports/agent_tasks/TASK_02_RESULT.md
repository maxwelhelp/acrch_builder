# TASK 02 RESULT — PARTIAL

Task: multi-cell anti-collapse on `two_diff`.

Baseline:
- rec=1.0
- choice_mass≈1.0
- active_cells=9 / program_active_cells=10
- program_expected_top_cells=6
- verdict=primitive_collapse

Stronger sparse config seed1:
- best_acc=0.9230
- expected_candidate_present=1.0
- expected_edge_choice_mass=0.99965
- expected_edge_recovery=1.0
- program_expected_top_cells=[4]
- program_active_cells=[8]
- primitive_top_share=0.2615
- verdict=sparse_or_partly_sparse_program
- sim_disabled_delta=+0.00475
- choice_without_sim_delta=0.0238
- loss_accounting_error=6.3e-10

Conclusion:
- Task 02 fix works on seed1.
- Need seeds 2/3 with same config before PASS.
