# TASK 03 RESULT — PASS

Task: learned sequential chain: `chain_diff_merge` then `chain_diff_product`.

| Type | Run | OK | best_acc | choice_mass | recovery | dep | ablate | active_cells | verdicts |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| merge | `agent_reports/task03_chain_diff_merge_seed1` | True | 0.9422 | 0.9995 | 1.000 | 0.4250 | 0.4418 | [8, 8] | ['sparse_or_partly_sparse_program', 'sparse_or_partly_sparse_program'] |
| merge | `agent_reports/task03_chain_diff_merge_seed2` | True | 0.9164 | 0.9992 | 1.000 | 0.4133 | 0.4305 | [8, 8] | ['sparse_or_partly_sparse_program', 'sparse_or_partly_sparse_program'] |
| merge | `agent_reports/task03_chain_diff_merge_seed3` | True | 0.8598 | 0.9977 | 1.000 | 0.3582 | 0.3492 | [8, 8] | ['sparse_or_partly_sparse_program', 'sparse_or_partly_sparse_program'] |
| product | `agent_reports/task03_chain_diff_product_seed1_6ep` | True | 0.7461 | 0.9994 | 1.000 | 0.2480 | 0.2613 | [8, 5] | ['sparse_or_partly_sparse_program', 'sparse_or_partly_sparse_program'] |
| product | `agent_reports/task03_chain_diff_product_seed2_6ep` | True | 0.7367 | 0.9984 | 1.000 | 0.2449 | 0.2359 | [8, 4] | ['sparse_or_partly_sparse_program', 'sparse_or_partly_sparse_program'] |
| product | `agent_reports/task03_chain_diff_product_seed3_6ep` | True | 0.7191 | 0.9990 | 1.000 | 0.2258 | 0.2199 | [8, 4] | ['sparse_or_partly_sparse_program', 'sparse_or_partly_sparse_program'] |

Verdict: PASS.
Next allowed task: 04_REAL_ABLATION_CREDIT if PASS.
