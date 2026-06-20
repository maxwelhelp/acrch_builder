# TASK 04 RESULT — PASS

Goal: check if learned credit paths are useful, not only decorative.

| Run | acc | recovery | choice | sim_delta | choice_delta | layer_dep | layer_ablate | active |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `agent_reports/task03_chain_diff_merge_seed1` | 0.9422 | 1.000 | 0.9995 | 0.0140 | 0.0205 | 0.4250 | 0.4418 | [8, 8] |
| `agent_reports/task03_chain_diff_merge_seed2` | 0.9164 | 1.000 | 0.9992 | 0.0075 | 0.0189 | 0.4133 | 0.4305 | [8, 8] |
| `agent_reports/task03_chain_diff_merge_seed3` | 0.8598 | 1.000 | 0.9977 | 0.1708 | 0.0381 | 0.3582 | 0.3492 | [8, 8] |
| `agent_reports/task03_chain_diff_product_seed1_6ep` | 0.7461 | 1.000 | 0.9994 | 0.0090 | 0.0252 | 0.2480 | 0.2613 | [8, 5] |
| `agent_reports/task03_chain_diff_product_seed2_6ep` | 0.7438 | 1.000 | 0.9981 | 0.0164 | 0.0287 | 0.2520 | 0.2430 | [8, 4] |
| `agent_reports/task03_chain_diff_product_seed3_6ep` | 0.7164 | 1.000 | 0.9991 | 0.0017 | 0.0293 | 0.2230 | 0.2172 | [8, 4] |

Conclusion:
- layer credit useful: True
- simulator useful in most runs: True
- choice/controller useful in most runs: True

Verdict: PASS
