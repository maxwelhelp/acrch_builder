# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_FAIL`
- honesty_score: `1.0`
- full_acc: `0.40703125`
- audit_acc: `0.40703125`
- deploy_acc: `0.40703125`
- frontend_params: `922.0`
- frontend_flops: `1163835.0`
- frontend_activation_bytes: `181868.0`
- model_params: `707411.0`
- model_flops: `1360443.0`
- model_activation_bytes: `183916.0`

## Checks
- deploy_above_random: `PASS`
- honesty_retained: `PASS`
- simulator_ce_ablation_positive: `FAIL`
- simulator_changes_choice: `PASS`
- non_grid_scanner_usage_positive: `PASS`
- layer0_ablation_positive: `PASS`

## Ablations
- sim_disabled_delta: `-0.09836649894714355`
- gain_disabled_delta: `-0.1103287935256958`
- sim_result_disabled_delta: `-0.0015072822570800781`
- slot_disabled_delta: `0.07340490818023682`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.6910936832427979`
- choice_without_sim_delta: `0.039570022374391556`
- grid_candidate_usage: `0.14806324243545532`
- semantic_candidate_usage: `0.6721744537353516`
- usage_candidate_usage: `0.14804816246032715`
- random_candidate_usage: `0.03171415999531746`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000186264515`

