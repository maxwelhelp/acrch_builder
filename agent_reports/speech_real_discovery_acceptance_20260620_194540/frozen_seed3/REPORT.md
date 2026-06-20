# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.32421875`
- audit_acc: `0.32421875`
- deploy_acc: `0.32421875`
- frontend_params: `922.0`
- frontend_flops: `1163835.0`
- frontend_activation_bytes: `181868.0`
- model_params: `707411.0`
- model_flops: `1360443.0`
- model_activation_bytes: `183916.0`

## Checks
- deploy_above_random: `PASS`
- honesty_retained: `PASS`
- simulator_ce_ablation_positive: `PASS`
- simulator_changes_choice: `PASS`
- non_grid_scanner_usage_positive: `PASS`
- layer0_ablation_positive: `PASS`

## Ablations
- sim_disabled_delta: `0.09929132461547852`
- gain_disabled_delta: `-0.06232893466949463`
- sim_result_disabled_delta: `-0.003125429153442383`
- slot_disabled_delta: `-0.01007688045501709`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.6819065809249878`
- choice_without_sim_delta: `0.03848586231470108`
- grid_candidate_usage: `0.8482016324996948`
- semantic_candidate_usage: `0.07751457393169403`
- usage_candidate_usage: `0.04426814243197441`
- random_candidate_usage: `0.030015679076313972`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000279396772`

