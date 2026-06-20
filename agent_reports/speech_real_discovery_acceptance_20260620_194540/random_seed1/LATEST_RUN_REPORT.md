# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_FAIL`
- honesty_score: `1.0`
- full_acc: `0.39375`
- audit_acc: `0.39375`
- deploy_acc: `0.39375`
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
- sim_disabled_delta: `-0.15374243259429932`
- gain_disabled_delta: `-0.07195055484771729`
- sim_result_disabled_delta: `-0.09682440757751465`
- slot_disabled_delta: `0.07969987392425537`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.481442928314209`
- choice_without_sim_delta: `0.032962847501039505`
- grid_candidate_usage: `0.42533546686172485`
- semantic_candidate_usage: `0.004520951770246029`
- usage_candidate_usage: `0.37640082836151123`
- random_candidate_usage: `0.19374272227287292`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `0.999999969266355`

