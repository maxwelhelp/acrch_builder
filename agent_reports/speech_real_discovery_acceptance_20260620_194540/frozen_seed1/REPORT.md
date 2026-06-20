# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_FAIL`
- honesty_score: `1.0`
- full_acc: `0.38359375`
- audit_acc: `0.38359375`
- deploy_acc: `0.38359375`
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
- sim_disabled_delta: `-0.2224646806716919`
- gain_disabled_delta: `-0.04414510726928711`
- sim_result_disabled_delta: `-0.2422645092010498`
- slot_disabled_delta: `0.240891695022583`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.5831506252288818`
- choice_without_sim_delta: `0.03833832964301109`
- grid_candidate_usage: `0.5232127904891968`
- semantic_candidate_usage: `0.0027525010518729687`
- usage_candidate_usage: `0.30248337984085083`
- random_candidate_usage: `0.17155134677886963`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000181607902`

