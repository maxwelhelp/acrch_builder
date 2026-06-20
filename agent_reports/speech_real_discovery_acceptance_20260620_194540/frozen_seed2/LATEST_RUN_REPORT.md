# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.38203125`
- audit_acc: `0.38203125`
- deploy_acc: `0.38203125`
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
- sim_disabled_delta: `0.006278634071350098`
- gain_disabled_delta: `-0.02019190788269043`
- sim_result_disabled_delta: `0.028638482093811035`
- slot_disabled_delta: `0.18434202671051025`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.7506917715072632`
- choice_without_sim_delta: `0.03949546813964844`
- grid_candidate_usage: `0.10434775054454803`
- semantic_candidate_usage: `0.7115823030471802`
- usage_candidate_usage: `0.15412400662899017`
- random_candidate_usage: `0.029945913702249527`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `0.9999999739229679`

