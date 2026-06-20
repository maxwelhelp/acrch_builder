# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_FAIL`
- honesty_score: `1.0`
- full_acc: `0.3671875`
- audit_acc: `0.3671875`
- deploy_acc: `0.3671875`
- frontend_params: `922.0`
- frontend_flops: `1163835.0`
- frontend_activation_bytes: `181868.0`
- model_params: `707411.0`
- model_flops: `1360443.0`
- model_activation_bytes: `183916.0`

## Checks
- deploy_above_random: `PASS`
- simulator_ce_ablation_positive: `FAIL`
- simulator_changes_choice: `PASS`
- non_grid_scanner_usage_positive: `PASS`
- layer0_ablation_positive: `PASS`
- credit_closed: `PASS`
- bounded_credit_budget: `PASS`
- joint_credit_measured: `PASS`
- random_credit_budget_nonzero: `PASS`
- unchosen_candidate_credit_measured: `PASS`
- no_primitive_collapse: `PASS`
- active_path_alive: `PASS`

## Ablations
- sim_disabled_delta: `-0.03133547306060791`
- gain_disabled_delta: `-0.0011893510818481445`
- sim_result_disabled_delta: `0.022968411445617676`
- slot_disabled_delta: `0.22856342792510986`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `1.1298002004623413`
- choice_without_sim_delta: `0.035855505615472794`
- grid_candidate_usage: `0.06404060125350952`
- semantic_candidate_usage: `0.025705689564347267`
- usage_candidate_usage: `0.7268800735473633`
- random_candidate_usage: `0.18337363004684448`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `0.9999999944120646`
- single_signed_projection_usage: `0.0`
- pair_jl16_usage: `0.0`

