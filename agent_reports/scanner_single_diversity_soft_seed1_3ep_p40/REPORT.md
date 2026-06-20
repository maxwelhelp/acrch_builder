# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.57109375`
- audit_acc: `0.57109375`
- deploy_acc: `0.57109375`
- frontend_params: `922.0`
- frontend_flops: `1163835.0`
- frontend_activation_bytes: `181868.0`
- model_params: `707411.0`
- model_flops: `1360443.0`
- model_activation_bytes: `183916.0`

## Checks
- deploy_above_random: `PASS`
- simulator_ce_ablation_positive: `PASS`
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
- sim_disabled_delta: `0.3674508333206177`
- gain_disabled_delta: `0.08495450019836426`
- sim_result_disabled_delta: `0.13148260116577148`
- slot_disabled_delta: `0.29551756381988525`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.9517067670822144`
- choice_without_sim_delta: `0.045154210180044174`
- grid_candidate_usage: `0.008285813964903355`
- semantic_candidate_usage: `0.022117923945188522`
- usage_candidate_usage: `0.8342631459236145`
- random_candidate_usage: `0.12964020669460297`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000079162419`
- single_signed_projection_usage: `0.005692917387932539`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.7540271878242493`
- single_signed_projection_signed_score_mean: `0.0917855054140091`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

