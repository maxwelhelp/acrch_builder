# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_FAIL`
- honesty_score: `1.0`
- full_acc: `0.39609375`
- audit_acc: `0.39609375`
- deploy_acc: `0.39609375`
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
- no_primitive_collapse: `FAIL`
- active_path_alive: `PASS`

## Ablations
- sim_disabled_delta: `0.17872858047485352`
- gain_disabled_delta: `-0.002797722816467285`
- sim_result_disabled_delta: `-0.004964709281921387`
- slot_disabled_delta: `0.03324019908905029`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.6251593828201294`
- choice_without_sim_delta: `0.050344184041023254`
- grid_candidate_usage: `0.04025786370038986`
- semantic_candidate_usage: `0.942484974861145`
- usage_candidate_usage: `0.008418949320912361`
- random_candidate_usage: `0.004783939570188522`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000107102096`
- single_signed_projection_usage: `0.004054283257573843`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.6969350576400757`
- single_signed_projection_signed_score_mean: `-0.011520572938024998`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`
- projection_logit_cap: `1.0`
- projection_logit_clipped_fraction: `0.0`

