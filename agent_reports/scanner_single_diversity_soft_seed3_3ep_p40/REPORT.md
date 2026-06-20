# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.60859375`
- audit_acc: `0.60859375`
- deploy_acc: `0.60859375`
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
- sim_disabled_delta: `0.454365611076355`
- gain_disabled_delta: `0.008446812629699707`
- sim_result_disabled_delta: `0.2003687620162964`
- slot_disabled_delta: `-0.03647899627685547`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `1.3533103466033936`
- choice_without_sim_delta: `0.051273420453071594`
- grid_candidate_usage: `0.7587502002716064`
- semantic_candidate_usage: `0.2098095864057541`
- usage_candidate_usage: `0.0098006222397089`
- random_candidate_usage: `0.021307390183210373`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000365253072`
- single_signed_projection_usage: `0.00033223742502741516`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.8037040829658508`
- single_signed_projection_signed_score_mean: `0.08714281767606735`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

