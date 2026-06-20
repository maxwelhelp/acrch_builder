# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
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
- sim_disabled_delta: `0.05269002914428711`
- gain_disabled_delta: `-0.004887700080871582`
- sim_result_disabled_delta: `0.023560285568237305`
- slot_disabled_delta: `0.07680869102478027`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.5923283100128174`
- choice_without_sim_delta: `0.0419880636036396`
- grid_candidate_usage: `0.7576330900192261`
- semantic_candidate_usage: `0.0003087936493102461`
- usage_candidate_usage: `0.041540488600730896`
- random_candidate_usage: `0.20051762461662292`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `0.9999999968858901`
- single_signed_projection_usage: `0.0`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.7928820252418518`
- single_signed_projection_signed_score_mean: `0.1380549967288971`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

