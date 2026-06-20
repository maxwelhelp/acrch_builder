# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.3828125`
- audit_acc: `0.3828125`
- deploy_acc: `0.3828125`
- frontend_params: `922.0`
- frontend_flops: `1163835.0`
- frontend_activation_bytes: `181868.0`
- model_params: `707414.0`
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
- sim_disabled_delta: `0.01837313175201416`
- gain_disabled_delta: `-0.0018557310104370117`
- sim_result_disabled_delta: `0.005137205123901367`
- slot_disabled_delta: `0.0016529560089111328`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.6374607086181641`
- choice_without_sim_delta: `0.04311847314238548`
- grid_candidate_usage: `0.03184225410223007`
- semantic_candidate_usage: `0.8695005178451538`
- usage_candidate_usage: `0.008000096306204796`
- random_candidate_usage: `0.07320376485586166`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000279396772`
- single_signed_projection_usage: `0.017453394830226898`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.6523111462593079`
- single_signed_projection_signed_score_mean: `-0.07781827449798584`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

