# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.38671875`
- audit_acc: `0.38671875`
- deploy_acc: `0.38671875`
- frontend_params: `922.0`
- frontend_flops: `1163835.0`
- frontend_activation_bytes: `181868.0`
- model_params: `1243097.0`
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
- sim_disabled_delta: `0.13582837581634521`
- gain_disabled_delta: `0.006305336952209473`
- sim_result_disabled_delta: `0.05818891525268555`
- slot_disabled_delta: `0.0982065200805664`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.691942572593689`
- choice_without_sim_delta: `0.03529665246605873`
- self_delta_disabled_delta: `0.0`
- self_delta_zero_delta: `0.0`
- self_delta_shuffle_delta: `0.001080632209777832`
- choice_without_self_delta_delta: `0.0`
- choice_zero_self_delta_delta: `0.0`
- choice_shuffle_self_delta_delta: `0.006607739254832268`
- grid_candidate_usage: `0.9019606709480286`
- semantic_candidate_usage: `0.0462474524974823`
- usage_candidate_usage: `0.004482968710362911`
- random_candidate_usage: `0.047249097377061844`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.000000008938514`
- single_signed_projection_usage: `5.981940557830967e-05`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.677004337310791`
- single_signed_projection_signed_score_mean: `-0.025841526687145233`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

