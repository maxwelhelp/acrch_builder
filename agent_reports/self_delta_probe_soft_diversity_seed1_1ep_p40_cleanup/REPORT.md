# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.240625`
- audit_acc: `0.240625`
- deploy_acc: `0.240625`
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
- sim_disabled_delta: `0.004891157150268555`
- gain_disabled_delta: `-0.040250420570373535`
- sim_result_disabled_delta: `-0.043160200119018555`
- slot_disabled_delta: `0.11405587196350098`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.3227806091308594`
- choice_without_sim_delta: `0.040034614503383636`
- self_delta_disabled_delta: `0.0`
- self_delta_zero_delta: `0.0`
- self_delta_shuffle_delta: `0.0010941028594970703`
- choice_without_self_delta_delta: `0.0`
- choice_zero_self_delta_delta: `0.0`
- choice_shuffle_self_delta_delta: `0.006353174336254597`
- grid_candidate_usage: `0.18050533533096313`
- semantic_candidate_usage: `0.23110081255435944`
- usage_candidate_usage: `0.5689479112625122`
- random_candidate_usage: `0.019440965726971626`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.000000024210749`
- single_signed_projection_usage: `4.999335942557082e-06`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.8728710412979126`
- single_signed_projection_signed_score_mean: `0.06639068573713303`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

