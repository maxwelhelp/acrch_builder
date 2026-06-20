# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.38984375`
- audit_acc: `0.38984375`
- deploy_acc: `0.38984375`
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
- sim_disabled_delta: `0.20759010314941406`
- gain_disabled_delta: `-0.006316184997558594`
- sim_result_disabled_delta: `0.1738346815109253`
- slot_disabled_delta: `0.10466146469116211`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.5475856065750122`
- choice_without_sim_delta: `0.04892595857381821`
- grid_candidate_usage: `0.002875467762351036`
- semantic_candidate_usage: `0.020884860306978226`
- usage_candidate_usage: `0.8734044432640076`
- random_candidate_usage: `0.10193352401256561`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `0.9999999913270585`
- single_signed_projection_usage: `0.000901695981156081`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.7869242429733276`
- single_signed_projection_signed_score_mean: `-0.4031149446964264`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

