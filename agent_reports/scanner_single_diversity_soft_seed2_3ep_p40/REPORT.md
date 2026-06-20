# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.58125`
- audit_acc: `0.58125`
- deploy_acc: `0.58125`
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
- sim_disabled_delta: `0.31019127368927`
- gain_disabled_delta: `0.08780884742736816`
- sim_result_disabled_delta: `0.03790116310119629`
- slot_disabled_delta: `0.21812033653259277`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `1.124912142753601`
- choice_without_sim_delta: `0.041616179049015045`
- grid_candidate_usage: `0.00356255448423326`
- semantic_candidate_usage: `0.21744942665100098`
- usage_candidate_usage: `0.6913793087005615`
- random_candidate_usage: `0.08727110922336578`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `0.9999999841966201`
- single_signed_projection_usage: `0.00033758513745851815`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.699507474899292`
- single_signed_projection_signed_score_mean: `-0.2134905755519867`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

