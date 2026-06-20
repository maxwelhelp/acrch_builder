# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.36484375`
- audit_acc: `0.36484375`
- deploy_acc: `0.36484375`
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
- sim_disabled_delta: `0.06087028980255127`
- gain_disabled_delta: `-0.04363274574279785`
- sim_result_disabled_delta: `0.021419048309326172`
- slot_disabled_delta: `0.03274548053741455`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.4368835687637329`
- choice_without_sim_delta: `0.03607453778386116`
- grid_candidate_usage: `0.005993406753987074`
- semantic_candidate_usage: `0.09721852093935013`
- usage_candidate_usage: `0.6609783172607422`
- random_candidate_usage: `0.23565241694450378`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000191357685`
- single_signed_projection_usage: `0.00015735723718535155`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.7805211544036865`
- single_signed_projection_signed_score_mean: `-0.3705277442932129`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

