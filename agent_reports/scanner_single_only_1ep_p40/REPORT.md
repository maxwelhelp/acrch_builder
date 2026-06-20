# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_FAIL`
- honesty_score: `1.0`
- full_acc: `0.43046875`
- audit_acc: `0.43046875`
- deploy_acc: `0.43046875`
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
- sim_disabled_delta: `0.12177908420562744`
- gain_disabled_delta: `0.012141704559326172`
- sim_result_disabled_delta: `0.10660815238952637`
- slot_disabled_delta: `0.14818918704986572`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.7688134908676147`
- choice_without_sim_delta: `0.03779294341802597`
- grid_candidate_usage: `0.007665532641112804`
- semantic_candidate_usage: `0.8904533386230469`
- usage_candidate_usage: `0.0901564210653305`
- random_candidate_usage: `0.007020832039415836`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000102445483`
- single_signed_projection_usage: `0.0047038858756423`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.7422447800636292`
- single_signed_projection_signed_score_mean: `0.09915029257535934`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

