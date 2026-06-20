# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.41015625`
- audit_acc: `0.41015625`
- deploy_acc: `0.41015625`
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
- sim_disabled_delta: `0.18399369716644287`
- gain_disabled_delta: `-0.01670098304748535`
- sim_result_disabled_delta: `0.17170941829681396`
- slot_disabled_delta: `0.34282422065734863`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.6258459091186523`
- choice_without_sim_delta: `0.04268745705485344`
- grid_candidate_usage: `0.0032003773376345634`
- semantic_candidate_usage: `0.013007540255784988`
- usage_candidate_usage: `0.684352695941925`
- random_candidate_usage: `0.269123911857605`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `0.9999999729916453`
- single_signed_projection_usage: `0.030315447598695755`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.7233800888061523`
- single_signed_projection_signed_score_mean: `0.014895686879754066`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

