# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.36953125`
- audit_acc: `0.36953125`
- deploy_acc: `0.36953125`
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
- sim_disabled_delta: `0.04642009735107422`
- gain_disabled_delta: `-0.01295769214630127`
- sim_result_disabled_delta: `0.047429442405700684`
- slot_disabled_delta: `0.10703504085540771`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.666319727897644`
- choice_without_sim_delta: `0.04013016074895859`
- grid_candidate_usage: `0.009378280490636826`
- semantic_candidate_usage: `0.03309651091694832`
- usage_candidate_usage: `0.8944628238677979`
- random_candidate_usage: `0.05834338441491127`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000214204192`
- single_signed_projection_usage: `0.00471902173012495`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.6861973404884338`
- single_signed_projection_signed_score_mean: `-0.01334733422845602`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

