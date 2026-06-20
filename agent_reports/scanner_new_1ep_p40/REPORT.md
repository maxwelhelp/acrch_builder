# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.40234375`
- audit_acc: `0.40234375`
- deploy_acc: `0.40234375`
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
- sim_disabled_delta: `0.22653186321258545`
- gain_disabled_delta: `0.05806779861450195`
- sim_result_disabled_delta: `-0.012293457984924316`
- slot_disabled_delta: `0.08195853233337402`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.6333932876586914`
- choice_without_sim_delta: `0.044099271297454834`
- grid_candidate_usage: `0.006886820774525404`
- semantic_candidate_usage: `0.8038226962089539`
- usage_candidate_usage: `0.13209056854248047`
- random_candidate_usage: `0.05651295930147171`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.000000043444743`
- single_signed_projection_usage: `0.0006533468258567154`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.6782807111740112`
- single_signed_projection_signed_score_mean: `-0.018945962190628052`
- pair_jl16_usage: `3.3651791454758495e-05`
- pair_jl16_candidate_count: `64.0`
- pair_jl16_top_score: `0.6456157565116882`
- pair_jl16_seconds: `0.0029486799612641335`
- pair_jl16_pairs_tested: `1024.0`

