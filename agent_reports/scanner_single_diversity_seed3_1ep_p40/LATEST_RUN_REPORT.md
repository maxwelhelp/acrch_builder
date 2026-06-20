# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.375`
- audit_acc: `0.375`
- deploy_acc: `0.375`
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
- sim_disabled_delta: `0.08366680145263672`
- gain_disabled_delta: `-0.06396794319152832`
- sim_result_disabled_delta: `-0.03031635284423828`
- slot_disabled_delta: `-0.02131032943725586`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.34605860710144043`
- choice_without_sim_delta: `0.04720383509993553`
- grid_candidate_usage: `0.10602904856204987`
- semantic_candidate_usage: `0.009521468542516232`
- usage_candidate_usage: `0.6022138595581055`
- random_candidate_usage: `0.27920669317245483`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `0.9999999403953552`
- single_signed_projection_usage: `0.0030288705602288246`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.8261743783950806`
- single_signed_projection_signed_score_mean: `0.3085363209247589`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

