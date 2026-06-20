# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.3375`
- audit_acc: `0.3375`
- deploy_acc: `0.3375`
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
- sim_disabled_delta: `0.0674903392791748`
- gain_disabled_delta: `0.022916316986083984`
- sim_result_disabled_delta: `0.010900139808654785`
- slot_disabled_delta: `0.09525561332702637`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.5936574935913086`
- choice_without_sim_delta: `0.033953484147787094`
- grid_candidate_usage: `0.007212703116238117`
- semantic_candidate_usage: `0.030708452686667442`
- usage_candidate_usage: `0.9162147045135498`
- random_candidate_usage: `0.04561493545770645`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000272411853`
- single_signed_projection_usage: `0.00024923146702349186`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.6495281457901001`
- single_signed_projection_signed_score_mean: `-0.0039055119268596172`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

