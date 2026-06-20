# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.3453125`
- audit_acc: `0.3453125`
- deploy_acc: `0.3453125`
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
- sim_disabled_delta: `0.04309821128845215`
- gain_disabled_delta: `-0.004141569137573242`
- sim_result_disabled_delta: `0.02423238754272461`
- slot_disabled_delta: `0.05082857608795166`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.4188135862350464`
- choice_without_sim_delta: `0.04517503082752228`
- self_delta_disabled_delta: `0.0`
- self_delta_zero_delta: `0.0`
- self_delta_shuffle_delta: `0.00013709068298339844`
- choice_without_self_delta_delta: `0.0`
- choice_zero_self_delta_delta: `0.0`
- choice_shuffle_self_delta_delta: `0.004334559664130211`
- grid_candidate_usage: `0.044594526290893555`
- semantic_candidate_usage: `0.06194707751274109`
- usage_candidate_usage: `0.6966794729232788`
- random_candidate_usage: `0.1953643560409546`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `0.9999999443534762`
- single_signed_projection_usage: `0.0014145115856081247`
- single_signed_projection_candidate_count: `32.0`
- single_signed_projection_top_score: `0.9060262441635132`
- single_signed_projection_signed_score_mean: `-0.18161511421203613`
- pair_jl16_usage: `0.0`
- pair_jl16_candidate_count: `0.0`
- pair_jl16_top_score: `0.0`
- pair_jl16_seconds: `0.0`
- pair_jl16_pairs_tested: `0.0`

