# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.66796875`
- audit_acc: `0.66796875`
- deploy_acc: `0.66796875`
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
- sim_disabled_delta: `0.10262542963027954`
- gain_disabled_delta: `0.05968743562698364`
- sim_result_disabled_delta: `0.02759373188018799`
- slot_disabled_delta: `0.034261465072631836`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `2.0268478393554688`
- choice_without_sim_delta: `0.03389231488108635`
- grid_candidate_usage: `0.8072059750556946`
- semantic_candidate_usage: `0.0029513707850128412`
- usage_candidate_usage: `0.18013326823711395`
- random_candidate_usage: `0.009709391742944717`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.000000005820766`

