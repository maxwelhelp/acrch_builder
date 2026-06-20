# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.59140625`
- audit_acc: `0.59140625`
- deploy_acc: `0.59140625`
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
- sim_disabled_delta: `0.14660954475402832`
- gain_disabled_delta: `-0.07437145709991455`
- sim_result_disabled_delta: `-0.0032346248626708984`
- slot_disabled_delta: `0.29026007652282715`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `1.363732933998108`
- choice_without_sim_delta: `0.03444889187812805`
- grid_candidate_usage: `0.016755111515522003`
- semantic_candidate_usage: `0.0017597529804334044`
- usage_candidate_usage: `0.9659082889556885`
- random_candidate_usage: `0.015576853416860104`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.000000006868504`

