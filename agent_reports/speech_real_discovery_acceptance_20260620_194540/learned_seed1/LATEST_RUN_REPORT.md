# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_PASS`
- honesty_score: `1.0`
- full_acc: `0.58359375`
- audit_acc: `0.58359375`
- deploy_acc: `0.58359375`
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
- sim_disabled_delta: `0.5811140537261963`
- gain_disabled_delta: `0.041054487228393555`
- sim_result_disabled_delta: `0.19343769550323486`
- slot_disabled_delta: `0.5010620355606079`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `1.5613442659378052`
- choice_without_sim_delta: `0.05359935387969017`
- grid_candidate_usage: `0.017205368727445602`
- semantic_candidate_usage: `0.021540699526667595`
- usage_candidate_usage: `0.9462040662765503`
- random_candidate_usage: `0.015049888752400875`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `1.0000000232830644`

