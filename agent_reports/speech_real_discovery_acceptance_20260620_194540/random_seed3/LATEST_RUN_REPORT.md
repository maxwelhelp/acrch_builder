# Audio frontend report

- dataset: `speechcommands`
- variant: `structured`
- status: `SMOKE_FAIL`
- honesty_score: `1.0`
- full_acc: `0.35546875`
- audit_acc: `0.35546875`
- deploy_acc: `0.35546875`
- frontend_params: `922.0`
- frontend_flops: `1163835.0`
- frontend_activation_bytes: `181868.0`
- model_params: `707411.0`
- model_flops: `1360443.0`
- model_activation_bytes: `183916.0`

## Checks
- deploy_above_random: `PASS`
- honesty_retained: `PASS`
- simulator_ce_ablation_positive: `FAIL`
- simulator_changes_choice: `PASS`
- non_grid_scanner_usage_positive: `PASS`
- layer0_ablation_positive: `PASS`

## Ablations
- sim_disabled_delta: `-0.15834200382232666`
- gain_disabled_delta: `-0.14438819885253906`
- sim_result_disabled_delta: `-0.04945707321166992`
- slot_disabled_delta: `0.03818035125732422`
- layer0_output_disabled_delta: `0.0`
- layer0_state_disabled_delta: `0.4967515468597412`
- choice_without_sim_delta: `0.03995461389422417`
- grid_candidate_usage: `0.843936562538147`
- semantic_candidate_usage: `0.05937238037586212`
- usage_candidate_usage: `0.06272414326667786`
- random_candidate_usage: `0.033966898918151855`
- global_candidate_usage: `0.0`
- grid_candidate_coverage: `1.0`
- semantic_candidate_coverage: `1.0`
- usage_candidate_coverage: `1.0`
- random_candidate_coverage: `1.0`
- global_candidate_coverage: `0.0`
- scanner_source_mass_sum: `0.9999999850988388`

