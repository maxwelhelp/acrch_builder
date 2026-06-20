# Latest vertical slice report

- report_dir: `agent_reports/task_config_diff_smoke`
- task: `diff`
- task_config_path: `/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/configs/tasks/diff.yml`
- task_config_name: `diff`
- task_config_digest: `1b2c75610ea7a43bde373cb23d606b27840ebccec8a09442e3bf7d4356d0daf4`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}]`
- pass_thresholds: `{'oracle_acc': {'min': 1.0}}`
- pass_threshold_results: `{'oracle_acc': {'value': 1.0, 'min': 1.0, 'passed': True}}`
- pass_thresholds_met: `True`
- epochs: `1`
- state_norm_mode: `none`
- best_acc: `0.375`
- last_acc: `0.375`
- program_recovery_rate: `0.0`
- expected_edge_recovery: `0.0`
- expected_any_recovery: `0.015625`
- expected_edge_active: `0.24408698081970215`
- expected_candidate_present: `0.375`
- expected_edge_choice_mass: `0.016559310257434845`
- gain_disabled_delta: `0.0007848143577575684`
- sim_result_disabled_delta: `0.012217164039611816`
- sim_disabled_delta: `0.01645958423614502`
- choice_without_sim_delta: `0.026864953339099884`
- semantic_grid_mismatch: `0.33984375`
- grid_candidate_usage: `0.3871384859085083`
- semantic_candidate_usage: `0.05750472843647003`
- usage_candidate_usage: `0.5547360181808472`
- random_candidate_usage: `0.0006207986152730882`
- scanner_source_mass_sum: `1.0000000311410986`
- skip_mass: `0.2878210246562958`
- transform_mass: `0.5189962387084961`
- disable_mass: `0.19318273663520813`
- oracle_acc: `1.0`
- edge_pair_bias_expected: `0.0005050016334280372`
- write_pair_bias_expected: `0.0005044666468165815`
- phase_pair_bias_expected: `0.0005054734647274017`
- cell_output_pair_bias_expected: `-0.0004687932669185102`
- edge_pair_bias_std: `0.0005177803104743361`
- write_pair_bias_std: `0.0005064281285740435`
- choice_entropy: `1.147475004196167`
- edge_scale_mean: `0.9466619491577148`
- cell_output_gate_mean: `0.5263203382492065`
- cell_tape_weight_mean: `0.061641789972782135`

## Conclusion
vertical slice completed

Program details: `PROGRAM_REPORT.md` and `program_epoch_XXX.json`.
