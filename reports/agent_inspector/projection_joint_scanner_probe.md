# Projection joint scanner probe

- status: `PASS`
- device: `cuda`
- measured_delta_loss_is_source_of_truth: `true`
- expected_actions_used_for_training: `false`

## Checks

- first_diff_true_rank_le_4: `PASS`
- first_diff_direction_sign_preserved: `PASS`
- first_diff_top32_linear_acc_gt_0_90: `PASS`
- chain_exact_true_pair_rank_le_8: `PASS`
- chain_jl16_true_pair_rank_le_16: `PASS`
- chain_jl16_speedup_ge_1_5: `PASS`

## Key measurements

- first_diff true rank: `1`
- first_diff top32 linear acc: `0.9883`
- exact true pair rank: `1`
- JL16 true pair rank: `1`
- JL16 speedup: `1.973x`

JL is a proposer only. Bounded measured counterfactual delta-loss remains the training truth.
