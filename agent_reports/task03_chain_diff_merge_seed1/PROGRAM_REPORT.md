# ActionMatrix Program Report

- task: `chain_diff_merge`
- batch_acc: `0.9296875`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'merge'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9995518922805786, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8478552103042603, 'action_L0_0_1_diff_tape': 0.009779071435332298, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9996253848075867, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.8509679436683655, 'action_L0_2_3_diff_tape': 0.0019659644458442926, 'action_L1_1_3_merge_present': 1.0, 'action_L1_1_3_merge_choice_mass': 0.9993162155151367, 'action_L1_1_3_merge_recovery': 1.0, 'action_L1_1_3_merge_active': 0.8432520031929016, 'action_L1_1_3_merge_tape': 0.008635833859443665}`

Flow:
```text
input -> Layer 0 -> Layer 1 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `4`
- active_cells: `8`
- active_edges_per_target: `2.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.62/a0.015 | *diff:1.00/a0.848 | ctx_matrix:0.54/a0.260 | diff:1.00/a0.841 |
| s1 | ctx_matrix:0.43/a0.007 | ctx_matrix:0.51/a0.032 | ctx_matrix:0.50/a0.010 | ctx_matrix:0.60/a0.323 |
| s2 | ctx_matrix:0.44/a0.214 | diff:1.00/a0.835 | ctx_matrix:0.58/a0.020 | *diff:1.00/a0.851 |
| s3 | ctx_matrix:0.44/a0.006 | ctx_matrix:0.40/a0.300 | ctx_matrix:0.51/a0.009 | ctx_matrix:0.56/a0.032 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.6249, active=0.015197, transform=0.0129, skip=0.7993, disable=0.1878, tape=0.000029
- edge 0->1 EXPECTED: top=diff mass=0.9996, expected=diff mass=0.9996, active=0.847855, transform=0.9084, skip=0.0667, disable=0.0250, tape=0.009779
- edge 0->2: top=ctx_matrix mass=0.5413, active=0.260399, transform=0.1659, skip=0.6964, disable=0.1377, tape=0.002552
- edge 0->3: top=diff mass=0.9996, active=0.840580, transform=0.9215, skip=0.0566, disable=0.0220, tape=0.004430
- edge 1->0: top=ctx_matrix mass=0.4312, active=0.007278, transform=0.1048, skip=0.2821, disable=0.6131, tape=0.000292
- edge 1->1: top=ctx_matrix mass=0.5084, active=0.031889, transform=0.2094, skip=0.0879, disable=0.7027, tape=0.000518
- edge 1->2: top=ctx_matrix mass=0.4976, active=0.009808, transform=0.1053, skip=0.2378, disable=0.6569, tape=0.000215
- edge 1->3: top=ctx_matrix mass=0.5961, active=0.322931, transform=0.8123, skip=0.0232, disable=0.1645, tape=0.005456
- edge 2->0: top=ctx_matrix mass=0.4377, active=0.213645, transform=0.1672, skip=0.7045, disable=0.1283, tape=0.002019
- edge 2->1: top=diff mass=0.9996, active=0.835400, transform=0.9237, skip=0.0521, disable=0.0242, tape=0.003812
- edge 2->2: top=ctx_matrix mass=0.5768, active=0.019632, transform=0.0191, skip=0.7293, disable=0.2516, tape=0.000017
- edge 2->3 EXPECTED: top=diff mass=0.9996, expected=diff mass=0.9996, active=0.850968, transform=0.9330, skip=0.0461, disable=0.0209, tape=0.001966
- edge 3->0: top=ctx_matrix mass=0.4444, active=0.006234, transform=0.0953, skip=0.3369, disable=0.5678, tape=0.000170
- edge 3->1: top=ctx_matrix mass=0.4014, active=0.300149, transform=0.7923, skip=0.0350, disable=0.1727, tape=0.006955
- edge 3->2: top=ctx_matrix mass=0.5089, active=0.009132, transform=0.1009, skip=0.2914, disable=0.6076, tape=0.000150
- edge 3->3: top=ctx_matrix mass=0.5576, active=0.031576, transform=0.2401, skip=0.1067, disable=0.6533, tape=0.000203

## Layer 1

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `1`
- active_cells: `8`
- active_edges_per_target: `2.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'merge'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.66/a0.002 | ctx_matrix:0.54/a0.027 | ctx_matrix:0.36/a0.007 | ctx_matrix:0.56/a0.206 |
| s1 | ctx_matrix:0.59/a0.222 | ctx_matrix:0.72/a0.323 | ctx_matrix:0.49/a0.235 | *merge:1.00/a0.843 |
| s2 | ctx_matrix:0.58/a0.009 | ctx_matrix:0.61/a0.041 | ctx_matrix:0.60/a0.004 | ctx_matrix:0.67/a0.261 |
| s3 | ctx_matrix:0.44/a0.034 | ctx_matrix:0.49/a0.187 | output_write:0.37/a0.037 | ctx_matrix:0.67/a0.333 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.6645, active=0.002020, transform=0.1079, skip=0.5426, disable=0.3496, tape=0.000172
- edge 0->1: top=ctx_matrix mass=0.5364, active=0.027297, transform=0.3032, skip=0.5413, disable=0.1554, tape=0.005377
- edge 0->2: top=ctx_matrix mass=0.3557, active=0.006683, transform=0.1879, skip=0.3883, disable=0.4238, tape=0.000770
- edge 0->3: top=ctx_matrix mass=0.5594, active=0.206266, transform=0.0969, skip=0.6459, disable=0.2572, tape=0.008156
- edge 1->0: top=ctx_matrix mass=0.5914, active=0.221812, transform=0.0868, skip=0.4392, disable=0.4740, tape=0.008379
- edge 1->1: top=ctx_matrix mass=0.7161, active=0.323499, transform=0.1326, skip=0.6196, disable=0.2478, tape=0.022308
- edge 1->2: top=ctx_matrix mass=0.4922, active=0.234576, transform=0.0961, skip=0.3481, disable=0.5558, tape=0.008679
- edge 1->3 EXPECTED: top=merge mass=0.9993, expected=merge mass=0.9993, active=0.843252, transform=0.0634, skip=0.5765, disable=0.3601, tape=0.008636
- edge 2->0: top=ctx_matrix mass=0.5835, active=0.008737, transform=0.1364, skip=0.5189, disable=0.3448, tape=0.000803
- edge 2->1: top=ctx_matrix mass=0.6113, active=0.041189, transform=0.2562, skip=0.5907, disable=0.1531, tape=0.007286
- edge 2->2: top=ctx_matrix mass=0.6021, active=0.003728, transform=0.0988, skip=0.4846, disable=0.4166, tape=0.000302
- edge 2->3: top=ctx_matrix mass=0.6680, active=0.261375, transform=0.0765, skip=0.6796, disable=0.2439, tape=0.008607
- edge 3->0: top=ctx_matrix mass=0.4410, active=0.034022, transform=0.2381, skip=0.2830, disable=0.4789, tape=0.005834
- edge 3->1: top=ctx_matrix mass=0.4868, active=0.187461, transform=0.4974, skip=0.2961, disable=0.2065, tape=0.068696
- edge 3->2: top=output_write mass=0.3662, active=0.036547, transform=0.2485, skip=0.2171, disable=0.5343, tape=0.006281
- edge 3->3: top=ctx_matrix mass=0.6737, active=0.332512, transform=0.1071, skip=0.4939, disable=0.3990, tape=0.020088

