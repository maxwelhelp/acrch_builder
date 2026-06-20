# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.7265625`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9994949102401733, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8707141876220703, 'action_L0_0_1_diff_tape': 0.030430499464273453, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9995462894439697, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.8636530637741089, 'action_L0_2_3_diff_tape': 0.014127811416983604, 'action_L1_1_3_product_present': 1.0, 'action_L1_1_3_product_choice_mass': 0.9945136904716492, 'action_L1_1_3_product_recovery': 0.99609375, 'action_L1_1_3_product_active': 0.8818128108978271, 'action_L1_1_3_product_tape': 0.3866000473499298}`

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
| s0 | contrast:0.30/a0.000 | *diff:1.00/a0.871 | edge_gate:0.33/a0.212 | diff:1.00/a0.839 |
| s1 | contrast:0.23/a0.014 | contrast:0.29/a0.001 | edge_gate:0.21/a0.012 | edge_gate:0.31/a0.393 |
| s2 | edge_gate:0.25/a0.215 | diff:1.00/a0.833 | contrast:0.38/a0.000 | *diff:1.00/a0.864 |
| s3 | output_write:0.29/a0.014 | edge_gate:0.31/a0.405 | edge_gate:0.25/a0.012 | contrast:0.22/a0.001 |

### Cells
- edge 0->0: top=contrast mass=0.2964, active=0.000255, transform=0.1265, skip=0.2772, disable=0.5963, tape=0.000008
- edge 0->1 EXPECTED: top=diff mass=0.9995, expected=diff mass=0.9995, active=0.870714, transform=0.9166, skip=0.0514, disable=0.0320, tape=0.030430
- edge 0->2: top=edge_gate mass=0.3315, active=0.211533, transform=0.4264, skip=0.4287, disable=0.1449, tape=0.011240
- edge 0->3: top=diff mass=0.9996, active=0.838697, transform=0.9128, skip=0.0482, disable=0.0389, tape=0.018816
- edge 1->0: top=contrast mass=0.2258, active=0.013912, transform=0.0386, skip=0.8755, disable=0.0860, tape=0.000323
- edge 1->1: top=contrast mass=0.2913, active=0.000766, transform=0.0810, skip=0.5815, disable=0.3375, tape=0.000011
- edge 1->2: top=edge_gate mass=0.2148, active=0.011687, transform=0.0260, skip=0.9112, disable=0.0627, tape=0.000191
- edge 1->3: top=edge_gate mass=0.3131, active=0.393431, transform=0.3504, skip=0.5601, disable=0.0896, tape=0.013652
- edge 2->0: top=edge_gate mass=0.2521, active=0.215083, transform=0.4945, skip=0.3610, disable=0.1445, tape=0.010603
- edge 2->1: top=diff mass=0.9995, active=0.832729, transform=0.9115, skip=0.0596, disable=0.0290, tape=0.018029
- edge 2->2: top=contrast mass=0.3796, active=0.000150, transform=0.0929, skip=0.3965, disable=0.5106, tape=0.000002
- edge 2->3 EXPECTED: top=diff mass=0.9995, expected=diff mass=0.9995, active=0.863653, transform=0.9129, skip=0.0523, disable=0.0347, tape=0.014128
- edge 3->0: top=output_write mass=0.2924, active=0.013579, transform=0.0341, skip=0.8668, disable=0.0991, tape=0.000285
- edge 3->1: top=edge_gate mass=0.3121, active=0.404985, transform=0.3027, skip=0.6045, disable=0.0928, tape=0.021198
- edge 3->2: top=edge_gate mass=0.2481, active=0.011949, transform=0.0220, skip=0.9061, disable=0.0719, tape=0.000162
- edge 3->3: top=contrast mass=0.2243, active=0.000954, transform=0.0702, skip=0.5090, disable=0.4208, tape=0.000012

## Layer 1

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `1`
- active_cells: `4`
- active_edges_per_target: `1.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | replace:0.41/a0.000 | split:0.33/a0.001 | replace:0.32/a0.000 | replace:0.33/a0.074 |
| s1 | replace:0.36/a0.055 | split:0.29/a0.007 | replace:0.39/a0.054 | *product:0.99/a0.882 |
| s2 | replace:0.22/a0.000 | split:0.26/a0.000 | replace:0.29/a0.000 | replace:0.22/a0.050 |
| s3 | replace:0.32/a0.001 | split:0.21/a0.048 | replace:0.40/a0.000 | replace:0.48/a0.008 |

### Cells
- edge 0->0: top=replace mass=0.4084, active=0.000010, transform=0.0009, skip=0.3828, disable=0.6163, tape=0.000000
- edge 0->1: top=split mass=0.3284, active=0.000601, transform=0.0095, skip=0.6509, disable=0.3396, tape=0.000011
- edge 0->2: top=replace mass=0.3195, active=0.000256, transform=0.0067, skip=0.4487, disable=0.5447, tape=0.000001
- edge 0->3: top=replace mass=0.3271, active=0.074018, transform=0.0439, skip=0.6210, disable=0.3350, tape=0.004366
- edge 1->0: top=replace mass=0.3609, active=0.054776, transform=0.0309, skip=0.3247, disable=0.6444, tape=0.001819
- edge 1->1: top=split mass=0.2926, active=0.007427, transform=0.0074, skip=0.4851, disable=0.5075, tape=0.000237
- edge 1->2: top=replace mass=0.3898, active=0.053733, transform=0.0284, skip=0.3229, disable=0.6487, tape=0.001257
- edge 1->3 EXPECTED: top=product mass=0.9945, expected=product mass=0.9945, active=0.881813, transform=0.5441, skip=0.2545, disable=0.2014, tape=0.386600
- edge 2->0: top=replace mass=0.2225, active=0.000151, transform=0.0051, skip=0.3326, disable=0.6623, tape=0.000000
- edge 2->1: top=split mass=0.2572, active=0.000200, transform=0.0072, skip=0.5312, disable=0.4616, tape=0.000002
- edge 2->2: top=replace mass=0.2850, active=0.000007, transform=0.0006, skip=0.2751, disable=0.7243, tape=0.000000
- edge 2->3: top=replace mass=0.2179, active=0.049564, transform=0.0340, skip=0.5123, disable=0.4538, tape=0.001725
- edge 3->0: top=replace mass=0.3158, active=0.000747, transform=0.0089, skip=0.4841, disable=0.5069, tape=0.000025
- edge 3->1: top=split mass=0.2065, active=0.048042, transform=0.1002, skip=0.6597, disable=0.2402, tape=0.006477
- edge 3->2: top=replace mass=0.3990, active=0.000417, transform=0.0072, skip=0.4866, disable=0.5062, tape=0.000003
- edge 3->3: top=replace mass=0.4780, active=0.008006, transform=0.0072, skip=0.6182, disable=0.3747, tape=0.000234

