# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.73046875`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.999151349067688, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8700353503227234, 'action_L0_0_1_diff_tape': 0.02978404238820076, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9992695450782776, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.862358808517456, 'action_L0_2_3_diff_tape': 0.013843598775565624, 'action_L1_1_3_product_present': 1.0, 'action_L1_1_3_product_choice_mass': 0.9926378130912781, 'action_L1_1_3_product_recovery': 0.99609375, 'action_L1_1_3_product_active': 0.8857581615447998, 'action_L1_1_3_product_tape': 0.40293216705322266}`

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
| s0 | edge_gate:0.30/a0.000 | *diff:1.00/a0.870 | edge_gate:0.38/a0.214 | diff:1.00/a0.838 |
| s1 | contrast:0.22/a0.014 | contrast:0.31/a0.001 | edge_gate:0.22/a0.012 | edge_gate:0.35/a0.396 |
| s2 | edge_gate:0.29/a0.213 | diff:1.00/a0.831 | contrast:0.42/a0.000 | *diff:1.00/a0.862 |
| s3 | output_write:0.27/a0.014 | edge_gate:0.39/a0.408 | edge_gate:0.29/a0.012 | edge_gate:0.25/a0.001 |

### Cells
- edge 0->0: top=edge_gate mass=0.2982, active=0.000293, transform=0.1236, skip=0.2881, disable=0.5883, tape=0.000008
- edge 0->1 EXPECTED: top=diff mass=0.9992, expected=diff mass=0.9992, active=0.870035, transform=0.9155, skip=0.0515, disable=0.0330, tape=0.029784
- edge 0->2: top=edge_gate mass=0.3848, active=0.213835, transform=0.4192, skip=0.4401, disable=0.1406, tape=0.011254
- edge 0->3: top=diff mass=0.9993, active=0.837884, transform=0.9132, skip=0.0466, disable=0.0402, tape=0.018578
- edge 1->0: top=contrast mass=0.2229, active=0.014106, transform=0.0417, skip=0.8701, disable=0.0881, tape=0.000345
- edge 1->1: top=contrast mass=0.3118, active=0.000942, transform=0.0819, skip=0.5838, disable=0.3343, tape=0.000013
- edge 1->2: top=edge_gate mass=0.2200, active=0.012208, transform=0.0268, skip=0.9125, disable=0.0607, tape=0.000207
- edge 1->3: top=edge_gate mass=0.3515, active=0.396401, transform=0.3714, skip=0.5362, disable=0.0923, tape=0.014189
- edge 2->0: top=edge_gate mass=0.2901, active=0.212921, transform=0.4929, skip=0.3616, disable=0.1454, tape=0.010405
- edge 2->1: top=diff mass=0.9991, active=0.831311, transform=0.9107, skip=0.0593, disable=0.0300, tape=0.017563
- edge 2->2: top=contrast mass=0.4243, active=0.000177, transform=0.0892, skip=0.4251, disable=0.4858, tape=0.000003
- edge 2->3 EXPECTED: top=diff mass=0.9993, expected=diff mass=0.9993, active=0.862359, transform=0.9138, skip=0.0505, disable=0.0357, tape=0.013844
- edge 3->0: top=output_write mass=0.2692, active=0.013663, transform=0.0363, skip=0.8647, disable=0.0990, tape=0.000303
- edge 3->1: top=edge_gate mass=0.3917, active=0.408141, transform=0.3132, skip=0.5940, disable=0.0928, tape=0.021221
- edge 3->2: top=edge_gate mass=0.2932, active=0.012426, transform=0.0224, skip=0.9094, disable=0.0681, tape=0.000171
- edge 3->3: top=edge_gate mass=0.2497, active=0.001159, transform=0.0727, skip=0.5116, disable=0.4157, tape=0.000015

## Layer 1

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `1`
- active_cells: `4`
- active_edges_per_target: `1.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | split:0.36/a0.000 | split:0.44/a0.001 | replace:0.29/a0.000 | split:0.39/a0.075 |
| s1 | split:0.31/a0.057 | split:0.57/a0.008 | replace:0.32/a0.055 | *product:0.99/a0.886 |
| s2 | split:0.26/a0.000 | split:0.34/a0.000 | split:0.33/a0.000 | split:0.40/a0.050 |
| s3 | replace:0.25/a0.001 | split:0.21/a0.050 | replace:0.36/a0.000 | replace:0.41/a0.009 |

### Cells
- edge 0->0: top=split mass=0.3644, active=0.000011, transform=0.0009, skip=0.3785, disable=0.6206, tape=0.000000
- edge 0->1: top=split mass=0.4390, active=0.000596, transform=0.0099, skip=0.6528, disable=0.3374, tape=0.000013
- edge 0->2: top=replace mass=0.2934, active=0.000257, transform=0.0062, skip=0.4428, disable=0.5510, tape=0.000001
- edge 0->3: top=split mass=0.3885, active=0.075419, transform=0.0485, skip=0.6242, disable=0.3274, tape=0.005021
- edge 1->0: top=split mass=0.3072, active=0.057493, transform=0.0323, skip=0.3154, disable=0.6523, tape=0.001947
- edge 1->1: top=split mass=0.5656, active=0.007807, transform=0.0079, skip=0.4698, disable=0.5223, tape=0.000240
- edge 1->2: top=replace mass=0.3249, active=0.055162, transform=0.0271, skip=0.3076, disable=0.6653, tape=0.001173
- edge 1->3 EXPECTED: top=product mass=0.9926, expected=product mass=0.9926, active=0.885758, transform=0.5642, skip=0.2424, disable=0.1935, tape=0.402932
- edge 2->0: top=split mass=0.2565, active=0.000157, transform=0.0051, skip=0.3285, disable=0.6664, tape=0.000001
- edge 2->1: top=split mass=0.3437, active=0.000210, transform=0.0073, skip=0.5272, disable=0.4654, tape=0.000002
- edge 2->2: top=split mass=0.3294, active=0.000007, transform=0.0005, skip=0.2605, disable=0.7390, tape=0.000000
- edge 2->3: top=split mass=0.3957, active=0.049878, transform=0.0362, skip=0.5120, disable=0.4518, tape=0.001855
- edge 3->0: top=replace mass=0.2487, active=0.000724, transform=0.0094, skip=0.4736, disable=0.5170, tape=0.000014
- edge 3->1: top=split mass=0.2090, active=0.049728, transform=0.1032, skip=0.6545, disable=0.2423, tape=0.006429
- edge 3->2: top=replace mass=0.3601, active=0.000449, transform=0.0071, skip=0.4687, disable=0.5242, tape=0.000003
- edge 3->3: top=replace mass=0.4065, active=0.009263, transform=0.0087, skip=0.6070, disable=0.3843, tape=0.000336

