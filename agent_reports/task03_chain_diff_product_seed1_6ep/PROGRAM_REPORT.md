# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.71875`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9995994567871094, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8481341004371643, 'action_L0_0_1_diff_tape': 0.019043371081352234, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9997128844261169, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.846482515335083, 'action_L0_2_3_diff_tape': 0.00713622011244297, 'action_L1_1_3_product_present': 1.0, 'action_L1_1_3_product_choice_mass': 0.9989333152770996, 'action_L1_1_3_product_recovery': 1.0, 'action_L1_1_3_product_active': 0.9009690284729004, 'action_L1_1_3_product_tape': 0.37007737159729004}`

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
| s0 | channel:0.64/a0.001 | *diff:1.00/a0.848 | channel:0.46/a0.250 | diff:1.00/a0.790 |
| s1 | channel:0.27/a0.011 | channel:0.53/a0.003 | channel:0.36/a0.021 | channel:0.40/a0.404 |
| s2 | channel:0.33/a0.169 | diff:1.00/a0.795 | channel:0.59/a0.001 | *diff:1.00/a0.846 |
| s3 | channel:0.32/a0.011 | channel:0.38/a0.420 | channel:0.37/a0.020 | channel:0.51/a0.003 |

### Cells
- edge 0->0: top=channel mass=0.6400, active=0.000520, transform=0.1230, skip=0.0354, disable=0.8415, tape=0.000009
- edge 0->1 EXPECTED: top=diff mass=0.9996, expected=diff mass=0.9996, active=0.848134, transform=0.9331, skip=0.0400, disable=0.0269, tape=0.019043
- edge 0->2: top=channel mass=0.4625, active=0.250161, transform=0.6125, skip=0.2504, disable=0.1371, tape=0.006926
- edge 0->3: top=diff mass=0.9996, active=0.790339, transform=0.9525, skip=0.0318, disable=0.0156, tape=0.008674
- edge 1->0: top=channel mass=0.2700, active=0.010760, transform=0.0471, skip=0.7876, disable=0.1653, tape=0.000163
- edge 1->1: top=channel mass=0.5288, active=0.003012, transform=0.1092, skip=0.1388, disable=0.7520, tape=0.000060
- edge 1->2: top=channel mass=0.3603, active=0.021497, transform=0.0514, skip=0.8260, disable=0.1227, tape=0.000268
- edge 1->3: top=channel mass=0.4021, active=0.403699, transform=0.4022, skip=0.5294, disable=0.0684, tape=0.010766
- edge 2->0: top=channel mass=0.3265, active=0.168873, transform=0.4427, skip=0.3275, disable=0.2299, tape=0.004130
- edge 2->1: top=diff mass=0.9996, active=0.795155, transform=0.8882, skip=0.0681, disable=0.0437, tape=0.010909
- edge 2->2: top=channel mass=0.5875, active=0.001028, transform=0.1130, skip=0.0554, disable=0.8316, tape=0.000010
- edge 2->3 EXPECTED: top=diff mass=0.9997, expected=diff mass=0.9997, active=0.846483, transform=0.9179, skip=0.0567, disable=0.0254, tape=0.007136
- edge 3->0: top=channel mass=0.3154, active=0.010797, transform=0.0304, skip=0.8768, disable=0.0928, tape=0.000131
- edge 3->1: top=channel mass=0.3775, active=0.420419, transform=0.2426, skip=0.6890, disable=0.0683, tape=0.011487
- edge 3->2: top=channel mass=0.3658, active=0.020134, transform=0.0326, skip=0.8963, disable=0.0711, tape=0.000158
- edge 3->3: top=channel mass=0.5057, active=0.002869, transform=0.1639, skip=0.2837, disable=0.5524, tape=0.000046

## Layer 1

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `1`
- active_cells: `5`
- active_edges_per_target: `1.25`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | gated_keep:0.24/a0.000 | contrast:0.23/a0.000 | split:0.15/a0.000 | split:0.35/a0.064 |
| s1 | split:0.22/a0.084 | smooth:0.30/a0.007 | split:0.28/a0.067 | *product:1.00/a0.901 |
| s2 | gated_keep:0.24/a0.000 | contrast:0.24/a0.000 | gated_keep:0.20/a0.000 | split:0.39/a0.079 |
| s3 | split:0.17/a0.001 | contrast:0.17/a0.040 | split:0.17/a0.000 | split:0.36/a0.006 |

### Cells
- edge 0->0: top=gated_keep mass=0.2353, active=0.000008, transform=0.0010, skip=0.5495, disable=0.4495, tape=0.000000
- edge 0->1: top=contrast mass=0.2300, active=0.000230, transform=0.0074, skip=0.7122, disable=0.2804, tape=0.000003
- edge 0->2: top=split mass=0.1495, active=0.000184, transform=0.0034, skip=0.4697, disable=0.5269, tape=0.000001
- edge 0->3: top=split mass=0.3537, active=0.064322, transform=0.0412, skip=0.5460, disable=0.4128, tape=0.002969
- edge 1->0: top=split mass=0.2237, active=0.084182, transform=0.0322, skip=0.5540, disable=0.4138, tape=0.002206
- edge 1->1: top=smooth mass=0.3000, active=0.007227, transform=0.0103, skip=0.7241, disable=0.2656, tape=0.000313
- edge 1->2: top=split mass=0.2842, active=0.066509, transform=0.0234, skip=0.4802, disable=0.4963, tape=0.001097
- edge 1->3 EXPECTED: top=product mass=0.9989, expected=product mass=0.9989, active=0.900969, transform=0.5226, skip=0.2818, disable=0.1955, tape=0.370077
- edge 2->0: top=gated_keep mass=0.2369, active=0.000490, transform=0.0025, skip=0.6727, disable=0.3249, tape=0.000001
- edge 2->1: top=contrast mass=0.2438, active=0.000370, transform=0.0034, skip=0.8090, disable=0.1876, tape=0.000001
- edge 2->2: top=gated_keep mass=0.2012, active=0.000016, transform=0.0004, skip=0.6015, disable=0.3982, tape=0.000000
- edge 2->3: top=split mass=0.3909, active=0.078656, transform=0.0210, skip=0.6781, disable=0.3008, tape=0.001823
- edge 3->0: top=split mass=0.1743, active=0.000824, transform=0.0064, skip=0.3712, disable=0.6224, tape=0.000006
- edge 3->1: top=contrast mass=0.1668, active=0.039692, transform=0.0722, skip=0.5075, disable=0.4203, tape=0.005542
- edge 3->2: top=split mass=0.1739, active=0.000237, transform=0.0034, skip=0.3018, disable=0.6947, tape=0.000001
- edge 3->3: top=split mass=0.3565, active=0.005885, transform=0.0068, skip=0.3934, disable=0.5997, tape=0.000113

