# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.75390625`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9400465488433838, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.9729090929031372, 'action_L0_0_1_diff_tape': 0.5265398025512695, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.5511260628700256, 'action_L0_2_3_diff_recovery': 0.82421875, 'action_L0_2_3_diff_active': 3.050417660688254e-07, 'action_L0_2_3_diff_tape': 1.500450821367849e-07, 'action_L1_1_3_product_present': 1.0, 'action_L1_1_3_product_choice_mass': 0.9974941611289978, 'action_L1_1_3_product_recovery': 1.0, 'action_L1_1_3_product_active': 0.591428279876709, 'action_L1_1_3_product_tape': 0.5363919138908386}`

Flow:
```text
input -> Layer 0 -> Layer 1 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `13`
- active_cells: `3`
- active_edges_per_target: `0.75`
- expected_active_mean: `0.48645469897245164`
- non_expected_active_mean: `0.027879224543842097`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | smooth:0.33/a0.001 | *diff:0.94/a0.973 | diff:0.86/a0.282 | diff:0.74/a0.001 |
| s1 | smooth:0.33/a0.000 | diff:0.63/a0.001 | diff:0.71/a0.007 | diff:0.35/a0.000 |
| s2 | diff:0.50/a0.000 | diff:0.93/a0.075 | diff:0.36/a0.000 | *diff:0.55/a0.000 |
| s3 | diff:0.38/a0.000 | diff:0.89/a0.023 | diff:0.73/a0.000 | smooth:0.44/a0.000 |

### Cells
- edge 0->0: top=smooth mass=0.3292, active=0.000554, transform=0.8674, skip=0.0086, disable=0.1241, tape=0.000203
- edge 0->1 EXPECTED: top=diff mass=0.9400, expected=diff mass=0.9400, active=0.972909, transform=0.9995, skip=0.0002, disable=0.0003, tape=0.526540
- edge 0->2: top=diff mass=0.8616, active=0.282179, transform=0.9972, skip=0.0020, disable=0.0008, tape=0.145802
- edge 0->3: top=diff mass=0.7441, active=0.000554, transform=0.9955, skip=0.0034, disable=0.0011, tape=0.000269
- edge 1->0: top=smooth mass=0.3340, active=0.000158, transform=0.3997, skip=0.3955, disable=0.2047, tape=0.000040
- edge 1->1: top=diff mass=0.6313, active=0.001035, transform=0.6719, skip=0.0056, disable=0.3226, tape=0.000408
- edge 1->2: top=diff mass=0.7143, active=0.007199, transform=0.7831, skip=0.1575, disable=0.0593, tape=0.003340
- edge 1->3: top=diff mass=0.3525, active=0.000004, transform=0.7184, skip=0.1966, disable=0.0850, tape=0.000002
- edge 2->0: top=diff mass=0.4966, active=0.000025, transform=0.7440, skip=0.0959, disable=0.1601, tape=0.000010
- edge 2->1: top=diff mass=0.9281, active=0.075031, transform=0.9802, skip=0.0019, disable=0.0179, tape=0.038971
- edge 2->2: top=diff mass=0.3593, active=0.000001, transform=0.8040, skip=0.0081, disable=0.1879, tape=0.000000
- edge 2->3 EXPECTED: top=diff mass=0.5511, expected=diff mass=0.5511, active=0.000000, transform=0.9382, skip=0.0264, disable=0.0354, tape=0.000000
- edge 3->0: top=diff mass=0.3764, active=0.000003, transform=0.4994, skip=0.1985, disable=0.3021, tape=0.000001
- edge 3->1: top=diff mass=0.8871, active=0.023317, transform=0.9435, skip=0.0076, disable=0.0489, tape=0.012207
- edge 3->2: top=diff mass=0.7287, active=0.000250, transform=0.8790, skip=0.0615, disable=0.0594, tape=0.000133
- edge 3->3: top=smooth mass=0.4440, active=0.000000, transform=0.4891, skip=0.0170, disable=0.4939, tape=0.000000

## Layer 1

- verdict: `primitive_collapse`
- expected_top_cells: `11`
- active_cells: `3`
- active_edges_per_target: `0.75`
- expected_active_mean: `0.591428279876709`
- non_expected_active_mean: `0.007662995028263471`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | diff:0.26/a0.000 | diff:0.20/a0.000 | product:0.49/a0.000 | product:1.00/a0.052 |
| s1 | diff:0.12/a0.000 | diff:0.12/a0.000 | product:0.54/a0.000 | *product:1.00/a0.591 |
| s2 | product:0.12/a0.000 | product:0.25/a0.000 | smooth:0.12/a0.000 | product:1.00/a0.062 |
| s3 | product:0.37/a0.000 | product:0.46/a0.000 | product:0.29/a0.000 | product:0.71/a0.000 |

### Cells
- edge 0->0: top=diff mass=0.2623, active=0.000000, transform=0.9045, skip=0.0198, disable=0.0757, tape=0.000000
- edge 0->1: top=diff mass=0.2042, active=0.000000, transform=0.9014, skip=0.0186, disable=0.0801, tape=0.000000
- edge 0->2: top=product mass=0.4935, active=0.000000, transform=0.8815, skip=0.0231, disable=0.0955, tape=0.000000
- edge 0->3: top=product mass=0.9984, active=0.052429, transform=0.8348, skip=0.0073, disable=0.1579, tape=0.030779
- edge 1->0: top=diff mass=0.1230, active=0.000048, transform=0.9810, skip=0.0143, disable=0.0047, tape=0.000044
- edge 1->1: top=diff mass=0.1161, active=0.000002, transform=0.9902, skip=0.0046, disable=0.0052, tape=0.000002
- edge 1->2: top=product mass=0.5440, active=0.000372, transform=0.9753, skip=0.0182, disable=0.0065, tape=0.000289
- edge 1->3 EXPECTED: top=product mass=0.9975, expected=product mass=0.9975, active=0.591428, transform=0.9821, skip=0.0051, disable=0.0128, tape=0.536392
- edge 2->0: top=product mass=0.1199, active=0.000003, transform=0.1172, skip=0.6081, disable=0.2747, tape=0.000000
- edge 2->1: top=product mass=0.2538, active=0.000046, transform=0.1589, skip=0.5630, disable=0.2781, tape=0.000002
- edge 2->2: top=smooth mass=0.1241, active=0.000000, transform=0.1156, skip=0.5571, disable=0.3273, tape=0.000000
- edge 2->3: top=product mass=0.9963, active=0.061629, transform=0.1451, skip=0.2463, disable=0.6086, tape=0.001572
- edge 3->0: top=product mass=0.3666, active=0.000004, transform=0.6084, skip=0.3165, disable=0.0751, tape=0.000001
- edge 3->1: top=product mass=0.4600, active=0.000078, transform=0.6181, skip=0.2895, disable=0.0925, tape=0.000034
- edge 3->2: top=product mass=0.2913, active=0.000000, transform=0.6359, skip=0.2868, disable=0.0773, tape=0.000000
- edge 3->3: top=product mass=0.7065, active=0.000333, transform=0.8578, skip=0.0495, disable=0.0926, tape=0.000227

