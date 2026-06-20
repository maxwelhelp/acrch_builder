# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.5078125`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 0.3203125, 'action_L0_0_1_diff_choice_mass': 0.024353498592972755, 'action_L0_0_1_diff_recovery': 0.015625, 'action_L0_0_1_diff_active': 0.3592700660228729, 'action_L0_0_1_diff_tape': 0.13144230842590332, 'action_L0_2_3_diff_present': 0.24609375, 'action_L0_2_3_diff_choice_mass': 0.017872784286737442, 'action_L0_2_3_diff_recovery': 0.01171875, 'action_L0_2_3_diff_active': 0.4398302435874939, 'action_L0_2_3_diff_tape': 0.20783057808876038, 'action_L1_1_3_product_present': 0.4765625, 'action_L1_1_3_product_choice_mass': 0.07854343205690384, 'action_L1_1_3_product_recovery': 0.0234375, 'action_L1_1_3_product_active': 0.574420690536499, 'action_L1_1_3_product_tape': 0.24844439327716827}`

Flow:
```text
input -> Layer 0 -> Layer 1 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `not_recovered`
- expected_top_cells: `0`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.3995501548051834`
- non_expected_active_mean: `0.49814215089593616`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `1.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.22/a0.780 | *ctx_matrix:0.11/a0.359 | ctx_matrix:0.12/a0.374 | ctx_matrix:0.11/a0.384 |
| s1 | ctx_matrix:0.12/a0.355 | ctx_matrix:0.19/a0.788 | smooth:0.13/a0.391 | ctx_matrix:0.12/a0.401 |
| s2 | ctx_matrix:0.10/a0.413 | skip:0.10/a0.436 | ctx_matrix:0.18/a0.827 | *disable:0.14/a0.440 |
| s3 | ctx_matrix:0.14/a0.330 | ctx_matrix:0.10/a0.347 | ctx_matrix:0.11/a0.353 | ctx_matrix:0.17/a0.793 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.2197, active=0.780473, transform=0.9386, skip=0.0370, disable=0.0243, tape=0.388167
- edge 0->1 EXPECTED: top=ctx_matrix mass=0.1111, expected=diff mass=0.0244, active=0.359270, transform=0.6928, skip=0.1854, disable=0.1218, tape=0.131442
- edge 0->2: top=ctx_matrix mass=0.1234, active=0.374388, transform=0.7602, skip=0.1618, disable=0.0780, tape=0.133929
- edge 0->3: top=ctx_matrix mass=0.1148, active=0.384131, transform=0.7503, skip=0.1613, disable=0.0884, tape=0.135569
- edge 1->0: top=ctx_matrix mass=0.1219, active=0.354609, transform=0.7470, skip=0.1316, disable=0.1214, tape=0.143265
- edge 1->1: top=ctx_matrix mass=0.1889, active=0.787650, transform=0.9487, skip=0.0281, disable=0.0233, tape=0.422986
- edge 1->2: top=smooth mass=0.1258, active=0.391204, transform=0.7960, skip=0.1263, disable=0.0777, tape=0.155994
- edge 1->3: top=ctx_matrix mass=0.1162, active=0.401172, transform=0.7907, skip=0.1189, disable=0.0905, tape=0.155929
- edge 2->0: top=ctx_matrix mass=0.0982, active=0.413491, transform=0.7790, skip=0.1195, disable=0.1015, tape=0.203116
- edge 2->1: top=skip mass=0.0996, active=0.436001, transform=0.7675, skip=0.1259, disable=0.1066, tape=0.215411
- edge 2->2: top=ctx_matrix mass=0.1762, active=0.827455, transform=0.9680, skip=0.0202, disable=0.0118, tape=0.495349
- edge 2->3 EXPECTED: top=disable mass=0.1360, expected=diff mass=0.0179, active=0.439830, transform=0.8055, skip=0.1118, disable=0.0827, tape=0.207831
- edge 3->0: top=ctx_matrix mass=0.1416, active=0.329667, transform=0.6952, skip=0.1796, disable=0.1253, tape=0.132581
- edge 3->1: top=ctx_matrix mass=0.1016, active=0.347191, transform=0.6858, skip=0.1872, disable=0.1270, tape=0.141327
- edge 3->2: top=ctx_matrix mass=0.1142, active=0.353109, transform=0.7395, skip=0.1767, disable=0.0838, tape=0.141302
- edge 3->3: top=ctx_matrix mass=0.1703, active=0.793448, transform=0.9436, skip=0.0361, disable=0.0203, tape=0.402986

## Layer 1

- verdict: `not_recovered`
- expected_top_cells: `0`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.574420690536499`
- non_expected_active_mean: `0.6488056222597758`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | diff:0.28/a0.610 | diff:0.36/a0.733 | diff:0.32/a0.726 | diff:0.33/a0.649 |
| s1 | ctx_matrix:0.20/a0.531 | ctx_matrix:0.32/a0.667 | ctx_matrix:0.22/a0.661 | *diff:0.17/a0.574 |
| s2 | diff:0.21/a0.569 | diff:0.32/a0.698 | diff:0.20/a0.693 | diff:0.26/a0.611 |
| s3 | forget:0.18/a0.572 | diff:0.32/a0.699 | diff:0.27/a0.697 | diff:0.27/a0.615 |

### Cells
- edge 0->0: top=diff mass=0.2774, active=0.609852, transform=0.8347, skip=0.0697, disable=0.0956, tape=0.297416
- edge 0->1: top=diff mass=0.3579, active=0.732856, transform=0.9227, skip=0.0313, disable=0.0460, tape=0.491272
- edge 0->2: top=diff mass=0.3211, active=0.726170, transform=0.9190, skip=0.0298, disable=0.0512, tape=0.403648
- edge 0->3: top=diff mass=0.3335, active=0.649405, transform=0.8097, skip=0.0804, disable=0.1099, tape=0.314525
- edge 1->0: top=ctx_matrix mass=0.2004, active=0.530832, transform=0.7769, skip=0.1163, disable=0.1068, tape=0.234152
- edge 1->1: top=ctx_matrix mass=0.3188, active=0.666702, transform=0.8929, skip=0.0540, disable=0.0532, tape=0.421930
- edge 1->2: top=ctx_matrix mass=0.2191, active=0.661340, transform=0.8890, skip=0.0516, disable=0.0595, tape=0.345373
- edge 1->3 EXPECTED: top=diff mass=0.1734, expected=product mass=0.0785, active=0.574421, transform=0.7467, skip=0.1321, disable=0.1212, tape=0.248444
- edge 2->0: top=diff mass=0.2050, active=0.568593, transform=0.7956, skip=0.1087, disable=0.0957, tape=0.232587
- edge 2->1: top=diff mass=0.3235, active=0.698260, transform=0.9027, skip=0.0500, disable=0.0472, tape=0.419633
- edge 2->2: top=diff mass=0.1978, active=0.693448, transform=0.8999, skip=0.0475, disable=0.0526, tape=0.335672
- edge 2->3: top=diff mass=0.2603, active=0.611306, transform=0.7669, skip=0.1239, disable=0.1093, tape=0.249229
- edge 3->0: top=forget mass=0.1847, active=0.572149, transform=0.8150, skip=0.0844, disable=0.1006, tape=0.258362
- edge 3->1: top=diff mass=0.3184, active=0.699412, transform=0.9128, skip=0.0383, disable=0.0489, tape=0.444725
- edge 3->2: top=diff mass=0.2692, active=0.696691, transform=0.9090, skip=0.0364, disable=0.0546, tape=0.365792
- edge 3->3: top=diff mass=0.2695, active=0.615067, transform=0.7885, skip=0.0965, disable=0.1150, tape=0.274383

