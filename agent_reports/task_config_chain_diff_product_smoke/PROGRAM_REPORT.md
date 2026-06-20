# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.5`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9992775917053223, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.7650565505027771, 'action_L0_0_1_diff_tape': 0.06068596988916397, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.998329222202301, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.7655416131019592, 'action_L0_2_3_diff_tape': 0.028326457366347313, 'action_L1_1_3_product_present': 0.6875, 'action_L1_1_3_product_choice_mass': 0.6872959136962891, 'action_L1_1_3_product_recovery': 0.6875, 'action_L1_1_3_product_active': 0.8880516290664673, 'action_L1_1_3_product_tape': 0.0493844598531723}`

Flow:
```text
input -> Layer 0 -> Layer 1 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `16`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.7652990818023682`
- non_expected_active_mean: `0.448379139815058`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | diff:0.97/a0.512 | *diff:1.00/a0.765 | diff:1.00/a0.650 | diff:0.99/a0.793 |
| s1 | diff:0.99/a0.133 | diff:0.99/a0.377 | diff:0.99/a0.262 | diff:0.93/a0.432 |
| s2 | diff:1.00/a0.397 | diff:1.00/a0.630 | diff:1.00/a0.560 | *diff:1.00/a0.766 |
| s3 | diff:0.99/a0.208 | diff:1.00/a0.461 | diff:1.00/a0.343 | diff:1.00/a0.518 |

### Cells
- edge 0->0: top=diff mass=0.9748, active=0.512067, transform=0.6365, skip=0.1860, disable=0.1775, tape=0.005747
- edge 0->1 EXPECTED: top=diff mass=0.9993, expected=diff mass=0.9993, active=0.765057, transform=0.5638, skip=0.2751, disable=0.1611, tape=0.060686
- edge 0->2: top=diff mass=0.9980, active=0.649763, transform=0.6204, skip=0.2548, disable=0.1248, tape=0.029236
- edge 0->3: top=diff mass=0.9868, active=0.792726, transform=0.5895, skip=0.2302, disable=0.1803, tape=0.027631
- edge 1->0: top=diff mass=0.9938, active=0.132664, transform=0.5689, skip=0.2375, disable=0.1937, tape=0.014910
- edge 1->1: top=diff mass=0.9933, active=0.377291, transform=0.5692, skip=0.2596, disable=0.1712, tape=0.010620
- edge 1->2: top=diff mass=0.9919, active=0.262484, transform=0.6075, skip=0.2386, disable=0.1538, tape=0.022458
- edge 1->3: top=diff mass=0.9341, active=0.432463, transform=0.5735, skip=0.2467, disable=0.1798, tape=0.030938
- edge 2->0: top=diff mass=0.9976, active=0.396788, transform=0.4889, skip=0.2590, disable=0.2521, tape=0.020214
- edge 2->1: top=diff mass=0.9982, active=0.630374, transform=0.4698, skip=0.3034, disable=0.2268, tape=0.038656
- edge 2->2: top=diff mass=0.9952, active=0.559603, transform=0.5846, skip=0.2366, disable=0.1787, tape=0.003916
- edge 2->3 EXPECTED: top=diff mass=0.9983, expected=diff mass=0.9983, active=0.765542, transform=0.4765, skip=0.2908, disable=0.2327, tape=0.028326
- edge 3->0: top=diff mass=0.9913, active=0.208384, transform=0.4749, skip=0.2596, disable=0.2655, tape=0.008000
- edge 3->1: top=diff mass=0.9980, active=0.461396, transform=0.4787, skip=0.3204, disable=0.2008, tape=0.029347
- edge 3->2: top=diff mass=0.9976, active=0.343489, transform=0.4996, skip=0.3005, disable=0.1999, tape=0.013866
- edge 3->3: top=diff mass=0.9951, active=0.517815, transform=0.5049, skip=0.2464, disable=0.2487, tape=0.003545

## Layer 1

- verdict: `primitive_collapse`
- expected_top_cells: `14`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.8880516290664673`
- non_expected_active_mean: `0.4549454073111216`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | product:0.62/a0.202 | product:0.87/a0.296 | product:0.80/a0.393 | product:0.81/a0.570 |
| s1 | product:0.73/a0.603 | product:0.94/a0.737 | product:0.69/a0.771 | *product:0.69/a0.888 |
| s2 | low_rank:0.64/a0.166 | product:0.81/a0.256 | product:0.75/a0.388 | product:0.49/a0.535 |
| s3 | low_rank:0.46/a0.287 | product:0.87/a0.401 | product:0.75/a0.508 | product:0.56/a0.714 |

### Cells
- edge 0->0: top=product mass=0.6191, active=0.202070, transform=0.0577, skip=0.8267, disable=0.1156, tape=0.003304
- edge 0->1: top=product mass=0.8737, active=0.295965, transform=0.1450, skip=0.5852, disable=0.2698, tape=0.013615
- edge 0->2: top=product mass=0.7972, active=0.392527, transform=0.0664, skip=0.6886, disable=0.2450, tape=0.008661
- edge 0->3: top=product mass=0.8117, active=0.569504, transform=0.1155, skip=0.5720, disable=0.3126, tape=0.023864
- edge 1->0: top=product mass=0.7302, active=0.603375, transform=0.0884, skip=0.8157, disable=0.0959, tape=0.015095
- edge 1->1: top=product mass=0.9371, active=0.737227, transform=0.1982, skip=0.5910, disable=0.2108, tape=0.042958
- edge 1->2: top=product mass=0.6868, active=0.770909, transform=0.0964, skip=0.7029, disable=0.2007, tape=0.022712
- edge 1->3 EXPECTED: top=product mass=0.6873, expected=product mass=0.6873, active=0.888052, transform=0.1720, skip=0.5737, disable=0.2543, tape=0.049384
- edge 2->0: top=low_rank mass=0.6357, active=0.165596, transform=0.1012, skip=0.6973, disable=0.2015, tape=0.004268
- edge 2->1: top=product mass=0.8107, active=0.256027, transform=0.1975, skip=0.4260, disable=0.3765, tape=0.013230
- edge 2->2: top=product mass=0.7494, active=0.387588, transform=0.0912, skip=0.5587, disable=0.3501, tape=0.009938
- edge 2->3: top=product mass=0.4902, active=0.534711, transform=0.1648, skip=0.4020, disable=0.4332, tape=0.027520
- edge 3->0: top=low_rank mass=0.4600, active=0.286515, transform=0.1144, skip=0.6899, disable=0.1956, tape=0.008717
- edge 3->1: top=product mass=0.8707, active=0.400828, transform=0.2268, skip=0.4138, disable=0.3594, tape=0.026649
- edge 3->2: top=product mass=0.7489, active=0.507598, transform=0.1150, skip=0.5250, disable=0.3600, tape=0.018470
- edge 3->3: top=product mass=0.5624, active=0.713741, transform=0.1394, skip=0.4203, disable=0.4404, tape=0.032380

