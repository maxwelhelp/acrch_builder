# ActionMatrix Program Report

- task: `chain_diff_merge`
- batch_acc: `0.45703125`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'merge'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 0.3828125, 'action_L0_0_1_diff_choice_mass': 0.38049498200416565, 'action_L0_0_1_diff_recovery': 0.3828125, 'action_L0_0_1_diff_active': 0.3057166337966919, 'action_L0_0_1_diff_tape': 0.09150981158018112, 'action_L0_2_3_diff_present': 0.36328125, 'action_L0_2_3_diff_choice_mass': 0.3607144355773926, 'action_L0_2_3_diff_recovery': 0.36328125, 'action_L0_2_3_diff_active': 0.2765435576438904, 'action_L0_2_3_diff_tape': 0.09179061651229858, 'action_L1_1_3_merge_present': 0.328125, 'action_L1_1_3_merge_choice_mass': 0.32525065541267395, 'action_L1_1_3_merge_recovery': 0.328125, 'action_L1_1_3_merge_active': 0.4745144844055176, 'action_L1_1_3_merge_tape': 0.09756515175104141}`

Flow:
```text
input -> Layer 0 -> Layer 1 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `15`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | diff:0.43/a0.362 | *diff:0.38/a0.306 | diff:0.30/a0.305 | diff:0.34/a0.280 |
| s1 | diff:0.35/a0.271 | diff:0.36/a0.327 | diff:0.32/a0.275 | diff:0.28/a0.251 |
| s2 | diff:0.36/a0.295 | diff:0.39/a0.298 | output_write:0.30/a0.349 | *diff:0.36/a0.277 |
| s3 | diff:0.39/a0.292 | diff:0.44/a0.297 | diff:0.37/a0.304 | diff:0.39/a0.337 |

### Cells
- edge 0->0: top=diff mass=0.4285, active=0.361786, transform=0.6088, skip=0.1720, disable=0.2192, tape=0.109503
- edge 0->1 EXPECTED: top=diff mass=0.3805, expected=diff mass=0.3805, active=0.305717, transform=0.5610, skip=0.2881, disable=0.1509, tape=0.091510
- edge 0->2: top=diff mass=0.3034, active=0.304833, transform=0.5625, skip=0.2465, disable=0.1910, tape=0.100497
- edge 0->3: top=diff mass=0.3380, active=0.279817, transform=0.6289, skip=0.2119, disable=0.1592, tape=0.098678
- edge 1->0: top=diff mass=0.3489, active=0.271254, transform=0.5485, skip=0.2896, disable=0.1618, tape=0.083270
- edge 1->1: top=diff mass=0.3615, active=0.327279, transform=0.6266, skip=0.1953, disable=0.1780, tape=0.078786
- edge 1->2: top=diff mass=0.3183, active=0.275284, transform=0.5506, skip=0.2688, disable=0.1806, tape=0.082207
- edge 1->3: top=diff mass=0.2834, active=0.251396, transform=0.6256, skip=0.2276, disable=0.1468, tape=0.082217
- edge 2->0: top=diff mass=0.3615, active=0.294529, transform=0.5914, skip=0.2572, disable=0.1514, tape=0.092260
- edge 2->1: top=diff mass=0.3913, active=0.298034, transform=0.5925, skip=0.2710, disable=0.1365, tape=0.083342
- edge 2->2: top=output_write mass=0.2989, active=0.348887, transform=0.6659, skip=0.1374, disable=0.1967, tape=0.093618
- edge 2->3 EXPECTED: top=diff mass=0.3607, expected=diff mass=0.3607, active=0.276544, transform=0.6628, skip=0.1966, disable=0.1405, tape=0.091791
- edge 3->0: top=diff mass=0.3877, active=0.292134, transform=0.5159, skip=0.2988, disable=0.1854, tape=0.083019
- edge 3->1: top=diff mass=0.4448, active=0.297397, transform=0.5210, skip=0.3191, disable=0.1599, tape=0.076664
- edge 3->2: top=diff mass=0.3679, active=0.303517, transform=0.5222, skip=0.2766, disable=0.2012, tape=0.085059
- edge 3->3: top=diff mass=0.3854, active=0.337456, transform=0.6659, skip=0.1431, disable=0.1910, tape=0.092387

## Layer 1

- verdict: `primitive_collapse`
- expected_top_cells: `6`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'merge'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | identity:0.26/a0.383 | identity:0.27/a0.336 | identity:0.30/a0.271 | identity:0.30/a0.428 |
| s1 | merge:0.35/a0.413 | identity:0.34/a0.389 | identity:0.37/a0.308 | *identity:0.34/a0.475 |
| s2 | merge:0.44/a0.372 | merge:0.28/a0.337 | merge:0.34/a0.283 | merge:0.43/a0.427 |
| s3 | merge:0.24/a0.422 | identity:0.25/a0.383 | identity:0.28/a0.318 | identity:0.30/a0.492 |

### Cells
- edge 0->0: top=identity mass=0.2606, active=0.382558, transform=0.4675, skip=0.3274, disable=0.2051, tape=0.103979
- edge 0->1: top=identity mass=0.2705, active=0.336245, transform=0.4354, skip=0.3630, disable=0.2016, tape=0.079426
- edge 0->2: top=identity mass=0.2999, active=0.270688, transform=0.4211, skip=0.3594, disable=0.2194, tape=0.069822
- edge 0->3: top=identity mass=0.2961, active=0.427699, transform=0.4255, skip=0.3502, disable=0.2243, tape=0.095549
- edge 1->0: top=merge mass=0.3515, active=0.412635, transform=0.4394, skip=0.3148, disable=0.2458, tape=0.106945
- edge 1->1: top=identity mass=0.3386, active=0.388557, transform=0.3841, skip=0.3698, disable=0.2461, tape=0.083201
- edge 1->2: top=identity mass=0.3688, active=0.308398, transform=0.3794, skip=0.3555, disable=0.2651, tape=0.072929
- edge 1->3 EXPECTED: top=identity mass=0.3434, expected=merge mass=0.3253, active=0.474514, transform=0.3841, skip=0.3451, disable=0.2708, tape=0.097565
- edge 2->0: top=merge mass=0.4381, active=0.372328, transform=0.4862, skip=0.3210, disable=0.1928, tape=0.109395
- edge 2->1: top=merge mass=0.2840, active=0.337148, transform=0.4379, skip=0.3693, disable=0.1928, tape=0.083882
- edge 2->2: top=merge mass=0.3388, active=0.283294, transform=0.4055, skip=0.3808, disable=0.2137, tape=0.073662
- edge 2->3: top=merge mass=0.4312, active=0.427280, transform=0.4280, skip=0.3564, disable=0.2156, tape=0.100460
- edge 3->0: top=merge mass=0.2419, active=0.421563, transform=0.4215, skip=0.3266, disable=0.2519, tape=0.105972
- edge 3->1: top=identity mass=0.2461, active=0.383497, transform=0.3779, skip=0.3720, disable=0.2500, tape=0.081132
- edge 3->2: top=identity mass=0.2764, active=0.317905, transform=0.3618, skip=0.3664, disable=0.2718, tape=0.072348
- edge 3->3: top=identity mass=0.2981, active=0.492413, transform=0.3513, skip=0.3686, disable=0.2801, tape=0.094673

