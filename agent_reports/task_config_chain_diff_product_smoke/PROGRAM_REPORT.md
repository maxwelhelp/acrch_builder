# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.625`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 0.9375, 'action_L0_0_1_diff_choice_mass': 0.20542562007904053, 'action_L0_0_1_diff_recovery': 0.25, 'action_L0_0_1_diff_active': 0.22412073612213135, 'action_L0_0_1_diff_tape': 0.054987832903862, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.4742199778556824, 'action_L0_2_3_diff_recovery': 0.6875, 'action_L0_2_3_diff_active': 0.2597217261791229, 'action_L0_2_3_diff_tape': 0.06266244500875473, 'action_L1_1_3_product_present': 0.25, 'action_L1_1_3_product_choice_mass': 0.057410210371017456, 'action_L1_1_3_product_recovery': 0.0625, 'action_L1_1_3_product_active': 0.20765294134616852, 'action_L1_1_3_product_tape': 0.06372439116239548}`

Flow:
```text
input -> Layer 0 -> Layer 1 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `12`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | diff:0.31/a0.236 | *merge:0.44/a0.224 | diff:0.24/a0.247 | diff:0.30/a0.252 |
| s1 | merge:0.23/a0.259 | diff:0.29/a0.178 | diff:0.36/a0.221 | diff:0.35/a0.223 |
| s2 | diff:0.34/a0.281 | merge:0.41/a0.228 | diff:0.34/a0.202 | *diff:0.47/a0.260 |
| s3 | diff:0.33/a0.288 | merge:0.34/a0.224 | diff:0.33/a0.264 | diff:0.27/a0.206 |

### Cells
- edge 0->0: top=diff mass=0.3103, active=0.236192, transform=0.2920, skip=0.4133, disable=0.2948, tape=0.035695
- edge 0->1 EXPECTED: top=merge mass=0.4364, expected=diff mass=0.2054, active=0.224121, transform=0.4844, skip=0.3403, disable=0.1753, tape=0.054988
- edge 0->2: top=diff mass=0.2406, active=0.246974, transform=0.3639, skip=0.3605, disable=0.2756, tape=0.054078
- edge 0->3: top=diff mass=0.2979, active=0.252165, transform=0.4159, skip=0.3430, disable=0.2411, tape=0.055527
- edge 1->0: top=merge mass=0.2347, active=0.258751, transform=0.4956, skip=0.2503, disable=0.2541, tape=0.069338
- edge 1->1: top=diff mass=0.2851, active=0.177674, transform=0.5912, skip=0.2372, disable=0.1716, tape=0.053950
- edge 1->2: top=diff mass=0.3577, active=0.221473, transform=0.5294, skip=0.2211, disable=0.2495, tape=0.064584
- edge 1->3: top=diff mass=0.3541, active=0.223399, transform=0.5707, skip=0.2181, disable=0.2113, tape=0.066373
- edge 2->0: top=diff mass=0.3441, active=0.280885, transform=0.4632, skip=0.3617, disable=0.1752, tape=0.071002
- edge 2->1: top=merge mass=0.4129, active=0.228427, transform=0.6250, skip=0.2721, disable=0.1028, tape=0.074572
- edge 2->2: top=diff mass=0.3416, active=0.201789, transform=0.4004, skip=0.3402, disable=0.2594, tape=0.033285
- edge 2->3 EXPECTED: top=diff mass=0.4742, expected=diff mass=0.4742, active=0.259722, transform=0.5259, skip=0.3326, disable=0.1415, tape=0.062662
- edge 3->0: top=diff mass=0.3305, active=0.287746, transform=0.4118, skip=0.3599, disable=0.2282, tape=0.070116
- edge 3->1: top=merge mass=0.3363, active=0.224155, transform=0.5598, skip=0.3154, disable=0.1248, tape=0.064455
- edge 3->2: top=diff mass=0.3335, active=0.263785, transform=0.4384, skip=0.3543, disable=0.2073, tape=0.058997
- edge 3->3: top=diff mass=0.2717, active=0.205591, transform=0.3931, skip=0.4075, disable=0.1994, tape=0.035314

## Layer 1

- verdict: `not_recovered`
- expected_top_cells: `3`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | product:0.21/a0.193 | smooth:0.15/a0.197 | product:0.18/a0.234 | ctx_matrix:0.14/a0.202 |
| s1 | write_gate:0.20/a0.200 | contrast:0.34/a0.194 | contrast:0.27/a0.237 | *channel:0.25/a0.208 |
| s2 | write_gate:0.21/a0.228 | output_write:0.19/a0.225 | product:0.23/a0.262 | write_gate:0.22/a0.236 |
| s3 | write_gate:0.39/a0.237 | write_gate:0.18/a0.235 | contrast:0.19/a0.281 | diff:0.16/a0.234 |

### Cells
- edge 0->0: top=product mass=0.2128, active=0.193414, transform=0.5023, skip=0.2924, disable=0.2053, tape=0.055636
- edge 0->1: top=smooth mass=0.1526, active=0.196512, transform=0.6129, skip=0.1861, disable=0.2009, tape=0.063560
- edge 0->2: top=product mass=0.1825, active=0.233857, transform=0.4939, skip=0.2468, disable=0.2593, tape=0.061506
- edge 0->3: top=ctx_matrix mass=0.1436, active=0.202456, transform=0.5212, skip=0.2436, disable=0.2352, tape=0.052800
- edge 1->0: top=write_gate mass=0.2013, active=0.200101, transform=0.5389, skip=0.3330, disable=0.1281, tape=0.067835
- edge 1->1: top=contrast mass=0.3405, active=0.193747, transform=0.6387, skip=0.2426, disable=0.1186, tape=0.070579
- edge 1->2: top=contrast mass=0.2658, active=0.237294, transform=0.5277, skip=0.3065, disable=0.1659, tape=0.071970
- edge 1->3 EXPECTED: top=channel mass=0.2536, expected=product mass=0.0574, active=0.207653, transform=0.5664, skip=0.2883, disable=0.1452, tape=0.063724
- edge 2->0: top=write_gate mass=0.2084, active=0.228228, transform=0.5420, skip=0.2736, disable=0.1844, tape=0.059209
- edge 2->1: top=output_write mass=0.1932, active=0.224925, transform=0.6405, skip=0.1818, disable=0.1778, tape=0.062702
- edge 2->2: top=product mass=0.2318, active=0.261842, transform=0.5106, skip=0.2640, disable=0.2254, tape=0.057144
- edge 2->3: top=write_gate mass=0.2242, active=0.236228, transform=0.5426, skip=0.2428, disable=0.2146, tape=0.051449
- edge 3->0: top=write_gate mass=0.3941, active=0.236657, transform=0.5523, skip=0.3036, disable=0.1442, tape=0.072398
- edge 3->1: top=write_gate mass=0.1752, active=0.235093, transform=0.6676, skip=0.1971, disable=0.1353, tape=0.079036
- edge 3->2: top=contrast mass=0.1941, active=0.281321, transform=0.5341, skip=0.2771, disable=0.1888, tape=0.075730
- edge 3->3: top=diff mass=0.1599, active=0.234190, transform=0.5735, skip=0.2690, disable=0.1575, tape=0.062358

