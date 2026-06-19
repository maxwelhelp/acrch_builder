# ActionMatrix Program Report

- task: `chain_diff_merge`
- batch_acc: `0.9296875`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'merge'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9994249939918518, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.9050956964492798, 'action_L0_0_1_diff_tape': 0.00636571180075407, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9995119571685791, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.8912861943244934, 'action_L0_2_3_diff_tape': 0.007781912572681904, 'action_L1_1_3_merge_present': 1.0, 'action_L1_1_3_merge_choice_mass': 0.9988332986831665, 'action_L1_1_3_merge_recovery': 1.0, 'action_L1_1_3_merge_active': 0.7890408039093018, 'action_L1_1_3_merge_tape': 0.011085272766649723}`

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
| s0 | ctx_matrix:0.30/a0.021 | *diff:1.00/a0.905 | ctx_matrix:0.55/a0.331 | diff:1.00/a0.890 |
| s1 | ctx_matrix:0.30/a0.002 | forget:0.33/a0.011 | ctx_matrix:0.42/a0.003 | forget:0.37/a0.188 |
| s2 | ctx_matrix:0.34/a0.310 | diff:1.00/a0.895 | ctx_matrix:0.40/a0.022 | *diff:1.00/a0.891 |
| s3 | ctx_matrix:0.35/a0.004 | ctx_matrix:0.45/a0.252 | ctx_matrix:0.40/a0.004 | forget:0.53/a0.011 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.2996, active=0.021409, transform=0.0404, skip=0.7104, disable=0.2492, tape=0.000043
- edge 0->1 EXPECTED: top=diff mass=0.9994, expected=diff mass=0.9994, active=0.905096, transform=0.9766, skip=0.0137, disable=0.0097, tape=0.006366
- edge 0->2: top=ctx_matrix mass=0.5512, active=0.330552, transform=0.2680, skip=0.6226, disable=0.1094, tape=0.003585
- edge 0->3: top=diff mass=0.9996, active=0.889634, transform=0.9751, skip=0.0134, disable=0.0116, tape=0.004647
- edge 1->0: top=ctx_matrix mass=0.3049, active=0.002104, transform=0.0341, skip=0.3054, disable=0.6605, tape=0.000027
- edge 1->1: top=forget mass=0.3264, active=0.011052, transform=0.1008, skip=0.0478, disable=0.8514, tape=0.000180
- edge 1->2: top=ctx_matrix mass=0.4221, active=0.002586, transform=0.0310, skip=0.2837, disable=0.6853, tape=0.000040
- edge 1->3: top=forget mass=0.3739, active=0.188483, transform=0.5865, skip=0.0295, disable=0.3840, tape=0.008555
- edge 2->0: top=ctx_matrix mass=0.3395, active=0.309960, transform=0.2349, skip=0.6852, disable=0.0799, tape=0.002686
- edge 2->1: top=diff mass=0.9994, active=0.894714, transform=0.9717, skip=0.0186, disable=0.0096, tape=0.008275
- edge 2->2: top=ctx_matrix mass=0.3995, active=0.021550, transform=0.0303, skip=0.7336, disable=0.2361, tape=0.000054
- edge 2->3 EXPECTED: top=diff mass=0.9995, expected=diff mass=0.9995, active=0.891286, transform=0.9703, skip=0.0183, disable=0.0114, tape=0.007782
- edge 3->0: top=ctx_matrix mass=0.3519, active=0.003539, transform=0.0368, skip=0.2708, disable=0.6924, tape=0.000035
- edge 3->1: top=ctx_matrix mass=0.4465, active=0.252453, transform=0.6159, skip=0.0272, disable=0.3570, tape=0.010357
- edge 3->2: top=ctx_matrix mass=0.3957, active=0.003965, transform=0.0330, skip=0.2507, disable=0.7163, tape=0.000062
- edge 3->3: top=forget mass=0.5288, active=0.011365, transform=0.0878, skip=0.0337, disable=0.8785, tape=0.000098

## Layer 1

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `1`
- active_cells: `8`
- active_edges_per_target: `2.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'merge'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.55/a0.003 | ctx_matrix:0.42/a0.041 | ctx_matrix:0.37/a0.004 | ctx_matrix:0.43/a0.191 |
| s1 | ctx_matrix:0.40/a0.221 | ctx_matrix:0.43/a0.248 | ctx_matrix:0.38/a0.208 | *merge:1.00/a0.789 |
| s2 | ctx_matrix:0.51/a0.005 | ctx_matrix:0.43/a0.040 | ctx_matrix:0.43/a0.002 | ctx_matrix:0.40/a0.183 |
| s3 | ctx_matrix:0.41/a0.049 | ctx_matrix:0.38/a0.246 | ctx_matrix:0.40/a0.050 | output_mix:0.32/a0.217 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.5476, active=0.002571, transform=0.0686, skip=0.5736, disable=0.3578, tape=0.000136
- edge 0->1: top=ctx_matrix mass=0.4235, active=0.041008, transform=0.2489, skip=0.4609, disable=0.2903, tape=0.006304
- edge 0->2: top=ctx_matrix mass=0.3713, active=0.004374, transform=0.1142, skip=0.5952, disable=0.2906, tape=0.000312
- edge 0->3: top=ctx_matrix mass=0.4258, active=0.190534, transform=0.0629, skip=0.5762, disable=0.3609, tape=0.005707
- edge 1->0: top=ctx_matrix mass=0.3977, active=0.221141, transform=0.0516, skip=0.5223, disable=0.4261, tape=0.006342
- edge 1->1: top=ctx_matrix mass=0.4329, active=0.247878, transform=0.0336, skip=0.5216, disable=0.4448, tape=0.006600
- edge 1->2: top=ctx_matrix mass=0.3808, active=0.207609, transform=0.0563, skip=0.5751, disable=0.3686, tape=0.006423
- edge 1->3 EXPECTED: top=merge mass=0.9988, expected=merge mass=0.9988, active=0.789041, transform=0.0372, skip=0.5404, disable=0.4224, tape=0.011085
- edge 2->0: top=ctx_matrix mass=0.5081, active=0.004623, transform=0.0953, skip=0.4952, disable=0.4095, tape=0.000321
- edge 2->1: top=ctx_matrix mass=0.4320, active=0.040317, transform=0.2308, skip=0.4210, disable=0.3483, tape=0.006554
- edge 2->2: top=ctx_matrix mass=0.4337, active=0.002287, transform=0.0680, skip=0.5610, disable=0.3710, tape=0.000139
- edge 2->3: top=ctx_matrix mass=0.3981, active=0.183057, transform=0.0581, skip=0.5130, disable=0.4289, tape=0.006111
- edge 3->0: top=ctx_matrix mass=0.4076, active=0.049003, transform=0.1654, skip=0.5151, disable=0.3196, tape=0.005291
- edge 3->1: top=ctx_matrix mass=0.3810, active=0.245970, transform=0.3875, skip=0.3824, disable=0.2301, tape=0.068586
- edge 3->2: top=ctx_matrix mass=0.3993, active=0.049634, transform=0.1801, skip=0.5482, disable=0.2717, tape=0.005878
- edge 3->3: top=output_mix mass=0.3226, active=0.216948, transform=0.0232, skip=0.5744, disable=0.4024, tape=0.003943

