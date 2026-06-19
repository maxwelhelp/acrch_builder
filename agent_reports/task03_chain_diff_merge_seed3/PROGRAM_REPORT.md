# ActionMatrix Program Report

- task: `chain_diff_merge`
- batch_acc: `0.8515625`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'merge'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.998909056186676, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8275988698005676, 'action_L0_0_1_diff_tape': 0.06399780511856079, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9990752935409546, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.8515810370445251, 'action_L0_2_3_diff_tape': 0.02922140061855316, 'action_L1_1_3_merge_present': 1.0, 'action_L1_1_3_merge_choice_mass': 0.9930006861686707, 'action_L1_1_3_merge_recovery': 1.0, 'action_L1_1_3_merge_active': 0.8116651773452759, 'action_L1_1_3_merge_tape': 0.0009089558152481914}`

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
| s0 | contrast:0.27/a0.040 | *diff:1.00/a0.828 | ctx_matrix:0.18/a0.377 | diff:1.00/a0.817 |
| s1 | contrast:0.25/a0.017 | contrast:0.42/a0.005 | contrast:0.24/a0.011 | forget:0.18/a0.185 |
| s2 | contrast:0.19/a0.455 | diff:1.00/a0.826 | contrast:0.48/a0.042 | *diff:1.00/a0.852 |
| s3 | contrast:0.23/a0.009 | forget:0.24/a0.123 | contrast:0.22/a0.006 | contrast:0.28/a0.004 |

### Cells
- edge 0->0: top=contrast mass=0.2672, active=0.040130, transform=0.0586, skip=0.9140, disable=0.0274, tape=0.000381
- edge 0->1 EXPECTED: top=diff mass=0.9989, expected=diff mass=0.9989, active=0.827599, transform=0.8824, skip=0.0790, disable=0.0386, tape=0.063998
- edge 0->2: top=ctx_matrix mass=0.1805, active=0.377374, transform=0.3805, skip=0.5486, disable=0.0709, tape=0.018460
- edge 0->3: top=diff mass=0.9991, active=0.816992, transform=0.8693, skip=0.0769, disable=0.0537, tape=0.041983
- edge 1->0: top=contrast mass=0.2515, active=0.016764, transform=0.1349, skip=0.7179, disable=0.1472, tape=0.000906
- edge 1->1: top=contrast mass=0.4192, active=0.005207, transform=0.0974, skip=0.8316, disable=0.0710, tape=0.000125
- edge 1->2: top=contrast mass=0.2404, active=0.011478, transform=0.0937, skip=0.7914, disable=0.1149, tape=0.000450
- edge 1->3: top=forget mass=0.1773, active=0.184846, transform=0.5138, skip=0.2872, disable=0.1991, tape=0.014465
- edge 2->0: top=contrast mass=0.1934, active=0.454599, transform=0.5000, skip=0.4269, disable=0.0732, tape=0.021712
- edge 2->1: top=diff mass=0.9989, active=0.826226, transform=0.8897, skip=0.0752, disable=0.0352, tape=0.040437
- edge 2->2: top=contrast mass=0.4844, active=0.041783, transform=0.0438, skip=0.9345, disable=0.0217, tape=0.000185
- edge 2->3 EXPECTED: top=diff mass=0.9991, expected=diff mass=0.9991, active=0.851581, transform=0.8792, skip=0.0744, disable=0.0464, tape=0.029221
- edge 3->0: top=contrast mass=0.2276, active=0.009355, transform=0.1103, skip=0.7311, disable=0.1587, tape=0.000559
- edge 3->1: top=forget mass=0.2353, active=0.123275, transform=0.4901, skip=0.3375, disable=0.1724, tape=0.013144
- edge 3->2: top=contrast mass=0.2235, active=0.006190, transform=0.0760, skip=0.8037, disable=0.1203, tape=0.000195
- edge 3->3: top=contrast mass=0.2818, active=0.003730, transform=0.0877, skip=0.8084, disable=0.1039, tape=0.000066

## Layer 1

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `1`
- active_cells: `8`
- active_edges_per_target: `2.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'merge'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | skip:0.21/a0.003 | ctx_matrix:0.27/a0.019 | output_write:0.23/a0.004 | ctx_matrix:0.22/a0.222 |
| s1 | ctx_matrix:0.37/a0.216 | ctx_matrix:0.27/a0.155 | ctx_matrix:0.32/a0.184 | *merge:0.99/a0.812 |
| s2 | ctx_matrix:0.30/a0.004 | ctx_matrix:0.38/a0.016 | ctx_matrix:0.20/a0.001 | ctx_matrix:0.37/a0.217 |
| s3 | ctx_matrix:0.44/a0.032 | ctx_matrix:0.39/a0.107 | ctx_matrix:0.30/a0.022 | ctx_matrix:0.27/a0.224 |

### Cells
- edge 0->0: top=skip mass=0.2061, active=0.002950, transform=0.0240, skip=0.4178, disable=0.5582, tape=0.000047
- edge 0->1: top=ctx_matrix mass=0.2684, active=0.018978, transform=0.0597, skip=0.5236, disable=0.4167, tape=0.000571
- edge 0->2: top=output_write mass=0.2267, active=0.003603, transform=0.0239, skip=0.4148, disable=0.5614, tape=0.000046
- edge 0->3: top=ctx_matrix mass=0.2175, active=0.221909, transform=0.0101, skip=0.5333, disable=0.4565, tape=0.000369
- edge 1->0: top=ctx_matrix mass=0.3712, active=0.216120, transform=0.0265, skip=0.4433, disable=0.5302, tape=0.002388
- edge 1->1: top=ctx_matrix mass=0.2652, active=0.154642, transform=0.0066, skip=0.4812, disable=0.5122, tape=0.000560
- edge 1->2: top=ctx_matrix mass=0.3208, active=0.184042, transform=0.0143, skip=0.4239, disable=0.5618, tape=0.000858
- edge 1->3 EXPECTED: top=merge mass=0.9930, expected=merge mass=0.9930, active=0.811665, transform=0.0095, skip=0.5544, disable=0.4361, tape=0.000909
- edge 2->0: top=ctx_matrix mass=0.3022, active=0.004348, transform=0.0610, skip=0.3829, disable=0.5562, tape=0.000189
- edge 2->1: top=ctx_matrix mass=0.3767, active=0.016146, transform=0.0842, skip=0.4705, disable=0.4453, tape=0.000816
- edge 2->2: top=ctx_matrix mass=0.1998, active=0.001498, transform=0.0180, skip=0.3544, disable=0.6276, tape=0.000019
- edge 2->3: top=ctx_matrix mass=0.3720, active=0.216892, transform=0.0151, skip=0.4894, disable=0.4955, tape=0.000829
- edge 3->0: top=ctx_matrix mass=0.4396, active=0.032303, transform=0.1062, skip=0.4606, disable=0.4332, tape=0.002189
- edge 3->1: top=ctx_matrix mass=0.3914, active=0.107242, transform=0.1816, skip=0.5166, disable=0.3019, tape=0.013113
- edge 3->2: top=ctx_matrix mass=0.3016, active=0.022144, transform=0.0637, skip=0.4598, disable=0.4765, tape=0.000794
- edge 3->3: top=ctx_matrix mass=0.2655, active=0.223745, transform=0.0051, skip=0.5280, disable=0.4669, tape=0.000378

