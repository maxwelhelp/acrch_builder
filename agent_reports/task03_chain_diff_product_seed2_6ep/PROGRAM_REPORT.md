# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.73828125`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9993199706077576, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8495742082595825, 'action_L0_0_1_diff_tape': 0.022294748574495316, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9995044469833374, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.8563516139984131, 'action_L0_2_3_diff_tape': 0.018258225172758102, 'action_L1_1_3_product_present': 1.0, 'action_L1_1_3_product_choice_mass': 0.994143009185791, 'action_L1_1_3_product_recovery': 1.0, 'action_L1_1_3_product_active': 0.8922293186187744, 'action_L1_1_3_product_tape': 0.30486953258514404}`

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
| s0 | skip:0.28/a0.000 | *diff:1.00/a0.850 | channel:0.30/a0.154 | diff:1.00/a0.801 |
| s1 | channel:0.24/a0.018 | skip:0.26/a0.002 | channel:0.24/a0.009 | skip:0.37/a0.423 |
| s2 | channel:0.24/a0.266 | diff:1.00/a0.821 | write_gate:0.26/a0.000 | *diff:1.00/a0.856 |
| s3 | split:0.18/a0.018 | skip:0.24/a0.406 | channel:0.21/a0.007 | skip:0.36/a0.002 |

### Cells
- edge 0->0: top=skip mass=0.2839, active=0.000499, transform=0.1721, skip=0.1834, disable=0.6445, tape=0.000008
- edge 0->1 EXPECTED: top=diff mass=0.9993, expected=diff mass=0.9993, active=0.849574, transform=0.9154, skip=0.0584, disable=0.0262, tape=0.022295
- edge 0->2: top=channel mass=0.2986, active=0.154076, transform=0.4180, skip=0.3662, disable=0.2158, tape=0.007217
- edge 0->3: top=diff mass=0.9995, active=0.800583, transform=0.8883, skip=0.0818, disable=0.0299, tape=0.018802
- edge 1->0: top=channel mass=0.2397, active=0.017865, transform=0.0356, skip=0.8946, disable=0.0697, tape=0.000169
- edge 1->1: top=skip mass=0.2596, active=0.002178, transform=0.1478, skip=0.3475, disable=0.5046, tape=0.000037
- edge 1->2: top=channel mass=0.2400, active=0.009441, transform=0.0333, skip=0.8385, disable=0.1281, tape=0.000123
- edge 1->3: top=skip mass=0.3707, active=0.422506, transform=0.2841, skip=0.6501, disable=0.0658, tape=0.012273
- edge 2->0: top=channel mass=0.2439, active=0.265511, transform=0.5233, skip=0.3558, disable=0.1209, tape=0.009011
- edge 2->1: top=diff mass=0.9994, active=0.820870, transform=0.9280, skip=0.0481, disable=0.0239, tape=0.019541
- edge 2->2: top=write_gate mass=0.2559, active=0.000233, transform=0.1215, skip=0.1029, disable=0.7756, tape=0.000005
- edge 2->3 EXPECTED: top=diff mass=0.9995, expected=diff mass=0.9995, active=0.856352, transform=0.9083, skip=0.0630, disable=0.0287, tape=0.018258
- edge 3->0: top=split mass=0.1816, active=0.017761, transform=0.0367, skip=0.8949, disable=0.0684, tape=0.000172
- edge 3->1: top=skip mass=0.2449, active=0.406015, transform=0.3256, skip=0.6000, disable=0.0745, tape=0.013289
- edge 3->2: top=channel mass=0.2149, active=0.007379, transform=0.0314, skip=0.8274, disable=0.1412, tape=0.000081
- edge 3->3: top=skip mass=0.3638, active=0.001849, transform=0.1222, skip=0.3813, disable=0.4965, tape=0.000027

## Layer 1

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `1`
- active_cells: `4`
- active_edges_per_target: `1.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.37/a0.000 | ctx_matrix:0.26/a0.001 | ctx_matrix:0.25/a0.000 | output_mix:0.22/a0.092 |
| s1 | ctx_matrix:0.26/a0.032 | ctx_matrix:0.20/a0.009 | ctx_matrix:0.22/a0.039 | *product:0.99/a0.892 |
| s2 | ctx_matrix:0.35/a0.000 | memory_gate:0.27/a0.001 | ctx_matrix:0.32/a0.000 | output_mix:0.25/a0.059 |
| s3 | ctx_matrix:0.29/a0.000 | ctx_matrix:0.29/a0.097 | ctx_matrix:0.24/a0.000 | output_mix:0.34/a0.014 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.3656, active=0.000006, transform=0.0015, skip=0.4407, disable=0.5578, tape=0.000000
- edge 0->1: top=ctx_matrix mass=0.2611, active=0.001171, transform=0.0129, skip=0.6806, disable=0.3065, tape=0.000007
- edge 0->2: top=ctx_matrix mass=0.2492, active=0.000178, transform=0.0223, skip=0.5859, disable=0.3918, tape=0.000003
- edge 0->3: top=output_mix mass=0.2238, active=0.092346, transform=0.0252, skip=0.6925, disable=0.2823, tape=0.001539
- edge 1->0: top=ctx_matrix mass=0.2622, active=0.032394, transform=0.0420, skip=0.3039, disable=0.6542, tape=0.001723
- edge 1->1: top=ctx_matrix mass=0.1972, active=0.008647, transform=0.0243, skip=0.4643, disable=0.5114, tape=0.000618
- edge 1->2: top=ctx_matrix mass=0.2181, active=0.038863, transform=0.1375, skip=0.3676, disable=0.4949, tape=0.006942
- edge 1->3 EXPECTED: top=product mass=0.9941, expected=product mass=0.9941, active=0.892229, transform=0.4284, skip=0.3385, disable=0.2331, tape=0.304870
- edge 2->0: top=ctx_matrix mass=0.3486, active=0.000093, transform=0.0044, skip=0.3988, disable=0.5968, tape=0.000000
- edge 2->1: top=memory_gate mass=0.2676, active=0.000921, transform=0.0102, skip=0.6045, disable=0.3853, tape=0.000005
- edge 2->2: top=ctx_matrix mass=0.3190, active=0.000001, transform=0.0028, skip=0.4564, disable=0.5408, tape=0.000000
- edge 2->3: top=output_mix mass=0.2472, active=0.059405, transform=0.0196, skip=0.6206, disable=0.3599, tape=0.000848
- edge 3->0: top=ctx_matrix mass=0.2904, active=0.000378, transform=0.0079, skip=0.5378, disable=0.4543, tape=0.000003
- edge 3->1: top=ctx_matrix mass=0.2905, active=0.097148, transform=0.0818, skip=0.7084, disable=0.2098, tape=0.007081
- edge 3->2: top=ctx_matrix mass=0.2380, active=0.000396, transform=0.0289, skip=0.6330, disable=0.3381, tape=0.000015
- edge 3->3: top=output_mix mass=0.3353, active=0.014223, transform=0.0088, skip=0.7092, disable=0.2819, tape=0.000253

