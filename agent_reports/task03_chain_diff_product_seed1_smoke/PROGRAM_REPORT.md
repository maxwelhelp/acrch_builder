# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.5625`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9995208382606506, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.7530193328857422, 'action_L0_0_1_diff_tape': 0.05681198835372925, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9996886253356934, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.7770975828170776, 'action_L0_2_3_diff_tape': 0.017761200666427612, 'action_L1_1_3_product_present': 1.0, 'action_L1_1_3_product_choice_mass': 0.9989837408065796, 'action_L1_1_3_product_recovery': 1.0, 'action_L1_1_3_product_active': 0.8158113360404968, 'action_L1_1_3_product_tape': 0.08973407745361328}`

Flow:
```text
input -> Layer 0 -> Layer 1 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `4`
- active_cells: `9`
- active_edges_per_target: `2.25`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | channel:0.49/a0.003 | *diff:1.00/a0.753 | channel:0.30/a0.330 | diff:1.00/a0.751 |
| s1 | channel:0.22/a0.032 | channel:0.38/a0.004 | channel:0.23/a0.053 | channel:0.27/a0.367 |
| s2 | memory_write:0.25/a0.244 | diff:1.00/a0.716 | channel:0.34/a0.005 | *diff:1.00/a0.777 |
| s3 | channel:0.23/a0.024 | channel:0.24/a0.279 | channel:0.22/a0.040 | channel:0.38/a0.004 |

### Cells
- edge 0->0: top=channel mass=0.4863, active=0.002648, transform=0.0886, skip=0.3369, disable=0.5744, tape=0.000046
- edge 0->1 EXPECTED: top=diff mass=0.9995, expected=diff mass=0.9995, active=0.753019, transform=0.6614, skip=0.2392, disable=0.0994, tape=0.056812
- edge 0->2: top=channel mass=0.3023, active=0.330086, transform=0.3733, skip=0.4859, disable=0.1408, tape=0.012495
- edge 0->3: top=diff mass=0.9996, active=0.751259, transform=0.6706, skip=0.2349, disable=0.0945, tape=0.030507
- edge 1->0: top=channel mass=0.2213, active=0.032082, transform=0.1214, skip=0.6472, disable=0.2313, tape=0.001344
- edge 1->1: top=channel mass=0.3756, active=0.004166, transform=0.0671, skip=0.4010, disable=0.5319, tape=0.000077
- edge 1->2: top=channel mass=0.2307, active=0.053423, transform=0.1138, skip=0.7163, disable=0.1699, tape=0.001570
- edge 1->3: top=channel mass=0.2731, active=0.366584, transform=0.3082, skip=0.5128, disable=0.1790, tape=0.016482
- edge 2->0: top=memory_write mass=0.2478, active=0.243974, transform=0.2740, skip=0.5323, disable=0.1937, tape=0.007941
- edge 2->1: top=diff mass=0.9995, active=0.716100, transform=0.5433, skip=0.3335, disable=0.1231, tape=0.031598
- edge 2->2: top=channel mass=0.3378, active=0.005443, transform=0.0573, skip=0.4753, disable=0.4674, tape=0.000025
- edge 2->3 EXPECTED: top=diff mass=0.9997, expected=diff mass=0.9997, active=0.777098, transform=0.5458, skip=0.3354, disable=0.1188, tape=0.017761
- edge 3->0: top=channel mass=0.2312, active=0.024184, transform=0.0807, skip=0.7860, disable=0.1333, tape=0.000730
- edge 3->1: top=channel mass=0.2410, active=0.278963, transform=0.2160, skip=0.6650, disable=0.1191, tape=0.014762
- edge 3->2: top=channel mass=0.2235, active=0.040131, transform=0.0699, skip=0.8315, disable=0.0986, tape=0.000580
- edge 3->3: top=channel mass=0.3805, active=0.003955, transform=0.0496, skip=0.5516, disable=0.3987, tape=0.000030

## Layer 1

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `1`
- active_cells: `7`
- active_edges_per_target: `1.75`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | smooth:0.31/a0.000 | smooth:0.19/a0.001 | split:0.19/a0.002 | split:0.30/a0.115 |
| s1 | split:0.26/a0.095 | smooth:0.51/a0.064 | split:0.29/a0.091 | *product:1.00/a0.816 |
| s2 | gated_keep:0.19/a0.002 | contrast:0.18/a0.002 | gated_keep:0.20/a0.000 | split:0.40/a0.126 |
| s3 | split:0.19/a0.002 | smooth:0.14/a0.006 | split:0.22/a0.001 | split:0.41/a0.063 |

### Cells
- edge 0->0: top=smooth mass=0.3088, active=0.000174, transform=0.0035, skip=0.5933, disable=0.4033, tape=0.000001
- edge 0->1: top=smooth mass=0.1904, active=0.001151, transform=0.0197, skip=0.7211, disable=0.2592, tape=0.000017
- edge 0->2: top=split mass=0.1863, active=0.002067, transform=0.0152, skip=0.5137, disable=0.4711, tape=0.000038
- edge 0->3: top=split mass=0.2978, active=0.115022, transform=0.0464, skip=0.5566, disable=0.3970, tape=0.003656
- edge 1->0: top=split mass=0.2568, active=0.094544, transform=0.0275, skip=0.6104, disable=0.3622, tape=0.001570
- edge 1->1: top=smooth mass=0.5125, active=0.063838, transform=0.0266, skip=0.7612, disable=0.2122, tape=0.001471
- edge 1->2: top=split mass=0.2940, active=0.090795, transform=0.0256, skip=0.5584, disable=0.4160, tape=0.001177
- edge 1->3 EXPECTED: top=product mass=0.9990, expected=product mass=0.9990, active=0.815811, transform=0.1823, skip=0.5117, disable=0.3060, tape=0.089734
- edge 2->0: top=gated_keep mass=0.1921, active=0.002481, transform=0.0078, skip=0.6620, disable=0.3303, tape=0.000016
- edge 2->1: top=contrast mass=0.1834, active=0.001570, transform=0.0087, skip=0.7976, disable=0.1936, tape=0.000009
- edge 2->2: top=gated_keep mass=0.2018, active=0.000216, transform=0.0018, skip=0.6314, disable=0.3668, tape=0.000000
- edge 2->3: top=split mass=0.3975, active=0.126486, transform=0.0219, skip=0.6594, disable=0.3187, tape=0.001587
- edge 3->0: top=split mass=0.1877, active=0.001533, transform=0.0114, skip=0.4039, disable=0.5847, tape=0.000012
- edge 3->1: top=smooth mass=0.1446, active=0.005757, transform=0.0446, skip=0.5506, disable=0.4048, tape=0.000194
- edge 3->2: top=split mass=0.2199, active=0.001031, transform=0.0100, skip=0.3526, disable=0.6374, tape=0.000006
- edge 3->3: top=split mass=0.4074, active=0.063272, transform=0.0240, skip=0.4136, disable=0.5625, tape=0.001323

