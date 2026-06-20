# ActionMatrix Program Report

- task: `two_diff`
- batch_acc: `0.82421875`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9991471171379089, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.5304476022720337, 'action_L0_0_1_diff_tape': 0.0026733819395303726, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9993894696235657, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.5315365791320801, 'action_L0_2_3_diff_tape': 0.004003459122031927}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `4`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.37/a0.057 | *diff:1.00/a0.530 | ctx_matrix:0.52/a0.286 | diff:1.00/a0.436 |
| s1 | ctx_matrix:0.49/a0.238 | ctx_matrix:0.33/a0.052 | ctx_matrix:0.50/a0.241 | ctx_matrix:0.31/a0.297 |
| s2 | ctx_matrix:0.46/a0.321 | diff:1.00/a0.486 | ctx_matrix:0.43/a0.062 | *diff:1.00/a0.532 |
| s3 | ctx_matrix:0.48/a0.241 | ctx_matrix:0.40/a0.303 | ctx_matrix:0.46/a0.232 | forget:0.27/a0.053 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.3691, active=0.057093, transform=0.0591, skip=0.3743, disable=0.5666, tape=0.001997
- edge 0->1 EXPECTED: top=diff mass=0.9991, expected=diff mass=0.9991, active=0.530448, transform=0.0282, skip=0.6119, disable=0.3599, tape=0.002673
- edge 0->2: top=ctx_matrix mass=0.5187, active=0.285757, transform=0.1386, skip=0.5024, disable=0.3590, tape=0.013552
- edge 0->3: top=diff mass=0.9993, active=0.435985, transform=0.0283, skip=0.6094, disable=0.3623, tape=0.002566
- edge 1->0: top=ctx_matrix mass=0.4944, active=0.238424, transform=0.5107, skip=0.3169, disable=0.1724, tape=0.072428
- edge 1->1: top=ctx_matrix mass=0.3292, active=0.052370, transform=0.0817, skip=0.3729, disable=0.5453, tape=0.002467
- edge 1->2: top=ctx_matrix mass=0.4954, active=0.240686, transform=0.5296, skip=0.2767, disable=0.1937, tape=0.073874
- edge 1->3: top=ctx_matrix mass=0.3106, active=0.297261, transform=0.1812, skip=0.5129, disable=0.3059, tape=0.018321
- edge 2->0: top=ctx_matrix mass=0.4639, active=0.321462, transform=0.1441, skip=0.6362, disable=0.2197, tape=0.017635
- edge 2->1: top=diff mass=0.9991, active=0.486084, transform=0.0329, skip=0.7025, disable=0.2646, tape=0.003214
- edge 2->2: top=ctx_matrix mass=0.4313, active=0.061667, transform=0.0835, skip=0.4319, disable=0.4846, tape=0.003254
- edge 2->3 EXPECTED: top=diff mass=0.9994, expected=diff mass=0.9994, active=0.531537, transform=0.0303, skip=0.6978, disable=0.2719, tape=0.004003
- edge 3->0: top=ctx_matrix mass=0.4761, active=0.241194, transform=0.4978, skip=0.3174, disable=0.1848, tape=0.070871
- edge 3->1: top=ctx_matrix mass=0.3966, active=0.302666, transform=0.1707, skip=0.5183, disable=0.3110, tape=0.015201
- edge 3->2: top=ctx_matrix mass=0.4562, active=0.231813, transform=0.5061, skip=0.2817, disable=0.2122, tape=0.070462
- edge 3->3: top=forget mass=0.2706, active=0.052861, transform=0.0789, skip=0.3717, disable=0.5494, tape=0.002584

