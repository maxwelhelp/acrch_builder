# ActionMatrix Program Report

- task: `two_diff`
- batch_acc: `0.81640625`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9988794326782227, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.48432856798171997, 'action_L0_0_1_diff_tape': 0.002792392624542117, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9991241693496704, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.4863770008087158, 'action_L0_2_3_diff_tape': 0.004110780544579029}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `4`
- active_cells: `13`
- active_edges_per_target: `3.25`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.36/a0.050 | *diff:1.00/a0.484 | ctx_matrix:0.39/a0.248 | diff:1.00/a0.384 |
| s1 | ctx_matrix:0.48/a0.210 | ctx_matrix:0.32/a0.046 | ctx_matrix:0.49/a0.211 | ctx_matrix:0.24/a0.257 |
| s2 | ctx_matrix:0.28/a0.285 | diff:1.00/a0.440 | ctx_matrix:0.43/a0.055 | *diff:1.00/a0.486 |
| s3 | ctx_matrix:0.46/a0.212 | ctx_matrix:0.33/a0.264 | ctx_matrix:0.44/a0.203 | forget:0.29/a0.045 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.3585, active=0.049528, transform=0.0636, skip=0.3689, disable=0.5675, tape=0.001992
- edge 0->1 EXPECTED: top=diff mass=0.9989, expected=diff mass=0.9989, active=0.484329, transform=0.0311, skip=0.6070, disable=0.3619, tape=0.002792
- edge 0->2: top=ctx_matrix mass=0.3923, active=0.247675, transform=0.1442, skip=0.4935, disable=0.3623, tape=0.012699
- edge 0->3: top=diff mass=0.9991, active=0.384303, transform=0.0308, skip=0.6024, disable=0.3668, tape=0.002535
- edge 1->0: top=ctx_matrix mass=0.4835, active=0.209907, transform=0.5286, skip=0.3072, disable=0.1642, tape=0.067057
- edge 1->1: top=ctx_matrix mass=0.3212, active=0.045874, transform=0.0882, skip=0.3721, disable=0.5397, tape=0.002478
- edge 1->2: top=ctx_matrix mass=0.4867, active=0.210873, transform=0.5380, skip=0.2713, disable=0.1907, tape=0.066706
- edge 1->3: top=ctx_matrix mass=0.2400, active=0.257118, transform=0.1925, skip=0.5056, disable=0.3019, tape=0.017042
- edge 2->0: top=ctx_matrix mass=0.2791, active=0.284880, transform=0.1552, skip=0.6307, disable=0.2141, tape=0.017522
- edge 2->1: top=diff mass=0.9989, active=0.439579, transform=0.0364, skip=0.7015, disable=0.2620, tape=0.003311
- edge 2->2: top=ctx_matrix mass=0.4267, active=0.054556, transform=0.0873, skip=0.4261, disable=0.4866, tape=0.003206
- edge 2->3 EXPECTED: top=diff mass=0.9991, expected=diff mass=0.9991, active=0.486377, transform=0.0332, skip=0.6951, disable=0.2717, tape=0.004111
- edge 3->0: top=ctx_matrix mass=0.4629, active=0.211661, transform=0.5173, skip=0.3084, disable=0.1743, tape=0.065985
- edge 3->1: top=ctx_matrix mass=0.3251, active=0.263615, transform=0.1841, skip=0.5144, disable=0.3015, tape=0.014538
- edge 3->2: top=ctx_matrix mass=0.4415, active=0.202819, transform=0.5171, skip=0.2765, disable=0.2064, tape=0.063942
- edge 3->3: top=forget mass=0.2915, active=0.045314, transform=0.0849, skip=0.3706, disable=0.5445, tape=0.002532

