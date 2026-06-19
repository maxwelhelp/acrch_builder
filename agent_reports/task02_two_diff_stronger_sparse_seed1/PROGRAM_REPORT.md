# ActionMatrix Program Report

- task: `two_diff`
- batch_acc: `0.92578125`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9995867013931274, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8046405911445618, 'action_L0_0_1_diff_tape': 0.2998242974281311, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9996709823608398, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.786007285118103, 'action_L0_2_3_diff_tape': 0.27525627613067627}`

Flow:
```text
input -> Layer 0 -> final_read:last
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
| s0 | ctx_matrix:0.43/a0.011 | *diff:1.00/a0.805 | ctx_matrix:0.37/a0.251 | diff:1.00/a0.805 |
| s1 | ctx_matrix:0.40/a0.007 | ctx_matrix:0.36/a0.014 | ctx_matrix:0.40/a0.014 | ctx_matrix:0.38/a0.242 |
| s2 | ctx_matrix:0.26/a0.145 | diff:1.00/a0.743 | ctx_matrix:0.37/a0.012 | *diff:1.00/a0.786 |
| s3 | ctx_matrix:0.36/a0.006 | ctx_matrix:0.33/a0.188 | ctx_matrix:0.37/a0.009 | ctx_matrix:0.35/a0.013 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.4346, active=0.010593, transform=0.0403, skip=0.3421, disable=0.6176, tape=0.000233
- edge 0->1 EXPECTED: top=diff mass=0.9996, expected=diff mass=0.9996, active=0.804641, transform=0.5618, skip=0.2971, disable=0.1411, tape=0.299824
- edge 0->2: top=ctx_matrix mass=0.3699, active=0.251348, transform=0.1586, skip=0.5266, disable=0.3148, tape=0.014909
- edge 0->3: top=diff mass=0.9996, active=0.805063, transform=0.5702, skip=0.2800, disable=0.1498, tape=0.267283
- edge 1->0: top=ctx_matrix mass=0.3985, active=0.007046, transform=0.0458, skip=0.3463, disable=0.6080, tape=0.000105
- edge 1->1: top=ctx_matrix mass=0.3646, active=0.013787, transform=0.0277, skip=0.3139, disable=0.6584, tape=0.000239
- edge 1->2: top=ctx_matrix mass=0.4027, active=0.013530, transform=0.0300, skip=0.4371, disable=0.5329, tape=0.000148
- edge 1->3: top=ctx_matrix mass=0.3792, active=0.242202, transform=0.1814, skip=0.3751, disable=0.4435, tape=0.018385
- edge 2->0: top=ctx_matrix mass=0.2628, active=0.144610, transform=0.2152, skip=0.3393, disable=0.4455, tape=0.013889
- edge 2->1: top=diff mass=0.9996, active=0.743090, transform=0.5458, skip=0.2646, disable=0.1895, tape=0.276982
- edge 2->2: top=ctx_matrix mass=0.3682, active=0.012279, transform=0.0245, skip=0.3556, disable=0.6199, tape=0.000159
- edge 2->3 EXPECTED: top=diff mass=0.9997, expected=diff mass=0.9997, active=0.786007, transform=0.5640, skip=0.2340, disable=0.2020, tape=0.275256
- edge 3->0: top=ctx_matrix mass=0.3649, active=0.005670, transform=0.0406, skip=0.5315, disable=0.4278, tape=0.000097
- edge 3->1: top=ctx_matrix mass=0.3283, active=0.187586, transform=0.1442, skip=0.5894, disable=0.2664, tape=0.014631
- edge 3->2: top=ctx_matrix mass=0.3668, active=0.009033, transform=0.0245, skip=0.6229, disable=0.3526, tape=0.000074
- edge 3->3: top=ctx_matrix mass=0.3516, active=0.013029, transform=0.0253, skip=0.4523, disable=0.5224, tape=0.000209

