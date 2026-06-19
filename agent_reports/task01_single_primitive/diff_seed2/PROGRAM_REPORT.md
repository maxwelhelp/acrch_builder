# ActionMatrix Program Report

- task: `diff`
- batch_acc: `0.8828125`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9995830655097961, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8731986284255981, 'action_L0_0_1_diff_tape': 0.36846446990966797}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `5`
- active_cells: `13`
- active_edges_per_target: `3.25`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.37/a0.336 | *diff:1.00/a0.873 | diff:0.99/a0.438 | diff:0.98/a0.410 |
| s1 | ctx_matrix:0.49/a0.483 | ctx_matrix:0.33/a0.341 | ctx_matrix:0.48/a0.210 | ctx_matrix:0.31/a0.179 |
| s2 | ctx_matrix:0.46/a0.189 | diff:1.00/a0.440 | ctx_matrix:0.40/a0.008 | ctx_matrix:0.29/a0.046 |
| s3 | ctx_matrix:0.45/a0.178 | diff:0.99/a0.432 | ctx_matrix:0.37/a0.052 | memory_read:0.20/a0.007 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.3683, active=0.336190, transform=0.0480, skip=0.1688, disable=0.7832, tape=0.015487
- edge 0->1 EXPECTED: top=diff mass=0.9996, expected=diff mass=0.9996, active=0.873199, transform=0.6063, skip=0.2269, disable=0.1669, tape=0.368464
- edge 0->2: top=diff mass=0.9945, active=0.437813, transform=0.2263, skip=0.4383, disable=0.3354, tape=0.051920
- edge 0->3: top=diff mass=0.9801, active=0.409591, transform=0.2417, skip=0.4731, disable=0.2852, tape=0.054243
- edge 1->0: top=ctx_matrix mass=0.4941, active=0.482758, transform=0.6014, skip=0.2214, disable=0.1771, tape=0.210661
- edge 1->1: top=ctx_matrix mass=0.3323, active=0.340537, transform=0.0631, skip=0.1357, disable=0.8012, tape=0.020676
- edge 1->2: top=ctx_matrix mass=0.4838, active=0.210186, transform=0.2646, skip=0.3892, disable=0.3462, tape=0.030523
- edge 1->3: top=ctx_matrix mass=0.3102, active=0.178652, transform=0.2745, skip=0.4217, disable=0.3039, tape=0.028183
- edge 2->0: top=ctx_matrix mass=0.4590, active=0.188748, transform=0.2274, skip=0.5594, disable=0.2131, tape=0.023811
- edge 2->1: top=diff mass=0.9979, active=0.440161, transform=0.2790, skip=0.5104, disable=0.2106, tape=0.060236
- edge 2->2: top=ctx_matrix mass=0.3951, active=0.008266, transform=0.0025, skip=0.2604, disable=0.7371, tape=0.000017
- edge 2->3: top=ctx_matrix mass=0.2868, active=0.045614, transform=0.0584, skip=0.6892, disable=0.2524, tape=0.001286
- edge 3->0: top=ctx_matrix mass=0.4500, active=0.178384, transform=0.2506, skip=0.4606, disable=0.2888, tape=0.021720
- edge 3->1: top=diff mass=0.9946, active=0.432392, transform=0.2943, skip=0.4218, disable=0.2839, tape=0.058049
- edge 3->2: top=ctx_matrix mass=0.3737, active=0.052283, transform=0.0606, skip=0.5475, disable=0.3920, tape=0.001146
- edge 3->3: top=memory_read mass=0.2008, active=0.006736, transform=0.0029, skip=0.2248, disable=0.7723, tape=0.000027

