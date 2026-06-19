# ActionMatrix Program Report

- task: `merge`
- batch_acc: `0.91015625`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'merge'}]`
- action_metrics: `{'action_L0_0_1_merge_present': 1.0, 'action_L0_0_1_merge_choice_mass': 0.9988864064216614, 'action_L0_0_1_merge_recovery': 1.0, 'action_L0_0_1_merge_active': 0.877249538898468, 'action_L0_0_1_merge_tape': 0.0289740189909935}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `1`
- active_cells: `13`
- active_edges_per_target: `3.25`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'merge'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.66/a0.695 | *merge:1.00/a0.877 | ctx_matrix:0.38/a0.589 | ctx_matrix:0.54/a0.546 |
| s1 | ctx_matrix:0.56/a0.674 | ctx_matrix:0.50/a0.639 | ctx_matrix:0.56/a0.379 | ctx_matrix:0.56/a0.324 |
| s2 | ctx_matrix:0.37/a0.170 | ctx_matrix:0.32/a0.320 | ctx_matrix:0.53/a0.038 | ctx_matrix:0.33/a0.039 |
| s3 | ctx_matrix:0.49/a0.208 | ctx_matrix:0.34/a0.359 | ctx_matrix:0.55/a0.072 | ctx_matrix:0.53/a0.032 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.6596, active=0.695137, transform=0.7511, skip=0.1553, disable=0.0936, tape=0.227653
- edge 0->1 EXPECTED: top=merge mass=0.9989, expected=merge mass=0.9989, active=0.877250, transform=0.1226, skip=0.6430, disable=0.2345, tape=0.028974
- edge 0->2: top=ctx_matrix mass=0.3771, active=0.588647, transform=0.0868, skip=0.6261, disable=0.2871, tape=0.011039
- edge 0->3: top=ctx_matrix mass=0.5364, active=0.546435, transform=0.1082, skip=0.6758, disable=0.2160, tape=0.015337
- edge 1->0: top=ctx_matrix mass=0.5568, active=0.673628, transform=0.8195, skip=0.0992, disable=0.0813, tape=0.444431
- edge 1->1: top=ctx_matrix mass=0.5032, active=0.639218, transform=0.5767, skip=0.2336, disable=0.1897, tape=0.165794
- edge 1->2: top=ctx_matrix mass=0.5568, active=0.379175, transform=0.2839, skip=0.4055, disable=0.3106, tape=0.048797
- edge 1->3: top=ctx_matrix mass=0.5599, active=0.323934, transform=0.3411, skip=0.4243, disable=0.2346, tape=0.056665
- edge 2->0: top=ctx_matrix mass=0.3668, active=0.170243, transform=0.3825, skip=0.2580, disable=0.3595, tape=0.038682
- edge 2->1: top=ctx_matrix mass=0.3239, active=0.319724, transform=0.0705, skip=0.4590, disable=0.4705, tape=0.007497
- edge 2->2: top=ctx_matrix mass=0.5256, active=0.037543, transform=0.1179, skip=0.3449, disable=0.5371, tape=0.001031
- edge 2->3: top=ctx_matrix mass=0.3337, active=0.039357, transform=0.0630, skip=0.4859, disable=0.4511, tape=0.000887
- edge 3->0: top=ctx_matrix mass=0.4885, active=0.207955, transform=0.3324, skip=0.4200, disable=0.2476, tape=0.038014
- edge 3->1: top=ctx_matrix mass=0.3403, active=0.359370, transform=0.0554, skip=0.6512, disable=0.2934, tape=0.006268
- edge 3->2: top=ctx_matrix mass=0.5481, active=0.072347, transform=0.0383, skip=0.6208, disable=0.3409, tape=0.000722
- edge 3->3: top=ctx_matrix mass=0.5336, active=0.031785, transform=0.1319, skip=0.5689, disable=0.2993, tape=0.001106

