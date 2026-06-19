# ActionMatrix Program Report

- task: `merge`
- batch_acc: `0.8828125`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'merge'}]`
- action_metrics: `{'action_L0_0_1_merge_present': 1.0, 'action_L0_0_1_merge_choice_mass': 0.999487578868866, 'action_L0_0_1_merge_recovery': 1.0, 'action_L0_0_1_merge_active': 0.916833758354187, 'action_L0_0_1_merge_tape': 0.31820982694625854}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `3`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'merge'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.61/a0.718 | *merge:1.00/a0.917 | ctx_matrix:0.41/a0.715 | ctx_matrix:0.38/a0.702 |
| s1 | ctx_matrix:0.44/a0.570 | ctx_matrix:0.49/a0.727 | ctx_matrix:0.31/a0.467 | skip:0.30/a0.444 |
| s2 | ctx_matrix:0.37/a0.114 | merge:0.99/a0.351 | ctx_matrix:0.30/a0.066 | ctx_matrix:0.24/a0.075 |
| s3 | ctx_matrix:0.34/a0.122 | merge:0.97/a0.362 | ctx_matrix:0.28/a0.083 | ctx_matrix:0.29/a0.071 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.6099, active=0.717844, transform=0.7230, skip=0.1796, disable=0.0974, tape=0.311130
- edge 0->1 EXPECTED: top=merge mass=0.9995, expected=merge mass=0.9995, active=0.916834, transform=0.5983, skip=0.2359, disable=0.1658, tape=0.318210
- edge 0->2: top=ctx_matrix mass=0.4145, active=0.715001, transform=0.5983, skip=0.3107, disable=0.0910, tape=0.260843
- edge 0->3: top=ctx_matrix mass=0.3811, active=0.701882, transform=0.5118, skip=0.3549, disable=0.1332, tape=0.189784
- edge 1->0: top=ctx_matrix mass=0.4374, active=0.570169, transform=0.6949, skip=0.1654, disable=0.1397, tape=0.276893
- edge 1->1: top=ctx_matrix mass=0.4926, active=0.727437, transform=0.6716, skip=0.1970, disable=0.1313, tape=0.265831
- edge 1->2: top=ctx_matrix mass=0.3061, active=0.466594, transform=0.6367, skip=0.2658, disable=0.0975, tape=0.193560
- edge 1->3: top=skip mass=0.3037, active=0.443918, transform=0.5472, skip=0.3095, disable=0.1433, tape=0.139927
- edge 2->0: top=ctx_matrix mass=0.3655, active=0.113639, transform=0.1570, skip=0.3193, disable=0.5237, tape=0.008314
- edge 2->1: top=merge mass=0.9921, active=0.351429, transform=0.1174, skip=0.3338, disable=0.5488, tape=0.016291
- edge 2->2: top=ctx_matrix mass=0.2953, active=0.066001, transform=0.1715, skip=0.5151, disable=0.3134, tape=0.005671
- edge 2->3: top=ctx_matrix mass=0.2394, active=0.075401, transform=0.0941, skip=0.4637, disable=0.4422, tape=0.002880
- edge 3->0: top=ctx_matrix mass=0.3394, active=0.122421, transform=0.1558, skip=0.2526, disable=0.5916, tape=0.008595
- edge 3->1: top=merge mass=0.9698, active=0.361731, transform=0.1204, skip=0.2582, disable=0.6214, tape=0.017465
- edge 3->2: top=ctx_matrix mass=0.2812, active=0.083072, transform=0.1460, skip=0.4048, disable=0.4492, tape=0.006232
- edge 3->3: top=ctx_matrix mass=0.2912, active=0.071385, transform=0.1278, skip=0.4037, disable=0.4685, tape=0.004079

