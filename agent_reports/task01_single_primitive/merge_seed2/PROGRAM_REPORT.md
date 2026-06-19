# ActionMatrix Program Report

- task: `merge`
- batch_acc: `0.8828125`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'merge'}]`
- action_metrics: `{'action_L0_0_1_merge_present': 1.0, 'action_L0_0_1_merge_choice_mass': 0.999550461769104, 'action_L0_0_1_merge_recovery': 1.0, 'action_L0_0_1_merge_active': 0.9073131084442139, 'action_L0_0_1_merge_tape': 0.12700912356376648}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `4`
- active_cells: `15`
- active_edges_per_target: `3.75`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'merge'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.35/a0.571 | *merge:1.00/a0.907 | ctx_matrix:0.52/a0.646 | ctx_matrix:0.34/a0.631 |
| s1 | ctx_matrix:0.44/a0.513 | merge:0.37/a0.655 | ctx_matrix:0.45/a0.412 | ctx_matrix:0.30/a0.394 |
| s2 | ctx_matrix:0.36/a0.144 | merge:1.00/a0.472 | ctx_matrix:0.35/a0.047 | ctx_matrix:0.30/a0.098 |
| s3 | ctx_matrix:0.38/a0.151 | merge:0.98/a0.479 | ctx_matrix:0.36/a0.103 | skip:0.19/a0.055 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.3522, active=0.570691, transform=0.3593, skip=0.2893, disable=0.3513, tape=0.136971
- edge 0->1 EXPECTED: top=merge mass=0.9996, expected=merge mass=0.9996, active=0.907313, transform=0.3162, skip=0.4161, disable=0.2677, tape=0.127009
- edge 0->2: top=ctx_matrix mass=0.5225, active=0.645588, transform=0.5272, skip=0.2641, disable=0.2087, tape=0.215435
- edge 0->3: top=ctx_matrix mass=0.3415, active=0.631479, transform=0.4361, skip=0.3478, disable=0.2161, tape=0.157655
- edge 1->0: top=ctx_matrix mass=0.4422, active=0.512659, transform=0.5701, skip=0.2343, disable=0.1957, tape=0.206098
- edge 1->1: top=merge mass=0.3684, active=0.655279, transform=0.2929, skip=0.3462, disable=0.3609, tape=0.109256
- edge 1->2: top=ctx_matrix mass=0.4486, active=0.411743, transform=0.6240, skip=0.2034, disable=0.1726, tape=0.176357
- edge 1->3: top=ctx_matrix mass=0.3023, active=0.393926, transform=0.5262, skip=0.2800, disable=0.1939, tape=0.133464
- edge 2->0: top=ctx_matrix mass=0.3566, active=0.144164, transform=0.1329, skip=0.5671, disable=0.2999, tape=0.008072
- edge 2->1: top=merge mass=0.9979, active=0.472150, transform=0.0737, skip=0.6339, disable=0.2924, tape=0.009575
- edge 2->2: top=ctx_matrix mass=0.3539, active=0.046656, transform=0.1262, skip=0.4838, disable=0.3900, tape=0.002814
- edge 2->3: top=ctx_matrix mass=0.3016, active=0.097663, transform=0.1157, skip=0.6116, disable=0.2727, tape=0.004553
- edge 3->0: top=ctx_matrix mass=0.3765, active=0.150517, transform=0.1668, skip=0.4758, disable=0.3574, tape=0.009809
- edge 3->1: top=merge mass=0.9806, active=0.478873, transform=0.0901, skip=0.5513, disable=0.3586, tape=0.011402
- edge 3->2: top=ctx_matrix mass=0.3613, active=0.103122, transform=0.2031, skip=0.4486, disable=0.3483, tape=0.008707
- edge 3->3: top=skip mass=0.1919, active=0.054801, transform=0.0983, skip=0.4712, disable=0.4305, tape=0.002620

