# ActionMatrix Program Report

- task: `two_diff`
- batch_acc: `0.8359375`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9989702105522156, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.6213074922561646, 'action_L0_0_1_diff_tape': 0.004385172389447689, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9992748498916626, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.6165809631347656, 'action_L0_2_3_diff_tape': 0.005916244350373745}`

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
| s0 | ctx_matrix:0.36/a0.088 | *diff:1.00/a0.621 | ctx_matrix:0.11/a0.377 | diff:1.00/a0.541 |
| s1 | ctx_matrix:0.49/a0.298 | ctx_matrix:0.31/a0.076 | ctx_matrix:0.49/a0.294 | replace:0.16/a0.378 |
| s2 | replace:0.14/a0.416 | diff:1.00/a0.583 | ctx_matrix:0.41/a0.088 | *diff:1.00/a0.617 |
| s3 | ctx_matrix:0.47/a0.300 | replace:0.15/a0.386 | ctx_matrix:0.45/a0.281 | forget:0.30/a0.075 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.3569, active=0.088489, transform=0.0599, skip=0.3903, disable=0.5498, tape=0.002934
- edge 0->1 EXPECTED: top=diff mass=0.9990, expected=diff mass=0.9990, active=0.621307, transform=0.0365, skip=0.6063, disable=0.3573, tape=0.004385
- edge 0->2: top=ctx_matrix mass=0.1122, active=0.376587, transform=0.1511, skip=0.5017, disable=0.3471, tape=0.019862
- edge 0->3: top=diff mass=0.9993, active=0.540690, transform=0.0344, skip=0.6093, disable=0.3563, tape=0.004080
- edge 1->0: top=ctx_matrix mass=0.4879, active=0.298174, transform=0.5091, skip=0.3160, disable=0.1750, tape=0.088386
- edge 1->1: top=ctx_matrix mass=0.3099, active=0.076230, transform=0.0806, skip=0.3880, disable=0.5315, tape=0.003288
- edge 1->2: top=ctx_matrix mass=0.4898, active=0.294241, transform=0.5215, skip=0.2838, disable=0.1947, tape=0.086840
- edge 1->3: top=replace mass=0.1588, active=0.377890, transform=0.1902, skip=0.5077, disable=0.3022, tape=0.024574
- edge 2->0: top=replace mass=0.1373, active=0.415698, transform=0.1606, skip=0.6172, disable=0.2222, tape=0.025943
- edge 2->1: top=diff mass=0.9990, active=0.582914, transform=0.0421, skip=0.6885, disable=0.2694, tape=0.005320
- edge 2->2: top=ctx_matrix mass=0.4099, active=0.087986, transform=0.0811, skip=0.4480, disable=0.4709, tape=0.004163
- edge 2->3 EXPECTED: top=diff mass=0.9993, expected=diff mass=0.9993, active=0.616581, transform=0.0369, skip=0.6894, disable=0.2736, tape=0.005916
- edge 3->0: top=ctx_matrix mass=0.4674, active=0.300355, transform=0.4976, skip=0.3162, disable=0.1862, tape=0.086357
- edge 3->1: top=replace mass=0.1546, active=0.386446, transform=0.1879, skip=0.5056, disable=0.3065, tape=0.022087
- edge 3->2: top=ctx_matrix mass=0.4480, active=0.281459, transform=0.4993, skip=0.2888, disable=0.2119, tape=0.082186
- edge 3->3: top=forget mass=0.3014, active=0.075127, transform=0.0754, skip=0.3909, disable=0.5337, tape=0.003208

