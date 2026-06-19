# ActionMatrix Program Report

- task: `two_diff`
- batch_acc: `0.921875`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9994852542877197, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8039756417274475, 'action_L0_0_1_diff_tape': 0.3416290283203125, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9996053576469421, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.8164999485015869, 'action_L0_2_3_diff_tape': 0.3409641981124878}`

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
| s0 | ctx_matrix:0.42/a0.018 | *diff:1.00/a0.804 | ctx_matrix:0.41/a0.203 | diff:1.00/a0.812 |
| s1 | contrast:0.23/a0.006 | ctx_matrix:0.33/a0.009 | contrast:0.23/a0.004 | replace:0.32/a0.186 |
| s2 | ctx_matrix:0.44/a0.219 | diff:1.00/a0.779 | contrast:0.29/a0.012 | *diff:1.00/a0.816 |
| s3 | replace:0.23/a0.004 | ctx_matrix:0.44/a0.134 | contrast:0.21/a0.003 | replace:0.35/a0.008 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.4223, active=0.017732, transform=0.0298, skip=0.8199, disable=0.1504, tape=0.000380
- edge 0->1 EXPECTED: top=diff mass=0.9995, expected=diff mass=0.9995, active=0.803976, transform=0.6665, skip=0.2137, disable=0.1198, tape=0.341629
- edge 0->2: top=ctx_matrix mass=0.4113, active=0.202637, transform=0.2002, skip=0.6370, disable=0.1628, tape=0.020821
- edge 0->3: top=diff mass=0.9995, active=0.811625, transform=0.6613, skip=0.2064, disable=0.1323, tape=0.332411
- edge 1->0: top=contrast mass=0.2253, active=0.006263, transform=0.0330, skip=0.5633, disable=0.4036, tape=0.000085
- edge 1->1: top=ctx_matrix mass=0.3277, active=0.009198, transform=0.0203, skip=0.6829, disable=0.2968, tape=0.000092
- edge 1->2: top=contrast mass=0.2299, active=0.004357, transform=0.0269, skip=0.6742, disable=0.2989, tape=0.000040
- edge 1->3: top=replace mass=0.3156, active=0.185702, transform=0.1665, skip=0.4048, disable=0.4287, tape=0.010555
- edge 2->0: top=ctx_matrix mass=0.4389, active=0.218702, transform=0.2364, skip=0.4956, disable=0.2680, tape=0.029616
- edge 2->1: top=diff mass=0.9994, active=0.779372, transform=0.6542, skip=0.1999, disable=0.1459, tape=0.333641
- edge 2->2: top=contrast mass=0.2908, active=0.012249, transform=0.0196, skip=0.8610, disable=0.1194, tape=0.000159
- edge 2->3 EXPECTED: top=diff mass=0.9996, expected=diff mass=0.9996, active=0.816500, transform=0.6387, skip=0.1953, disable=0.1660, tape=0.340964
- edge 3->0: top=replace mass=0.2333, active=0.004370, transform=0.0258, skip=0.4533, disable=0.5209, tape=0.000055
- edge 3->1: top=ctx_matrix mass=0.4414, active=0.133638, transform=0.1454, skip=0.3184, disable=0.5362, tape=0.007518
- edge 3->2: top=contrast mass=0.2150, active=0.002674, transform=0.0219, skip=0.5602, disable=0.4180, tape=0.000021
- edge 3->3: top=replace mass=0.3491, active=0.008238, transform=0.0163, skip=0.5429, disable=0.4408, tape=0.000078

