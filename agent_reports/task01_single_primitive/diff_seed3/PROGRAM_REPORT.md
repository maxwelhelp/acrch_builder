# ActionMatrix Program Report

- task: `diff`
- batch_acc: `0.94921875`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9998278617858887, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.867458164691925, 'action_L0_0_1_diff_tape': 0.4798450767993927}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `5`
- active_cells: `6`
- active_edges_per_target: `1.5`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | replace:0.16/a0.091 | *diff:1.00/a0.867 | diff:0.95/a0.406 | diff:0.98/a0.464 |
| s1 | contrast:0.16/a0.033 | contrast:0.21/a0.041 | contrast:0.16/a0.026 | replace:0.14/a0.033 |
| s2 | contrast:0.17/a0.022 | diff:0.98/a0.261 | contrast:0.30/a0.001 | replace:0.13/a0.022 |
| s3 | replace:0.22/a0.024 | diff:0.98/a0.250 | replace:0.15/a0.015 | replace:0.26/a0.001 |

### Cells
- edge 0->0: top=replace mass=0.1588, active=0.090546, transform=0.0329, skip=0.8451, disable=0.1220, tape=0.002800
- edge 0->1 EXPECTED: top=diff mass=0.9998, expected=diff mass=0.9998, active=0.867458, transform=0.7108, skip=0.1793, disable=0.1099, tape=0.479845
- edge 0->2: top=diff mass=0.9534, active=0.405912, transform=0.1322, skip=0.6952, disable=0.1726, tape=0.026266
- edge 0->3: top=diff mass=0.9753, active=0.464064, transform=0.1555, skip=0.5760, disable=0.2686, tape=0.030513
- edge 1->0: top=contrast mass=0.1628, active=0.032729, transform=0.1365, skip=0.5320, disable=0.3315, tape=0.002543
- edge 1->1: top=contrast mass=0.2081, active=0.040697, transform=0.0484, skip=0.7196, disable=0.2320, tape=0.002117
- edge 1->2: top=contrast mass=0.1552, active=0.025996, transform=0.0427, skip=0.6983, disable=0.2590, tape=0.000448
- edge 1->3: top=replace mass=0.1421, active=0.032726, transform=0.0401, skip=0.5748, disable=0.3851, tape=0.000480
- edge 2->0: top=contrast mass=0.1682, active=0.021694, transform=0.0556, skip=0.4632, disable=0.4812, tape=0.000542
- edge 2->1: top=diff mass=0.9778, active=0.260692, transform=0.2177, skip=0.3024, disable=0.4799, tape=0.031075
- edge 2->2: top=contrast mass=0.3031, active=0.000713, transform=0.0008, skip=0.8270, disable=0.1722, tape=0.000001
- edge 2->3: top=replace mass=0.1262, active=0.022125, transform=0.0164, skip=0.4519, disable=0.5318, tape=0.000103
- edge 3->0: top=replace mass=0.2169, active=0.023902, transform=0.0549, skip=0.4104, disable=0.5346, tape=0.000651
- edge 3->1: top=diff mass=0.9831, active=0.249747, transform=0.2153, skip=0.2547, disable=0.5299, tape=0.029123
- edge 3->2: top=replace mass=0.1457, active=0.014774, transform=0.0163, skip=0.5434, disable=0.4402, tape=0.000081
- edge 3->3: top=replace mass=0.2577, active=0.000648, transform=0.0009, skip=0.6557, disable=0.3434, tape=0.000000

