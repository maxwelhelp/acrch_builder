# ActionMatrix Program Report

- task: `diff`
- batch_acc: `0.4375`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 0.3125, 'action_L0_0_1_diff_choice_mass': 0.035896338522434235, 'action_L0_0_1_diff_recovery': 0.0, 'action_L0_0_1_diff_active': 0.24055646359920502, 'action_L0_0_1_diff_tape': 0.0831468254327774}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `not_recovered`
- expected_top_cells: `0`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | smooth:0.28/a0.260 | *smooth:0.26/a0.241 | smooth:0.37/a0.250 | smooth:0.34/a0.246 |
| s1 | smooth:0.36/a0.251 | smooth:0.26/a0.148 | smooth:0.41/a0.180 | smooth:0.44/a0.235 |
| s2 | smooth:0.40/a0.295 | smooth:0.27/a0.205 | smooth:0.39/a0.180 | smooth:0.35/a0.258 |
| s3 | smooth:0.38/a0.264 | smooth:0.32/a0.219 | smooth:0.40/a0.218 | smooth:0.38/a0.205 |

### Cells
- edge 0->0: top=smooth mass=0.2802, active=0.260370, transform=0.3077, skip=0.4088, disable=0.2834, tape=0.044611
- edge 0->1 EXPECTED: top=smooth mass=0.2616, expected=diff mass=0.0359, active=0.240556, transform=0.5764, skip=0.2739, disable=0.1497, tape=0.083147
- edge 0->2: top=smooth mass=0.3682, active=0.249954, transform=0.4421, skip=0.2830, disable=0.2749, tape=0.062456
- edge 0->3: top=smooth mass=0.3362, active=0.246244, transform=0.4915, skip=0.2649, disable=0.2436, tape=0.065342
- edge 1->0: top=smooth mass=0.3649, active=0.251183, transform=0.5627, skip=0.2199, disable=0.2174, tape=0.086786
- edge 1->1: top=smooth mass=0.2577, active=0.148421, transform=0.6416, skip=0.2124, disable=0.1460, tape=0.049664
- edge 1->2: top=smooth mass=0.4106, active=0.180289, transform=0.6307, skip=0.1658, disable=0.2035, tape=0.065131
- edge 1->3: top=smooth mass=0.4380, active=0.234991, transform=0.6378, skip=0.1749, disable=0.1873, tape=0.082547
- edge 2->0: top=smooth mass=0.4038, active=0.294834, transform=0.4870, skip=0.3553, disable=0.1577, tape=0.067907
- edge 2->1: top=smooth mass=0.2688, active=0.205191, transform=0.6603, skip=0.2555, disable=0.0842, tape=0.068063
- edge 2->2: top=smooth mass=0.3939, active=0.180417, transform=0.5129, skip=0.2828, disable=0.2043, tape=0.039858
- edge 2->3: top=smooth mass=0.3483, active=0.257883, transform=0.5994, skip=0.2578, disable=0.1428, tape=0.073327
- edge 3->0: top=smooth mass=0.3782, active=0.264136, transform=0.4945, skip=0.3277, disable=0.1778, tape=0.080372
- edge 3->1: top=smooth mass=0.3209, active=0.218812, transform=0.6275, skip=0.2785, disable=0.0940, tape=0.078213
- edge 3->2: top=smooth mass=0.4000, active=0.217777, transform=0.5696, skip=0.2456, disable=0.1848, tape=0.074133
- edge 3->3: top=smooth mass=0.3781, active=0.204906, transform=0.5064, skip=0.3317, disable=0.1619, tape=0.050167

