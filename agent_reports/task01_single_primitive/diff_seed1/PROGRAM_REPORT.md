# ActionMatrix Program Report

- task: `diff`
- batch_acc: `0.96484375`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9997667670249939, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8946657776832581, 'action_L0_0_1_diff_tape': 0.4447152614593506}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `5`
- active_cells: `5`
- active_edges_per_target: `1.25`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.30/a0.042 | *diff:1.00/a0.895 | diff:0.93/a0.519 | diff:0.90/a0.499 |
| s1 | ctx_matrix:0.21/a0.014 | ctx_matrix:0.23/a0.032 | ctx_matrix:0.22/a0.037 | ctx_matrix:0.22/a0.027 |
| s2 | ctx_matrix:0.16/a0.008 | diff:0.94/a0.267 | ctx_matrix:0.27/a0.000 | ctx_matrix:0.14/a0.017 |
| s3 | ctx_matrix:0.20/a0.016 | diff:0.94/a0.326 | ctx_matrix:0.21/a0.037 | ctx_matrix:0.25/a0.001 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.3003, active=0.041973, transform=0.0761, skip=0.3477, disable=0.5763, tape=0.002378
- edge 0->1 EXPECTED: top=diff mass=0.9998, expected=diff mass=0.9998, active=0.894666, transform=0.6602, skip=0.2211, disable=0.1188, tape=0.444715
- edge 0->2: top=diff mass=0.9310, active=0.519470, transform=0.1113, skip=0.6885, disable=0.2001, tape=0.015436
- edge 0->3: top=diff mass=0.9025, active=0.499282, transform=0.1189, skip=0.7242, disable=0.1569, tape=0.017670
- edge 1->0: top=ctx_matrix mass=0.2127, active=0.014366, transform=0.1051, skip=0.3170, disable=0.5779, tape=0.000679
- edge 1->1: top=ctx_matrix mass=0.2262, active=0.032323, transform=0.0246, skip=0.1004, disable=0.8750, tape=0.000661
- edge 1->2: top=ctx_matrix mass=0.2214, active=0.037065, transform=0.0226, skip=0.4511, disable=0.5263, tape=0.000224
- edge 1->3: top=ctx_matrix mass=0.2232, active=0.026888, transform=0.0224, skip=0.5003, disable=0.4772, tape=0.000203
- edge 2->0: top=ctx_matrix mass=0.1637, active=0.008043, transform=0.0355, skip=0.3748, disable=0.5897, tape=0.000103
- edge 2->1: top=diff mass=0.9360, active=0.267342, transform=0.1107, skip=0.3093, disable=0.5800, tape=0.017061
- edge 2->2: top=ctx_matrix mass=0.2748, active=0.000444, transform=0.0007, skip=0.2219, disable=0.7773, tape=0.000000
- edge 2->3: top=ctx_matrix mass=0.1380, active=0.016775, transform=0.0081, skip=0.5506, disable=0.4413, tape=0.000033
- edge 3->0: top=ctx_matrix mass=0.2024, active=0.015536, transform=0.0388, skip=0.5608, disable=0.4004, tape=0.000205
- edge 3->1: top=diff mass=0.9407, active=0.325871, transform=0.1066, skip=0.4735, disable=0.4199, tape=0.019804
- edge 3->2: top=ctx_matrix mass=0.2050, active=0.037336, transform=0.0067, skip=0.6806, disable=0.3126, tape=0.000057
- edge 3->3: top=ctx_matrix mass=0.2455, active=0.000591, transform=0.0008, skip=0.4156, disable=0.5836, tape=0.000000

