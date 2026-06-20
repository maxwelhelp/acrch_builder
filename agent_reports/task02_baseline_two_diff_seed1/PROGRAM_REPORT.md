# ActionMatrix Program Report

- task: `two_diff`
- batch_acc: `0.9296875`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9997841715812683, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8494822978973389, 'action_L0_0_1_diff_tape': 0.38230419158935547, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9991718530654907, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.8477767705917358, 'action_L0_2_3_diff_tape': 0.33201146125793457}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `6`
- active_cells: `10`
- active_edges_per_target: `2.5`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.62/a0.036 | *diff:1.00/a0.849 | ctx_matrix:0.41/a0.419 | diff:1.00/a0.863 |
| s1 | ctx_matrix:0.45/a0.024 | ctx_matrix:0.46/a0.045 | ctx_matrix:0.47/a0.051 | diff:0.79/a0.381 |
| s2 | ctx_matrix:0.33/a0.260 | diff:1.00/a0.820 | ctx_matrix:0.53/a0.057 | *diff:1.00/a0.848 |
| s3 | ctx_matrix:0.42/a0.018 | diff:0.45/a0.284 | ctx_matrix:0.49/a0.036 | ctx_matrix:0.50/a0.041 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.6231, active=0.035717, transform=0.0608, skip=0.3026, disable=0.6366, tape=0.000892
- edge 0->1 EXPECTED: top=diff mass=0.9998, expected=diff mass=0.9998, active=0.849482, transform=0.6512, skip=0.2209, disable=0.1279, tape=0.382304
- edge 0->2: top=ctx_matrix mass=0.4079, active=0.419077, transform=0.2617, skip=0.4261, disable=0.3121, tape=0.040556
- edge 0->3: top=diff mass=0.9998, active=0.863450, transform=0.6391, skip=0.2149, disable=0.1460, tape=0.330862
- edge 1->0: top=ctx_matrix mass=0.4542, active=0.024352, transform=0.0624, skip=0.3389, disable=0.5987, tape=0.000417
- edge 1->1: top=ctx_matrix mass=0.4647, active=0.044769, transform=0.0251, skip=0.2695, disable=0.7054, tape=0.000479
- edge 1->2: top=ctx_matrix mass=0.4725, active=0.051209, transform=0.0418, skip=0.3963, disable=0.5619, tape=0.000568
- edge 1->3: top=diff mass=0.7905, active=0.381015, transform=0.1796, skip=0.3418, disable=0.4786, tape=0.024192
- edge 2->0: top=ctx_matrix mass=0.3316, active=0.259559, transform=0.3295, skip=0.2865, disable=0.3840, tape=0.037940
- edge 2->1: top=diff mass=0.9992, active=0.820161, transform=0.6334, skip=0.2006, disable=0.1660, tape=0.362267
- edge 2->2: top=ctx_matrix mass=0.5308, active=0.057225, transform=0.0377, skip=0.2981, disable=0.6642, tape=0.000793
- edge 2->3 EXPECTED: top=diff mass=0.9992, expected=diff mass=0.9992, active=0.847777, transform=0.6309, skip=0.1811, disable=0.1880, tape=0.332011
- edge 3->0: top=ctx_matrix mass=0.4236, active=0.017672, transform=0.0446, skip=0.5238, disable=0.4315, tape=0.000254
- edge 3->1: top=diff mass=0.4460, active=0.284077, transform=0.1313, skip=0.5692, disable=0.2995, tape=0.017004
- edge 3->2: top=ctx_matrix mass=0.4926, active=0.035587, transform=0.0286, skip=0.5826, disable=0.3888, tape=0.000257
- edge 3->3: top=ctx_matrix mass=0.5016, active=0.041334, transform=0.0166, skip=0.3888, disable=0.5945, tape=0.000260

