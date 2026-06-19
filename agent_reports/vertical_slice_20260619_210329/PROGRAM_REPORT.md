# ActionMatrix Program Report

- task: `diff`
- batch_acc: `0.95703125`
- expected: `diff` on edge `0->1`
- verdict: `primitive_collapse`
- expected_top_cells: `15`
- active_cells: `13`

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | diff:0.59/a0.114 | *diff:0.51/a0.705 | diff:0.51/a0.230 | diff:0.51/a0.265 |
| s1 | diff:0.86/a0.301 | diff:0.73/a0.110 | diff:0.69/a0.093 | diff:0.76/a0.116 |
| s2 | diff:0.76/a0.118 | diff:0.60/a0.263 | diff:0.69/a0.003 | diff:0.56/a0.055 |
| s3 | diff:0.63/a0.057 | diff:0.55/a0.149 | ctx_matrix:0.53/a0.015 | diff:0.66/a0.002 |

## Cells

- edge 0->0: top=diff mass=0.5923, expected_mass=0.5923, active=0.113679, transform=0.0115, skip=0.8924, disable=0.0960, tape=0.000840
- edge 0->1 EXPECTED: top=diff mass=0.5097, expected_mass=0.5097, active=0.704996, transform=0.1030, skip=0.5953, disable=0.3017, tape=0.035512
- edge 0->2: top=diff mass=0.5097, expected_mass=0.5097, active=0.230415, transform=0.0195, skip=0.7306, disable=0.2499, tape=0.002616
- edge 0->3: top=diff mass=0.5130, expected_mass=0.5130, active=0.265076, transform=0.0316, skip=0.8422, disable=0.1262, tape=0.003876
- edge 1->0: top=diff mass=0.8579, expected_mass=0.8579, active=0.301404, transform=0.2159, skip=0.5416, disable=0.2425, tape=0.042995
- edge 1->1: top=diff mass=0.7315, expected_mass=0.7315, active=0.109748, transform=0.0185, skip=0.8979, disable=0.0835, tape=0.002046
- edge 1->2: top=diff mass=0.6871, expected_mass=0.6871, active=0.093008, transform=0.0396, skip=0.7758, disable=0.1846, tape=0.002800
- edge 1->3: top=diff mass=0.7564, expected_mass=0.7564, active=0.116314, transform=0.0555, skip=0.8469, disable=0.0975, tape=0.004602
- edge 2->0: top=diff mass=0.7575, expected_mass=0.7575, active=0.117677, transform=0.0301, skip=0.5196, disable=0.4503, tape=0.002420
- edge 2->1: top=diff mass=0.6000, expected_mass=0.6000, active=0.262860, transform=0.0261, skip=0.5859, disable=0.3880, tape=0.005134
- edge 2->2: top=diff mass=0.6887, expected_mass=0.6887, active=0.002713, transform=0.0004, skip=0.9310, disable=0.0685, tape=0.000001
- edge 2->3: top=diff mass=0.5643, expected_mass=0.5643, active=0.055262, transform=0.0065, skip=0.8206, disable=0.1729, tape=0.000182
- edge 3->0: top=diff mass=0.6319, expected_mass=0.6319, active=0.056860, transform=0.0329, skip=0.2745, disable=0.6925, tape=0.001422
- edge 3->1: top=diff mass=0.5486, expected_mass=0.5486, active=0.149428, transform=0.0208, skip=0.3254, disable=0.6538, tape=0.001626
- edge 3->2: top=ctx_matrix mass=0.5314, expected_mass=0.4563, active=0.014928, transform=0.0052, skip=0.4279, disable=0.5669, tape=0.000060
- edge 3->3: top=diff mass=0.6639, expected_mass=0.6639, active=0.001542, transform=0.0010, skip=0.8581, disable=0.1409, tape=0.000001
