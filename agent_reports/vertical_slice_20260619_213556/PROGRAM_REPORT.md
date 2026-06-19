# ActionMatrix Program Report

- task: `two_diff`
- batch_acc: `0.9140625`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `16`
- active_cells: `16`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | diff:1.00/a0.205 | *diff:0.98/a0.711 | diff:1.00/a0.418 | diff:0.98/a0.674 |
| s1 | diff:1.00/a0.270 | diff:0.99/a0.134 | diff:1.00/a0.220 | diff:0.98/a0.324 |
| s2 | diff:1.00/a0.485 | diff:0.97/a0.687 | diff:1.00/a0.193 | *diff:0.97/a0.752 |
| s3 | diff:1.00/a0.198 | diff:0.99/a0.253 | diff:1.00/a0.196 | diff:0.98/a0.081 |

### Cells
- edge 0->0: top=diff mass=0.9992, active=0.204937, transform=0.0235, skip=0.8060, disable=0.1705, tape=0.001136
- edge 0->1 EXPECTED: top=diff mass=0.9842, expected=diff mass=0.9842, active=0.711273, transform=0.0735, skip=0.6011, disable=0.3254, tape=0.022093
- edge 0->2: top=diff mass=0.9966, active=0.418489, transform=0.0551, skip=0.7467, disable=0.1982, tape=0.009994
- edge 0->3: top=diff mass=0.9768, active=0.674485, transform=0.0945, skip=0.6460, disable=0.2595, tape=0.022353
- edge 1->0: top=diff mass=0.9989, active=0.270112, transform=0.1890, skip=0.6091, disable=0.2018, tape=0.026456
- edge 1->1: top=diff mass=0.9944, active=0.134083, transform=0.0314, skip=0.8105, disable=0.1581, tape=0.000984
- edge 1->2: top=diff mass=0.9970, active=0.220001, transform=0.1425, skip=0.7265, disable=0.1310, tape=0.019336
- edge 1->3: top=diff mass=0.9809, active=0.324410, transform=0.1423, skip=0.6683, disable=0.1895, tape=0.017741
- edge 2->0: top=diff mass=0.9956, active=0.484608, transform=0.0873, skip=0.5446, disable=0.3681, tape=0.018050
- edge 2->1: top=diff mass=0.9732, active=0.687447, transform=0.0848, skip=0.5334, disable=0.3818, tape=0.027885
- edge 2->2: top=diff mass=0.9978, active=0.193241, transform=0.0153, skip=0.8594, disable=0.1253, tape=0.000850
- edge 2->3 EXPECTED: top=diff mass=0.9703, expected=diff mass=0.9703, active=0.751777, transform=0.0923, skip=0.5633, disable=0.3444, tape=0.022565
- edge 3->0: top=diff mass=0.9985, active=0.198248, transform=0.1858, skip=0.5179, disable=0.2962, tape=0.024660
- edge 3->1: top=diff mass=0.9867, active=0.253150, transform=0.1092, skip=0.5506, disable=0.3402, tape=0.016548
- edge 3->2: top=diff mass=0.9974, active=0.196430, transform=0.1299, skip=0.6531, disable=0.2170, tape=0.018691
- edge 3->3: top=diff mass=0.9834, active=0.080572, transform=0.0370, skip=0.7645, disable=0.1985, tape=0.000711

