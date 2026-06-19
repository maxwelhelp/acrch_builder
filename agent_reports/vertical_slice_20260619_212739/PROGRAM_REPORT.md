# ActionMatrix Program Report

- task: `two_diff`
- batch_acc: `0.91796875`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `16`
- active_cells: `16`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | diff:1.00/a0.287 | *diff:0.90/a0.735 | diff:0.96/a0.460 | diff:0.81/a0.700 |
| s1 | diff:0.98/a0.295 | diff:0.97/a0.185 | diff:0.97/a0.239 | diff:0.88/a0.345 |
| s2 | diff:0.98/a0.531 | diff:0.89/a0.717 | diff:1.00/a0.277 | *diff:0.85/a0.774 |
| s3 | diff:0.99/a0.214 | diff:0.95/a0.269 | diff:0.99/a0.211 | diff:0.97/a0.114 |

### Cells
- edge 0->0: top=diff mass=0.9996, active=0.286631, transform=0.0202, skip=0.8254, disable=0.1543, tape=0.001263
- edge 0->1 EXPECTED: top=diff mass=0.8975, expected=diff mass=0.8975, active=0.735340, transform=0.0768, skip=0.5955, disable=0.3277, tape=0.023783
- edge 0->2: top=diff mass=0.9631, active=0.460304, transform=0.0540, skip=0.7455, disable=0.2005, tape=0.010311
- edge 0->3: top=diff mass=0.8139, active=0.700189, transform=0.1002, skip=0.6371, disable=0.2628, tape=0.024259
- edge 1->0: top=diff mass=0.9833, active=0.294527, transform=0.1819, skip=0.6155, disable=0.2026, tape=0.026369
- edge 1->1: top=diff mass=0.9741, active=0.185088, transform=0.0276, skip=0.8287, disable=0.1437, tape=0.001123
- edge 1->2: top=diff mass=0.9691, active=0.239322, transform=0.1353, skip=0.7326, disable=0.1321, tape=0.019150
- edge 1->3: top=diff mass=0.8780, active=0.345333, transform=0.1446, skip=0.6679, disable=0.1875, tape=0.018484
- edge 2->0: top=diff mass=0.9780, active=0.531464, transform=0.0864, skip=0.5416, disable=0.3720, tape=0.018774
- edge 2->1: top=diff mass=0.8897, active=0.716590, transform=0.0887, skip=0.5249, disable=0.3864, tape=0.030182
- edge 2->2: top=diff mass=0.9992, active=0.277083, transform=0.0132, skip=0.8727, disable=0.1141, tape=0.000965
- edge 2->3 EXPECTED: top=diff mass=0.8536, expected=diff mass=0.8536, active=0.774409, transform=0.0984, skip=0.5536, disable=0.3480, tape=0.024807
- edge 3->0: top=diff mass=0.9854, active=0.213690, transform=0.1767, skip=0.5252, disable=0.2980, tape=0.024152
- edge 3->1: top=diff mass=0.9496, active=0.269342, transform=0.1081, skip=0.5550, disable=0.3369, tape=0.016842
- edge 3->2: top=diff mass=0.9861, active=0.211160, transform=0.1217, skip=0.6605, disable=0.2178, tape=0.018109
- edge 3->3: top=diff mass=0.9730, active=0.114353, transform=0.0325, skip=0.7895, disable=0.1780, tape=0.000820

