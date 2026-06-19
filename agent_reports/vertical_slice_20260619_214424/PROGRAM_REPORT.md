# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.6796875`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `16`
- active_cells: `9`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | diff:1.00/a0.001 | *diff:1.00/a0.975 | diff:1.00/a0.652 | diff:1.00/a0.927 |
| s1 | diff:1.00/a0.004 | diff:1.00/a0.001 | diff:1.00/a0.019 | diff:1.00/a0.174 |
| s2 | diff:1.00/a0.221 | diff:1.00/a0.943 | diff:1.00/a0.001 | *diff:1.00/a0.882 |
| s3 | diff:1.00/a0.027 | diff:1.00/a0.694 | diff:1.00/a0.125 | diff:1.00/a0.000 |

### Cells
- edge 0->0: top=diff mass=0.9998, active=0.001068, transform=0.4968, skip=0.4886, disable=0.0146, tape=0.000379
- edge 0->1 EXPECTED: top=diff mass=0.9993, expected=diff mass=0.9993, active=0.975261, transform=0.0630, skip=0.0566, disable=0.8804, tape=0.017561
- edge 0->2: top=diff mass=0.9983, active=0.652256, transform=0.0709, skip=0.1518, disable=0.7773, tape=0.015216
- edge 0->3: top=diff mass=0.9987, active=0.927046, transform=0.0379, skip=0.1120, disable=0.8501, tape=0.007866
- edge 1->0: top=diff mass=0.9973, active=0.003840, transform=0.0151, skip=0.4325, disable=0.5523, tape=0.000009
- edge 1->1: top=diff mass=0.9998, active=0.001251, transform=0.5595, skip=0.4221, disable=0.0184, tape=0.000502
- edge 1->2: top=diff mass=0.9956, active=0.018524, transform=0.0287, skip=0.2818, disable=0.6895, tape=0.000107
- edge 1->3: top=diff mass=0.9984, active=0.173505, transform=0.0278, skip=0.4014, disable=0.5708, tape=0.000621
- edge 2->0: top=diff mass=0.9989, active=0.221002, transform=0.0274, skip=0.4627, disable=0.5099, tape=0.001152
- edge 2->1: top=diff mass=0.9992, active=0.943217, transform=0.0581, skip=0.1353, disable=0.8066, tape=0.017212
- edge 2->2: top=diff mass=0.9996, active=0.000894, transform=0.3591, skip=0.6324, disable=0.0085, tape=0.000214
- edge 2->3 EXPECTED: top=diff mass=0.9987, expected=diff mass=0.9987, active=0.882382, transform=0.0252, skip=0.3128, disable=0.6620, tape=0.003732
- edge 3->0: top=diff mass=0.9982, active=0.026555, transform=0.0258, skip=0.2118, disable=0.7624, tape=0.000121
- edge 3->1: top=diff mass=0.9992, active=0.694449, transform=0.0795, skip=0.1028, disable=0.8177, tape=0.017215
- edge 3->2: top=diff mass=0.9969, active=0.124507, transform=0.0356, skip=0.1615, disable=0.8029, tape=0.000846
- edge 3->3: top=diff mass=0.9998, active=0.000497, transform=0.5692, skip=0.4135, disable=0.0172, tape=0.000190

## Layer 1

- verdict: `not_recovered`
- expected_top_cells: `0`
- active_cells: `10`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | smooth:0.21/a0.001 | diff:0.15/a0.061 | forget:0.16/a0.077 | diff:0.15/a0.618 |
| s1 | diff:1.00/a0.354 | diff:0.99/a0.010 | diff:1.00/a0.351 | *diff:0.99/a0.900 |
| s2 | diff:0.35/a0.146 | diff:0.29/a0.056 | diff:0.23/a0.001 | diff:0.30/a0.571 |
| s3 | diff:1.00/a0.147 | diff:0.99/a0.037 | diff:0.99/a0.036 | diff:0.98/a0.039 |

### Cells
- edge 0->0: top=smooth mass=0.2056, active=0.001431, transform=0.0000, skip=0.7508, disable=0.2492, tape=0.000000
- edge 0->1: top=diff mass=0.1507, active=0.060768, transform=0.0183, skip=0.8396, disable=0.1421, tape=0.000488
- edge 0->2: top=forget mass=0.1567, active=0.076886, transform=0.0115, skip=0.6628, disable=0.3256, tape=0.000437
- edge 0->3: top=diff mass=0.1534, active=0.618298, transform=0.0048, skip=0.9605, disable=0.0347, tape=0.000523
- edge 1->0: top=diff mass=0.9981, active=0.354374, transform=0.0475, skip=0.3507, disable=0.6018, tape=0.009927
- edge 1->1: top=diff mass=0.9851, active=0.010267, transform=0.0001, skip=0.3352, disable=0.6647, tape=0.000000
- edge 1->2: top=diff mass=0.9963, active=0.351387, transform=0.0833, skip=0.1858, disable=0.7310, tape=0.019909
- edge 1->3 EXPECTED: top=diff mass=0.9928, expected=product mass=0.0011, active=0.899706, transform=0.0521, skip=0.6708, disable=0.2772, tape=0.017468
- edge 2->0: top=diff mass=0.3546, active=0.145518, transform=0.0049, skip=0.8539, disable=0.1412, tape=0.000434
- edge 2->1: top=diff mass=0.2933, active=0.055917, transform=0.0252, skip=0.8615, disable=0.1133, tape=0.000863
- edge 2->2: top=diff mass=0.2298, active=0.000716, transform=0.0000, skip=0.6558, disable=0.3442, tape=0.000000
- edge 2->3: top=diff mass=0.3017, active=0.570620, transform=0.0054, skip=0.9646, disable=0.0300, tape=0.001022
- edge 3->0: top=diff mass=0.9969, active=0.146877, transform=0.0490, skip=0.1911, disable=0.7599, tape=0.004739
- edge 3->1: top=diff mass=0.9884, active=0.037365, transform=0.1019, skip=0.1943, disable=0.7039, tape=0.002245
- edge 3->2: top=diff mass=0.9945, active=0.035651, transform=0.0483, skip=0.0780, disable=0.8737, tape=0.000972
- edge 3->3: top=diff mass=0.9810, active=0.038692, transform=0.0000, skip=0.3463, disable=0.6536, tape=0.000000

