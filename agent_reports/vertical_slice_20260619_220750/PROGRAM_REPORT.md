# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.6796875`
- final_read: `last`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

Flow:
```text
input -> Layer 0 -> Layer 1 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `not_recovered`
- expected_top_cells: `0`
- active_cells: `8`
- active_edges_per_target: `2.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | forget:0.23/a0.001 | *ctx_matrix:0.38/a0.972 | ctx_matrix:0.49/a0.452 | ctx_matrix:0.51/a0.971 |
| s1 | ctx_matrix:0.54/a0.011 | ctx_matrix:0.34/a0.001 | ctx_matrix:0.58/a0.008 | ctx_matrix:0.48/a0.374 |
| s2 | ctx_matrix:0.43/a0.403 | ctx_matrix:0.40/a0.961 | forget:0.36/a0.001 | *ctx_matrix:0.54/a0.974 |
| s3 | ctx_matrix:0.47/a0.002 | ctx_matrix:0.41/a0.253 | ctx_matrix:0.54/a0.004 | forget:0.33/a0.000 |

### Cells
- edge 0->0: top=forget mass=0.2347, active=0.001477, transform=0.0015, skip=0.2741, disable=0.7243, tape=0.000001
- edge 0->1 EXPECTED: top=ctx_matrix mass=0.3793, expected=diff mass=0.0044, active=0.972322, transform=0.1542, skip=0.0135, disable=0.8323, tape=0.001270
- edge 0->2: top=ctx_matrix mass=0.4892, active=0.452087, transform=0.0388, skip=0.6645, disable=0.2967, tape=0.001222
- edge 0->3: top=ctx_matrix mass=0.5076, active=0.971434, transform=0.6754, skip=0.0158, disable=0.3088, tape=0.000609
- edge 1->0: top=ctx_matrix mass=0.5363, active=0.010778, transform=0.0273, skip=0.6101, disable=0.3626, tape=0.000175
- edge 1->1: top=ctx_matrix mass=0.3408, active=0.001378, transform=0.0024, skip=0.0387, disable=0.9589, tape=0.000001
- edge 1->2: top=ctx_matrix mass=0.5759, active=0.007876, transform=0.0123, skip=0.8371, disable=0.1506, tape=0.000057
- edge 1->3: top=ctx_matrix mass=0.4787, active=0.374312, transform=0.5343, skip=0.0878, disable=0.3779, tape=0.002162
- edge 2->0: top=ctx_matrix mass=0.4285, active=0.402877, transform=0.0968, skip=0.3667, disable=0.5366, tape=0.001967
- edge 2->1: top=ctx_matrix mass=0.3960, active=0.961330, transform=0.2265, skip=0.0121, disable=0.7614, tape=0.001189
- edge 2->2: top=forget mass=0.3572, active=0.001374, transform=0.0022, skip=0.5495, disable=0.4482, tape=0.000000
- edge 2->3 EXPECTED: top=ctx_matrix mass=0.5402, expected=diff mass=0.0004, active=0.973552, transform=0.8023, skip=0.0109, disable=0.1868, tape=0.000823
- edge 3->0: top=ctx_matrix mass=0.4721, active=0.002057, transform=0.0068, skip=0.3414, disable=0.6518, tape=0.000023
- edge 3->1: top=ctx_matrix mass=0.4104, active=0.252622, transform=0.0120, skip=0.0142, disable=0.9737, tape=0.001173
- edge 3->2: top=ctx_matrix mass=0.5365, active=0.003771, transform=0.0060, skip=0.6112, disable=0.3828, tape=0.000033
- edge 3->3: top=forget mass=0.3272, active=0.000396, transform=0.0055, skip=0.0290, disable=0.9656, tape=0.000001

## Layer 1

- verdict: `primitive_collapse`
- expected_top_cells: `11`
- active_cells: `8`
- active_edges_per_target: `2.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | product:0.57/a0.000 | product:0.56/a0.098 | ctx_matrix:0.62/a0.005 | product:0.59/a0.669 |
| s1 | product:0.57/a0.124 | product:0.55/a0.003 | ctx_matrix:0.60/a0.138 | *product:0.58/a0.932 |
| s2 | product:0.53/a0.021 | product:0.55/a0.191 | ctx_matrix:0.46/a0.000 | product:0.68/a0.767 |
| s3 | ctx_matrix:0.52/a0.006 | product:0.55/a0.069 | ctx_matrix:0.66/a0.005 | product:0.51/a0.020 |

### Cells
- edge 0->0: top=product mass=0.5660, active=0.000002, transform=0.0000, skip=0.6990, disable=0.3010, tape=0.000000
- edge 0->1: top=product mass=0.5598, active=0.097789, transform=0.0463, skip=0.7386, disable=0.2152, tape=0.001945
- edge 0->2: top=ctx_matrix mass=0.6214, active=0.004554, transform=0.0078, skip=0.5331, disable=0.4592, tape=0.000022
- edge 0->3: top=product mass=0.5876, active=0.669365, transform=0.0035, skip=0.9470, disable=0.0495, tape=0.000379
- edge 1->0: top=product mass=0.5689, active=0.123774, transform=0.0211, skip=0.5338, disable=0.4452, tape=0.001183
- edge 1->1: top=product mass=0.5520, active=0.003097, transform=0.0002, skip=0.4154, disable=0.5844, tape=0.000000
- edge 1->2: top=ctx_matrix mass=0.5974, active=0.137748, transform=0.0590, skip=0.2506, disable=0.6904, tape=0.005120
- edge 1->3 EXPECTED: top=product mass=0.5849, expected=product mass=0.5849, active=0.932435, transform=0.0137, skip=0.8184, disable=0.1679, tape=0.002669
- edge 2->0: top=product mass=0.5272, active=0.020526, transform=0.0033, skip=0.8708, disable=0.1259, tape=0.000032
- edge 2->1: top=product mass=0.5529, active=0.191148, transform=0.0733, skip=0.8010, disable=0.1257, tape=0.008014
- edge 2->2: top=ctx_matrix mass=0.4577, active=0.000008, transform=0.0000, skip=0.5919, disable=0.4080, tape=0.000000
- edge 2->3: top=product mass=0.6767, active=0.766547, transform=0.0043, skip=0.9702, disable=0.0256, tape=0.000750
- edge 3->0: top=ctx_matrix mass=0.5214, active=0.006447, transform=0.0265, skip=0.3498, disable=0.6237, tape=0.000160
- edge 3->1: top=product mass=0.5476, active=0.068948, transform=0.1330, skip=0.2820, disable=0.5850, tape=0.006478
- edge 3->2: top=ctx_matrix mass=0.6605, active=0.005067, transform=0.0517, skip=0.1487, disable=0.7996, tape=0.000227
- edge 3->3: top=product mass=0.5065, active=0.020384, transform=0.0001, skip=0.5571, disable=0.4428, tape=0.000000

