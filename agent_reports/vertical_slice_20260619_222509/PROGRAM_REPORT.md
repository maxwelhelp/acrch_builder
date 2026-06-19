# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.5546875`
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
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.23/a0.890 | *ctx_matrix:0.39/a0.983 | ctx_matrix:0.44/a0.789 | ctx_matrix:0.48/a0.982 |
| s1 | ctx_matrix:0.57/a0.345 | ctx_matrix:0.34/a0.876 | ctx_matrix:0.61/a0.274 | ctx_matrix:0.50/a0.814 |
| s2 | ctx_matrix:0.47/a0.830 | ctx_matrix:0.40/a0.982 | ctx_matrix:0.35/a0.873 | *ctx_matrix:0.51/a0.986 |
| s3 | ctx_matrix:0.56/a0.202 | ctx_matrix:0.46/a0.718 | ctx_matrix:0.58/a0.191 | ctx_matrix:0.34/a0.808 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.2266, active=0.890469, transform=0.0305, skip=0.8233, disable=0.1462, tape=0.000161
- edge 0->1 EXPECTED: top=ctx_matrix mass=0.3877, expected=diff mass=0.0728, active=0.983456, transform=0.5675, skip=0.2676, disable=0.1649, tape=0.005910
- edge 0->2: top=ctx_matrix mass=0.4448, active=0.789454, transform=0.2896, skip=0.4659, disable=0.2445, tape=0.005552
- edge 0->3: top=ctx_matrix mass=0.4828, active=0.982134, transform=0.8079, skip=0.1102, disable=0.0820, tape=0.004545
- edge 1->0: top=ctx_matrix mass=0.5721, active=0.345388, transform=0.2833, skip=0.4831, disable=0.2336, tape=0.003038
- edge 1->1: top=ctx_matrix mass=0.3435, active=0.876330, transform=0.0251, skip=0.9071, disable=0.0678, tape=0.000162
- edge 1->2: top=ctx_matrix mass=0.6057, active=0.274449, transform=0.1927, skip=0.6297, disable=0.1776, tape=0.002310
- edge 1->3: top=ctx_matrix mass=0.4970, active=0.814225, transform=0.6426, skip=0.2640, disable=0.0934, tape=0.007709
- edge 2->0: top=ctx_matrix mass=0.4674, active=0.829803, transform=0.3664, skip=0.2728, disable=0.3608, tape=0.005823
- edge 2->1: top=ctx_matrix mass=0.4003, active=0.981593, transform=0.5599, skip=0.2306, disable=0.2095, tape=0.005314
- edge 2->2: top=ctx_matrix mass=0.3485, active=0.873160, transform=0.0116, skip=0.8649, disable=0.1235, tape=0.000072
- edge 2->3 EXPECTED: top=ctx_matrix mass=0.5118, expected=diff mass=0.0612, active=0.985702, transform=0.7584, skip=0.1108, disable=0.1308, tape=0.004873
- edge 3->0: top=ctx_matrix mass=0.5610, active=0.202190, transform=0.2406, skip=0.4736, disable=0.2858, tape=0.002151
- edge 3->1: top=ctx_matrix mass=0.4601, active=0.717919, transform=0.3156, skip=0.4966, disable=0.1878, tape=0.008154
- edge 3->2: top=ctx_matrix mass=0.5818, active=0.190759, transform=0.1397, skip=0.6186, disable=0.2417, tape=0.001780
- edge 3->3: top=ctx_matrix mass=0.3364, active=0.807676, transform=0.0482, skip=0.8562, disable=0.0956, tape=0.000243

## Layer 1

- verdict: `not_recovered`
- expected_top_cells: `0`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | diff:0.99/a0.877 | diff:0.98/a0.699 | diff:0.96/a0.650 | diff:0.97/a0.943 |
| s1 | diff:0.94/a0.756 | diff:0.96/a0.954 | diff:0.92/a0.755 | *diff:0.95/a0.984 |
| s2 | diff:0.98/a0.598 | diff:0.98/a0.635 | diff:0.99/a0.876 | diff:0.98/a0.935 |
| s3 | diff:0.92/a0.455 | diff:0.95/a0.674 | diff:0.93/a0.475 | diff:0.97/a0.984 |

### Cells
- edge 0->0: top=diff mass=0.9854, active=0.876744, transform=0.5384, skip=0.3873, disable=0.0743, tape=0.350982
- edge 0->1: top=diff mass=0.9760, active=0.698843, transform=0.6324, skip=0.2570, disable=0.1106, tape=0.289563
- edge 0->2: top=diff mass=0.9605, active=0.650171, transform=0.6608, skip=0.2182, disable=0.1211, tape=0.266210
- edge 0->3: top=diff mass=0.9700, active=0.942690, transform=0.3692, skip=0.5217, disable=0.1091, tape=0.162838
- edge 1->0: top=diff mass=0.9428, active=0.755640, transform=0.6519, skip=0.2179, disable=0.1302, tape=0.305113
- edge 1->1: top=diff mass=0.9621, active=0.953894, transform=0.7529, skip=0.1761, disable=0.0710, tape=0.490282
- edge 1->2: top=diff mass=0.9241, active=0.755089, transform=0.7135, skip=0.1351, disable=0.1513, tape=0.298328
- edge 1->3 EXPECTED: top=diff mass=0.9460, expected=product mass=0.0124, active=0.983891, transform=0.5824, skip=0.3176, disable=0.1000, tape=0.266919
- edge 2->0: top=diff mass=0.9827, active=0.598040, transform=0.5781, skip=0.3335, disable=0.0883, tape=0.247062
- edge 2->1: top=diff mass=0.9843, active=0.635102, transform=0.6324, skip=0.2642, disable=0.1034, tape=0.279535
- edge 2->2: top=diff mass=0.9893, active=0.876248, transform=0.6656, skip=0.2625, disable=0.0719, tape=0.451097
- edge 2->3: top=diff mass=0.9847, active=0.934936, transform=0.4251, skip=0.4861, disable=0.0888, tape=0.214111
- edge 3->0: top=diff mass=0.9204, active=0.454877, transform=0.6730, skip=0.1539, disable=0.1731, tape=0.201593
- edge 3->1: top=diff mass=0.9548, active=0.674141, transform=0.8134, skip=0.0852, disable=0.1015, tape=0.380060
- edge 3->2: top=diff mass=0.9295, active=0.475151, transform=0.7664, skip=0.0801, disable=0.1535, tape=0.223162
- edge 3->3: top=diff mass=0.9662, active=0.983974, transform=0.7300, skip=0.1996, disable=0.0704, tape=0.476526

