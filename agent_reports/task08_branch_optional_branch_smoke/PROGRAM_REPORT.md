# ActionMatrix Program Report

- task: `branch_optional_branch`
- batch_acc: `0.4765625`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'route'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'edge_gate'}, {'layer': 2, 'src': 1, 'tgt': 3, 'primitive': 'channel'}]`
- action_metrics: `{'action_L0_0_1_route_present': 0.453125, 'action_L0_0_1_route_choice_mass': 0.4208875596523285, 'action_L0_0_1_route_recovery': 0.453125, 'action_L0_0_1_route_active': 0.2504166066646576, 'action_L0_0_1_route_tape': 0.06796366721391678, 'action_L1_1_3_edge_gate_present': 0.2734375, 'action_L1_1_3_edge_gate_choice_mass': 0.2636632025241852, 'action_L1_1_3_edge_gate_recovery': 0.2734375, 'action_L1_1_3_edge_gate_active': 0.30393117666244507, 'action_L1_1_3_edge_gate_tape': 0.0769081562757492, 'action_L2_1_3_channel_present': 0.3671875, 'action_L2_1_3_channel_choice_mass': 0.3626064360141754, 'action_L2_1_3_channel_recovery': 0.3671875, 'action_L2_1_3_channel_active': 0.4147660434246063, 'action_L2_1_3_channel_tape': 0.1506851464509964}`

Flow:
```text
input -> Layer 0 -> Layer 1 -> Layer 2 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `16`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.2504166066646576`
- non_expected_active_mean: `0.26507995227972664`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'route'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | route:0.47/a0.323 | *route:0.42/a0.250 | route:0.36/a0.265 | route:0.43/a0.224 |
| s1 | route:0.50/a0.260 | route:0.54/a0.291 | route:0.42/a0.256 | route:0.41/a0.220 |
| s2 | route:0.46/a0.256 | route:0.44/a0.239 | route:0.36/a0.281 | route:0.39/a0.207 |
| s3 | route:0.56/a0.290 | route:0.47/a0.277 | route:0.44/a0.287 | route:0.51/a0.302 |

### Cells
- edge 0->0: top=route mass=0.4694, active=0.322699, transform=0.5685, skip=0.1980, disable=0.2335, tape=0.111474
- edge 0->1 EXPECTED: top=route mass=0.4209, expected=route mass=0.4209, active=0.250417, transform=0.5524, skip=0.2976, disable=0.1499, tape=0.067964
- edge 0->2: top=route mass=0.3640, active=0.264547, transform=0.5550, skip=0.2516, disable=0.1933, tape=0.071918
- edge 0->3: top=route mass=0.4276, active=0.223955, transform=0.5972, skip=0.2397, disable=0.1631, tape=0.072487
- edge 1->0: top=route mass=0.4993, active=0.260201, transform=0.5426, skip=0.2845, disable=0.1728, tape=0.068235
- edge 1->1: top=route mass=0.5383, active=0.290535, transform=0.5192, skip=0.2942, disable=0.1866, tape=0.084663
- edge 1->2: top=route mass=0.4210, active=0.255870, transform=0.5401, skip=0.2781, disable=0.1817, tape=0.064732
- edge 1->3: top=route mass=0.4054, active=0.220176, transform=0.5768, skip=0.2688, disable=0.1544, tape=0.065218
- edge 2->0: top=route mass=0.4620, active=0.256347, transform=0.5646, skip=0.2635, disable=0.1719, tape=0.080136
- edge 2->1: top=route mass=0.4365, active=0.238733, transform=0.5551, skip=0.3060, disable=0.1388, tape=0.068654
- edge 2->2: top=route mass=0.3610, active=0.280576, transform=0.5820, skip=0.1972, disable=0.2207, tape=0.101053
- edge 2->3: top=route mass=0.3889, active=0.207114, transform=0.6024, skip=0.2458, disable=0.1519, tape=0.072956
- edge 3->0: top=route mass=0.5598, active=0.290189, transform=0.4854, skip=0.3064, disable=0.2082, tape=0.077741
- edge 3->1: top=route mass=0.4708, active=0.276958, transform=0.4714, skip=0.3605, disable=0.1681, tape=0.069287
- edge 3->2: top=route mass=0.4416, active=0.286654, transform=0.4809, skip=0.2981, disable=0.2210, tape=0.073842
- edge 3->3: top=route mass=0.5138, active=0.301646, transform=0.5306, skip=0.2328, disable=0.2366, tape=0.109434

## Layer 1

- verdict: `primitive_collapse`
- expected_top_cells: `16`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.30393117666244507`
- non_expected_active_mean: `0.26550823350747427`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'edge_gate'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | edge_gate:0.61/a0.276 | edge_gate:0.51/a0.260 | edge_gate:0.45/a0.282 | edge_gate:0.40/a0.299 |
| s1 | edge_gate:0.38/a0.267 | edge_gate:0.43/a0.271 | edge_gate:0.36/a0.283 | *edge_gate:0.26/a0.304 |
| s2 | edge_gate:0.47/a0.251 | edge_gate:0.48/a0.245 | edge_gate:0.59/a0.275 | edge_gate:0.47/a0.286 |
| s3 | edge_gate:0.43/a0.232 | edge_gate:0.58/a0.229 | edge_gate:0.49/a0.246 | edge_gate:0.61/a0.281 |

### Cells
- edge 0->0: top=edge_gate mass=0.6115, active=0.275822, transform=0.6126, skip=0.1976, disable=0.1898, tape=0.085484
- edge 0->1: top=edge_gate mass=0.5144, active=0.259967, transform=0.5113, skip=0.2481, disable=0.2406, tape=0.067787
- edge 0->2: top=edge_gate mass=0.4493, active=0.281671, transform=0.5085, skip=0.2763, disable=0.2152, tape=0.085772
- edge 0->3: top=edge_gate mass=0.4043, active=0.299315, transform=0.4972, skip=0.2980, disable=0.2048, tape=0.078026
- edge 1->0: top=edge_gate mass=0.3779, active=0.266940, transform=0.6182, skip=0.2245, disable=0.1573, tape=0.080390
- edge 1->1: top=edge_gate mass=0.4292, active=0.270758, transform=0.5297, skip=0.2821, disable=0.1881, tape=0.070474
- edge 1->2: top=edge_gate mass=0.3608, active=0.282968, transform=0.5159, skip=0.3133, disable=0.1708, tape=0.084408
- edge 1->3 EXPECTED: top=edge_gate mass=0.2637, expected=edge_gate mass=0.2637, active=0.303931, transform=0.5010, skip=0.3378, disable=0.1613, tape=0.076908
- edge 2->0: top=edge_gate mass=0.4689, active=0.250850, transform=0.6307, skip=0.2478, disable=0.1215, tape=0.079338
- edge 2->1: top=edge_gate mass=0.4772, active=0.245152, transform=0.5371, skip=0.3125, disable=0.1505, tape=0.066667
- edge 2->2: top=edge_gate mass=0.5948, active=0.274610, transform=0.5312, skip=0.3416, disable=0.1272, tape=0.086072
- edge 2->3: top=edge_gate mass=0.4652, active=0.285937, transform=0.5100, skip=0.3658, disable=0.1242, tape=0.075375
- edge 3->0: top=edge_gate mass=0.4279, active=0.232448, transform=0.6000, skip=0.2618, disable=0.1382, tape=0.066746
- edge 3->1: top=edge_gate mass=0.5771, active=0.229105, transform=0.5045, skip=0.3277, disable=0.1679, tape=0.055957
- edge 3->2: top=edge_gate mass=0.4947, active=0.246384, transform=0.4952, skip=0.3572, disable=0.1475, tape=0.070007
- edge 3->3: top=edge_gate mass=0.6139, active=0.280696, transform=0.4831, skip=0.3826, disable=0.1343, tape=0.067071

## Layer 2

- verdict: `primitive_collapse`
- expected_top_cells: `16`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.4147660434246063`
- non_expected_active_mean: `0.32056166330973307`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 2, 'src': 1, 'tgt': 3, 'primitive': 'channel'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | channel:0.60/a0.293 | channel:0.58/a0.294 | channel:0.58/a0.282 | channel:0.43/a0.367 |
| s1 | channel:0.37/a0.333 | channel:0.39/a0.342 | channel:0.41/a0.328 | *channel:0.36/a0.415 |
| s2 | channel:0.39/a0.270 | channel:0.45/a0.275 | channel:0.41/a0.268 | channel:0.37/a0.343 |
| s3 | channel:0.55/a0.332 | channel:0.43/a0.341 | channel:0.43/a0.324 | channel:0.37/a0.415 |

### Cells
- edge 0->0: top=channel mass=0.5978, active=0.292662, transform=0.5776, skip=0.2744, disable=0.1480, tape=0.084396
- edge 0->1: top=channel mass=0.5805, active=0.294425, transform=0.6008, skip=0.2413, disable=0.1579, tape=0.098678
- edge 0->2: top=channel mass=0.5783, active=0.281896, transform=0.6158, skip=0.2243, disable=0.1599, tape=0.095407
- edge 0->3: top=channel mass=0.4317, active=0.366892, transform=0.6218, skip=0.2249, disable=0.1533, tape=0.126936
- edge 1->0: top=channel mass=0.3748, active=0.333468, transform=0.6099, skip=0.2370, disable=0.1531, tape=0.101100
- edge 1->1: top=channel mass=0.3921, active=0.342489, transform=0.6343, skip=0.2048, disable=0.1609, tape=0.121553
- edge 1->2: top=channel mass=0.4101, active=0.327520, transform=0.6457, skip=0.1907, disable=0.1635, tape=0.116347
- edge 1->3 EXPECTED: top=channel mass=0.3626, expected=channel mass=0.3626, active=0.414766, transform=0.6526, skip=0.1907, disable=0.1567, tape=0.150685
- edge 2->0: top=channel mass=0.3916, active=0.270368, transform=0.6239, skip=0.2495, disable=0.1266, tape=0.083181
- edge 2->1: top=channel mass=0.4482, active=0.275498, transform=0.6483, skip=0.2174, disable=0.1344, tape=0.098772
- edge 2->2: top=channel mass=0.4140, active=0.268081, transform=0.6639, skip=0.2008, disable=0.1352, tape=0.097311
- edge 2->3: top=channel mass=0.3668, active=0.342755, transform=0.6682, skip=0.2018, disable=0.1300, tape=0.126265
- edge 3->0: top=channel mass=0.5458, active=0.332346, transform=0.5663, skip=0.2566, disable=0.1771, tape=0.107522
- edge 3->1: top=channel mass=0.4286, active=0.341103, transform=0.5892, skip=0.2232, disable=0.1877, tape=0.126592
- edge 3->2: top=channel mass=0.4349, active=0.323768, transform=0.6023, skip=0.2076, disable=0.1901, tape=0.121120
- edge 3->3: top=channel mass=0.3668, active=0.415154, transform=0.6107, skip=0.2077, disable=0.1816, tape=0.159279

