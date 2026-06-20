# ActionMatrix Program Report

- task: `branch_split_merge`
- batch_acc: `0.47265625`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'route'}, {'layer': 0, 'src': 0, 'tgt': 2, 'primitive': 'route'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'edge_gate'}, {'layer': 1, 'src': 2, 'tgt': 3, 'primitive': 'edge_gate'}, {'layer': 2, 'src': 1, 'tgt': 3, 'primitive': 'channel'}, {'layer': 2, 'src': 2, 'tgt': 3, 'primitive': 'channel'}]`
- action_metrics: `{'action_L0_0_1_route_present': 0.453125, 'action_L0_0_1_route_choice_mass': 0.4210206866264343, 'action_L0_0_1_route_recovery': 0.44921875, 'action_L0_0_1_route_active': 0.2624732255935669, 'action_L0_0_1_route_tape': 0.07261201739311218, 'action_L0_0_2_route_present': 0.3828125, 'action_L0_0_2_route_choice_mass': 0.363273561000824, 'action_L0_0_2_route_recovery': 0.3828125, 'action_L0_0_2_route_active': 0.28879332542419434, 'action_L0_0_2_route_tape': 0.0821910947561264, 'action_L1_1_3_edge_gate_present': 0.2578125, 'action_L1_1_3_edge_gate_choice_mass': 0.2528521716594696, 'action_L1_1_3_edge_gate_recovery': 0.2578125, 'action_L1_1_3_edge_gate_active': 0.3059931993484497, 'action_L1_1_3_edge_gate_tape': 0.07938594371080399, 'action_L1_2_3_edge_gate_present': 0.4921875, 'action_L1_2_3_edge_gate_choice_mass': 0.47838670015335083, 'action_L1_2_3_edge_gate_recovery': 0.4921875, 'action_L1_2_3_edge_gate_active': 0.32559797167778015, 'action_L1_2_3_edge_gate_tape': 0.08616408705711365, 'action_L2_1_3_channel_present': 0.35546875, 'action_L2_1_3_channel_choice_mass': 0.34757199883461, 'action_L2_1_3_channel_recovery': 0.35546875, 'action_L2_1_3_channel_active': 0.39539188146591187, 'action_L2_1_3_channel_tape': 0.14122259616851807, 'action_L2_2_3_channel_present': 0.3515625, 'action_L2_2_3_channel_choice_mass': 0.34204766154289246, 'action_L2_2_3_channel_recovery': 0.3515625, 'action_L2_2_3_channel_active': 0.39232033491134644, 'action_L2_2_3_channel_tape': 0.14005157351493835}`

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
- expected_active_mean: `0.2756332755088806`
- non_expected_active_mean: `0.27788453868457247`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'route'}, {'layer': 0, 'src': 0, 'tgt': 2, 'primitive': 'route'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | route:0.48/a0.336 | *route:0.42/a0.262 | *route:0.36/a0.289 | route:0.43/a0.241 |
| s1 | route:0.50/a0.270 | route:0.53/a0.302 | route:0.42/a0.276 | route:0.40/a0.235 |
| s2 | route:0.47/a0.268 | route:0.44/a0.251 | route:0.37/a0.307 | route:0.39/a0.224 |
| s3 | route:0.56/a0.291 | route:0.47/a0.279 | route:0.45/a0.299 | route:0.51/a0.310 |

### Cells
- edge 0->0: top=route mass=0.4759, active=0.335664, transform=0.6040, skip=0.1769, disable=0.2190, tape=0.126854
- edge 0->1 EXPECTED: top=route mass=0.4210, expected=route mass=0.4210, active=0.262473, transform=0.5586, skip=0.3021, disable=0.1394, tape=0.072612
- edge 0->2 EXPECTED: top=route mass=0.3633, expected=route mass=0.3633, active=0.288793, transform=0.5710, skip=0.2537, disable=0.1754, tape=0.082191
- edge 0->3: top=route mass=0.4288, active=0.241414, transform=0.6114, skip=0.2396, disable=0.1490, tape=0.079675
- edge 1->0: top=route mass=0.5029, active=0.270214, transform=0.5526, skip=0.2808, disable=0.1666, tape=0.073705
- edge 1->1: top=route mass=0.5337, active=0.302189, transform=0.5632, skip=0.2614, disable=0.1754, tape=0.097494
- edge 1->2: top=route mass=0.4183, active=0.276236, transform=0.5599, skip=0.2734, disable=0.1667, tape=0.073666
- edge 1->3: top=route mass=0.4024, active=0.235323, transform=0.5953, skip=0.2615, disable=0.1432, tape=0.071747
- edge 2->0: top=route mass=0.4701, active=0.268101, transform=0.5801, skip=0.2585, disable=0.1614, tape=0.087725
- edge 2->1: top=route mass=0.4368, active=0.250705, transform=0.5710, skip=0.3009, disable=0.1281, tape=0.074891
- edge 2->2: top=route mass=0.3678, active=0.307211, transform=0.6363, skip=0.1698, disable=0.1939, tape=0.124350
- edge 2->3: top=route mass=0.3946, active=0.223867, transform=0.6255, skip=0.2377, disable=0.1368, tape=0.081966
- edge 3->0: top=route mass=0.5588, active=0.290970, transform=0.4923, skip=0.3037, disable=0.2040, tape=0.079235
- edge 3->1: top=route mass=0.4699, active=0.278732, transform=0.4794, skip=0.3593, disable=0.1613, tape=0.070564
- edge 3->2: top=route mass=0.4458, active=0.299365, transform=0.4974, skip=0.2958, disable=0.2068, tape=0.079811
- edge 3->3: top=route mass=0.5129, active=0.310394, transform=0.5764, skip=0.2032, disable=0.2204, tape=0.122123

## Layer 1

- verdict: `primitive_collapse`
- expected_top_cells: `16`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.31579558551311493`
- non_expected_active_mean: `0.27353167108127047`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'edge_gate'}, {'layer': 1, 'src': 2, 'tgt': 3, 'primitive': 'edge_gate'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | edge_gate:0.63/a0.281 | edge_gate:0.52/a0.264 | edge_gate:0.45/a0.285 | edge_gate:0.39/a0.308 |
| s1 | edge_gate:0.41/a0.264 | edge_gate:0.41/a0.271 | edge_gate:0.35/a0.280 | *edge_gate:0.25/a0.306 |
| s2 | edge_gate:0.47/a0.281 | edge_gate:0.49/a0.277 | edge_gate:0.58/a0.308 | *edge_gate:0.48/a0.326 |
| s3 | edge_gate:0.43/a0.236 | edge_gate:0.61/a0.233 | edge_gate:0.51/a0.249 | edge_gate:0.62/a0.291 |

### Cells
- edge 0->0: top=edge_gate mass=0.6286, active=0.281127, transform=0.6252, skip=0.1866, disable=0.1883, tape=0.088460
- edge 0->1: top=edge_gate mass=0.5222, active=0.264499, transform=0.5229, skip=0.2381, disable=0.2391, tape=0.070013
- edge 0->2: top=edge_gate mass=0.4536, active=0.285207, transform=0.5216, skip=0.2617, disable=0.2167, tape=0.088779
- edge 0->3: top=edge_gate mass=0.3883, active=0.308416, transform=0.5058, skip=0.2887, disable=0.2055, tape=0.081042
- edge 1->0: top=edge_gate mass=0.4051, active=0.263893, transform=0.6296, skip=0.2157, disable=0.1548, tape=0.081455
- edge 1->1: top=edge_gate mass=0.4122, active=0.270983, transform=0.5469, skip=0.2680, disable=0.1851, tape=0.073179
- edge 1->2: top=edge_gate mass=0.3507, active=0.280442, transform=0.5316, skip=0.2978, disable=0.1705, tape=0.086967
- edge 1->3 EXPECTED: top=edge_gate mass=0.2529, expected=edge_gate mass=0.2529, active=0.305993, transform=0.5117, skip=0.3278, disable=0.1604, tape=0.079386
- edge 2->0: top=edge_gate mass=0.4709, active=0.281155, transform=0.6388, skip=0.2402, disable=0.1210, tape=0.089349
- edge 2->1: top=edge_gate mass=0.4940, active=0.277070, transform=0.5473, skip=0.3030, disable=0.1496, tape=0.076041
- edge 2->2: top=edge_gate mass=0.5849, active=0.307526, transform=0.5466, skip=0.3249, disable=0.1286, tape=0.098881
- edge 2->3 EXPECTED: top=edge_gate mass=0.4784, expected=edge_gate mass=0.4784, active=0.325598, transform=0.5169, skip=0.3586, disable=0.1246, tape=0.086164
- edge 3->0: top=edge_gate mass=0.4271, active=0.235672, transform=0.6083, skip=0.2519, disable=0.1397, tape=0.069111
- edge 3->1: top=edge_gate mass=0.6113, active=0.233038, transform=0.5149, skip=0.3153, disable=0.1698, tape=0.058324
- edge 3->2: top=edge_gate mass=0.5084, active=0.249061, transform=0.5077, skip=0.3411, disable=0.1513, tape=0.072942
- edge 3->3: top=edge_gate mass=0.6215, active=0.291354, transform=0.4949, skip=0.3679, disable=0.1372, tape=0.071617

## Layer 2

- verdict: `primitive_collapse`
- expected_top_cells: `15`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.39385610818862915`
- non_expected_active_mean: `0.3248645450387682`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `1.0`
- expected_actions: `[{'layer': 2, 'src': 1, 'tgt': 3, 'primitive': 'channel'}, {'layer': 2, 'src': 2, 'tgt': 3, 'primitive': 'channel'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | channel:0.60/a0.292 | channel:0.59/a0.292 | channel:0.55/a0.280 | channel:0.42/a0.366 |
| s1 | channel:0.38/a0.315 | channel:0.37/a0.322 | channel:0.44/a0.309 | *channel:0.35/a0.395 |
| s2 | disable:0.37/a0.315 | channel:0.45/a0.319 | channel:0.40/a0.312 | *channel:0.34/a0.392 |
| s3 | channel:0.54/a0.337 | channel:0.42/a0.343 | channel:0.44/a0.328 | channel:0.36/a0.419 |

### Cells
- edge 0->0: top=channel mass=0.5975, active=0.291595, transform=0.5731, skip=0.2764, disable=0.1505, tape=0.082914
- edge 0->1: top=channel mass=0.5856, active=0.291971, transform=0.5840, skip=0.2519, disable=0.1641, tape=0.093592
- edge 0->2: top=channel mass=0.5454, active=0.280416, transform=0.6152, skip=0.2253, disable=0.1595, tape=0.094586
- edge 0->3: top=channel mass=0.4234, active=0.365621, transform=0.6164, skip=0.2286, disable=0.1550, tape=0.124552
- edge 1->0: top=channel mass=0.3848, active=0.314997, transform=0.6047, skip=0.2395, disable=0.1558, tape=0.094206
- edge 1->1: top=channel mass=0.3705, active=0.321690, transform=0.6158, skip=0.2161, disable=0.1681, tape=0.108995
- edge 1->2: top=channel mass=0.4379, active=0.308915, transform=0.6439, skip=0.1926, disable=0.1635, tape=0.109281
- edge 1->3 EXPECTED: top=channel mass=0.3476, expected=channel mass=0.3476, active=0.395392, transform=0.6460, skip=0.1951, disable=0.1589, tape=0.141223
- edge 2->0: top=disable mass=0.3704, active=0.315391, transform=0.6143, skip=0.2553, disable=0.1304, tape=0.094105
- edge 2->1: top=channel mass=0.4480, active=0.319081, transform=0.6263, skip=0.2318, disable=0.1420, tape=0.107813
- edge 2->2: top=channel mass=0.4020, active=0.312046, transform=0.6571, skip=0.2058, disable=0.1371, tape=0.110815
- edge 2->3 EXPECTED: top=channel mass=0.3420, expected=channel mass=0.3420, active=0.392320, transform=0.6574, skip=0.2092, disable=0.1335, tape=0.140052
- edge 3->0: top=channel mass=0.5418, active=0.336752, transform=0.5668, skip=0.2556, disable=0.1776, tape=0.109226
- edge 3->1: top=channel mass=0.4172, active=0.343310, transform=0.5766, skip=0.2308, disable=0.1926, tape=0.123608
- edge 3->2: top=channel mass=0.4386, active=0.327664, transform=0.6062, skip=0.2066, disable=0.1872, tape=0.123840
- edge 3->3: top=channel mass=0.3615, active=0.418655, transform=0.6089, skip=0.2095, disable=0.1816, tape=0.160081

