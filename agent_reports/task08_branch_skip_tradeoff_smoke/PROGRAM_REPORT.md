# ActionMatrix Program Report

- task: `branch_skip_tradeoff`
- batch_acc: `0.53125`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'skip'}, {'layer': 0, 'src': 0, 'tgt': 2, 'primitive': 'skip'}, {'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'route'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'channel'}]`
- action_metrics: `{'action_L0_0_1_route_present': 0.5078125, 'action_L0_0_1_route_choice_mass': 0.4147595763206482, 'action_L0_0_1_route_recovery': 0.421875, 'action_L0_0_1_route_active': 0.3292391896247864, 'action_L0_0_1_route_tape': 0.1101137325167656, 'action_L0_0_2_skip_present': 0.3359375, 'action_L0_0_2_skip_choice_mass': 0.31060364842414856, 'action_L0_0_2_skip_recovery': 0.328125, 'action_L0_0_2_skip_active': 0.2689533829689026, 'action_L0_0_2_skip_tape': 0.07611744850873947, 'action_L0_2_3_skip_present': 0.2109375, 'action_L0_2_3_skip_choice_mass': 0.19337648153305054, 'action_L0_2_3_skip_recovery': 0.203125, 'action_L0_2_3_skip_active': 0.2705492377281189, 'action_L0_2_3_skip_tape': 0.08688239753246307, 'action_L1_1_3_channel_present': 0.421875, 'action_L1_1_3_channel_choice_mass': 0.4143647849559784, 'action_L1_1_3_channel_recovery': 0.421875, 'action_L1_1_3_channel_active': 0.5368654727935791, 'action_L1_1_3_channel_tape': 0.15288445353507996}`

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
- expected_active_mean: `0.2895806034406026`
- non_expected_active_mean: `0.2482403149971595`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'skip'}, {'layer': 0, 'src': 0, 'tgt': 2, 'primitive': 'skip'}, {'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'route'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | route:0.44/a0.220 | *route:0.41/a0.329 | *skip:0.31/a0.269 | route:0.38/a0.268 |
| s1 | route:0.45/a0.259 | route:0.43/a0.253 | route:0.37/a0.238 | route:0.37/a0.257 |
| s2 | route:0.39/a0.269 | route:0.36/a0.306 | route:0.45/a0.212 | *route:0.35/a0.271 |
| s3 | route:0.39/a0.236 | route:0.38/a0.287 | route:0.40/a0.231 | route:0.36/a0.192 |

### Cells
- edge 0->0: top=route mass=0.4408, active=0.219874, transform=0.5885, skip=0.1437, disable=0.2678, tape=0.068636
- edge 0->1 EXPECTED: top=route mass=0.4148, expected=route mass=0.4148, active=0.329239, transform=0.5965, skip=0.2551, disable=0.1485, tape=0.110114
- edge 0->2 EXPECTED: top=skip mass=0.3106, expected=skip mass=0.3106, active=0.268953, transform=0.5595, skip=0.2566, disable=0.1838, tape=0.076117
- edge 0->3: top=route mass=0.3779, active=0.267885, transform=0.5815, skip=0.2740, disable=0.1444, tape=0.087934
- edge 1->0: top=route mass=0.4481, active=0.259029, transform=0.5197, skip=0.2569, disable=0.2234, tape=0.070854
- edge 1->1: top=route mass=0.4340, active=0.253085, transform=0.6064, skip=0.1706, disable=0.2230, tape=0.088383
- edge 1->2: top=route mass=0.3670, active=0.237532, transform=0.5319, skip=0.2561, disable=0.2120, tape=0.063383
- edge 1->3: top=route mass=0.3687, active=0.256526, transform=0.5585, skip=0.2805, disable=0.1610, tape=0.082379
- edge 2->0: top=route mass=0.3926, active=0.269233, transform=0.5461, skip=0.3128, disable=0.1411, tape=0.074553
- edge 2->1: top=route mass=0.3574, active=0.305773, transform=0.5824, skip=0.3072, disable=0.1104, tape=0.093936
- edge 2->2: top=route mass=0.4483, active=0.212043, transform=0.6222, skip=0.1756, disable=0.2021, tape=0.065574
- edge 2->3 EXPECTED: top=route mass=0.3502, expected=skip mass=0.1934, active=0.270549, transform=0.5935, skip=0.3033, disable=0.1032, tape=0.086882
- edge 3->0: top=route mass=0.3898, active=0.235775, transform=0.5781, skip=0.2722, disable=0.1498, tape=0.070307
- edge 3->1: top=route mass=0.3832, active=0.287458, transform=0.6222, skip=0.2721, disable=0.1057, tape=0.098945
- edge 3->2: top=route mass=0.4002, active=0.230879, transform=0.6091, skip=0.2553, disable=0.1356, tape=0.069738
- edge 3->3: top=route mass=0.3576, active=0.192032, transform=0.6963, skip=0.1560, disable=0.1477, tape=0.079381

## Layer 1

- verdict: `primitive_collapse`
- expected_top_cells: `15`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.5368654727935791`
- non_expected_active_mean: `0.3875255028406779`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `0.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'channel'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | channel:0.40/a0.408 | channel:0.36/a0.388 | channel:0.30/a0.324 | smooth:0.31/a0.470 |
| s1 | channel:0.48/a0.444 | channel:0.39/a0.469 | channel:0.38/a0.385 | *channel:0.41/a0.537 |
| s2 | channel:0.58/a0.338 | channel:0.49/a0.347 | channel:0.57/a0.315 | channel:0.48/a0.432 |
| s3 | channel:0.59/a0.354 | channel:0.58/a0.363 | channel:0.46/a0.298 | channel:0.58/a0.478 |

### Cells
- edge 0->0: top=channel mass=0.4011, active=0.408083, transform=0.5328, skip=0.2955, disable=0.1717, tape=0.092731
- edge 0->1: top=channel mass=0.3598, active=0.388434, transform=0.5282, skip=0.2938, disable=0.1780, tape=0.093640
- edge 0->2: top=channel mass=0.2993, active=0.323524, transform=0.5231, skip=0.2707, disable=0.2062, tape=0.083396
- edge 0->3: top=smooth mass=0.3073, active=0.470219, transform=0.5352, skip=0.2791, disable=0.1857, tape=0.117162
- edge 1->0: top=channel mass=0.4814, active=0.443975, transform=0.6137, skip=0.2435, disable=0.1428, tape=0.120466
- edge 1->1: top=channel mass=0.3925, active=0.468964, transform=0.6033, skip=0.2445, disable=0.1523, tape=0.125805
- edge 1->2: top=channel mass=0.3832, active=0.385378, transform=0.5995, skip=0.2251, disable=0.1754, tape=0.112710
- edge 1->3 EXPECTED: top=channel mass=0.4144, expected=channel mass=0.4144, active=0.536865, transform=0.6133, skip=0.2299, disable=0.1568, tape=0.152884
- edge 2->0: top=channel mass=0.5758, active=0.338287, transform=0.6311, skip=0.2629, disable=0.1061, tape=0.091031
- edge 2->1: top=channel mass=0.4865, active=0.347431, transform=0.6243, skip=0.2640, disable=0.1117, tape=0.095701
- edge 2->2: top=channel mass=0.5714, active=0.314643, transform=0.6175, skip=0.2493, disable=0.1332, tape=0.088429
- edge 2->3: top=channel mass=0.4798, active=0.431634, transform=0.6335, skip=0.2504, disable=0.1161, tape=0.122540
- edge 3->0: top=channel mass=0.5851, active=0.353510, transform=0.6411, skip=0.2451, disable=0.1138, tape=0.089538
- edge 3->1: top=channel mass=0.5752, active=0.363092, transform=0.6357, skip=0.2444, disable=0.1199, tape=0.093678
- edge 3->2: top=channel mass=0.4633, active=0.298095, transform=0.6322, skip=0.2273, disable=0.1405, tape=0.082924
- edge 3->3: top=channel mass=0.5812, active=0.477614, transform=0.6397, skip=0.2331, disable=0.1272, tape=0.122010

## Layer 2

- verdict: `no_expected_actions_for_this_layer`
- expected_top_cells: `0`
- active_cells: `16`
- active_edges_per_target: `4.0`
- expected_active_mean: `0.0`
- non_expected_active_mean: `0.2576452884823084`
- non_expected_top_split: `0.0`
- non_expected_top_skip: `0.0`
- non_expected_top_disable: `3.0`
- expected_actions: `[]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.35/a0.233 | ctx_matrix:0.32/a0.251 | ctx_matrix:0.40/a0.245 | ctx_matrix:0.34/a0.277 |
| s1 | disable:0.35/a0.214 | diff:0.34/a0.233 | diff:0.36/a0.226 | diff:0.35/a0.259 |
| s2 | disable:0.52/a0.227 | ctx_matrix:0.35/a0.247 | ctx_matrix:0.33/a0.244 | diff:0.28/a0.273 |
| s3 | disable:0.50/a0.277 | ctx_matrix:0.56/a0.300 | ctx_matrix:0.49/a0.287 | ctx_matrix:0.42/a0.328 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.3545, active=0.232870, transform=0.4686, skip=0.3611, disable=0.1703, tape=0.061130
- edge 0->1: top=ctx_matrix mass=0.3226, active=0.251229, transform=0.4503, skip=0.3854, disable=0.1643, tape=0.062970
- edge 0->2: top=ctx_matrix mass=0.3956, active=0.245153, transform=0.4492, skip=0.3611, disable=0.1898, tape=0.060540
- edge 0->3: top=ctx_matrix mass=0.3448, active=0.277445, transform=0.4286, skip=0.3904, disable=0.1809, tape=0.058716
- edge 1->0: top=disable mass=0.3543, active=0.213559, transform=0.5203, skip=0.3002, disable=0.1796, tape=0.062879
- edge 1->1: top=diff mass=0.3429, active=0.233236, transform=0.5025, skip=0.3237, disable=0.1739, tape=0.066163
- edge 1->2: top=diff mass=0.3595, active=0.225997, transform=0.4988, skip=0.3012, disable=0.2001, tape=0.062690
- edge 1->3: top=diff mass=0.3458, active=0.258929, transform=0.4796, skip=0.3283, disable=0.1921, tape=0.062268
- edge 2->0: top=disable mass=0.5239, active=0.227222, transform=0.5156, skip=0.3133, disable=0.1711, tape=0.064000
- edge 2->1: top=ctx_matrix mass=0.3508, active=0.246940, transform=0.4975, skip=0.3367, disable=0.1658, tape=0.066765
- edge 2->2: top=ctx_matrix mass=0.3285, active=0.244173, transform=0.4942, skip=0.3154, disable=0.1904, tape=0.064524
- edge 2->3: top=diff mass=0.2820, active=0.273408, transform=0.4750, skip=0.3420, disable=0.1830, tape=0.062386
- edge 3->0: top=disable mass=0.5014, active=0.276963, transform=0.5941, skip=0.2458, disable=0.1602, tape=0.091619
- edge 3->1: top=ctx_matrix mass=0.5617, active=0.300482, transform=0.5775, skip=0.2663, disable=0.1562, tape=0.096253
- edge 3->2: top=ctx_matrix mass=0.4857, active=0.286786, transform=0.5726, skip=0.2480, disable=0.1794, tape=0.089601
- edge 3->3: top=ctx_matrix mass=0.4164, active=0.327931, transform=0.5547, skip=0.2723, disable=0.1730, tape=0.089412

