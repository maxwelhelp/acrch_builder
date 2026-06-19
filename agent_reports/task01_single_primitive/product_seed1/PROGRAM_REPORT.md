# ActionMatrix Program Report

- task: `product`
- batch_acc: `0.9375`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_product_present': 1.0, 'action_L0_0_1_product_choice_mass': 0.9995290040969849, 'action_L0_0_1_product_recovery': 1.0, 'action_L0_0_1_product_active': 0.8977384567260742, 'action_L0_0_1_product_tape': 0.4471818506717682}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `12`
- active_cells: `5`
- active_edges_per_target: `1.25`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | low_rank:0.30/a0.018 | *product:1.00/a0.898 | product:0.83/a0.452 | product:0.62/a0.391 |
| s1 | product:0.26/a0.012 | product:0.19/a0.020 | product:0.24/a0.029 | product:0.26/a0.016 |
| s2 | product:0.14/a0.003 | product:0.26/a0.189 | low_rank:0.18/a0.000 | edge_gate:0.16/a0.005 |
| s3 | product:0.20/a0.007 | product:0.41/a0.263 | product:0.20/a0.021 | low_rank:0.18/a0.000 |

### Cells
- edge 0->0: top=low_rank mass=0.2952, active=0.018147, transform=0.2198, skip=0.5180, disable=0.2622, tape=0.001883
- edge 0->1 EXPECTED: top=product mass=0.9995, expected=product mass=0.9995, active=0.897738, transform=0.6752, skip=0.1792, disable=0.1457, tape=0.447182
- edge 0->2: top=product mass=0.8329, active=0.451806, transform=0.0568, skip=0.8177, disable=0.1255, tape=0.004010
- edge 0->3: top=product mass=0.6242, active=0.391044, transform=0.0588, skip=0.8742, disable=0.0671, tape=0.004215
- edge 1->0: top=product mass=0.2573, active=0.012008, transform=0.1025, skip=0.3540, disable=0.5436, tape=0.000684
- edge 1->1: top=product mass=0.1860, active=0.020431, transform=0.0401, skip=0.0435, disable=0.9164, tape=0.000832
- edge 1->2: top=product mass=0.2367, active=0.029165, transform=0.0079, skip=0.4309, disable=0.5612, tape=0.000059
- edge 1->3: top=product mass=0.2599, active=0.015918, transform=0.0089, skip=0.5635, disable=0.4276, tape=0.000053
- edge 2->0: top=product mass=0.1414, active=0.002786, transform=0.0117, skip=0.3336, disable=0.6547, tape=0.000011
- edge 2->1: top=product mass=0.2575, active=0.188948, transform=0.0222, skip=0.1027, disable=0.8751, tape=0.002755
- edge 2->2: top=low_rank mass=0.1849, active=0.000081, transform=0.0003, skip=0.1819, disable=0.8178, tape=0.000000
- edge 2->3: top=edge_gate mass=0.1626, active=0.005052, transform=0.0011, skip=0.5069, disable=0.4920, tape=0.000001
- edge 3->0: top=product mass=0.2009, active=0.007386, transform=0.0126, skip=0.6433, disable=0.3441, tape=0.000028
- edge 3->1: top=product mass=0.4065, active=0.262578, transform=0.0275, skip=0.2904, disable=0.6821, tape=0.004503
- edge 3->2: top=product mass=0.1966, active=0.020761, transform=0.0008, skip=0.6826, disable=0.3166, tape=0.000003
- edge 3->3: top=low_rank mass=0.1824, active=0.000114, transform=0.0004, skip=0.5640, disable=0.4357, tape=0.000000

