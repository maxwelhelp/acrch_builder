# ActionMatrix Program Report

- task: `product`
- batch_acc: `0.92578125`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_product_present': 1.0, 'action_L0_0_1_product_choice_mass': 0.9997340440750122, 'action_L0_0_1_product_recovery': 1.0, 'action_L0_0_1_product_active': 0.8627256155014038, 'action_L0_0_1_product_tape': 0.4520806074142456}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `13`
- active_cells: `6`
- active_edges_per_target: `1.5`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | product:0.31/a0.076 | *product:1.00/a0.863 | product:0.59/a0.279 | product:0.65/a0.376 |
| s1 | product:0.25/a0.038 | smooth:0.31/a0.031 | product:0.20/a0.014 | product:0.22/a0.022 |
| s2 | product:0.25/a0.010 | product:0.46/a0.138 | smooth:0.38/a0.000 | product:0.24/a0.006 |
| s3 | product:0.16/a0.012 | product:0.29/a0.137 | product:0.14/a0.003 | smooth:0.20/a0.000 |

### Cells
- edge 0->0: top=product mass=0.3084, active=0.076030, transform=0.0236, skip=0.9575, disable=0.0189, tape=0.001461
- edge 0->1 EXPECTED: top=product mass=0.9997, expected=product mass=0.9997, active=0.862726, transform=0.6680, skip=0.1582, disable=0.1738, tape=0.452081
- edge 0->2: top=product mass=0.5900, active=0.278625, transform=0.0494, skip=0.6907, disable=0.2599, tape=0.006718
- edge 0->3: top=product mass=0.6479, active=0.375597, transform=0.0450, skip=0.4564, disable=0.4986, tape=0.006556
- edge 1->0: top=product mass=0.2524, active=0.037711, transform=0.0924, skip=0.7085, disable=0.1992, tape=0.002758
- edge 1->1: top=smooth mass=0.3132, active=0.031477, transform=0.0856, skip=0.8234, disable=0.0910, tape=0.002454
- edge 1->2: top=product mass=0.1954, active=0.013939, transform=0.0171, skip=0.6916, disable=0.2913, tape=0.000121
- edge 1->3: top=product mass=0.2222, active=0.022125, transform=0.0124, skip=0.4580, disable=0.5297, tape=0.000193
- edge 2->0: top=product mass=0.2497, active=0.010002, transform=0.0170, skip=0.4681, disable=0.5149, tape=0.000090
- edge 2->1: top=product mass=0.4587, active=0.137503, transform=0.0982, skip=0.1621, disable=0.7397, tape=0.007723
- edge 2->2: top=smooth mass=0.3836, active=0.000140, transform=0.0003, skip=0.8456, disable=0.1541, tape=0.000000
- edge 2->3: top=product mass=0.2444, active=0.005922, transform=0.0014, skip=0.1644, disable=0.8342, tape=0.000002
- edge 3->0: top=product mass=0.1628, active=0.012161, transform=0.0138, skip=0.3047, disable=0.6814, tape=0.000073
- edge 3->1: top=product mass=0.2900, active=0.136514, transform=0.0764, skip=0.0817, disable=0.8420, tape=0.005294
- edge 3->2: top=product mass=0.1362, active=0.003078, transform=0.0018, skip=0.2280, disable=0.7702, tape=0.000001
- edge 3->3: top=smooth mass=0.2018, active=0.000142, transform=0.0003, skip=0.4713, disable=0.5284, tape=0.000000

