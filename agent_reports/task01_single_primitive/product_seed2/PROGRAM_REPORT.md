# ActionMatrix Program Report

- task: `product`
- batch_acc: `0.88671875`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_product_present': 1.0, 'action_L0_0_1_product_choice_mass': 0.9998430013656616, 'action_L0_0_1_product_recovery': 1.0, 'action_L0_0_1_product_active': 0.8508261442184448, 'action_L0_0_1_product_tape': 0.43080079555511475}`

Flow:
```text
input -> Layer 0 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `primitive_collapse`
- expected_top_cells: `9`
- active_cells: `5`
- active_edges_per_target: `1.25`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | split:0.18/a0.042 | *product:1.00/a0.851 | product:0.41/a0.223 | product:0.67/a0.218 |
| s1 | product:0.20/a0.036 | split:0.31/a0.045 | product:0.17/a0.021 | output_mix:0.19/a0.019 |
| s2 | product:0.18/a0.032 | product:0.85/a0.327 | split:0.20/a0.000 | split:0.17/a0.015 |
| s3 | product:0.17/a0.024 | product:0.70/a0.292 | split:0.18/a0.014 | split:0.25/a0.000 |

### Cells
- edge 0->0: top=split mass=0.1837, active=0.042272, transform=0.0513, skip=0.1448, disable=0.8039, tape=0.003301
- edge 0->1 EXPECTED: top=product mass=0.9998, expected=product mass=0.9998, active=0.850826, transform=0.6784, skip=0.2098, disable=0.1118, tape=0.430801
- edge 0->2: top=product mass=0.4056, active=0.223377, transform=0.1033, skip=0.3953, disable=0.5014, tape=0.012332
- edge 0->3: top=product mass=0.6747, active=0.217700, transform=0.0993, skip=0.5091, disable=0.3916, tape=0.010730
- edge 1->0: top=product mass=0.2046, active=0.036450, transform=0.0777, skip=0.4871, disable=0.4352, tape=0.002115
- edge 1->1: top=split mass=0.3064, active=0.044745, transform=0.0769, skip=0.3032, disable=0.6199, tape=0.003673
- edge 1->2: top=product mass=0.1733, active=0.021211, transform=0.0192, skip=0.4804, disable=0.5004, tape=0.000147
- edge 1->3: top=output_mix mass=0.1936, active=0.019312, transform=0.0195, skip=0.5954, disable=0.3851, tape=0.000169
- edge 2->0: top=product mass=0.1837, active=0.031655, transform=0.0160, skip=0.8101, disable=0.1739, tape=0.000208
- edge 2->1: top=product mass=0.8451, active=0.327264, transform=0.0878, skip=0.8235, disable=0.0887, tape=0.012001
- edge 2->2: top=split mass=0.1971, active=0.000469, transform=0.0007, skip=0.4692, disable=0.5301, tape=0.000000
- edge 2->3: top=split mass=0.1733, active=0.015423, transform=0.0026, skip=0.8570, disable=0.1403, tape=0.000011
- edge 3->0: top=product mass=0.1723, active=0.023573, transform=0.0170, skip=0.6920, disable=0.2910, tape=0.000159
- edge 3->1: top=product mass=0.7030, active=0.292373, transform=0.0996, skip=0.7439, disable=0.1566, tape=0.011162
- edge 3->2: top=split mass=0.1816, active=0.014203, transform=0.0033, skip=0.6582, disable=0.3385, tape=0.000010
- edge 3->3: top=split mass=0.2527, active=0.000371, transform=0.0007, skip=0.4487, disable=0.5506, tape=0.000001

