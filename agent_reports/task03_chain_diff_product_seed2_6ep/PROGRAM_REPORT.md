# ActionMatrix Program Report

- task: `chain_diff_product`
- batch_acc: `0.73828125`
- final_read: `last`
- state_norm: `none`
- slot_address_used_by_controller: `True`
- slot_address_used_by_executor: `False`
- final_read_weights: `[0.0, 1.0]`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`
- action_metrics: `{'action_L0_0_1_diff_present': 1.0, 'action_L0_0_1_diff_choice_mass': 0.9995142817497253, 'action_L0_0_1_diff_recovery': 1.0, 'action_L0_0_1_diff_active': 0.8468485474586487, 'action_L0_0_1_diff_tape': 0.02264651656150818, 'action_L0_2_3_diff_present': 1.0, 'action_L0_2_3_diff_choice_mass': 0.9996646046638489, 'action_L0_2_3_diff_recovery': 1.0, 'action_L0_2_3_diff_active': 0.8575511574745178, 'action_L0_2_3_diff_tape': 0.01862052083015442, 'action_L1_1_3_product_present': 1.0, 'action_L1_1_3_product_choice_mass': 0.995698869228363, 'action_L1_1_3_product_recovery': 1.0, 'action_L1_1_3_product_active': 0.8943297863006592, 'action_L1_1_3_product_tape': 0.28535810112953186}`

Flow:
```text
input -> Layer 0 -> Layer 1 -> final_read:last
```

Cell format: `primitive:choice_mass/active`. `*` marks expected edge.

## Layer 0

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `4`
- active_cells: `8`
- active_edges_per_target: `2.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'diff'}, {'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'diff'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | skip:0.26/a0.000 | *diff:1.00/a0.847 | channel:0.39/a0.146 | diff:1.00/a0.798 |
| s1 | channel:0.29/a0.019 | channel:0.25/a0.002 | channel:0.32/a0.009 | skip:0.39/a0.429 |
| s2 | channel:0.29/a0.271 | diff:1.00/a0.822 | channel:0.30/a0.000 | *diff:1.00/a0.858 |
| s3 | channel:0.23/a0.018 | channel:0.32/a0.403 | channel:0.30/a0.007 | skip:0.32/a0.002 |

### Cells
- edge 0->0: top=skip mass=0.2594, active=0.000443, transform=0.1664, skip=0.1766, disable=0.6570, tape=0.000007
- edge 0->1 EXPECTED: top=diff mass=0.9995, expected=diff mass=0.9995, active=0.846849, transform=0.9147, skip=0.0584, disable=0.0269, tape=0.022647
- edge 0->2: top=channel mass=0.3888, active=0.146292, transform=0.4104, skip=0.3669, disable=0.2226, tape=0.006846
- edge 0->3: top=diff mass=0.9997, active=0.797775, transform=0.8809, skip=0.0888, disable=0.0304, tape=0.019102
- edge 1->0: top=channel mass=0.2900, active=0.018910, transform=0.0351, skip=0.8944, disable=0.0705, tape=0.000172
- edge 1->1: top=channel mass=0.2466, active=0.002086, transform=0.1448, skip=0.3354, disable=0.5198, tape=0.000036
- edge 1->2: top=channel mass=0.3181, active=0.009472, transform=0.0324, skip=0.8355, disable=0.1322, tape=0.000119
- edge 1->3: top=skip mass=0.3870, active=0.429027, transform=0.2695, skip=0.6665, disable=0.0640, tape=0.012077
- edge 2->0: top=channel mass=0.2931, active=0.271392, transform=0.5261, skip=0.3531, disable=0.1208, tape=0.009225
- edge 2->1: top=diff mass=0.9996, active=0.822276, transform=0.9299, skip=0.0465, disable=0.0236, tape=0.019717
- edge 2->2: top=channel mass=0.2969, active=0.000201, transform=0.1163, skip=0.0961, disable=0.7876, tape=0.000004
- edge 2->3 EXPECTED: top=diff mass=0.9997, expected=diff mass=0.9997, active=0.857551, transform=0.9051, skip=0.0662, disable=0.0286, tape=0.018621
- edge 3->0: top=channel mass=0.2259, active=0.017857, transform=0.0377, skip=0.8923, disable=0.0700, tape=0.000174
- edge 3->1: top=channel mass=0.3197, active=0.403232, transform=0.3334, skip=0.5910, disable=0.0756, tape=0.013317
- edge 3->2: top=channel mass=0.2977, active=0.007007, transform=0.0318, skip=0.8207, disable=0.1475, tape=0.000075
- edge 3->3: top=skip mass=0.3159, active=0.001687, transform=0.1178, skip=0.3821, disable=0.5002, tape=0.000024

## Layer 1

- verdict: `sparse_or_partly_sparse_program`
- expected_top_cells: `1`
- active_cells: `4`
- active_edges_per_target: `1.0`
- expected_actions: `[{'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'product'}]`

| src\\tgt | t0 | t1 | t2 | t3 |
|---|---|---|---|---|
| s0 | ctx_matrix:0.39/a0.000 | ctx_matrix:0.29/a0.001 | ctx_matrix:0.28/a0.000 | output_mix:0.18/a0.090 |
| s1 | ctx_matrix:0.27/a0.034 | ctx_matrix:0.21/a0.009 | ctx_matrix:0.24/a0.042 | *product:1.00/a0.894 |
| s2 | ctx_matrix:0.36/a0.000 | memory_gate:0.30/a0.001 | ctx_matrix:0.33/a0.000 | output_mix:0.21/a0.059 |
| s3 | ctx_matrix:0.29/a0.000 | ctx_matrix:0.33/a0.103 | ctx_matrix:0.26/a0.000 | output_mix:0.28/a0.014 |

### Cells
- edge 0->0: top=ctx_matrix mass=0.3856, active=0.000005, transform=0.0020, skip=0.4385, disable=0.5595, tape=0.000000
- edge 0->1: top=ctx_matrix mass=0.2950, active=0.001307, transform=0.0151, skip=0.6777, disable=0.3072, tape=0.000008
- edge 0->2: top=ctx_matrix mass=0.2772, active=0.000182, transform=0.0264, skip=0.5889, disable=0.3847, tape=0.000004
- edge 0->3: top=output_mix mass=0.1816, active=0.090298, transform=0.0257, skip=0.6938, disable=0.2805, tape=0.001427
- edge 1->0: top=ctx_matrix mass=0.2748, active=0.034259, transform=0.0472, skip=0.3057, disable=0.6471, tape=0.001844
- edge 1->1: top=ctx_matrix mass=0.2126, active=0.009324, transform=0.0309, skip=0.4596, disable=0.5096, tape=0.000756
- edge 1->2: top=ctx_matrix mass=0.2390, active=0.041584, transform=0.1552, skip=0.3689, disable=0.4759, tape=0.008131
- edge 1->3 EXPECTED: top=product mass=0.9957, expected=product mass=0.9957, active=0.894330, transform=0.4039, skip=0.3589, disable=0.2373, tape=0.285358
- edge 2->0: top=ctx_matrix mass=0.3611, active=0.000089, transform=0.0051, skip=0.3954, disable=0.5995, tape=0.000000
- edge 2->1: top=memory_gate mass=0.2990, active=0.000992, transform=0.0117, skip=0.5991, disable=0.3891, tape=0.000006
- edge 2->2: top=ctx_matrix mass=0.3319, active=0.000000, transform=0.0036, skip=0.4541, disable=0.5423, tape=0.000000
- edge 2->3: top=output_mix mass=0.2058, active=0.058612, transform=0.0196, skip=0.6196, disable=0.3608, tape=0.000786
- edge 3->0: top=ctx_matrix mass=0.2911, active=0.000381, transform=0.0095, skip=0.5313, disable=0.4592, tape=0.000003
- edge 3->1: top=ctx_matrix mass=0.3260, active=0.102656, transform=0.0835, skip=0.7041, disable=0.2123, tape=0.007315
- edge 3->2: top=ctx_matrix mass=0.2559, active=0.000439, transform=0.0350, skip=0.6293, disable=0.3357, tape=0.000020
- edge 3->3: top=output_mix mass=0.2778, active=0.014488, transform=0.0105, skip=0.7033, disable=0.2862, tape=0.000265

