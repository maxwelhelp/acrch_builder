# Branching and variable output proof

- status: `PASS`

## branch_split_merge

- status: `PASS`
- oracle_acc: `1.0`
- slot_alive_mean: `0.5069548785686493`
- slot_alive_count: `2.2200520833333335`
- split_none_mass: `0.32303351163864136`
- split_one_mass: `0.32922104001045227`
- split_two_mass: `0.347745418548584`
- collector_mass_mean: `0.24783561627070108`
- non_expected_active_mean(layer0): `0.27788453868457247`
- non_expected_top_split(layer0): `0.0`
- non_expected_top_skip(layer0): `0.0`
- non_expected_top_disable(layer0): `0.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'route'}, {'layer': 0, 'src': 0, 'tgt': 2, 'primitive': 'route'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'edge_gate'}, {'layer': 1, 'src': 2, 'tgt': 3, 'primitive': 'edge_gate'}, {'layer': 2, 'src': 1, 'tgt': 3, 'primitive': 'channel'}, {'layer': 2, 'src': 2, 'tgt': 3, 'primitive': 'channel'}]`

### Checks

- oracle_pass: `PASS`
- expected_actions_live: `PASS`
- slot_alive_bounded: `PASS`
- collector_live: `PASS`
- branch_metric_live: `PASS`
- split_two_child_active: `PASS`
- required_branch_credit: `PASS`

## branch_optional_branch

- status: `PASS`
- oracle_acc: `1.0`
- slot_alive_mean: `0.5043383042017618`
- slot_alive_count: `2.1171875`
- split_none_mass: `0.3232479691505432`
- split_one_mass: `0.3292089601357778`
- split_two_mass: `0.34754308064778644`
- collector_mass_mean: `0.24773168563842773`
- non_expected_active_mean(layer0): `0.26507995227972664`
- non_expected_top_split(layer0): `0.0`
- non_expected_top_skip(layer0): `0.0`
- non_expected_top_disable(layer0): `0.0`
- expected_actions: `[{'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'route'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'edge_gate'}, {'layer': 2, 'src': 1, 'tgt': 3, 'primitive': 'channel'}]`

### Checks

- oracle_pass: `PASS`
- expected_actions_live: `PASS`
- slot_alive_bounded: `PASS`
- collector_live: `PASS`
- branch_metric_live: `PASS`
- optional_branch_suppressed: `PASS`
- non_expected_split_suppressed: `PASS`

## branch_skip_tradeoff

- status: `PASS`
- oracle_acc: `1.0`
- slot_alive_mean: `0.4922626515229543`
- slot_alive_count: `1.7265625`
- split_none_mass: `0.3125225206216176`
- split_one_mass: `0.3332941035429637`
- split_two_mass: `0.3541833857695262`
- collector_mass_mean: `0.25316445529460907`
- non_expected_active_mean(layer0): `0.2482403149971595`
- non_expected_top_split(layer0): `0.0`
- non_expected_top_skip(layer0): `0.0`
- non_expected_top_disable(layer0): `0.0`
- expected_actions: `[{'layer': 0, 'src': 2, 'tgt': 3, 'primitive': 'skip'}, {'layer': 0, 'src': 0, 'tgt': 2, 'primitive': 'skip'}, {'layer': 0, 'src': 0, 'tgt': 1, 'primitive': 'route'}, {'layer': 1, 'src': 1, 'tgt': 3, 'primitive': 'channel'}]`

### Checks

- oracle_pass: `PASS`
- expected_actions_live: `PASS`
- slot_alive_bounded: `PASS`
- collector_live: `PASS`
- branch_metric_live: `PASS`
- skip_branch_live: `PASS`
- harmful_skip_suppressed: `PASS`

## Overall checks

- all_tasks_pass: `PASS`
- split_merge_pass: `PASS`
- optional_branch_pass: `PASS`
- skip_tradeoff_pass: `PASS`
