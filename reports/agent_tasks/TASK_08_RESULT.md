# TASK 08 RESULT — PASS

Task: branching and variable output count.

Proof:
- `python -m tools.project_probe.branching_output_proof`
- report: `reports/agent_inspector/BRANCHING_OUTPUT_PROOF.md`

What changed:
- added branch-aware trace metrics in `ActionMatrixLayer`
- added `slot_alive`, `split_count`, `child_gate`, `merge_gate`, `collector_mass`
- added branch-aware loss terms for split/merge/collector supervision
- added three branch configs:
  - `configs/tasks/branch_split_merge.yml`
  - `configs/tasks/branch_optional_branch.yml`
  - `configs/tasks/branch_skip_tradeoff.yml`
- added branch proof command:
  - `commands/probe_branching_output.sh`

Proof summary:
- all three branch subtests passed
- fixed max slots remained bounded
- final collector stayed single
- required branch actions carried positive credit
- optional branch suppression passed
- skip tradeoff passed after a slightly longer smoke and stronger branch regularization

Key proof values:
- `branch_split_merge`: `PASS`
  - `oracle_acc`: `1.0`
  - `slot_alive_mean`: `0.5069548785686493`
  - `slot_alive_count`: `2.2200520833333335`
  - `split_two_mass`: `0.347745418548584`
  - `collector_mass_mean`: `0.24783561627070108`
- `branch_optional_branch`: `PASS`
  - `oracle_acc`: `1.0`
  - `slot_alive_mean`: `0.5043383042017618`
  - `slot_alive_count`: `2.1171875`
  - `collector_mass_mean`: `0.24773168563842773`
- `branch_skip_tradeoff`: `PASS`
  - `oracle_acc`: `1.0`
  - `slot_alive_mean`: `0.4922626515229543`
  - `slot_alive_count`: `1.7265625`
  - `collector_mass_mean`: `0.25316445529460907`

Conclusion:
- Task 08 is complete.
- Next allowed task on the board is Task 09.
