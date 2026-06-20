# TASK 07 RESULT — PASS

Task: layer listening and physical specialization on the learned chain.

- Previous-layer context is now injected through separate projections in `ActionMatrixLayer`.
- `layer_listen_score` is above the fixed baseline.
- `layer_action_similarity` is low, so adjacent layers are not copying the same action mix.
- `layer_dependency_delta` and `external_delete_delta` are both positive.
- Both layers carry positive expected-action credit.
- `state_ablation_delta` is reported and is slightly negative in this short smoke; the direct dependency and delete checks still pass.

| Check | Result |
|---|---|
| oracle_pass | PASS |
| chain_readable | PASS |
| layer_listen_alive | PASS |
| adjacent_actions_non_identical | PASS |
| dependency_delta_positive | PASS |
| external_delete_delta_positive | PASS |
| internal_skip_delta_positive | PASS |
| layer0_credit_positive | PASS |
| layer1_credit_positive | PASS |

Key values:

- `oracle_acc`: `1.0`
- `program_recovery_rate`: `0.3424479166666667`
- `layer_listen_score`: `0.6303771138191223`
- `layer_action_similarity`: `0.14653953909873962`
- `layer_dependency_delta`: `0.028124999999999956`
- `external_delete_delta`: `0.020312499999999956`
- `internal_skip_delta`: `0.01171875`
- `state_ablation_delta`: `-0.005468750000000022`

Layer credit:

- layer0 `recovery=0.373046875`, `choice_mass=0.3706047087907791`
- layer1 `recovery=0.328125`, `choice_mass=0.32525065541267395`

Program heuristic note:

- `program_verdicts` still show `primitive_collapse` in this short smoke, but the expected-action metrics and specialization/dependency checks are positive.

No long training was added beyond the short smoke already run.
