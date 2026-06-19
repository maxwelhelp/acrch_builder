# Task 05 — simulator trained from real gain

## Objective

Train simulator targets from delayed real credit and prove useful choice impact.

## Preconditions

Task 04 PASS.

## Allowed files

```text
arch_builder/simulator.py
arch_builder/credit.py
arch_builder/model.py (sim components only)
arch_builder/train_vertical_slice.py
arch_builder/reporting.py
commands/ simulator proofs
docs/ and reports/agent_tasks/
```

## Required work

- Match each target to exact layer/cell/candidate/credit mode and age.
- Synthetic expected-action sim target remains only under explicit debug flag;
  default off in real-credit runs.
- Preserve normalized context/gain/sim-result/proposal choice components.
- Log component logit and gradient norms.
- Use Stage 4 deterministic independent ablations.

## Required metrics

```text
sim_pred_vs_real_corr
predicted_gain_choice_corr
gain_disabled_delta
sim_result_disabled_delta
sim_disabled_delta
choice_without_sim_delta
sim_vs_context logit/grad ratios
credit target age/mode/count
```

## Acceptance across repeated held-out events

```text
sim_pred_vs_real_corr > 0
choice_without_sim_delta > fixed nonzero threshold
full sim_disabled_delta reproducibly positive
at least one component CE delta positive
Task03 program acceptance preserved
```

Loss decrease or gradient alone is not success.

