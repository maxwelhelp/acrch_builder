# Task 01 — single primitive learned baselines

## Objective

Prove `diff`, `merge`, and `product` learn independently with corrected
transport. Establish baselines; do not solve multi-cell collapse here.

## Allowed files

```text
arch_builder/train_vertical_slice.py
arch_builder/synthetic_tasks.py
arch_builder/reporting.py
commands/ non-pushing baseline commands
docs/ and reports/agent_tasks/
```

Do not change model/executor/scanner/simulator/loss formulas. If forced oracle
passes but execution cannot train, stop with diagnostics.

## Required work

- One fixed non-pushing run matrix for `diff`, `merge`, `product`, one layer.
- Same dim/batch/steps, `slots=4`, `input_norm=none`, `state_norm=none`,
  `final_read=last`, `top_k=25`.
- At least three seeds per task (or explicitly report resource limitation).
- Comparison JSON/Markdown with every seed and mean; never choose best only.

## Acceptance per primitive

```text
oracle accuracy = 1.0
expected_candidate_present = 1.0
expected_edge_choice_mass >= 0.70
expected_edge_recovery >= 0.80
validation accuracy reproducibly above random
all report/regression contracts pass
```

Record collapse metrics, but Task 02 owns collapse. Stop after these baselines.

