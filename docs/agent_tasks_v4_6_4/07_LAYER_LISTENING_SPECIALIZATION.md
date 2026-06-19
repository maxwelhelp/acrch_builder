# Task 07 — layer listening and physical specialization

## Objective

Make sequential layers depend on previous state and specialize without hard
roles.

## Preconditions

Task 06 PASS; learned chain accepted.

## Allowed files

```text
arch_builder/model.py
arch_builder/credit.py
arch_builder/train_vertical_slice.py
arch_builder/reporting.py
commands/ layer proofs
docs/ and reports/agent_tasks/
```

## Required work

- Feed previous action/layer context through its separate projection; never read
  next-layer weights.
- Report listening score and adjacent action similarity.
- Physical hard-delete: identity bridge if shapes match, otherwise one shared
  fixed non-learned projection. Compare delete, internal skip, state ablation.
- Weak signal-gated anti-copy pressure only after dependency is alive.

## Acceptance

```text
layer_listen_score above fixed baseline
layer_dependency_delta > 0
external_delete_delta > 0 for useful layer
adjacent action distributions non-identical
both layers positive useful credit
chain remains readable/accepted
```

Forbidden: manual alternating roles, hard masks, learned delete adapter.

