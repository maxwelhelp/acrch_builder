# Task 08 — branching and variable output count

## Objective

Add soft branching/merge with fixed maximum slots and one final output state.

## Preconditions

Task 07 PASS.

## Allowed files

```text
arch_builder/model.py
arch_builder/executor.py
arch_builder/synthetic_tasks.py
arch_builder/credit.py
arch_builder/train_vertical_slice.py
arch_builder/reporting.py
commands/ branch proofs
docs/ and reports/agent_tasks/
```

## Required design

Fixed max slots, `slot_alive`, split/child gate, merge gate, per-layer/slot
output write, dynamic output tape, one final collector. Add tasks: useful
two-branch split/merge; harmful optional branch; local useful skip/global harmful
skip. Extend bounded credit to branches/outputs.

## Acceptance

```text
forced branch programs = 100%
required branches positive credit
harmful optional branch suppressed
active slots/split count nontrivial and bounded
one final output state
no all-alive/all-dead collapse
prior chain tasks pass
```

No hard branch count or hard role masks.

