# Task 03 — learned sequential chain

## Objective

Prove Layer 1 learns from Layer 0 content. Pass an easier merge chain before
product.

## Preconditions

Task 02 PASS; forced oracle still fully passing.

## Allowed files

```text
arch_builder/synthetic_tasks.py
arch_builder/train_vertical_slice.py
arch_builder/reporting.py
commands/ proof commands
docs/ and reports/agent_tasks/
```

Do not change Stage 3 state semantics.

## Required work

Add `chain_diff_merge`:

```text
Layer0: 0->1 diff, 2->3 diff
Layer1: 1->3 merge
label: sign(mean(diff01 + diff23))
```

Add CLI/oracle/reports/commands. Forced raw/addressed accuracy must be 100%
before training. Train merge chain, then product chain, always `final_read=last`.
Use fixed-batch/RNG dependency tests: zero Layer0 state, ablate Layer1 output,
and prove Layer0 output tape is ignored.

## Acceptance across repeated seeds

```text
all actions present=1, choice_mass>=0.60, recovery>=0.70
Layer1 terminal action recovered
layer_dependency_delta > 0
layer_ablation_delta > 0
validation accuracy > random
no primitive collapse
active_cells < 16 per layer
```

Product starts only after merge PASS. Stop before real credit.

