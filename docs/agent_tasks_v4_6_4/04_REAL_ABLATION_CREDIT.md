# Task 04 — budgeted delayed real ablation credit

## Objective

Replace placeholder credit with bounded held-out hierarchical ablations.

## Preconditions

Task 03 PASS.

## Allowed files

```text
arch_builder/credit.py
arch_builder/model.py (ablation hooks only)
arch_builder/train_vertical_slice.py
arch_builder/reporting.py
commands/ credit proofs
docs/ and reports/agent_tasks/
```

## Required design

Levels: layer, cell, primitive, mode, output, scanner source, simulation, memory
spot checks. Measure on held-out microbatches:

```text
delta_CE = CE(ablated) - CE(full)
positive = useful; negative = suspicious
```

Fixed reported budget split:

```text
40% suspicious hierarchy
30% high activity/uncertainty
20% random cells/primitives independent of layer result
10% output/memory/scanner/simulator
```

Store EMA, count, age, staleness, level, and mode. Credit from batch/event N may
affect only a later event. Never ablate everything.

## Acceptance

```text
budget never exceeded; random budget nonzero
fixed-seed credit finite/reproducible
known useful cells positive on average
irrelevant/random cells lower on average
age/staleness correct
delayed-application proof present
Task03 recovery survives credit penalty
```

Forbidden: same-batch credit, full combinatorial ablation, activity-only credit.

