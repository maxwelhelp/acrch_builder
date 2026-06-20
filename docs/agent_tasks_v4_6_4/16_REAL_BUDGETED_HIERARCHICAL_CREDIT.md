# Task 16 — real budgeted hierarchical credit

## Objective

Implement the credit contract that Task04 claimed but did not implement.

## Required work

- On fixed held-out microbatches measure `delta_CE=CE(ablated)-CE(full)` for layer,
  cell, primitive, mode, output, memory, scanner source and simulator.
- Enforce and report a fixed per-event budget: 40% suspicious hierarchy, 30%
  high activity/uncertainty, 20% independent random, 10% system spot checks.
- Store level, target, mode, signed value, EMA, count, global age and staleness.
- Queue event N results; they may affect selection only from event N+1 onward.
- Remove current correctness-to-all-chosen-edges usage credit.

## Acceptance

Budget never exceeds limit; random budget is nonzero; paired fixed-seed results
are reproducible; a forced useful component is positive and a forced harmful
component negative; delayed-use assertion passes; credit survives save/load.

Forbidden: activity-only or same-batch credit, and full combinatorial ablation.
