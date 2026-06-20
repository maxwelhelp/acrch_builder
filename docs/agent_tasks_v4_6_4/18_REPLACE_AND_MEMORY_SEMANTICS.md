# Task 18 — replace topology and real memory semantics

## Objective

Implement the plan's meanings instead of runtime no-op aliases.

## Required work

- Remove `replace -> target` as an executable primitive behavior. Implement a
  `replace_distribution` that changes candidate/topology selection.
- Make memory_read/write/forget/recall/gate control actual memory state. Remove
  unconditional fixed-EMA memory update as the only memory mechanism.
- Add explicit ablations for replace selection, memory read and memory write.
- Add a synthetic task that cannot pass without write in one layer and read in
  the next.

## Acceptance

Forced oracles prove replace affects selection but is not a hidden output bypass;
memory-required task passes 3 seeds; memory ablation causes a positive accuracy
drop; gradients reach memory controller and primitive parameters.
