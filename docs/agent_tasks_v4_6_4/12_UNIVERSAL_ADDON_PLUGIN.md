# Task 12 — universal add-on plug-in modes

## Objective

Wrap accepted core after Attention, then before Attention. No replacement yet.

## Preconditions

Task 11 PASS.

## Allowed files

New adapters/wrappers, small reference Transformer, plug-in reports/commands,
docs/status. Core semantics unchanged.

## Required modes and comparisons

```text
after-attention add-on
before-attention preparation
with mechanism vs identity/random/frozen vs attention-only
```

Report params/FLOPs/memory/latency, output norm, credit, readability.

## Acceptance

```text
after: mechanism > identity/random/frozen reproducibly
before: downstream task/attention improves matched baseline
shape/device/dtype tests pass
gradient/credit closure retained
mechanism not zero/residual-only
```

No causal replacement or TokenSlotAdapter.

