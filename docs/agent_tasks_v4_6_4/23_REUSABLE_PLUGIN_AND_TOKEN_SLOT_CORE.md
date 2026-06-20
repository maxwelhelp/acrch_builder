# Task 23 — real reusable plugin and token-slot core

## Objective

Use the actual PrimitiveMatrixScanner core in add-on and attention-replacement
experiments, and replace tautological tests with behavioral tests.

## Required work

- Remove the independent `LocalMechanism`/gated-MLP substitutes.
- Expose one reusable core adapter used by standalone, Transformer add-on and
  token-slot replacement paths.
- Fix ragged masking so inactive samples retain their previous slot state.
- Implement persistent `init_cache`, `step` and `reset`; do not reprocess the full
  prefix in `forward_incremental`.
- Compare prefix states/logits under changed future suffixes; test cache isolation,
  chunked==full and mixed precision. Fix the causal baseline to pool a token that
  can see the sequence.

## Acceptance

Type/parameter identity proves the real core is used; no-future leakage is a real
prefix equality test; unequal-length batched results equal individual truncation;
incremental cache matches full execution and does less repeated work; no hardcoded
`True` acceptance fields remain.
