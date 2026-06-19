# Task 13 — TokenSlotAdapter attention replacement

## Objective

Implement genuine causal Attention/block replacement with incremental slots.

## Preconditions

Task 12 PASS.

## Allowed files

New token-slot adapters, replacement wrapper, causal/incremental tests and
reports. Accepted standalone core semantics stay unchanged.

## Required modules

```text
TokenToSlotAdapter
PrimitiveMatrixScannerCore wrapper
SlotToTokenAdapter
incremental slot cache/update
```

Requirements: no future leakage; causal routing/masks; prefix/chunk support;
`slots_t = update(slots_{t-1}, token_t)`; no full-prefix recompute per token.

## Required tests

```text
future-token perturbation
full vs incremental equivalence
cache reset/isolation
variable lengths/padding
mixed precision/device/dtype
token-slot reconstruction
decode cost scaling
```

## Acceptance

```text
causal leakage exact pass
incremental/full match tolerance
no O(n^2) full recomputation
replacement near baseline or explicit cost advantage
mechanism > identity/random/frozen
readability and credit/simulator diagnostics retained
```

