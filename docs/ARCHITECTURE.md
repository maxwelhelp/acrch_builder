# Architecture map

## Current vertical slice

```text
raw content_grid[B, slots, D]
  -> content state (slot address stays separate)
  -> slot address enters controller/scanner context only
  -> PrimitiveMatrix5x5
  -> HybridScanner
       local grid candidates
       semantic top-k candidates
       usage candidates
       random exploration
  -> LowRankSimulator
       preview candidates
       predicted_gain
  -> ActionMatrixController
       choice_logits uses context + predicted_gain + sim + proposal
  -> ActionExecutor
       transform / skip / disable
  -> normalized target write
       disable = zero write mass
       bounded soft-OR target gate
  -> output_state
  -> classifier
  -> reports
```

## Important current fixes

```text
slot identity is required
trainable sim tensors must not be detached
program recovery is edge-specific for expected source->target
sim target is edge-specific
AMP dtype mismatch is fixed inside ActionExecutor
```
