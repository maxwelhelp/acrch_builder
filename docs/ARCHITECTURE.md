# Architecture map

## Current vertical slice

```text
state_grid[B, slots, D]
  -> slot identity embedding
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
