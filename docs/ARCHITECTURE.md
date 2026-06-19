# Architecture map

## Core loop

```text
input state_grid[B, slots, D]
  -> PrimitiveMatrix5x5
  -> HybridScanner
       local grid window
       semantic top-k
       usage/credit top-k
       random exploration
  -> LowRankSimulator
       preview top-K candidates
       predicted_gain
  -> ActionMatrixController
       choice_logits = LN(context) + LN(predicted_gain) + LN(sim) + LN(proposal)
  -> ActionExecutor
       transform / skip / disable mode
       replace_distribution selects candidate mixture
  -> state_next + output_tape
  -> classifier + task loss
  -> bounded diagnostics / credit report
```

## Important design fixes

### HybridScanner, not pure WindowScanner

Pure grid windows can trap candidates in the initial seed layout. HybridScanner uses:

```text
candidate_set =
  local_grid_3x3
  + semantic_topk_by_embedding
  + usage_topk_by_credit
  + random_explore
```

### Simulator must affect choice

The simulator is not allowed to be decorative. `predicted_gain` enters `choice_logits` after normalization, and reports include:

```text
sim_disabled_delta
choice_without_sim_delta
predicted_gain_choice_corr
sim_pred_vs_real_corr
```

### Credit is bounded

No combinatorial full ablation. Credit/report checks are budgeted and hierarchical.

### Replace / skip / disable are separated

```text
cell_mode = softmax([transform, skip, disable])
replace_distribution = candidate selection, not another no-op
```

### Plug-in mode comes later

Attention replacement requires a TokenSlotAdapter and incremental slot update. It is intentionally delayed until standalone proof passes.
