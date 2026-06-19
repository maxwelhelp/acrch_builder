# v4.6.4 implementation plan

Canonical plan: `docs/V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md`.

## Stage A: vertical slice

Implemented in this archive:

```text
PrimitiveMatrix5x5
HybridScanner
Top-K simulator
ActionMatrix executor
slot identity embedding
edge-specific known-program diagnostics
reporting script
```

## Stage B: after first reports

Check:

```text
val_acc above 50 on diff
expected_edge_recovery > 0
sim_disabled_delta meaningful
skip_mass not collapsed to 1
```

If not, next debug target is candidate path and executor path.

## Stage C: later

```text
budgeted hierarchical credit
Teacher/Audit/Deploy
structured matrix frontend
real audio
plug-in wrappers
TokenSlotAdapter
```


## v5 proof-slice correction

The v4 run proved slot identity and expected-edge primitive recovery, but accuracy stayed random. Diagnosis:

```text
edge_prog high + val_acc random =
  primitive choice was supervised by sim target,
  but the selected cell result was diluted before the classifier.
```

v5 fixes:

```text
positive non-zero edge_scale instead of tanh-zero start
small transform prior
direct cell output tape
metrics: expected_edge_choice_mass, edge_scale_mean, cell_output_gate_mean, cell_tape_weight_mean
```


## v6 output tape shape fix

v5 introduced a direct cell output tape but had a broadcasting bug:

```text
cell_tape.sum(dim=(1,2)) -> [B, D]
denom.squeeze(-1)        -> [B]
```

PyTorch aligns `[B]` with the last dimension `D`, so it crashed. v6 keeps denom as `[B, 1]`:

```text
output_tape_state = [B, D] / [B, 1]
```


## v7 proof-slice correction

The v6 report showed:

```text
edge_prog dropped to 0
val_acc stayed random
sim_delta became slightly positive
```

Diagnosis:

```text
1. Classifier LayerNorm removed the mean signal of TASK=diff.
   y = sign(mean(x0-x1)), so LayerNorm over features destroys this simple proof signal.

2. Sim target trained predicted_gain, but the choice path still did not have a
   direct known-program teacher signal for the expected edge during the proof slice.
```

v7 fixes:

```text
classifier is Linear(dim, classes), no LayerNorm
expected_edge_choice_loss trains choice mass on expected primitive at expected edge
reports expected_candidate_present and expected_edge_choice_mass
```

This choice loss is only for the synthetic known-program proof slice. It must be annealed or removed in later real-task stages.


## v8 raw-input and pair-bias correction

Diagnosis from v6/v7 discussion:

```text
TASK=diff label is sign(mean(x0-x1)).
Feature LayerNorm on input removes per-sample feature mean.
Therefore input_norm=layernorm can destroy the label signal before the program sees it.
```

v8 changes:

```text
input_norm defaults to none for proof-slice
--input-norm none/layernorm switch added
oracle_acc metric added to verify synthetic task itself
learnable source->target pair biases added to edge/write/phase/output gates
expected_candidate_present and expected_edge_choice_mass logged every epoch
```

Universal rule:

```text
normalization is a domain adapter choice, not part of the core ActionMatrix.
For synthetic proof tasks use raw/basic input. For real tasks add normalization only when honesty audits show it does not remove task signal.
```
