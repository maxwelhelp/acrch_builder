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
