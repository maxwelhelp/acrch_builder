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
