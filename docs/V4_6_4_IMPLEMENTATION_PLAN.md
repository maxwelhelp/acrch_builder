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


## v9 readable ActionMatrix report

v8 proved the synthetic proof-slice can reach high accuracy, but the report did not answer the most important question clearly:

```text
what program was actually assembled?
```

v9 adds:

```text
PROGRAM_REPORT.md
program_epoch_XXX.json
per-cell top primitive
per-cell choice mass
per-cell active mass
per-cell transform/skip/disable mass
expected primitive mass on every edge
expected edge marked in the table
```

This is required before claiming the system built a logical program.


## v10 anti-collapse proof-slice

Problem after v8:

```text
high accuracy but expected_any_recovery≈1.0
=> the proof-slice likely selected diff in most cells, not a sparse program.
```

v10 adds small synthetic-only structure losses:

```text
expected_choice_loss:
  choose expected primitive on expected edge

non_expected_primitive_loss:
  penalize expected primitive mass on non-expected edges

expected_active_loss:
  keep expected edge active

non_expected_active_loss:
  reduce active mass outside expected edge

non_expected_tape_loss:
  reduce output-tape write outside expected edge

non_expected_transform_loss:
  reduce transform mode outside expected edge
```

It also adds a readable program verdict:

```text
program_verdict = primitive_collapse / sparse_or_partly_sparse_program / not_recovered
expected_top_cells
active_cells
```

These losses are only for known-program synthetic proof. They are not the final unsupervised real-task training rule.


## v12_multilayer_fixed_20260619_2136

Packaging fix: previous v11 archive did not actually expose `two_diff` in CLI. This archive is verified by `python -m arch_builder.train_vertical_slice --help`.

Tasks:

```text
TASK=two_diff LAYERS=1:
  expected layer0: 0->1 diff and 2->3 diff

TASK=chain_diff_product LAYERS=2:
  expected layer0: 0->1 diff and 2->3 diff
  expected layer1: 1->3 product
```

`PROGRAM_REPORT.md` now reports every layer separately.
