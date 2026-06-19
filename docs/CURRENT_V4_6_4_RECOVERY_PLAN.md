# Current v4.6.4 recovery plan

Remaining implementation work is split into sequential agent tasks in
`docs/agent_tasks_v4_6_4/00_INDEX_AND_PROTOCOL.md`.

Scope: close the current synthetic vertical slice before adding real credit,
branching, honest-input curriculum, audio, or Transformer plug-in modes.

Canonical architecture contract:
`docs/V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md`.

This document is intentionally limited to the next empirical milestone. Each
stage must be implemented and tested separately. Do not combine the stages into
one large change.

## Current verified diagnosis

The latest `chain_diff_product` report is not blocked by primitive availability:

- `diff`, `merge`, and `product` exist in `PrimitiveMatrix5x5`;
- `ActionExecutor` implements all three;
- `expected_candidate_present = 1.0`;
- ordinary gradients reach controller/scanner/simulator/executor/classifier;
- program recovery and task accuracy remain near random.

Four blockers were identified before loss-weight tuning:

1. Choice/recovery losses were detached (fixed in Stage 1).
   `ActionMatrixLayer` stores `trace["choice"] = choice.detach()`, while
   `proof_slice_structure_losses` and the choice-based generic losses consume
   that detached tensor. Verified at runtime: `expected_choice_loss`,
   `non_expected_primitive_loss`, `primitive_usage_diversity`,
   `cell_choice_diversity`, and `layer_action_diversity` have
   `requires_grad=False`. Stage 1 added a separate live loss tensor; the current
   inspector now reports `recovery_loss_connected=true`.

2. The declared sequential program is not faithfully represented by the state
   transition. Messages are summed and divided by the number of slots before a
   residual update. With one forced `0->1 diff` cell and four slots, target 1
   receives approximately
   `x1 + (x0 - x1) / 4 = 0.25*x0 + 0.75*x1`, followed by LayerNorm. With several
   active incoming cells it receives an even broader mixture. The next layer
   therefore does not receive `diff01` in slot 1.

3. Slot address and slot value are mixed. `slot_embed` is added directly to
   operands, so executor primitives operate on `x + address`, not raw task
   values. Address information should guide the controller without silently
   changing the mathematical program.

4. Inspector integration was stale/incomplete (fixed in Stage 0). The restored
   probe now uses the current API and tests the actual autograd connection from
   recovery loss into the choice path.

## Stage 0: restore a truthful probe

Status: completed on 2026-06-19. Baseline result:

```text
gradient_closed = true
credit_closed = false
recovery_loss_connected = false
detached_enabled_losses =
  expected_choice_loss
  non_expected_primitive_loss
  primitive_usage_diversity
  cell_choice_diversity
  layer_action_diversity
```

Commands restored:

```text
bash commands/inspect_with_probe.sh
bash commands/build_focused_report.sh
```

Goal: obtain a reliable pre-change baseline and make future before/after checks
reproducible.

Files in scope:

- `commands/inspect_with_probe.sh`
- `commands/build_focused_report.sh`
- `tools/project_probe/probe_learning_loop.py`
- `tools/agent_inspector/runtime_probe.py` or remove the obsolete duplicate
- inspector configuration and generated summaries

Required behavior:

- one command runs the current project API on a small CPU batch;
- reports per-loss `requires_grad` and gradient norm into the intended module;
- distinguishes aggregate task gradient from recovery-loss gradient;
- `recovery_loss_connected` is based on autograd, not `isfinite(loss)`;
- does not report `choice_without_sim_delta` as an alias unless it performs a
  distinct ablation;
- exits non-zero on stale imports/signatures.

Acceptance:

- documented inspector command exists and completes;
- baseline explicitly reports the five detached choice-based losses;
- generated `SUMMARY.md`, gradient graph, and credit graph agree.

Stop after this stage and inspect the report.

## Stage 1: reconnect choice/recovery losses

Status: completed on 2026-06-19. Inspector after the change:

```text
gradient_closed = true
credit_closed = true
recovery_loss_connected = true
detached_enabled_losses = []
```

Short CPU `diff` smoke (3 epochs, 20 steps/epoch) confirmed that the
previously-dead choice supervision now moves the intended path:

```text
expected_choice_mass: 0.355 -> 0.893 -> 0.980
expected_edge_recovery: 0.516 -> 1.000 -> 1.000
train_loss: 9.570 -> 2.586 -> 1.118
last val_acc: 0.5625
```

This is only a recovery-path proof. It still shows all-cell primitive collapse
and negative simulator ablation, so it is not architecture acceptance.

Goal: make the already-computed recovery supervision actually train the choice
path. Do not change loss weights or add schedules in this stage.

Files in scope:

- `arch_builder/model.py`
- `arch_builder/train_vertical_slice.py`
- probe assertions from Stage 0

Required change:

- keep detached trace fields for reports;
- add a live choice tensor specifically for differentiable losses;
- make all choice-based losses consume the live tensor;
- keep signal gates detached intentionally so they control weights without
  creating second-order feedback.

Acceptance before a training run:

- each enabled loss documents whether it is intentionally live or report-only;
- expected-choice-only backward produces non-zero gradients in the actual
  choice path (`context_logits`, selected scanner proposal path, simulator
  choice components);
- reporter tensors remain detached;
- inspector says `recovery_loss_connected=true`.

Then run only the one-layer `diff` proof. Acceptance:

- expected choice mass and expected-edge recovery rise from their initial
  values;
- validation accuracy exceeds random;
- no NaN/Inf;
- report and probe agree.

If this fails, diagnose gradients/logit competition. Do not proceed to state
semantics or tune all regularizers at once.

## Stage 2: prove the task/program contract with a forced oracle

Status: completed on 2026-06-19. The forced oracle runs without training and
uses `ActionExecutor` plus the extracted real `apply_state_update` operation.

```text
state_update_behavior_preserved = true (max_abs_error = 0)
raw diff/merge/product accuracy = 1.0
all action invariants = true

address label flip rate:
  diff    = 0.0469
  merge   = 0.0254
  product = 0.1758

chain_diff_product final accuracy:
  raw-content forced path = 0.5801
  addressed forced path   = 0.5215
```

The first raw-content divergence is already at Layer 0 target-state transport:
primitive execution error is zero, but target-slot MAE after the current state
update is about `1.66`/`1.64` for the two diff writes. Layer 1 consequently
receives corrupted operands and its product output MAE is about `1.42`. This is
direct evidence for Stage 3; it is not a scanner, primitive availability, or
executor-math failure.

Artifacts:

```text
reports/agent_inspector/forced_program_oracle.json
reports/agent_inspector/FORCED_PROGRAM_ORACLE.md
```

Goal: test whether the exact declared expected actions, when forced through the
real executor and state transition, reproduce each task label. Separate raw
primitive correctness from address contamination and sequential state
semantics. Learning must not be used to hide an invalid synthetic contract.

Files in scope:

- `arch_builder/synthetic_tasks.py`
- `arch_builder/executor.py`
- `arch_builder/model.py` only for a behavior-preserving extraction of the
  existing state-update operation, if needed to avoid duplicating it
- `tools/project_probe/forced_program_oracle.py`
- `commands/probe_forced_program.sh`

Required invariants:

- every expected primitive exists in `name_to_id`;
- every expected primitive is executable;
- layer/source/target indices are valid;
- the oracle uses `ActionExecutor`, not a second handwritten primitive library;
- forced multi-layer programs call the exact same state-update operation as the
  learned model;
- the extraction of that operation, if added, must be behavior-preserving and
  covered by an equality regression test.

The oracle must produce three separate audits.

### 2A. Raw primitive audit

Run `ActionExecutor` on raw task operands, without `slot_embed`, controller,
scanner, gates, residual state update, or classifier.

Report for `diff`, `merge`, and `product`:

```text
primitive_available
primitive_executed
max_abs_error_vs_task_formula
label_accuracy_from_primitive_output
```

This audit answers only whether primitive ids and executor math match the task.

### 2B. Address contamination audit

Execute the same forced primitives twice:

```text
raw operands
raw operands + current slot_embed
```

Report:

```text
raw_label_accuracy
addressed_label_accuracy
address_output_delta
address_label_flip_rate
```

Do not require the current addressed path to pass. Its purpose is to measure
whether address information changes the mathematical value.

### 2C. Sequential state audit

Force only the declared expected cells. Use transform mode, unit write mass,
unit edge scale, and no unrelated cells. Feed layer output state into the next
real layer state-update operation. Use the final expected action output as the
oracle readout; for multiple terminal actions use the task-declared merge/read
rule explicitly.

Run both:

```text
raw-content path
current model-addressed path
```

Report per layer/action:

```text
input operand error
primitive output error
target slot error after state update
final label accuracy
```

This must expose whether failure begins in primitive execution, address mixing,
or target-state transport.

Acceptance for completing the diagnostic stage:

- all task/action invariants pass;
- raw primitive audit reaches 100% label accuracy (within numerical tolerance)
  for `diff`, `merge`, and `product`;
- address contamination is measured rather than hidden;
- the sequential audit uses the real current state-update code and identifies
  the first divergent layer/action;
- current multi-layer failure is recorded as expected evidence for Stage 3, not
  treated as an oracle implementation failure;
- JSON and concise Markdown reports are produced without training or optimizer
  steps.

Stop and save the oracle report.

## Stage 3: correct content/address and target-write semantics

Status: completed on 2026-06-19.

Forced oracle after the change:

```text
state_update_contract_pass = true
raw_primitive_pass = true
invariants_pass = true
address_content_separated = true
controller_address_choice_delta = 0.02386
sequential_contract_pass = true

executor address output delta = 0
executor address label flip rate = 0
diff/merge/product/two_diff forced accuracy = 1.0
chain_diff_product raw/addressed forced accuracy = 1.0 / 1.0
all expected target-slot errors after unit writes = 0
```

Stage 0/1 inspector remains fully closed. A short learned `diff` smoke stayed
finite and recovered the expected edge (`choice_mass=0.970`,
`expected_edge_recovery=1.0`). It still collapsed to `diff` in all 16 cells and
the simulator ablation remained negative; those are explicitly outside Stage 3.

Goal: make a selected ActionMatrix cell write a usable operation result into the
target slot for the next layer.

This stage requires a small design change, not a coefficient change.

### Stage 3 implementation lock

Do not choose these semantics during coding; use the contract below.

#### 3A. Separate content from address

Keep `slot_embed` as the learned slot-address parameter for compatibility, but
do not add it to `state`.

```text
content_state = input_norm(x)
slot_address = slot_embed
```

`ActionExecutor` and memory receive only `content_state`. Address enters only
the controller/scanner context. Preserve the current context width by using
addressed controller views for the first two context fields:

```text
control_src = content_src + source_slot_address
control_tgt = content_tgt + target_slot_address

controller_context = concat(
  control_src,
  control_tgt,
  content_src - content_tgt,
  content_src * content_tgt,
  content_memory,
)
```

This keeps slot identity visible without changing primitive operands. Do not
increase context dimension or redesign scanner heads in this stage.

Add a diagnostic address ablation that compares controller choice with real vs
zero slot address in eval mode. Required result:

```text
executor_address_output_delta = 0
executor_address_label_flip_rate = 0
controller_address_choice_delta > 0
```

The first two prove content honesty; the third proves address information was
not accidentally removed.

#### 3B. Define disable/write/value semantics

The current unnormalized `cell_out` lets disable probability shrink a value
while still writing. Replace that with a conditional enabled value and explicit
write mass:

```text
enabled_mode_mass = mode_transform + mode_skip

cell_value = (
  mode_transform * transformed
  + mode_skip * source_content
) / clamp_min(enabled_mode_mass, eps)

cell_write_mass = active * enabled_mode_mass
scaled_cell_value = edge_scale * cell_value
```

Therefore:

```text
transform -> write transformed value
skip      -> write source content unchanged (apart from edge_scale)
disable   -> zero write mass
```

Do not change gate priors, loss weights, Gumbel/top-k behavior, or edge-scale
range in this stage.

#### 3C. Normalize incoming values and use bounded soft-OR write gate

For each target:

```text
mass_sum[target] = sum_source(cell_write_mass[source,target])

write_value[target] =
  sum_source(cell_write_mass * scaled_cell_value)
  / clamp_min(mass_sum[target], eps)

target_write_gate[target] =
  1 - product_source(1 - clamp(cell_write_mass, 0, 1))

next_state[target] =
  (1 - target_write_gate[target]) * old_content_state[target]
  + target_write_gate[target] * write_value[target]
```

Properties required by tests:

```text
no active write       -> preserve old target exactly
one forced unit write -> replace target with cell value exactly
disable-only cell     -> preserve target exactly
multiple writes       -> normalized soft mixture, never divide by slot count
```

Change `apply_state_update` to accept value and write-mass grids explicitly.
The Stage 2 oracle must call this same method with unit mass for expected cells.

#### 3D. Make state normalization explicit

Add `state_norm=none|layernorm` through layer, model, CLI, command environment,
and reports. Default `none` for synthetic proof tasks. `layernorm` remains an
optional domain adapter experiment, not part of the proof contract.

Do not add a new normalization type in this stage.

#### 3E. Required trace fields

Keep existing trace compatibility and add detached report tensors:

```text
cell_write_mass
target_write_gate
write_value
state_norm_mode
slot_address_used_by_controller
slot_address_used_by_executor = false
```

Do not add new losses for these fields yet.

Design constraints:

- slot address embeddings influence controller/scanner context but do not alter
  executor operands;
- a transform write to target slot can expose the transformed value to the next
  layer;
- residual preservation is gated: preserve the old target when there is no
  write, interpolate/replace when a write is active;
- normalize multiple incoming writes by write mass instead of always dividing
  by the number of slots;
- synthetic proof mode must not apply feature LayerNorm that destroys the
  mathematical signal; make state normalization explicit/configurable;
- retain soft writes during training; do not introduce hard top-k edge masks.

Locked state rule for the proof slice:

```text
write_value[target] = sum(write_mass[cell] * scaled_cell_value[cell])
                      / clamp_min(sum(write_mass[cell]), eps)
target_write_gate = 1 - product(1 - clamp(write_mass[cell], 0, 1))
next_state[target] = (1 - target_write_gate) * old_state[target]
                     + target_write_gate * write_value[target]
```

Do not substitute another gate/interpolation formula based only on training
accuracy.

Acceptance:

- Stage 2 forced oracle now reaches 100% on the multi-layer contract;
- raw and addressed forced paths both reach 100% on `chain_diff_product` when
  address is routed only into controller context;
- every expected target-slot error after a forced unit write is at numerical
  zero with `state_norm=none`;
- controller address choice delta is positive;
- executor address output delta and label flip rate are zero;
- no hidden Layer 0 output enters `final_read=last`;
- Stage 0/1 inspector remains fully closed;
- `diff`, `merge`, `product`, and `two_diff` oracle regressions remain at 100%;
- a short learned one-layer `diff` smoke remains finite and keeps expected
  recovery alive. Primitive collapse is recorded but not solved in Stage 3.

Stop after oracle and one-layer regression tests.

## Stage 4: make diagnostics action-specific

Status: completed on 2026-06-19.

Validated on a short two-layer `chain_diff_product` report-contract run:

```text
all 15 per-action fields present in:
  metrics.csv
  final_report.json
  program_epoch_001.json
  PROGRAM_REPORT.md

scanner source mass:
  grid     = 0.24245
  semantic = 0.15365
  usage    = 0.58610
  random   = 0.01780
  sum      = 1.00000

independent simulator diagnostics:
  gain_disabled_delta       = +0.00282
  sim_result_disabled_delta = +0.00089
  sim_disabled_delta        = +0.00346
  choice_without_sim_delta  = 0.03133

weighted loss fields = 14
loss_accounting_error = 4.43e-7
```

Training and the project probe now call the same canonical objective function.
Stage 0/1 inspector remains closed and the Stage 3 forced oracle remains fully
passing. These smoke values prove report correctness only; they are not a model
quality claim.

Goal: remove averages that hide which sequential action is dead.

Stage 4 is instrumentation only. Do not change architecture, losses, loss
weights, adaptive gates, scanner candidate policy, or training schedule.

### Stage 4 implementation lock

#### 4A. One canonical training-objective function

Extract the current objective construction from the training loop into one
function in `train_vertical_slice.py` that returns:

```text
total_loss
raw_losses
signals
adaptive_gates_and_effective_lambdas
weighted_loss_contributions
```

Both the real training loop and `tools/project_probe/probe_learning_loop.py`
must call this same function. The probe must not maintain a copied loss formula.
This prevents the inspector API from becoming stale again.

Per epoch, average every returned scalar over actual optimizer steps. Add:

```text
loss_accounting_error = abs(
  mean(total_loss) - sum(mean(weighted_loss_contributions))
)
```

Required `loss_accounting_error <= 1e-5` (allow a small float accumulation
tolerance).

Do not backpropagate through metric accumulation.

#### 4B. Stable per-action metric keys

For every declared expected action, emit these flat keys:

```text
action_L{layer}_{src}_{tgt}_{primitive}_present
action_L{layer}_{src}_{tgt}_{primitive}_choice_mass
action_L{layer}_{src}_{tgt}_{primitive}_recovery
action_L{layer}_{src}_{tgt}_{primitive}_active
action_L{layer}_{src}_{tgt}_{primitive}_tape
```

Primitive names must be sanitized to `[a-z0-9_]+`; current names already fit.
Metrics are batch means and report-only detached values.

Write them to:

```text
metrics.csv
final_report.json
program_epoch_XXX.json
PROGRAM_REPORT.md
```

Keep aggregate metrics for continuity, but never use them to infer which action
failed when per-action metrics exist.

#### 4C. Real scanner source usage

`HybridScanner` must return source ids alongside candidate ids:

```text
0 = grid
1 = semantic
2 = usage
3 = random
```

After proposal top-k, gather matching source ids. After controller choice,
measure actual probability mass assigned to each source position:

```text
grid_candidate_usage
semantic_candidate_usage
usage_candidate_usage
random_candidate_usage
```

These four masses must be computed from final `choice`, averaged across cells
and batch, and sum to approximately `1.0`. Candidate duplication across sources
is intentional: mass belongs to the source position that delivered the
candidate.

Remove the hardcoded zero usage metrics from `HybridScanner`. Keep
`semantic_grid_mismatch` there because it is a proposal-set property.

Do not add `global_rescue_rate` yet; it requires real gain/credit and belongs to
the later scanner/credit milestone.

#### 4D. Simulator ablations must be distinct and deterministic

Add independent choice-logit ablations:

```text
gain disabled       -> zero predicted_gain component only
sim-result disabled -> zero sim_result component only
full sim disabled   -> zero both components
```

Use one fixed eval batch and restore identical CPU/CUDA RNG state before every
forward so random scanner candidates are identical. Compare primitive-id
distributions, not raw candidate positions.

Definitions:

```text
gain_disabled_delta = CE(no_gain) - CE(full)
sim_result_disabled_delta = CE(no_sim_result) - CE(full)
sim_disabled_delta = CE(no_gain_and_no_sim_result) - CE(full)

choice_without_sim_delta = mean absolute difference between
  full primitive-choice distribution and full-sim-disabled
  primitive-choice distribution
```

`choice_without_sim_delta` is therefore not a CE alias. Positive
`choice_without_sim_delta` proves influence on choice; positive CE deltas prove
that the corresponding influence is useful.

Keep the old `disable_sim=True` API as a compatibility alias for full disable,
but implement new explicit flags/mode underneath it.

#### 4E. Required loss fields

Report raw values using stable names:

```text
ce_loss
sim_loss
expected_choice_loss
expected_active_loss
non_expected_primitive_loss
non_expected_active_loss
non_expected_tape_loss
non_expected_transform_loss
primitive_usage_diversity
cell_choice_diversity
active_budget
tape_budget
layer_action_diversity
min_transform_loss
```

Report every effective lambda returned by the signal controller and every
weighted contribution with prefix:

```text
weighted_<raw_loss_name>
```

The last epoch's averaged accounting must also be copied into
`final_report.json`; do not report only the last minibatch.

Add for every expected action:

```text
action_L{layer}_{src}_{tgt}_{primitive}_present
action_L{layer}_{src}_{tgt}_{primitive}_choice_mass
action_L{layer}_{src}_{tgt}_{primitive}_recovery
action_L{layer}_{src}_{tgt}_{primitive}_active
action_L{layer}_{src}_{tgt}_{primitive}_tape
```

Also report:

- every raw loss;
- every effective lambda;
- every weighted contribution to total loss;
- gradient norm per loss family;
- actual scanner source mass after top-k/choice;
- separate gain-head, simulated-result, and full-simulator ablations.

Acceptance:

- metrics are present in `metrics.csv`, `final_report.json`, per-epoch program
  JSON, and `PROGRAM_REPORT.md` where appropriate;
- `choice_without_sim_delta` and `sim_disabled_delta` are distinct experiments;
- scanner source usage is measured rather than hardcoded to zero.

Additional acceptance:

- on `chain_diff_product`, all three expected actions have separate non-empty
  metrics in all required artifacts;
- scanner source masses are finite, non-negative, and sum to `1 ± 1e-4`;
- the three CE ablations and the choice-distribution ablation are separately
  computed fields, not copied aliases;
- training and project probe import the same objective function;
- Stage 0/1 inspector remains closed;
- forced Stage 3 oracle remains fully passing;
- a short smoke produces finite loss accounting and does not need to improve
  accuracy. Stage 4 measures collapse/simulator weakness; it does not fix them.

Stop after report-contract validation.

## Stage 5: climb the synthetic ladder

Run one test at a time and do not change architecture while comparing runs.

1. `diff`, one layer: correct edge/primitive is recovered and sparse enough to
   read.
2. `merge`, one layer: proves a second primitive independently.
3. `product`, one layer: proves multiplication independently.
4. `two_diff`, one layer: proves two useful cells without all-cell diff collapse.
5. Add `chain_diff_merge`, two layers: easier sequential dependency proof.
6. `chain_diff_product`, two layers: only after merge chain works.

For every run require:

- forced oracle passed first;
- expected actions individually recovered;
- `active_cells < all cells` after recovery is alive;
- primitive collapse verdict is false;
- `layer_dependency_delta > 0` for sequential tasks;
- `final_read=last`;
- simulator ablation has a reproducible positive effect before calling it useful.

If recovery is low, do not enable sparsity/collapse pressure. If recovery is
high but structure is dense, then tune the already signal-gated structural
losses. Never gate by epoch number.

## Milestone boundary

The current vertical slice is complete only after Stage 5 passes. Only then plan
the next bounded milestone:

- replace synthetic expected-action targets with budgeted real ablation credit;
- train simulator targets from real gain;
- validate HybridScanner semantic rescue with a real candidate cutoff;
- add credit EMA/age/staleness and random independent credit budget;
- then continue the canonical stages for branching, honesty curriculum, real
  frontends, and plug-in wrappers.

Do not start those items while synthetic sequential execution is invalid.

## Reusable lessons from the old `test2` project

Reuse principles, not its large implementation:

- tensors used for differentiable losses must stay live; detach only report
  copies (the old v4.3 head fixed exactly this class of bug);
- residual paths need an explicit dominance metric and a bounded/gated write;
- structure pressure should be soft, signal-driven, and introduced only after
  the useful path is alive;
- downstream-consumer/dependency metrics are more informative than raw activity.
