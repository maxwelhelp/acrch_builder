# Task 15 — honest discovery runner and acceptance semantics

## Objective

Make discovery train without oracle actions and make every synthetic config pass
only on learned behavior. Current code already separates `oracle|discovery`, fixes
final serialization, and has `probe_discovery_contract.sh`; finish the closure.

## Required work

1. Keep `expected_actions` out of every discovery loss, simulator target, adaptive
   loss gate and branch target. It is evaluation/report data only.
2. Replace config thresholds based only on `oracle_acc` with task-specific learned
   `val_acc`, recovery, non-collapse and dependency thresholds.
3. Use paired fixed held-out batches and identical RNG for full/ablated evaluation.
4. Add a task-gradient/outcome credit signal that names no expected primitive.
   For compositional/parity-like tasks this must evaluate bounded joint/pair
   interventions, not only single candidates: an individual Layer0 `diff` has
   near-zero first-order correlation with
   `sign(mean((s0-s1)*(s2-s3)))`. Fit/apply pair credit only on later events.
5. Run a short 3-seed discovery test for `chain_diff_product`; do not use long
   training until the short curve rises reliably.

## Acceptance

- `probe_discovery_contract.sh` PASS;
- final report never crashes and records `supervision_mode=discovery`;
- changing `expected_actions` leaves loss and gradients bit-identical;
- three seeds beat 60% validation accuracy in the agreed short budget;
- a diagnostic report shows that useful joint Layer0 interventions receive
  higher held-out delta-CE than their individual zero-credit components;
- real write metrics remain finite/nonzero; expected recovery is report-only;
- no config can PASS solely because `oracle_acc=1`.

Forbidden: reintroducing oracle losses, lowering chance threshold, or marking a
50% run PASS.
