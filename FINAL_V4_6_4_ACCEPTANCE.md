# v4.6.4 final acceptance audit

Status: **FAIL / NOT READY**.

The previous PASS was invalidated by a source-and-runtime audit on 2026-06-20.
The direct Layer0-output bypass is fixed and its forced oracle is green, but that
single invariant does not prove the complete v4.6.4 plan.

Confirmed working:

- PrimitiveMatrix 5x5 and the four scanner proposal sources exist;
- simulator outputs participate in choice logits;
- sequential state reaches the next layer;
- `final_read=last` has no direct Layer0 output bypass;
- config loading and legacy CLI compatibility exist;
- structured audio feature extraction exists;
- `validate`, forced-program, hybrid-scanner, and discovery-contract probes pass.

Release blockers:

1. Task04 is not budgeted hierarchical ablation credit. Training correctness is
   assigned to every selected edge and primitive.
2. Simulator targets still come from `expected_actions`, not held-out real gain.
3. Runtime `replace`, memory, split/merge, and variable-output semantics do not
   implement the canonical plan.
4. Several acceptance proofs are stale, tautological, or accept chance-level
   learned accuracy. Task configs currently pass on formula `oracle_acc` alone.
5. Transformer and token-slot demonstrations use separate MLP mechanisms rather
   than the reusable PrimitiveMatrixScanner core.
6. Real-data results do not establish a same-split, multi-seed comparison with
   v4.6.3.

The concrete remediation sequence is Tasks 15-24 in
`docs/agent_tasks_v4_6_4/`. Missing evidence must remain FAIL; no threshold or
probe weakening is allowed.

Evidence: `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md`.

Final verdict: FAIL.
Release status: not ready.
