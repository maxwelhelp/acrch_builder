# V4.6.4 plan conformance audit — 2026-06-20

Verdict: **NOT READY**. The previous final PASS is invalid.

## Confirmed fixes in this audit

- fixed `ModeCreditLedger.values` final-report crash with a serializable ledger state;
- added explicit `oracle|discovery` supervision mode;
- discovery loss construction does not read `expected_actions`;
- added oracle-free active/write floors and choice exploration floor;
- reports now state whether expected actions were used for training;
- evaluation receives the actual curriculum phase;
- internal previous-layer action/activity/write context remains available in deploy;
- previous-action context is live across layers and reaches scanner `before_proj`;
- clarified adaptive gates versus real runtime write metrics in epoch output;
- added `commands/probe_discovery_contract.sh`.

## PASS / PARTIAL / MISSING summary

| Plan area | Status | Reason |
|---|---|---|
| PrimitiveMatrix 5x5 | PASS | matrix, descriptors, topology and embeddings exist |
| HybridScanner sources | PARTIAL | sources exist; candidates are not deduplicated, anchor uses non-differentiable argmax, required health metrics incomplete |
| Simulator in choice | PARTIAL | path is live, but training target is oracle expected action rather than measured real gain |
| Budgeted hierarchical credit | MISSING | no fixed budget, component ablations, random allocation, delayed application or negative credit |
| ActionMatrix gates/modes | PARTIAL | basic gates exist; sign/group/rank/compose semantics incomplete |
| Replace topology semantics | MISSING | `replace` is executed as `return target` runtime primitive |
| Memory read/write | MISSING | memory update is unconditional fixed EMA, not controlled by selected memory actions |
| Sequential listening | PARTIAL | state and internal context are live; full credit/proof remains missing |
| Branching/variable outputs | MISSING | split/merge heads do not change topology/state; split executor is identity |
| Final output contract | PASS | last/mean/learned read exists; direct Layer0 output bypass fixed |
| Honest curriculum | MISSING | real hints/scaffolds and strict chance checks are not implemented |
| Standalone real data | MISSING | no honest matched multi-seed v4.6.3 comparison |
| Reusable plugin core | MISSING | demonstrations use separate MLP mechanisms |
| Token-slot replacement | MISSING | incremental cache, ragged masking and leakage checks are invalid/incomplete |
| Final reproducible audit | MISSING | canonical reproducer does not run all evidence and accepts stale reports |

## Runtime evidence

- `bash commands/validate.sh`: PASS
- `bash commands/probe_forced_program.sh`: PASS, direct bypass delta `0.0`
- `bash commands/probe_hybrid_scanner.sh`: PASS for its current limited contract
- `bash commands/probe_discovery_contract.sh`: PASS
- one-step discovery smoke: completes and writes both final reports
- inspector remains insufficient for release: simulator ablation deltas can be negative while summary is PASS

## Required continuation

Execute Tasks 15-24 in order. An agent may mark a task PASS only from newly
generated evidence satisfying that task's acceptance criteria.
