# v4.6.4 remaining agent task pack

Canonical contract: `docs/V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md`.

Completed foundation: truthful inspector, live recovery losses, forced oracle,
content/address target-write semantics, and action-specific diagnostics. Do not
reimplement them. Read:

```text
docs/CURRENT_V4_6_4_RECOVERY_PLAN.md
reports/agent_inspector/SUMMARY.md
reports/agent_inspector/FORCED_PROGRAM_ORACLE.md
```

## Mandatory order

```text
01 single primitive baselines
02 multi-cell anti-collapse
03 learned sequential chain
04 real ablation credit
05 simulator from real gain
06 HybridScanner proof
07 layer specialization
08 branching/output count
09 Teacher/Audit/Deploy
10 structured frontend/audio
11 standalone real-data acceptance
12 add-on plug-ins
13 TokenSlot replacement
14 final audit
```

Task N starts only after task N-1 writes a `PASS` handoff.

## Task files

1. `01_SINGLE_PRIMITIVE_BASELINES.md`
2. `02_MULTI_CELL_ANTI_COLLAPSE.md`
3. `03_LEARNED_SEQUENTIAL_CHAIN.md`
4. `04_REAL_ABLATION_CREDIT.md`
5. `05_SIMULATOR_REAL_GAIN.md`
6. `06_HYBRID_SCANNER_PROOF.md`
7. `07_LAYER_LISTENING_SPECIALIZATION.md`
8. `08_BRANCHING_VARIABLE_OUTPUT.md`
9. `09_HONEST_CURRICULUM_MODE_CREDIT.md`
10. `10_STRUCTURED_FRONTEND_AUDIO.md`
11. `11_STANDALONE_REAL_DATA_ACCEPTANCE.md`
12. `12_UNIVERSAL_ADDON_PLUGIN.md`
13. `13_TOKEN_SLOT_ATTENTION_REPLACEMENT.md`
14. `14_FINAL_ACCEPTANCE_AUDIT.md`

Shared progress board: `STATUS_BOARD.md`.

## Rules for every agent

1. Read this index, assigned task, canonical plan, current recovery plan, and
   current inspector/oracle reports.
2. Inspect `git status --short`; preserve unrelated/user work. Never reset,
   clean, commit, push, or pull unless explicitly requested.
3. Before and after edits run:

```bash
bash commands/inspect_with_probe.sh
bash commands/probe_forced_program.sh
bash commands/validate.sh
```

4. Edit only allowed files. Stop before expanding scope.
5. Never implement the next task early.
6. No epoch schedules, answer hints, expected-edge logic in model forward, hard
   masks, or weakened acceptance thresholds.
7. Save `reports/agent_tasks/TASK_<NN>_RESULT.md` using
   `99_STATUS_TEMPLATE.md`.
8. Mark PASS only when every acceptance item passes; otherwise preserve evidence
   and stop.

## Permanent regression gates

```text
gradient_closed = true
credit_closed = true
recovery_loss_connected = true
detached_enabled_losses = []
forced oracle summary = all true
loss_accounting_error <= 1e-5
scanner source masses sum to 1 ± 1e-4
no NaN/Inf
```

One coding agent at a time. Read-only analysis may be parallel, but agents must
not concurrently edit model/training/report files.
