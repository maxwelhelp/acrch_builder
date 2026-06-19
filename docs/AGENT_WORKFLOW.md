# Agent workflow

This repository is now the main workspace for the Primitive Matrix Scanner / ActionMatrix Program project.

## Mandatory work cycle

Every agent or assistant must follow this loop.

```text
1. Sync first
   git checkout main
   git pull --rebase --autostash origin main

2. Inspect before editing
   read README.md
   read docs/AGENT_WORKFLOW.md
   read docs/ARCHITECTURE.md
   read docs/V4_6_4_IMPLEMENTATION_PLAN.md
   inspect the exact files that will be changed

3. Edit locally
   make changes in a local working tree
   keep modules small and readable
   do not create weights/checkpoints

4. Validate locally
   bash commands/validate.sh
   run the relevant smoke or train command

5. Sync changes back to GitHub
   git status
   git add only source/docs/commands/reports
   never stage weights/cache/checkpoints
   git commit
   git push origin main

6. Give the user run commands
   include sync command
   include test command
   include report-read command

7. User runs tests
   run script commits reports only and pushes them

8. Analyze reports
   sync again
   read LATEST_RUN_REPORT.md
   read latest agent_reports/*/metrics.csv
   read final_report.json
   decide next change

9. Repeat
   sync -> edit locally -> validate -> push -> user tests -> analyze
```

## Hard rules

```text
Always sync before analysis or editing.
Always edit locally, then push to GitHub.
Do not rely on GitHub web editing for large changes.
Do not commit model weights, checkpoints, safetensors, pyc, or cache files.
Do not move to real audio or plug-in transformer work until the vertical slice is understood.
Every run must produce reports that can be committed without weights.
```

## Current first target

The current target is the minimal vertical slice:

```text
1-2 layers
4 slots
PrimitiveMatrix5x5
HybridScanner
Top-K simulator
mandatory simulator influence
ActionMatrix executor
bounded credit/reporting
known synthetic tasks
```

Acceptance before expansion:

```text
program_recovery_rate above random
sim_disabled_delta positive
choice_without_sim_delta positive
no skip-all collapse
bounded credit/reporting cost
readable ActionMatrix-style report
```

## Commands for the user

Update local repo:

```bash
git checkout main
git pull --rebase --autostash origin main
```

Validate:

```bash
bash commands/validate.sh
```

CPU smoke:

```bash
EPOCHS=1 BATCH_SIZE=32 DIM=32 DEVICE=cpu MAX_STEPS=20 \
  bash commands/run_vertical_slice_push_reports.sh
```

GPU run:

```bash
EPOCHS=5 BATCH_SIZE=128 DIM=64 DEVICE=cuda AMP=fp16 TASK=diff \
  bash commands/run_vertical_slice_push_reports.sh
```

Read reports:

```bash
cat LATEST_RUN_REPORT.md
LAST_DIR=$(ls -td agent_reports/vertical_slice_* | head -1)
echo "$LAST_DIR"
cat "$LAST_DIR/final_report.json"
tail -n +1 "$LAST_DIR/metrics.csv"
```
