# Agent workflow

This repository is the main workspace for the Primitive Matrix Scanner / ActionMatrix Program project.

## Mandatory work cycle

```text
1. Sync first
   git checkout main
   git pull --rebase --autostash origin main

2. Inspect before editing
   read README.md
   read docs/AGENT_WORKFLOW.md
   read docs/ARCHITECTURE.md
   read docs/V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md
   read docs/V4_6_4_IMPLEMENTATION_PLAN.md
   inspect exact files that will be changed

3. Edit locally
   make changes in a local working tree
   keep modules small and readable
   do not create weights/checkpoints

4. Validate locally
   bash commands/validate.sh

5. Sync changes back to GitHub
   git status
   git add only source/docs/commands/reports
   never stage weights/cache/checkpoints
   git commit
   git push origin main

6. User runs tests
   run script commits reports only and pushes them

7. Analyze reports
   sync again
   read LATEST_RUN_REPORT.md
   read latest agent_reports/*/metrics.csv
   read final_report.json
   decide next change

8. Repeat
   sync -> edit locally -> validate -> push -> user tests -> reports -> analyze
```

## Hard rules

```text
Always sync before analysis or editing.
Main development is local files + git sync.
Do not commit model weights, checkpoints, safetensors, pyc, or cache files.
Do not move to real audio or plug-in transformer work until the vertical slice is understood.
Every run must produce reports that can be committed without weights.
```

## Canonical architecture plan

`docs/V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md` is the design contract. Keep it in sync with implementation decisions.
