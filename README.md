# acrch_builder

Clean modular workspace for the **Primitive Matrix Scanner / ActionMatrix Program** architecture.

This repo starts from the corrected v4.6.4 plan but does **not** try to build the whole giant system in one jump. The first real code target is a minimal vertical slice that proves the core loop can work:

```text
PrimitiveMatrix -> HybridScanner -> Top-K Simulator -> ActionMatrix Controller -> Executor -> bounded Credit -> Reports
```

## Why this repo exists

The previous experiments showed an important problem:

```text
Conv frontend works -> edge/memory/output credit appears
no-conv raw pooling collapses -> near random
```

So the new project separates:

```text
1. standalone proof:
   tiny synthetic known-program tasks where the correct program is known

2. real input:
   audio / structured matrix frontend later

3. plug-in mode:
   after/before/replace Attention later, only after standalone proof
```

## Project map

```text
arch_builder/
  primitive_matrix.py      # 5x5 PrimitiveMatrix + functional embedding init
  hybrid_scanner.py        # grid + semantic + usage + random candidate sources
  simulator.py             # low-rank candidate preview + predicted gain
  executor.py              # actual primitive operations
  model.py                 # ActionMatrix vertical-slice model
  synthetic_tasks.py       # known-program synthetic tasks
  credit.py                # bounded credit buffers / diagnostic deltas
  reporting.py             # csv/json/markdown reports
  train_vertical_slice.py  # CLI train/eval/report entrypoint

commands/
  validate.sh
  run_vertical_slice_push_reports.sh

docs/
  ARCHITECTURE.md
  V4_6_4_IMPLEMENTATION_PLAN.md

agent_reports/
  generated run reports only; no weights
```

## Install

```bash
git clone git@github.com:maxwelhelp/acrch_builder.git
cd acrch_builder

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash commands/validate.sh
```

## First run

GPU:

```bash
EPOCHS=5 BATCH_SIZE=128 DIM=64 DEVICE=cuda AMP=fp16 \
bash commands/run_vertical_slice_push_reports.sh
```

CPU smoke:

```bash
EPOCHS=1 BATCH_SIZE=32 DIM=32 DEVICE=cpu MAX_STEPS=20 \
bash commands/run_vertical_slice_push_reports.sh
```

The sync script validates, runs, writes reports, refuses to commit checkpoints/weights, commits only source/docs/reports, and pushes.

## First acceptance target

Do not move to real audio or Transformer plug-ins until the vertical slice shows:

```text
program_recovery_rate > random
sim_disabled_delta > 0
choice_without_sim_delta > 0
no skip-all collapse
bounded credit cost
ActionMatrix report is readable
```
