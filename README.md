# acrch_builder

Clean modular workspace for the **Primitive Matrix Scanner / ActionMatrix Program** architecture.

Main workflow is in:

```text
docs/AGENT_WORKFLOW.md

docs/V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md
```

Current target is not the full huge architecture. Current target is the first proof slice:

```text
PrimitiveMatrix5x5
-> HybridScanner
-> Top-K Simulator
-> ActionMatrix Controller
-> Executor
-> known synthetic task
-> reports
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash commands/validate.sh
```

## CPU smoke

```bash
EPOCHS=1 BATCH_SIZE=32 DIM=32 DEVICE=cpu MAX_STEPS=20 \
bash commands/run_vertical_slice_push_reports.sh
```

## GPU proof run

```bash
EPOCHS=5 BATCH_SIZE=128 DIM=64 DEVICE=cuda AMP=fp16 TASK=diff \
bash commands/run_vertical_slice_push_reports.sh
```

The run script commits and pushes reports only. No weights/checkpoints are committed.

## v7 note

The synthetic proof slice uses an explicit expected-edge choice loss to verify that the ActionMatrix can execute a known program before moving to unsupervised credit.

## v8 run note

For synthetic proof-slice use `INPUT_NORM=none` because `TASK=diff` depends on feature mean.

## Program report

Each run writes `PROGRAM_REPORT.md` inside the run directory. It shows the ActionMatrix table: which primitive each source->target cell selected, choice mass, and active mass.

## v10 anti-collapse

The synthetic proof run can use small anti-collapse losses so the expected primitive is not selected in every cell. Check `PROGRAM_REPORT.md` for `program_verdict`, `expected_top_cells`, and `active_cells`.

## v11 harder proof tasks

Use `TASK=two_diff` for two useful diff cells in one layer, and `TASK=chain_diff_product LAYERS=2` to test whether the second ActionMatrix layer does something different from the first.
