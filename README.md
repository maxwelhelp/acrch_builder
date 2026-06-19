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
