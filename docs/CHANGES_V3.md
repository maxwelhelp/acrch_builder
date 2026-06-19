# Changes v3

This archive edits the uploaded project locally and replaces source files, not patch snippets.

Fixed for the first vertical slice:

- Added slot/address embeddings so edge `(slot0 -> slot1)` is identifiable.
- Kept trainable `predicted_gain_for_loss` and `mode_for_loss` tensors non-detached.
- Program recovery is now edge-specific: expected source/target edge is tracked.
- Synthetic batches now carry `expected_src` and `expected_tgt`.
- Sim target is positive only for the expected primitive on the expected edge.
- Default `TOP_K` is 25 for the first proof slice so correct primitives are not cut away at startup.
- Usage candidates include stable rescue primitives: diff, merge, product, memory_write, memory_read.
- AMP dtype mismatch in executor is fixed.
- Validate and run scripts clean pycache before forbidden-file checks.

Run order:

```bash
~/bin/apply_acrch_builder_archive.sh /home/maxwelhelp/Загрузки/acrch_builder_clean_v3.zip

EPOCHS=1 BATCH_SIZE=32 DIM=32 DEVICE=cpu MAX_STEPS=20 \
  bash commands/run_vertical_slice_push_reports.sh

EPOCHS=5 BATCH_SIZE=128 DIM=64 DEVICE=cuda AMP=fp16 TASK=diff \
  bash commands/run_vertical_slice_push_reports.sh
```
