# Task 05 — Config-driven tasks

Goal: remove hardcoded synthetic task logic from Python.

Need:
1. Add `configs/tasks/*.yml`.
2. Each config describes:
   - task name
   - slots
   - layers
   - target formula
   - expected actions
   - pass thresholds
3. Loader reads config and builds task.
4. Existing tasks must still work:
   - diff
   - two_diff
   - merge
   - product
   - chain_diff_merge
   - chain_diff_product
5. Add command:
   `bash commands/run_task_config.sh configs/tasks/chain_diff_product.yml`

PASS:
- old CLI tasks still work
- config task run matches old task result
- validate passes
- report writes config name and expected actions
