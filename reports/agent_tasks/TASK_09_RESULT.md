# TASK 09 RESULT — PASS

```text
status: PASS
task_file: docs/agent_tasks_v4_6_4/09_HONEST_CURRICULUM_MODE_CREDIT.md
agent/model: Codex GPT-5
started_at: not captured
finished_at: 2026-06-20T05:24:27+03:00
git_head_before: 2ae51bc9234944c8ae04b069f40d8a3cc8a2c1dd
git_head_after: 2ae51bc9234944c8ae04b069f40d8a3cc8a2c1dd
```

## Scope changed

- `arch_builder/model.py`
- `arch_builder/credit.py`
- `arch_builder/reporting.py`
- `arch_builder/train_vertical_slice.py`
- `tools/project_probe/probe_learning_loop.py`
- `tools/project_probe/honesty_curriculum_proof.py`
- `commands/probe_honesty_curriculum.sh`

Notes:
- `model.py` got curriculum-mode plumbing and trimmed hint-context handling.
- `credit.py` got separate teacher/audit/deploy credit ledgers.
- `train_vertical_slice.py` now records phased curriculum metadata and credit buffers.
- `probe_learning_loop.py` was fixed for the current branch-loss schema so inspector stays green.

## Commands/configs/seeds

- `bash commands/inspect_with_probe.sh`
- `bash commands/probe_forced_program.sh`
- `bash commands/validate.sh`
- `bash commands/probe_honesty_curriculum.sh`

Proof command defaults:
- task config: `configs/tasks/chain_diff_product.yml`
- transfer task config: `configs/tasks/chain_diff_merge.yml`
- seed: `123`
- device: `cpu`
- smoke training: `epochs=1`, `max_steps=4`, `batch_size=16`
- audit batch: `eval_batch_size=32`
- model smoke: `dim=16`, `top_k=8`, `sim_rank=8`

## Acceptance checklist

- no static/runtime leakage — PASS
- random/shuffled hints do not beat honest deploy — PASS
- deploy credit contains deploy observations only — PASS
- honesty_score >= 0.80, preferably >= 0.90 — PASS (`1.0`)
- deploy retains readable program/diversity — PASS
- deploy random means FAIL regardless of Teacher accuracy — PASS

## Metrics

Proof report: `reports/agent_inspector/HONESTY_CURRICULUM_PROOF.md`

- `honesty_score`: `1.0`
- `honesty_floor`: `0.8`
- `full_acc`: `0.5625`
- `no_conv_acc`: `0.625`
- `no_hints_acc`: `0.5625`
- `deploy_acc`: `0.5625`
- `shuffled_acc`: `0.53125`
- `random_acc`: `0.5625`
- `transfer_acc`: `0.375`
- `deploy_credit_written`: `True`
- `program_recovery_rate` (full): `0.13541666666666666`
- `primitive_top_share` (full): `0.16636326164007187`
- `layer_action_similarity` (full): `0.23988310992717743`
- `program_diversity`: `PASS`

## Permanent regression gates

- inspector: `PASS` (`gradient_closed=True`, `credit_closed=True`, `bad=[]`)
- forced oracle: `PASS with existing baseline caveat` (`state_update_contract_pass=True`, `raw_primitive_pass=True`, `invariants_pass=True`, `address_content_separated=True`, `controller_address_alive=True`, `sequential_contract_pass=True`; known pre-existing `final_read_last_has_no_layer0_output_bypass=False`)
- validation: `PASS`
- loss accounting: `PASS` (`inspect_with_probe.sh` completed; no loss-accounting regression reported)
- scanner source mass sum: `PASS` (`1.0000000026066118` in proof; within tolerance)
- NaN/Inf: `PASS` (none observed in proof/validation)

## Artifacts and risks

- `reports/agent_inspector/HONESTY_CURRICULUM_PROOF.md`
- `reports/agent_inspector/honesty_curriculum_proof.json`
- `reports/agent_inspector/SUMMARY.md`
- `reports/agent_inspector/FORCED_PROGRAM_ORACLE.md`

Risks:
- `forced oracle` still has the pre-existing `final_read_last_has_no_layer0_output_bypass=False` caveat.
- Task09 proof uses a short smoke train; it proves honesty/mode separation, not final-scale performance.

## Next-task permission

`next_task_allowed: yes`
`reason: Task 09 PASS and proof artifacts are written.`
