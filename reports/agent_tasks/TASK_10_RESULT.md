# TASK 10 RESULT — PASS

```text
status: PASS
task_file: docs/agent_tasks_v4_6_4/10_STRUCTURED_FRONTEND_AUDIO.md
agent/model: Codex GPT-5
started_at: not captured
finished_at: 2026-06-20T06:08:35+03:00
git_head_before: 2ae51bc9234944c8ae04b069f40d8a3cc8a2c1dd
git_head_after: 2ae51bc9234944c8ae04b069f40d8a3cc8a2c1dd
```

## Scope changed

- `arch_builder/audio_frontend.py`
- `arch_builder/train_audio_frontend.py`
- `tools/project_probe/audio_frontend_proof.py`
- `commands/probe_audio_frontend.sh`
- `reports/agent_tasks/TASK_10_RESULT.md`
- `docs/agent_tasks_v4_6_4/STATUS_BOARD.md`

Notes:
- core vertical-slice model stayed shared
- audio behavior lives in wrappers/entrypoint only
- Conv is used only as scaffold baseline, not as deploy proof

## Commands/configs/seeds

- `bash commands/validate.sh`
- `bash commands/inspect_with_probe.sh`
- `bash commands/probe_forced_program.sh`
- `bash commands/probe_audio_frontend.sh`

Proof defaults:
- seed: `7`
- device: `cpu`
- epochs: `3`
- steps_per_epoch: `18`
- eval_steps: `6`
- batch_size: `48`
- eval_batch_size: `96`
- length: `256`
- slots: `4`
- dim: `24`
- layers: `1`
- top_k: `8`
- sim_rank: `8`

## Acceptance checklist

- all variants reported with params/FLOPs/memory — PASS
- structured frontend reproducibly beats raw failed baseline — PASS
- Deploy above random with honesty retained — PASS
- core program/credit/scanner/simulator acceptance retained — PASS
- Conv explicitly scaffold-only — PASS

## Metrics

Proof report: `reports/agent_inspector/AUDIO_FRONTEND_PROOF.md`

- raw:
  - `full_acc`: `0.5451388888888888`
  - `audit_acc`: `0.5486111111111112`
  - `deploy_acc`: `0.5381944444444444`
  - `honesty_score`: `0.9872611464968153`
  - `frontend_params`: `120.0`
  - `frontend_flops`: `2144.0`
  - `frontend_activation_bytes`: `1504.0`
- conv:
  - `full_acc`: `0.6215277777777778`
  - `audit_acc`: `0.6197916666666666`
  - `deploy_acc`: `0.5885416666666666`
  - `honesty_score`: `0.946927374301676`
  - `frontend_params`: `2160.0`
  - `frontend_flops`: `105216.0`
  - `frontend_activation_bytes`: `7936.0`
- structured:
  - `full_acc`: `0.9791666666666666`
  - `audit_acc`: `0.9774305555555556`
  - `deploy_acc`: `0.9861111111111112`
  - `honesty_score`: `1.0070921985815604`
  - `frontend_params`: `362.0`
  - `frontend_flops`: `9675.0`
  - `frontend_activation_bytes`: `3084.0`

## Permanent regression gates

- inspector: `PASS` (`gradient_closed=True`, `credit_closed=True`, `bad=[]`)
- forced oracle: `PASS with known caveat` (`state_update_contract_pass=True`, `raw_primitive_pass=True`, `invariants_pass=True`, `address_content_separated=True`, `controller_address_alive=True`, `sequential_contract_pass=True`, `final_read_last_has_no_layer0_output_bypass=False`)
- validation: `PASS`
- scanner source mass sum: `PASS` (`1.0000000026066118` in audio proof, within tolerance)
- NaN/Inf: `PASS`

## Artifacts and risks

- `reports/agent_inspector/AUDIO_FRONTEND_PROOF.md`
- `reports/agent_inspector/audio_frontend_proof.json`
- `reports/agent_inspector/FORCED_PROGRAM_ORACLE.md`
- `reports/agent_inspector/SUMMARY.md`

Risks:
- audio proof uses a synthetic order task, not SpeechCommands yet
- forced oracle still has the known `final_read` caveat from earlier tasks

## Next-task permission

`next_task_allowed: yes`
`reason: Task 10 PASS and audio proof artifacts are written.`
