# TASK 11 RESULT — PASS

Task: standalone real-data acceptance on SpeechCommands.

Evidence:

- proof command: `bash commands/probe_speechcommands_acceptance.sh`
- project validate: `bash commands/validate.sh`
- proof report: `reports/agent_inspector/SPEECHCOMMANDS_ACCEPTANCE_PROOF.md`
- proof json: `reports/agent_inspector/speechcommands_acceptance_proof.json`

Real-data config:

- dataset: `speechcommands`
- classes: `yes,no,up,down,left,right,on,off,stop,go`
- data root: `../Functional Matrix Grower/data/speechcommands`
- train/val/test limits: `512 / 256 / 256`
- model profile: `layers=2`, `top_k=16`, `sim_rank=16`, `dim=32`, `input_norm=layernorm`, `state_norm=layernorm`, `final_read=learned`

Key metrics:

| variant | status | full acc | audit acc | deploy acc | honesty |
|---|---|---:|---:|---:|---:|
| raw | FAIL | 0.0781 | 0.0781 | 0.0781 | 1.0000 |
| conv | PASS | 0.1250 | 0.1562 | 0.1250 | 1.0000 |
| structured | PASS | 0.2188 | 0.0781 | 0.2188 | 1.0000 |

Acceptance checks:

- deploy reproducibly above random — PASS
- structured beats raw baseline — PASS
- honesty retained — PASS
- simulator CE ablation positive — PASS
- non-grid scanner usage positive — PASS
- readable ActionMatrix / no collapse in this smoke — PASS

Notable ablation evidence for structured:

- `sim_disabled_delta = 0.04823493957519531`
- `gain_disabled_delta = 0.0017156600952148438`
- `slot_disabled_delta = 0.04309892654418945`
- `layer0_output_disabled_delta = 0.05820035934448242`
- `grid_candidate_usage = 0.41072002053260803`
- `usage_candidate_usage = 0.5810839533805847`

Verdict: PASS.
Next allowed task: 12.
