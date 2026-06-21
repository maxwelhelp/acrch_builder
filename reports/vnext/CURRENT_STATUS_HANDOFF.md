# vNext current status handoff

Date: 2026-06-21
Branch: `vnext_utility_critic_diagnostic`
Baseline tag: `baseline_real_discovery_3seed_acc0592_20260621`
Current HEAD before this handoff: `ed44b48 fix(vnext): report off-diagonal MMR similarity`

## Baseline contract

Do not roll the branch back to baseline. Keep the comparison contract explicit:

```text
ENABLE_VNEXT=0 -> baseline/comparison path
ENABLE_VNEXT=1 -> active vNext development path
```

Unsafe or immature vNext pieces should be gated, scheduled, instrumented, and tested. They should not be deleted as a substitute for fixing the learning loop.

## What works now

- Branch is synchronized with `origin/vnext_utility_critic_diagnostic` at `ed44b48`.
- `python -m py_compile` passes for the core edited modules in the MCP Python environment.
- Previous user-side `validate.sh` run was reported as OK before this handoff.
- Tiny vNext and 20-step credit smoke runs complete technically: UtilityCritic is instantiated, active choice/MMR can run, gradient-credit NLL/rank are logged, and forward/backward does not crash in the user CUDA environment.
- Delayed counterfactual credit closes: the current debug run reports `credit_closed=1`, `credit_total_measurements=12`, `credit_items=8`.

## Current 20-step debug result

Latest available debug run: `agent_reports/vnext_debug_20steps_credit.log`.

```text
status=SMOKE_FAIL
test_acc=0.0859
train_acc=0.1031
val_acc=0.1094
train_samples_per_second=13.4/s
credit_closed=1
credit_total_measurements=12
credit_items=8
primitive_top_share=0.6144
primitive_entropy=0.4753
utility_gain_corr=-1.0
sim_pred_real_corr=-1.0
mmr_selected_similarity≈0.961
single_signed_projection_usage=0.0
grad_credit_nll=-2.604
grad_credit_rank=0.0148
```

This is not a reason to revert vNext. It is the current active development failure point.

## Known failures / risks

1. `mmr_selected_similarity≈0.961` means behavior-space MMR is not separating candidates enough, or the behavior features are collapsed, or utility dominance overwhelms the diversity penalty.
2. `primitive_top_share≈0.614 > 0.60` and `primitive_entropy≈0.475 < 0.65` show early primitive collapse pressure.
3. `utility_gain_corr=-1.0` and `sim_pred_real_corr=-1.0` show current critic/simulator ranking is not aligned with delayed measured gains.
4. Hard MMR masking currently can remove CE-gradient paths to unselected candidates unless a policy/advantage log-prob path is added.
5. Utility choice is enabled immediately in the smoke, while the critic is still cold-started; this likely destabilizes early steps.
6. `single_signed_projection_usage=0.0` despite projection candidates being present means final choice does not currently use that scanner source.
7. MCP validation in this session is environment-blocked at import smoke because the connector Python has no `torch` installed. Syntax compile still works.

## Next development plan

1. Add fast standalone probes for controller/MMR and policy-gradient learning contracts.
2. Extend MMR diagnostics: pre/post off-diagonal similarity, behavior feature pair stats, selected utility, utility drop versus top-k.
3. Add a delayed policy/advantage learning path using candidate log-probs from the pre-hard-mask controller logits.
4. Add warmup/schedule flags so the critic can train/probe before hard MMR/utility choice dominates forward selection.
5. Add behavior decorrelation diagnostic/trainable loss behind a coefficient.
6. Rerun `validate.sh`; if the user Python with torch is available, rerun the 20-step credit smoke and inspect the new metrics.

## Do not commit

- `agent_reports/*` run directories/logs.
- `model_last.pt` and any model/checkpoint/cache files.
- `__pycache__`, `.pyc`, `.pyo`.

