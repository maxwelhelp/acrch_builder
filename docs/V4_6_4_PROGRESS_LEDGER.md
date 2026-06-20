# V4.6.4 Progress Ledger

This file is the compact source of truth for what was fixed, what was proven, and what is still open. Detailed run artifacts live in `reports/agent_tasks/`, `agent_reports/`, and `reports/agent_inspector/`.

## Current state

- Inspector/probe is live and useful: `gradient_closed=True`, `credit_closed=True`, `recovery_loss_connected=True`.
- Forced oracle is passing: raw primitives, content/address separation, target-write contract, and sequential forced paths pass.
- Single primitives work.
- Multi-cell anti-collapse works on `two_diff`.
- Learned sequential merge chain works across repeated seeds.
- Learned sequential product chain has a passing structural smoke on seed1, but full repeated-seed PASS is not recorded yet.

## Stage / task board

| Step | Status | What was proven | Key result | Next |
|---|---|---|---|---|
| Stage 0 inspector/probe | PASS | Runtime probe and reports were restored against current project API. | `gradient_closed=True`, `credit_closed=True` after fixes; inspector/validate commands work. | done |
| Stage 1 live choice loss | PASS | Choice/recovery losses now connect to autograd. | `choice_for_loss` is live; report choice remains detached; `recovery_loss_connected=True`; `detached_enabled_losses=[]`. | done |
| Stage 2 forced oracle | PASS | Executor math and forced known-program paths are testable. | raw `diff/merge/product` oracle = 100%; action invariants pass. | done |
| Stage 3 content/address + target-write | PASS | Slot address is separated from executor operands; target writes obey contract. | executor gets clean content; controller still sees slot address; target writes/soft-OR gate pass; forced chain oracle 100%. | done |
| Stage 4 diagnostics/accounting | PASS | Training and inspector use the same objective/reporting path. | per-action metrics, scanner source mass, independent simulator ablations, and loss accounting added; accounting error ~1e-7/1e-9. | done |
| Task 01 single primitive baselines | PASS | `diff`, `merge`, and `product` learn individually. | all seeds: oracle=1.0, candidate=1.0, choice_mass≈1.0, recovery=1.0, val well above random. | done |
| Task 02 two_diff anti-collapse | PASS | Two expected `diff` actions recover without global primitive collapse. | seed1/2/3 pass; active cells < 16; top_cells=4; verdict `sparse_or_partly_sparse_program`. | done |
| Task 03 chain_diff_merge | MERGE_PASS | Layer1 learns from Layer0 content on merge chain. | seeds 1/2/3 pass; best_acc 0.8598-0.9422; dependency_delta 0.358-0.425; ablation_delta 0.349-0.442; active=[8,8]. | done |
| Task 03 chain_diff_product | SMOKE_PASS_SEED1 | Product chain structure works and starts learning. | seed1 6ep: best_acc=0.7461, recovery=1.0, choice_mass=0.9994, dependency_delta=0.2480, ablation_delta=0.2613, active=[8,5], no collapse. | run product seeds 2/3 6ep |

## Code changes that matter

| Area | Change | Why it mattered |
|---|---|---|
| `arch_builder/model.py` | Added live `choice_for_loss`; kept detached `choice` for reports/signal gates. | Fixed detached choice/recovery losses: expected choice loss now actually trains controller choice. |
| `arch_builder/model.py` | Split slot address from executor operands. | Primitives now operate on clean content, while controller/scanner can still use address features. |
| `arch_builder/model.py` | Added/used target-write semantics: disable has zero write mass; unit write replaces target; multi-write normalizes and gates with bounded soft-OR. | Fixed broken state transport between layers and made forced sequential programs meaningful. |
| `arch_builder/train_vertical_slice.py` | Unified training objective used by real training and inspector probe. | Prevented inspector from checking a different loss path than actual training. |
| `arch_builder/train_vertical_slice.py` | Added per-loss autograd diagnostics and loss accounting. | Made detached or decorative losses visible instead of hidden. |
| `arch_builder/train_vertical_slice.py` | Added action-specific metrics for each expected action. | Multi-action tasks can now show which exact edge/action failed. |
| `arch_builder/train_vertical_slice.py` | Reworked active budget to penalize soft active-cell count, not only mean activity. | Fixed Task 02 seed2 failure where 16 weak active cells passed mean budget but failed acceptance. |
| `arch_builder/synthetic_tasks.py` | Added `chain_diff_merge`. | Enables Task 03: Layer0 diff+diff -> Layer1 merge. |
| `tools/project_probe/forced_program_oracle.py` | Added `chain_diff_merge` to forced sequential audit. | Forced oracle now covers merge chain before product chain. |
| `tools/project_probe/*`, `commands/*` | Restored probe/forced-oracle/validate compatibility. | Stage 0 reports became runnable again. |
| `reports/agent_tasks/*` | Added task result files for Task 01/02/03. | Agents can read concise PASS/PARTIAL state without parsing full logs. |

## Important results

### Task 01: single primitive baselines

| Task | Seeds | Result |
|---|---:|---|
| `diff` | 1/2/3 | PASS; best val roughly 0.8824-0.9434; recovery=1.0; choice_mass≈1.0. |
| `merge` | 1/2/3 | PASS; best val roughly 0.8949-0.9012; recovery=1.0; choice_mass≈1.0. |
| `product` | 1/2/3 | PASS; best val roughly 0.9004-0.9172; recovery=1.0; choice_mass≈1.0. |

### Task 02: two_diff anti-collapse

| Run | Status | Key metrics |
|---|---|---|
| seed1 stronger sparse | PASS | best_acc=0.9230, top_cells=4, active=8, top_share=0.2615, verdict=sparse. |
| seed2 softcount fix | PASS | best_acc=0.8184, top_cells=4, active=13, top_share≈0.279, verdict=sparse. |
| seed3 stronger sparse | PASS | best_acc=0.9309, top_cells=4, active=8, top_share=0.2532, verdict=sparse. |

The important bug here was not recovery. Recovery was already 1.0. The failure was that `active_budget` optimized mean activity, while acceptance checked active cell count. The soft active-cell-count budget fixed that mismatch.

### Task 03: learned sequential chain

| Subtask | Status | Key metrics |
|---|---|---|
| `chain_diff_merge` seed1 | PASS | best_acc=0.9422, dep=0.4250, ablate=0.4418, active=[8,8], no collapse. |
| `chain_diff_merge` seed2 | PASS | best_acc=0.9164, dep=0.4133, ablate=0.4305, active=[8,8], no collapse. |
| `chain_diff_merge` seed3 | PASS | best_acc=0.8598, dep=0.3582, ablate=0.3492, active=[8,8], no collapse. |
| `chain_diff_product` seed1 smoke 4ep | STRUCTURAL PASS / weak accuracy | best_acc=0.5934, dep=0.0809, ablate=0.0938, active=[9,7], no collapse. |
| `chain_diff_product` seed1 6ep | SMOKE PASS | best_acc=0.7461, recovery=1.0, choice_mass=0.9994, dep=0.2480, ablate=0.2613, active=[8,5], no collapse. |

Interpretation: `chain_diff_merge` proves the sequential contour. Product is harder: controller/recovery works early, but classifier/value path needs more epochs. Six epochs are enough for seed1 smoke; repeated seeds are still needed before Task 03 full PASS.

## Why early epochs are weak

This is not a hardcoded task schedule. There is only normal tau annealing and signal-gated structure regularization:

- `tau = max(tau_min, tau_start * tau_decay ** (epoch - 1))` softens/hardens choice gradually.
- recovery/collapse/sparse gates depend on live signals: candidate presence, expected choice mass, primitive top share, and active-cell count.
- in `chain_diff_product`, recovery can reach 1.0 before classification improves, because selecting the correct program and training the numeric/product/classifier path are different phases.

## Current next command

Run product seeds 2/3 with 6 epochs, then update `TASK_03_RESULT.md`:

```bash
cd ~/test/sience/experiments/math_search/WORKING_BEST/acrch_builder
# run chain_diff_product seeds 2 and 3 with the same 6ep config as seed1_6ep
```

Task 04 starts only after `chain_diff_product` repeated seeds pass.
