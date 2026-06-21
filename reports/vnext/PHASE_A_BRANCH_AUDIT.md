# Phase A — Current vNext Branch Audit

Date: 2026-06-21  
Branch: `vnext_utility_critic_diagnostic`  
Baseline: `baseline_real_discovery_3seed_acc0592_20260621` (`b1d4ed0`)  
Audited HEAD: `fb73b27`

## 1. Repository state and existing commits

The local branch and cached remote ref both point to `fb73b27`:

```text
fb73b27 feat: implement phase 4 credit upgrade (gradient trace, rank-based loss, Shapley-lite triples)
c44529d feat(vnext): implement active UtilityCritic choice, feedback memory, MMR controller, and lazy executor
ecbd3c9 feat: implement safety infrastructure and Phase 1 UtilityCritic diagnostic-only mode
b1d4ed0 baseline_real_discovery_3seed_acc0592_20260621
```

`git fetch`/`git pull` was attempted before the audit but the SSH operation did
not complete in the available environment. The pre-existing local
`origin/vnext_utility_critic_diagnostic` ref exactly matches local `HEAD`.

The tracked worktree was clean at audit start. There are many untracked run
directories under `agent_reports/`, `LATEST_AUDIO_FRONTEND_REPORT.md`, and two
untracked reports (`PHASE_3_INTEGRATION.md`, `PHASE_4_INTEGRATION.md`). They are
not part of this audit commit and must not be added accidentally.

## 2. Roadmap phases already partially implemented

| Roadmap area | Current implementation | Audit status |
|---|---|---|
| Phase 1 UtilityCritic diagnostic | `UtilityCritic`, report metrics, run flags | Partial; not truly diagnostic-only |
| Utility choice | Additive normalized utility score | Behavior-changing, default-off |
| Feedback memory | Global primitive gain/regret EMA | Wrong address space; update runs even when source is off |
| MMR controller | Batched hard candidate mask using primitive embeddings | Not behavior-feature MMR; nondifferentiable pool selection |
| Lazy executor | Executes selected top-B candidate subset | Experimental; assumes exact fixed selected count |
| Learnable output mix | Enabled by `enable_vnext` | Behavior-changing during training |
| Category scanner | Category candidates | Default-off source, but expanded bank is default-on |
| Primitive expansion | Spectral, attention-like and mined primitives | Incorrectly active in the default bank |
| Phase 4 credit | gradient trace, rank loss, triples | Partly unconditional and ahead of Learning Contract |

## 3. Flag defaults and actual effects

All CLI `store_true` flags and corresponding environment variables default to
off in `run_speechcommands_real_discovery.sh`:

```text
ENABLE_VNEXT=0
ENABLE_UTILITY_CRITIC_PROBE=0
ENABLE_UTILITY_CRITIC_CHOICE=0
ENABLE_SCANNER_FEEDBACK_MEMORY=0
ENABLE_MMR_CONTROLLER=0
ENABLE_LAZY_EXECUTOR=0
ENABLE_CATEGORY_SCANNER=0
ENABLE_AUTO_MINED_ATOMS=0
```

However, flag defaults do **not** provide baseline equivalence:

- the primitive bank is unconditionally expanded from 25 to 40;
- the executor unconditionally allocates spectral/attention/mined parameters;
- the executor implementation is unconditionally replaced with per-primitive
  `mask.any()`/subselect branches;
- feedback gain/regret buffers are updated by every measured usage-credit
  update even when the feedback scanner source is disabled;
- triple interventions are enabled in counterfactual collection without a
  Phase-4 flag.

Flag-specific effects:

- `ENABLE_VNEXT=1` creates a UtilityCritic and learnable output mixing weights.
  The latter immediately receive CE gradients and therefore change training.
- `ENABLE_UTILITY_CRITIC_PROBE=1` creates the critic, but also installs gradient
  hooks and trains it with gradient-credit NLL/ranking on later steps. This is
  not metrics-only diagnostic behavior.
- `ENABLE_UTILITY_CRITIC_CHOICE=1` disables current simulator/gain components
  and adds UtilityCritic logits to choice.
- `ENABLE_SCANNER_FEEDBACK_MEMORY=1` adds feedback candidates, but the memory is
  global primitive-only and source id `4` overlaps the existing global source.
- `ENABLE_MMR_CONTROLLER=1` hard-masks candidates using primitive embedding
  similarity. The mask selection is nondifferentiable.
- `ENABLE_LAZY_EXECUTOR=1` executes only a selected budget, but depends on the
  hard selection contract and fixed cardinality assumptions.
- `ENABLE_CATEGORY_SCANNER=1` adds category candidates. New categories and new
  primitive parameters already exist even when it is off.
- `ENABLE_AUTO_MINED_ATOMS` is stored but does not gate the unconditionally
  expanded primitive bank or executor parameters.

The vNext smoke script defaults `ENABLE_VNEXT=1` and critic probe on. The vNext
3-seed script also defaults vNext and probe on, while utility choice remains off.

## 4. Default-off equivalence result

A deterministic CPU model smoke used identical seed/config/input on current
HEAD and an archive of `main`:

| Measurement | `main` | current HEAD, all vNext flags off |
|---|---:|---:|
| Primitive count | 25 | 40 |
| Parameters | 30,841 | 71,701 |
| Logit sum | -2.09030 | -1.62644 |
| Logit std | 0.16778 | 0.27067 |

Result: **FAIL**. Parameter count, RNG consumption, candidate space and forward
outputs differ with all vNext flags disabled.

## 5. Executor performance and implementation risk

Static audit found one `mask.any()` branch per primitive and Python dispatch for
all 40 primitives. This causes device-to-host synchronization risk on CUDA/P40.
It also imports NumPy and constructs a DCT matrix with nested Python loops at
module construction time.

A single-thread CPU microbenchmark (`N=512`, `K=8`, `D=64`, old candidate ids)
measured:

| Executor | Mean forward |
|---|---:|
| `main` vectorized-all-branches | 47.25 ms |
| current subselect implementation | 10.71 ms |

The CPU result favors subselect execution, but it does not clear the CUDA risk:
CUDA is unavailable in the audit environment, so P40 synchronization cost is
unmeasured. The current implementation must not be accepted on CPU timing alone.

## 6. Credit and learning-loop risks

1. UtilityCritic receives `target_address` as `head_vector`; this is a slot
   address, not real output-head context.
2. Gradient-trace hooks activate whenever a critic exists, including probe mode.
   The trainer consumes these records and adds NLL/rank loss without an explicit
   Phase-4 enable flag.
3. Counterfactual triples are inserted unconditionally and their synergy is
   reduced into the same scalar records used for ordinary utility alignment.
4. Hard MMR selection blocks CE gradient through candidate membership. There is
   no formal policy/propensity contract for the hard selection step.
5. MMR uses primitive embeddings, not critic behavior features.
6. Feedback EMA is global primitive-only, has no layer address, no staleness
   decay and no low-count UCB contract.
7. `utility_pool_size` is reported but does not define the actual scanner pool.
8. Several Phase-2/4 metrics are synthesized post hoc from sparse measured
   records and do not prove that runtime controller behavior uses them.

## 7. Generated and large-file audit

No new weight file was added by the three vNext commits. The repository already
tracks historical `agent_reports` artifacts and multi-megabyte backup patches
from earlier history. Current untracked vNext run directories contain generated
reports/checkpoints and must remain untracked.

`commands/validate.sh` was changed to ignore forbidden model/cache files under
all `agent_reports/`. This prevents current local checkpoints from failing
validation but weakens the repository guard: accidentally tracked weights under
that directory would no longer be detected by the validation command.

Validation result before stabilization:

```text
bash commands/validate.sh -> PASS
```

This PASS confirms syntax/import hygiene only; it does not establish baseline
equivalence, CUDA performance, or a closed vNext learning contract.

## 8. Dangerous files

| File | Risk |
|---|---|
| `arch_builder/executor.py` | unconditional expansion; `mask.any()` CUDA sync risk |
| `arch_builder/primitive_matrix.py` | default bank expanded to 40; global feedback mutation |
| `arch_builder/model.py` | hard MMR/lazy path; false head context; automatic grad hooks |
| `arch_builder/credit.py` | unconditional triples; mixed target semantics; Python pair loops |
| `arch_builder/train_audio_frontend.py` | automatic gradient-credit training in probe mode |
| `arch_builder/hybrid_scanner.py` | feedback/global source id collision; expanded-bank coupling |
| `commands/validate.sh` | ignores forbidden files inside `agent_reports` |

## 9. Required corrections before Phase D

1. Add and approve the Phase 1.5 Learning Contract.
2. Restore 25-primitive baseline bank and baseline executor when expansion flags
   are off; expanded primitives must be explicitly gated.
3. Restore exact default-off parameter/RNG/forward equivalence.
4. Put gradient trace, rank loss and Shapley-lite triples behind explicit
   diagnostic flags; do not activate them from critic probe mode.
5. Keep UtilityCritic probe metrics-only with zero effect on runtime choice and
   baseline optimizer paths except its explicitly isolated critic loss.
6. Define inference-safe head context and critic target semantics.
7. Keep hard MMR diagnostic-only until policy-gradient/soft-ST closure is tested.
8. Make feedback memory at least layer × primitive and update it only from
   measured gain when explicitly enabled.
9. Benchmark executor on P40; reject a regression greater than 15–20%.
10. Restore a validation rule that rejects tracked/generated weight files rather
    than excluding the entire `agent_reports` subtree.

Phase A conclusion: the branch is recoverable, but it is not safe to proceed to
new roadmap features until Learning Contract and stabilization are completed.
