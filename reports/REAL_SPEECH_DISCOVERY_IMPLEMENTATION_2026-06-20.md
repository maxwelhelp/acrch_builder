# Real SpeechCommands discovery implementation — 2026-06-20

## Scope

Implemented generic path B on the real structured SpeechCommands input. The real
trainer does not import the synthetic vertical-slice trainer, does not read
`expected_actions`, and does not assign roles to layers.

## Mechanism

- bounded high-activity plus independent-random interventions;
- single and joint primitive-removal counterfactuals;
- all interventions evaluated in one expanded batch;
- paired `delta_CE = CE(ablated) - CE(full)` on a separate microbatch;
- joint synergy separated from the sum of individual gains;
- credit applied only on later optimizer steps and retained until refreshed;
- normalized signed credit trains controller choice and scanner anchor;
- raw measured gain calibrates simulator prediction;
- measured credit updates usage candidates;
- a separate unchosen budget forces 1-2 top-k alternatives and measures
  `CE(full)-CE(forced)` instead of pointlessly ablating an unused action;
- source quotas expose grid/semantic/usage/random candidates without forcing choice;
- generic sparsity, topology consistency, exploration and anti-collapse losses.

## Real input/head

- structured energy/delta/onset/DCT frontend retained;
- multiclass LayerNorm/MLP head replaces the weak linear audio readout;
- discovery always runs deploy mode; teacher/audit/layer-role schedules are absent.

## Performance work

- precomputed on-device local topology lookup replaces per-row Python/CPU code;
- scanner metrics and `.cpu()` synchronizations disabled during training/credit;
- primitive executor removes 25 `mask.any()` host synchronizations per layer;
- ablations vectorized over an intervention batch dimension;
- DataLoader supports pinned memory, persistent workers and prefetch;
- CPU 20-step smoke improved from about 6.1s to 5.3s before later additions.

## Evidence

- `bash commands/probe_real_discovery_credit.sh`: PASS
- `bash commands/probe_forced_program.sh`: PASS
- `bash commands/probe_hybrid_scanner.sh`: PASS
- `bash commands/validate.sh`: PASS
- `bash commands/inspect_with_probe.sh`: gradient/credit closed
- real SpeechCommands 20-step smoke: complete, correctly FAIL at chance;
  simulator and choices were non-decorative in the first smoke, while all
  acceptance gates remained strict.

The real smoke is not an acceptance result. A full/multi-seed P40 run is still
required before any universal discovery PASS.

## Acceptance boundary

- one report is only `SMOKE_PASS` or `SMOKE_FAIL`, with `acceptance_status=NOT_RUN`;
- `global_candidate_usage` is reported as `global_scan_usage` and is excluded
  from the semantic+usage+random non-grid gate;
- universal acceptance requires learned/frozen/random controller runs over three
  unique seeds through `commands/run_speechcommands_real_discovery_acceptance.sh`.
