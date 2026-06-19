# Task 14 — final acceptance audit

## Objective

Run canonical acceptance matrix and produce reproducible final status. Do not
invent fixes here.

## Preconditions

Tasks 01–13 PASS.

## Allowed files

Evaluation/report aggregation, documentation, reproduction commands, status.
No architecture/loss edits; failures return to owning task.

## Required audit groups

Synthetic recovery/collapse/dependency; real credit; simulator; HybridScanner;
specialization/hard-delete; branching/output; honesty; frontend/SpeechCommands;
add-on plug-ins; causal replacement; speed/params/memory; artifact completeness.

## Required artifacts

```text
FINAL_V4_6_4_ACCEPTANCE.md
final_v4_6_4_acceptance.json
experiment_manifest.json
commands_to_reproduce.md
failed_checks.json
```

Every claim links report/config/seed and distinguishes fact/inference/missing
evidence.

## PASS rule

PASS only if every mandatory canonical check passes. Otherwise status FAIL,
name owning task, attach evidence, and never weaken threshold or claim release.

