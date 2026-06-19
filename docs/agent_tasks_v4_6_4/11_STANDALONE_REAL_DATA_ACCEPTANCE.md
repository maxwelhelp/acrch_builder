# Task 11 — standalone real-data acceptance

## Objective

Accept full standalone v4.6.4 on SpeechCommands and held-out transfer/config
before any Transformer integration.

## Preconditions

Task 10 PASS.

## Allowed files

Training/evaluation commands, dataset wrappers, reporting, and small audit-proven
bug fixes. No new major subsystem.

## Required comparisons

```text
available prior baseline
Conv scaffold; structured frontend
without scanner/simulation/layer listening/memory/credit penalty
Teacher vs Audit vs Deploy
```

Use fixed splits/config/seeds and paired eval where possible.

## Acceptance

```text
Deploy reproducibly above random
full system matches/improves declared baseline or explicit cost-quality tradeoff
readable ActionMatrix; no primitive/skip/disable/slot collapse
nontrivial layer specialization and component credit
simulator CE ablation reproducibly positive
non-grid scanner rescue/real gain positive
honesty_score >= 0.80
complete reproducible artifacts
```

Any failed core gate returns to its owning task. No plug-in work on failure.

