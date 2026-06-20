# Task 21 — production HybridScanner health

## Objective

Make proposal generation trainable, deduplicated, measurable and credit-driven.

## Required work

- Fix the hard `anchor_logits.argmax` dead-gradient path using a defensible STE,
  learned credit objective or other bounded differentiable estimator.
- Deduplicate the grid/semantic/usage/random union while preserving source
  attribution and valid masks; use plan default `usage_k=2` unless justified.
- Drive usage candidates only from delayed real Task16 credit.
- Add global rescue, semantic quality/collapse, source coverage and source-ablation
  metrics to every run, not only a standalone proof.

## Acceptance

Anchor has nonzero task gradient or measured delayed credit; no duplicate receives
double probability mass; each source's mass sums correctly; non-grid rescue is
positive on a held-out semantic task; scanner-source ablations are paired.
