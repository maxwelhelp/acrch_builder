# Task 17 — simulator real-gain closure

## Objective

Train `predicted_gain` from delayed mode-specific measured credit, not from
`expected_actions` targets `1/-0.2`.

## Required work

- Build stop-gradient targets from Task16 real ablation gains.
- Distinguish unavailable/stale credit from zero gain; mask unavailable targets.
- Report target coverage, calibration loss, Pearson/Spearman correlation,
  predicted-gain/choice correlation and sim/context logit and gradient ratios.
- Pair full, no-gain, no-result and no-simulator ablations on identical batches/RNG.

## Acceptance

No expected action is read by the canonical simulator target; coverage and age
are reported; correlation is positive on held-out data in 3 seeds; simulator
ablation has positive median delta-CE and changes choices without destroying
Task15 validation accuracy.
