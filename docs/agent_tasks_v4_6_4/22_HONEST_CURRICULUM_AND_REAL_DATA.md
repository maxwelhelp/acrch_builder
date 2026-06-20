# Task 22 — honest curriculum and matched real-data acceptance

## Objective

Test real scaffolds/hints and establish standalone value on SpeechCommands.

## Required work

- Define actual Conv and hint inputs with explicit weights and decay; do not call
  internal recurrent context a hint.
- Evaluate teacher, audit, deploy, no-conv, no-hints, shuffled hints and random
  hints on the same held-out examples.
- Compare structured v4.6.4 against v4.6.3 on identical split, preprocessing,
  seed set, optimization budget and evaluation sample count.
- Include test accuracy, speed, parameter count and memory.

## Acceptance

Deploy is significantly above chance and reproducible across at least 3 seeds;
random/shuffled hints cannot pass by equality at chance; structured model meets
the plan's matched baseline rule; all data split hashes are recorded.
