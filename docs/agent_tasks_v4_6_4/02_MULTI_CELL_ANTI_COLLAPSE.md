# Task 02 — multi-cell recovery without primitive collapse

## Objective

Make `two_diff` recover its two useful cells without selecting `diff` in all
cells or leaving all 16 cells active.

## Preconditions

Task 01 PASS for all three single primitives.

## Allowed files

```text
arch_builder/train_vertical_slice.py
arch_builder/reporting.py
commands/ proof commands
docs/ and reports/agent_tasks/
```

Do not change executor, transport, addresses, scanner policy, or simulator.

## Required method

1. Preserve an unchanged baseline.
2. Use Stage 4 loss accounting to find the ineffective live term.
3. Change one signal-gated mechanism per experiment: existing gate threshold/
   sharpness, existing effective scaling, active/tape target, or normalization
   of a live diversity loss.
4. Anti-collapse/sparse pressure opens only after both expected actions live.
5. Provide an ablation proving structure improves without killing recovery.

## Acceptance across at least three seeds

```text
both expected actions: present=1, choice_mass>=0.70, recovery>=0.80
expected_top_cells <= 4
active_cells < 16
primitive_top_share < 0.85
program verdict != primitive_collapse
validation accuracy > random
single diff regression passes
```

Forbidden: task-specific bans, expected-edge reads in forward, hard masks, and
epoch calendars. Stop after `two_diff`.

