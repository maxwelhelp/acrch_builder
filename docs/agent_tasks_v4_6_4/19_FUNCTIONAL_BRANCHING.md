# Task 19 — functional branching and variable outputs

## Objective

Make split/child/merge alter computation, rather than remain auxiliary heads.

## Required work

- Define bounded child-slot allocation and alive masks in the state transition.
- Make `split_count` and `child_gate` create distinct child states.
- Make `merge_gate` collect those states into the declared collector.
- Replace the current branch config with a real split -> two transforms -> merge
  formula and expected evaluation program.
- Keep tensor sizes bounded and report active children/collectors.

## Acceptance

Learned accuracy exceeds chance in 3 seeds; deleting either child or merge lowers
accuracy on paired data; branch heads receive task gradients; no PASS from branch
auxiliary loss alone; program report shows actual split and merge actions.
