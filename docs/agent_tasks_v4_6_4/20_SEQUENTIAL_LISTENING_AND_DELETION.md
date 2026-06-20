# Task 20 — honest sequential listening and physical deletion

## Objective

Close the internal Layer0 -> Layer1 path while preserving `final_read=last`.

## Required work

- Keep live previous action/activity/write context in teacher, audit and deploy.
- Prove scanner `before_proj` and Layer1 controller receive gradients from final CE
  through Layer0 choices; add explicit health metrics.
- Pair normal, state-ablate, context-ablate, output-ablate and physical-delete
  evaluations using the same batches and RNG.
- Keep the direct Layer0 output -> final bypass forbidden.

## Acceptance

Forced bypass probe remains PASS; Layer0 controller and scanner-before grad norms
are nonzero; context/state ablations have positive median effect in 3 seeds;
physical deletion is worse than learned full model; no primitive collapse.
