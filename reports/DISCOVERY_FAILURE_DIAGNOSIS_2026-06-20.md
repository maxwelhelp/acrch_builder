# Discovery failure diagnosis — chain_diff_product

## Observed failure

Ten epochs remain at binary chance (`~0.50`, CE `~0.693`). Runtime write is not
dead: it grows, and all 16 cells are active. The old `gateR/C/S` fields were
adaptive regularizer gates, not runtime gates.

## Confirmed implementation defects

1. `top_k=25` previously scanned only 19 proposals with duplicates. Required
   primitives could be absent. Full-scan now exposes all 25 exactly once.
2. Discovery used noisy Gumbel choices and annealed before task credit existed.
   It now uses a high-temperature differentiable softmax.
3. No stable cell-to-primitive parameter existed; position logits referred to a
   changing proposal order. A primitive-identity pair bias now exists.
4. Generic sparsity was disabled and expected-action adaptive gates reduced it
   to zero. Discovery now has direct oracle-free alive, coverage, stable-topology
   and tail/concentration losses.
5. Reusing one output directory appended incompatible CSV schemas. Discovery
   runs now default to timestamped directories.

## Remaining mathematical blocker

The target is

`sign(mean((s0-s1)*(s2-s3)))`.

Each individual Layer0 difference has approximately zero first-order correlation
with the label. Credit appears only for a joint intervention containing both
differences and a later product. CE, single-cell ablation, and an independent
single-candidate critic therefore cannot distinguish the useful Layer0 cells at
the symmetric/chance solution. Tests confirmed the single critic can inflate
expected recovery by choosing `diff`/`product` everywhere while its own CE and
model validation CE remain at `~0.693`; that approach was rejected.

## Required fix

Implement Task15/16 bounded hierarchical counterfactual credit with joint/pair
probes on fixed held-out microbatches:

- sample a bounded set of sparse Layer0 cell/action pairs plus Layer1 actions;
- compare paired full/intervened CE with identical data and RNG;
- store signed joint gain separately from individual gain;
- assign synergy credit only on a later optimizer event;
- reserve an independent random exploration budget;
- use the credit to update cell/primitive priors and simulator targets;
- accept only learned validation accuracy, never expected recovery alone.

Until that exists, longer CE-only runs are not expected to solve this task.
