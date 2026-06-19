# Task 06 — HybridScanner semantic and usage proof

## Objective

Prove non-grid sources rescue useful candidates under a real cutoff; usage must
come from real credit, not a hardcoded primitive list.

## Preconditions

Tasks 04–05 PASS.

## Allowed files

```text
arch_builder/primitive_matrix.py
arch_builder/hybrid_scanner.py
arch_builder/model.py (proposal plumbing only)
arch_builder/synthetic_tasks.py
arch_builder/train_vertical_slice.py
arch_builder/reporting.py
commands/ scanner proofs
docs/ and reports/agent_tasks/
```

## Required work

- Replace stable usage list with delayed credit/usage ranking; cold start may be
  uniform/random only.
- Preserve source IDs and actual source choice mass.
- Build semantic rescue where correct primitive is outside grid and
  `top_k < total candidates`.
- Paired comparisons: grid; grid+semantic; +usage; full hybrid+random.
- Report semantic entropy/rank/mismatch, global rescue rate, semantic candidate
  real gain/quality, source masses/gains, collapse flag.

## Acceptance

```text
correct primitive absent grid-only by construction
present through non-grid source under cutoff
global_rescue_rate > 0
non-grid held-out real gain > 0
full hybrid recovery > grid-only
embedding rank/entropy healthy
random mass nonzero and bounded
```

Candidate presence alone is not success.

