# Task 24 — rebuild the final acceptance audit

## Objective

Generate release status from fresh evidence, never from stale result prose.

## Required work

- One canonical command must run validation and every Task15-23 proof.
- Encode every required plan rule as machine-readable PASS/FAIL/MISSING.
- Verify per-run artifacts, config/code digests, seeds, split hashes and commands.
- FAIL on missing metrics, stale timestamps, chance-level equality, collapse,
  negative required ablations or non-reproducible evidence.
- Regenerate final Markdown/JSON and status board from the machine result.

## Acceptance

Clean checkout reproduces all evidence; intentionally removing one required file
or metric makes final audit FAIL; all plan comparisons and 3-seed aggregates are
present; only then may `release_ready` become true.

Forbidden: manual PASS edits, caveat-based PASS, threshold weakening, or relying
on formula oracle accuracy as learned evidence.
