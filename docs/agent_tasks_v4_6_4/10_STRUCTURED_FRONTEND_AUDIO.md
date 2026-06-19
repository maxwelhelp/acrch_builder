# Task 10 — structured frontend and audio wrapper

## Objective

Connect raw/basic audio without hiding the task in Conv.

## Preconditions

Task 09 Deploy PASS.

## Allowed files

```text
new domain wrappers under arch_builder/
audio dataset/training entrypoint
reporting and commands
generic adapter interfaces only in core
docs/ and reports/agent_tasks/
```

## Required comparisons

```text
temporary Conv scaffold
no-conv raw/basic baseline
structured matrix frontend
```

Structured operations may use frames/windows, log-energy, fixed DCT-like
transforms, delta/onset, and learned projections; never labels/actions. Keep the
accepted core shared and domain behavior in wrappers.

## Acceptance

```text
all variants reported with params/FLOPs/memory
structured frontend reproducibly beats raw failed baseline
Deploy above random with honesty retained
core program/credit/scanner/simulator acceptance retained
Conv explicitly scaffold-only
```

