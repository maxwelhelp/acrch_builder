# Task 09 — Teacher/Audit/Deploy honesty curriculum

## Objective

Add honest scaffold curriculum and mode-specific credit without answer leakage.

## Preconditions

Task 08 PASS.

## Allowed files

```text
new curriculum module under arch_builder/
arch_builder/credit.py
arch_builder/model.py (mode plumbing only)
arch_builder/train_vertical_slice.py
arch_builder/reporting.py
commands/ honesty audits
docs/ and reports/agent_tasks/
```

## Required design

Modes: Teacher (meta-process scaffold), Audit (reduced scaffold), Deploy (none).
Hints may encode process/budget/mode only. Ban labels, expected primitive/edge,
answer-correlated features, operation choice, and next-layer weights. Decay by
honesty signals, never epoch.

Separate `credit_teacher`, `credit_audit`, `credit_deploy`. Deploy writes only
above explicit honesty floor; Teacher credit never targets Deploy simulator.

## Required audits

```text
teacher, no_conv, no_hints, deploy
shuffled hints, random hints, transfer structure
```

## Acceptance

```text
no static/runtime leakage
random/shuffled hints do not beat honest deploy
deploy credit contains deploy observations only
honesty_score >= 0.80 (preferred >=0.90)
deploy retains readable program/diversity
```

Deploy random means FAIL regardless of Teacher accuracy.

