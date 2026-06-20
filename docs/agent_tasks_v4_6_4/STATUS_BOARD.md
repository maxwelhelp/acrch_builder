# v4.6.4 agent status board

Agents update only their row after writing the required result artifact.

| Task | Status | Result artifact | Next allowed |
|---:|---|---|---|
| 01 | PASS | `reports/agent_tasks/TASK_01_RESULT.md` | yes |
| 02 | PASS | `reports/agent_tasks/TASK_02_RESULT.md` | yes |
| 03 | PASS | `reports/agent_tasks/TASK_03_RESULT.md` | yes |
| 04 | FAIL | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | no: real ablation credit missing |
| 05 | PASS | `reports/agent_tasks/TASK_05_RESULT.md` | yes |
| 06 | FAIL | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | no: simulator real-gain closure missing |
| 07 | FAIL | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | no: collapse and unpaired ablations |
| 08 | FAIL | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | no: branching heads are decorative |
| 09 | FAIL | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | no: chance-level honesty can pass |
| 10 | FAIL | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | no: proof expects obsolete caveat |
| 11 | FAIL | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | no: real-data evidence insufficient |
| 12 | FAIL | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | no: plugin does not reuse core |
| 13 | FAIL | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | no: leakage/cache checks invalid |
| 14 | FAIL | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | no: final PASS was not reproducible |
| 15 | IN_PROGRESS | `reports/PLAN_CONFORMANCE_AUDIT_2026-06-20.md` | yes |
| 16 | PENDING | — | after 15 PASS |
| 17 | PENDING | — | after 16 PASS |
| 18 | PENDING | — | after 17 PASS |
| 19 | PENDING | — | after 18 PASS |
| 20 | PENDING | — | after 19 PASS |
| 21 | PENDING | — | after 20 PASS |
| 22 | PENDING | — | after 21 PASS |
| 23 | PENDING | — | after 22 PASS |
| 24 | PENDING | — | after 23 PASS |

Allowed statuses: `PENDING`, `IN_PROGRESS`, `PASS`, `FAIL`, `BLOCKED`.

Only task 01 is initially allowed. A later task changes `Next allowed` to `yes`
only after the prior task is PASS.
