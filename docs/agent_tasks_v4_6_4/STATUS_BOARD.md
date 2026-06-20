# v4.6.4 agent status board

Agents update only their row after writing the required result artifact.

| Task | Status | Result artifact | Next allowed |
|---:|---|---|---|
| 01 | PASS | `reports/agent_tasks/TASK_01_RESULT.md` | yes |
| 02 | PASS | `reports/agent_tasks/TASK_02_RESULT.md` | yes |
| 03 | PASS | `reports/agent_tasks/TASK_03_RESULT.md` | yes |
| 04 | PASS | `reports/agent_tasks/TASK_04_RESULT.md` | yes |
| 05 | PASS | `reports/agent_tasks/TASK_05_RESULT.md` | yes |
| 06 | PASS | `reports/agent_tasks/TASK_06_RESULT.md` | yes |
| 07 | PASS | `reports/agent_tasks/TASK_07_RESULT.md` | yes |
| 08 | PASS | `reports/agent_tasks/TASK_08_RESULT.md` | yes |
| 09 | PASS | `reports/agent_tasks/TASK_09_RESULT.md` | yes |
| 10 | PASS | `reports/agent_tasks/TASK_10_RESULT.md` | yes |
| 11 | PASS | `reports/agent_tasks/TASK_11_RESULT.md` | yes |
| 12 | PASS | `reports/agent_tasks/TASK_12_RESULT.md` | yes |
| 13 | PASS | `reports/agent_tasks/TASK_13_RESULT.md` | yes |
| 14 | PASS | `reports/agent_tasks/TASK_14_RESULT.md` | no |

Allowed statuses: `PENDING`, `IN_PROGRESS`, `PASS`, `FAIL`, `BLOCKED`.

Only task 01 is initially allowed. A later task changes `Next allowed` to `yes`
only after the prior task is PASS.
