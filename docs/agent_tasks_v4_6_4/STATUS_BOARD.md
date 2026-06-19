# v4.6.4 agent status board

Agents update only their row after writing the required result artifact.

| Task | Status | Result artifact | Next allowed |
|---:|---|---|---|
| 01 | PENDING | `reports/agent_tasks/TASK_01_RESULT.md` | no |
| 02 | PENDING | `reports/agent_tasks/TASK_02_RESULT.md` | no |
| 03 | PENDING | `reports/agent_tasks/TASK_03_RESULT.md` | no |
| 04 | PENDING | `reports/agent_tasks/TASK_04_RESULT.md` | no |
| 05 | PENDING | `reports/agent_tasks/TASK_05_RESULT.md` | no |
| 06 | PENDING | `reports/agent_tasks/TASK_06_RESULT.md` | no |
| 07 | PENDING | `reports/agent_tasks/TASK_07_RESULT.md` | no |
| 08 | PENDING | `reports/agent_tasks/TASK_08_RESULT.md` | no |
| 09 | PENDING | `reports/agent_tasks/TASK_09_RESULT.md` | no |
| 10 | PENDING | `reports/agent_tasks/TASK_10_RESULT.md` | no |
| 11 | PENDING | `reports/agent_tasks/TASK_11_RESULT.md` | no |
| 12 | PENDING | `reports/agent_tasks/TASK_12_RESULT.md` | no |
| 13 | PENDING | `reports/agent_tasks/TASK_13_RESULT.md` | no |
| 14 | PENDING | `reports/agent_tasks/TASK_14_RESULT.md` | no |

Allowed statuses: `PENDING`, `IN_PROGRESS`, `PASS`, `FAIL`, `BLOCKED`.

Only task 01 is initially allowed. A later task changes `Next allowed` to `yes`
only after the prior task is PASS.

