# acrch_builder inspector v2: runtime probe + focus requests

Добавлено:

```text
reports/agent_inspector/runtime_gradient_graph.json
reports/agent_inspector/runtime_credit_graph.json
reports/agent_inspector/runtime_graph_summary.md
reports/agent_inspector/runtime_probe.json
reports/agent_inspector/focused_report.md
inspector_request.yml
```

## 1) Обычный анализ без обучения

```bash
cd ~/test/sience/experiments/math_search/WORKING_BEST/acrch_builder && \
unzip -o ~/Загрузки/acrch_builder_inspector_probe_v2.zip -d . && \
chmod +x commands/*.sh && \
bash commands/inspect_all.sh && \
xdg-open reports/agent_inspector/static_dashboard.html
```

## 2) Анализ + маленький runtime-probe с backward/grad_norm

Это НЕ большой train. Это один маленький batch, чтобы доказать, куда идёт градиент.

```bash
cd ~/test/sience/experiments/math_search/WORKING_BEST/acrch_builder && \
bash commands/inspect_with_probe.sh
```

## 3) Агент просит сфокусированный анализ

Агент/ты редактируешь:

```text
inspector_request.yml
```

Потом:

```bash
bash commands/build_focused_report.sh
```

Получишь:

```text
reports/agent_inspector/focused_report.md
```

## Агенту давать

```text
reports/agent_inspector/AGENT_CONTEXT.md
reports/agent_inspector/holes_report.md
reports/agent_inspector/runtime_summary.md
reports/agent_inspector/runtime_graph_summary.md
reports/agent_inspector/runtime_gradient_graph.json
reports/agent_inspector/runtime_credit_graph.json
reports/agent_inspector/focused_report.md
reports/agent_inspector/static_graph.json
.inspector.yml
inspector_request.yml
```

## Универсальность

Для другого проекта меняешь `.inspector.yml`:

```text
runtime_probe.adapter
runtime_probe.command или свой adapter
runtime_graphs.gradient_graph.groups
runtime_graphs.credit_graph.stages
closed_loops
roles
```
