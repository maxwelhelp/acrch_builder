# acrch_builder probe contract v1 GITFIXED

Проверено под текущий GitHub API `arch_builder/train_vertical_slice.py`:
- `summarize_trace(trace, batch, model)`
- `proof_slice_structure_losses(...)`
- `generic_anti_collapse_losses(...)`
- `sim_targets_for_expected_actions(...)`

## Важно

Основные файлы проекта менять не надо.
Этот архив добавляет только inspector/probe файлы и команды.

## Запуск

```bash
cd ~/test/sience/experiments/math_search/WORKING_BEST/acrch_builder && \
unzip -o ~/Загрузки/acrch_builder_probe_contract_v1_gitfixed.zip -d . && \
chmod +x commands/*.sh && \
bash commands/inspect_with_probe.sh && \
xdg-open reports/agent_inspector/static_dashboard.html
```

## Агенту давать

```text
reports/agent_inspector/AGENT_CONTEXT.md
reports/agent_inspector/SUMMARY.md
reports/agent_inspector/runtime_gradient_graph.json
reports/agent_inspector/runtime_credit_graph.json
reports/agent_inspector/runtime_probe.json
reports/agent_inspector/static_graph.json
.inspector.json
docs/RUNTIME_PROBE_CONTRACT.md
```
