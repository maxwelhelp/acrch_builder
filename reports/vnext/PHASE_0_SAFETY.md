# Phase 0 — Safety Infrastructure Report

This document describes the safety infrastructure, validation scripts, comparison tools, and rollback mechanisms established for the vNext architecture development branch `vnext_utility_critic_diagnostic`.

## 1. Команды запуска и тестирования

В рамках Phase 0 были созданы и интегрированы следующие команды (находятся в директории `commands/` и `tools/`):

1. **`commands/run_speechcommands_vnext_smoke.sh`**
   - Запускает быстрый проверочный проход (smoke test) длительностью в 1 эпоху (200 шагов обучения) с включенной vNext-диагностикой (`ENABLE_VNEXT=1` и `ENABLE_UTILITY_CRITIC_PROBE=1`).
   - Используется для проверки отсутствия критических ошибок выполнения и падений памяти на Tesla P40.
   
2. **`commands/run_speechcommands_vnext_3seed.sh`**
   - Запускает полное 3-seed сравнение на датасете SpeechCommands.
   - По завершении всех трех запусков автоматически вызывает скрипт сравнения для агрегации метрик.

3. **`tools/project_probe/compare_vnext_to_baseline.py`**
   - Скрипт сравнения, который считывает один или несколько файлов `final_report.json` и рассчитывает среднее/стандартное отклонение, после чего строит таблицу сравнения с историческим baseline-тегом проекта.

---

## 2. Сравнение с Baseline

Зафиксированный baseline (`baseline_real_discovery_3seed_acc0592_20260621`) имеет следующие значения:

- **Accuracy (validation) mean**: `0.5919`
- **Test accuracy mean**: `0.5794`
- **Speed (samples/sec) mean**: `136.3`
- **sim_delta (LowRankSimulator CE ablation delta)**: `+0.3193`
- **choice_sim (choice change without simulator)**: `0.0407`
- **primitive_top_share (Gini-dominance proxy)**: `0.4799`
- **credit_closed**: `1.0` (все кредитные интервалы закрыты)

Скрипт сравнения автоматически вычисляет разницу (Delta) по каждому из этих показателей и выводит в консоль наглядную таблицу.

---

## 3. Флаги Rollback и vNext

Для обеспечения безопасности и поэтапного включения логики vNext используются следующие переменные окружения (флаги):

* `ENABLE_VNEXT=0` (по умолчанию `0` — полностью отключает все изменения vNext и гарантирует выполнение оригинального baseline-кода).
* `ENABLE_UTILITY_CRITIC_PROBE=0` (включает/выключает расчет диагностических метрик нового `UtilityCritic` без влияния на логику принятия решений `choice_logits`).
* `ENABLE_UTILITY_CRITIC_CHOICE=0` (включает влияние `UtilityCritic` на `choice_logits`. Должен быть равен `0` в фазе 1).

Если при запуске с `ENABLE_VNEXT=1` точность падает или возникают аномалии, сброс всех флагов в `0` гарантирует 100% возврат к исходному поведению.
