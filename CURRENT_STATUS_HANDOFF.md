# CURRENT_STATUS_HANDOFF: Universal Adaptive Layer & Active Program Search

Этот документ содержит контекст и инструкции для продолжения разработки и стабилизации vNext в ветке `vnext_utility_critic_diagnostic`.

---

## Текущий статус

Мы полностью завершили интеграцию **Active Program Search Loop** (активного цикла поиска программ) и закрыли все сопутствующие задачи по роудмапу. 

Все тесты и валидация (`validate.sh`, а также все старые и новые диагностические зонды) успешно проходят с кодом 0.

---

## Что было сделано в этой сессии

1. **Step-Based Program Search Loop** (`arch_builder/program_search.py`, `arch_builder/train_audio_frontend.py`):
   - Перевели логику `ProgramSearchState` с эпох на шаги/окна (`--program-search-update-every`).
   - Реализовали триггеры **Exploration Bursts** при плато точности, доминировании примитивов или падении энтропии выбора.
   - Во время burst динамически изменяются `tau_multiplier`, `random_k_multiplier` и `exploration_mode`.

2. **Развязка бюджетов Joint Credit** (`arch_builder/credit.py`):
   - Разделили лимиты для одиночных и совместных кредитных замеров для предотвращения взаимного вытеснения.

3. **Снапшоты роутинга и динамики** (`arch_builder/train_audio_frontend.py`):
   - Реализовали накопление фактических путей маршрутизации в валидации.
   - Добавили вычисление и вывод в `PROGRAM_DYNAMICS.md` метрик:
     - **Route Jaccard similarity**: стабильность роутинга между эпохами.
     - **Cell Drift**: количество изменивших выбор ячеек.
     - **Source Mix**: процентное распределение вкладов всех 9 источников.
   - Реализовали сохранение топологии в `program_snapshot_epoch_{epoch}.json`.

4. **Три новых диагностических зонда**:
   - `tools/project_probe/probe_feedback_memory_closure.py` (контроль старения и top-k памяти обратной связи).
   - `tools/project_probe/probe_program_search_loop.py` (тестирование автомата поиска и burst режимов).
   - `tools/project_probe/probe_program_export.py` (контроль экспорта модели в markdown).

Все новые пробы успешно прописаны и выполняются в `commands/validate.sh`.

---

## Что нужно делать дальше

1. **Запуск длинного GPU-эксперимента** на Tesla P40 для замера качества сходимости при включенном поиске программ.
2. **Анализ стабильности роутинга**: отслеживание сходимости Jaccard-коэффициента к высоким значениям (например, > 0.85) по мере роста crystallization.
3. **Тюнинг гиперпараметров**: регулировка размера окна обновления и длительности burst.
