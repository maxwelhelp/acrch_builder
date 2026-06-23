# ИНСТРУКЦИИ ДЛЯ СЛЕДУЮЩЕГО АГЕНТА (NEXT AGENT INSTRUCTIONS)

Этот документ подготовлен для быстрой передачи контекста и начала работы следующего AI-агента над репозиторием `arch_builder` по ветке `vnext_utility_critic_diagnostic`.

---

## Текущий статус сессии (Active Program Search)

Мы успешно реализовали и интегрировали **Active Program Search Loop** (активный цикл поиска программ) во всей архитектуре vNext, устранили диагностические пропуски, развязали бюджеты совместных кредитов и добавили отслеживание динамики роутинга в реальном времени.

Все изменения закоммичены в git в текущую ветку `vnext_utility_critic_diagnostic`.

### Основные документы сессии:
- [Рабочий план реализации (implementation_plan.md)](file:///home/maxwelhelp/.gemini/antigravity/brain/1b481e63-9148-4409-ac6d-bfb22c4e3f62/implementation_plan.md)
- [Чек-лист выполненных задач (task.md)](file:///home/maxwelhelp/.gemini/antigravity/brain/1b481e63-9148-4409-ac6d-bfb22c4e3f62/task.md)
- [Итоговый отчет изменений (walkthrough.md)](file:///home/maxwelhelp/.gemini/antigravity/brain/1b481e63-9148-4409-ac6d-bfb22c4e3f62/walkthrough.md)
- [Динамика программы (PROGRAM_DYNAMICS.md)](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/agent_reports/vnext_smoke_seed1_1ep/PROGRAM_DYNAMICS.md) (создается автоматически во время тренировки в `--out-dir`).

---

## Что было сделано и проверено в этой сессии

1. **Step-Based Program Search Loop** (`arch_builder/program_search.py`):
   - Перевели логику `ProgramSearchState` на шаги (steps/windows) вместо эпох.
   - Шаг обновления задается через CLI-флаг `--program-search-update-every` (по умолчанию 50 шагов). Накопление статистики идет через скользящее среднее.
   - Добавлен автоматический триггер **Exploration Burst** при плато точности, доминировании одного примитива (collapse > 0.70) или затухании энтропии выбора (< 0.1).
   - В режиме burst временно повышается температура роутинга `tau_multiplier`, увеличивается количество случайных кандидатов `random_k_multiplier` и активируется `exploration_mode`.

2. **Развязка бюджетов Joint Credit** (`arch_builder/credit.py`):
   - Разделили бюджеты одиночных контрфактуальных замеров и совместных (pairwise/ Shapley-lite) замеров, чтобы исключить взаимное вытеснение при оценке кредитов.
   - Добавлены CLI-аргументы `--enable-joint-credit`, `--joint-credit-interval`, `--joint-credit-extra-budget`.

3. **Снапшоты роутинга и динамики** (`arch_builder/train_audio_frontend.py`):
   - Во время валидации собираются фактические пути прохождения тензоров (`trace`).
   - На основе собранных трейсов вычисляются:
     - **Route Jaccard similarity**: мера стабильности роутинга между эпохами.
     - **Cell Drift**: количество ячеек памяти, изменивших выбранный примитив или источник.
     - **Source Mix**: процентное распределение вкладов всех 9 источников (включая добавленный `exploration` источник).
   - Все эти показатели записываются в файл `PROGRAM_DYNAMICS.md` на каждой эпохе.
   - Снапшоты топологии программы сохраняются в JSON файлы вида `program_snapshot_epoch_{epoch}.json`.

4. **Exporter CLI `--run-dir`**:
   - Скрипт `tools/project_probe/export_program_report.py` теперь принимает параметр `--run-dir` для автоматического нахождения сохраненных чекпоинтов модели (`model_last.pt` / `model_best.pt`).

5. **Три новых диагностических зонда**:
   - `tools/project_probe/probe_feedback_memory_closure.py`: проверяет накопление EMA оценок, старение/forgetting и работу top-k по памяти обратной связи.
   - `tools/project_probe/probe_program_search_loop.py`: тестирует конечный автомат поиска программ, переходы в burst и возвращение к базовым параметрам.
   - `tools/project_probe/probe_program_export.py`: гарантирует корректность генерации и экспорта markdown- blueprint'а программы.

---

## Как запускать валидацию и тесты

1. **Полная валидация проекта** (запускает компиляцию, CLI-тесты и все 3 новых диагностических зонда):
   ```bash
   bash commands/validate.sh
   ```
   *Ожидаемый результат*: Скрипт должен вывести `[validate] OK` и завершиться с кодом 0.

2. **Запуск интеграционного smoke-теста (SpeechCommands)**:
   ```bash
   ENABLE_UTILITY_CRITIC_CHOICE=1 ENABLE_LAZY_EXECUTOR=1 ENABLE_SCANNER_FEEDBACK_MEMORY=1 ENABLE_CATEGORY_SCANNER=1 ENABLE_MMR_CONTROLLER=1 \
   bash commands/run_speechcommands_vnext_smoke.sh \
     --enable-program-search-loop \
     --program-search-update-every 10 \
     --program-plateau-windows 2 \
     --program-burst-duration 5 \
     --enable-joint-credit \
     --joint-credit-interval 10 \
     --joint-credit-extra-budget 2 \
     --train-limit 300 --val-limit 150 --test-limit 100 --epochs 2 --steps-per-epoch 50
   ```
   *Результат*: В папке `agent_reports/vnext_smoke_seed1_1ep/` создадутся файлы `final_report.json`, `PROGRAM_DYNAMICS.md`, а также JSON-снапшоты программ.

---

## Что делать дальше (Задание для следующего агента)

Ядро поискового цикла полностью стабилизировано и покрыто тестами. Для следующего шага рекомендуется:

1. **Запуск длинного эксперимента (Long Run) на GPU (Tesla P40)**:
   - Запустите SpeechCommands на 8-15 эпох с включенным Active Program Search, используя MCP-задания или фоновые команды.
   - Параметры запуска:
     ```bash
     ENABLE_VNEXT=1 ENABLE_UTILITY_CRITIC_CHOICE=1 ENABLE_MMR_CONTROLLER=1 ENABLE_LAZY_EXECUTOR=1 ENABLE_SCANNER_FEEDBACK_MEMORY=1 ENABLE_CATEGORY_SCANNER=1 \
     bash commands/run_speechcommands_real_discovery.sh \
       --enable-program-search-loop \
       --program-search-update-every 50 \
       --enable-joint-credit \
       --joint-credit-interval 25 \
       --joint-credit-extra-budget 2
     ```

2. **Мониторинг динамики и стабилизация**:
   - Анализируйте `PROGRAM_DYNAMICS.md` во время обучения:
     - Растет ли точность (`Val Acc`) и кристаллизация (`Crystallization`) после окончания exploration bursts?
     - Стабилизируется ли маршрутизация (повышается ли `Route Jaccard` до значений > 0.80 к концу обучения)?
     - Помогает ли `exploration` источник находить редкие эффективные примитивы в ячейках.

3. **Тюнинг Burst-параметров**:
   - Настройте `--program-search-update-every`, `patience`, `burst_duration` и мультипликаторы шума для максимизации качества обучения аудио-классификатора.
