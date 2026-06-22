# ИНСТРУКЦИИ ДЛЯ СЛЕДУЮЩЕГО АГЕНТА (NEXT AGENT INSTRUCTIONS)

Этот документ подготовлен для быстрой передачи контекста и начала работы следующего AI-агента над репозиторием `arch_builder`.

---

## Текущий статус

1. **Роудмап выполнен на 100%**:
   Все 5 фаз (20 шагов) роудмапа [ARCHITECTURE_ROADMAP_UNIVERSAL_ADAPTIVE_LAYER.md](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/reports/ARCHITECTURE_ROADMAP_UNIVERSAL_ADAPTIVE_LAYER.md) полностью завершены, интегрированы и покрыты тестами.
   
2. **Ветка разработки**: `vnext_utility_critic_diagnostic`.
   Все измененные файлы и новые пробы закоммичены в git.

3. **Грязные файлы (Dirty-file policy)**:
   Файлы `arch_builder/credit.py`, `arch_builder/utility_critic.py`, `arch_builder/vnext_controller.py` содержат стабильную логику vNext и не должны перезаписываться или форматироваться без явного согласования.

---

## Что было сделано и проверено

Мы успешно реализовали, интегрировали и протестировали все 11 ключевых нововведений vNext через соответствующие пробы:

1. **Category-Aware Scanner (Шаг 9)**:
   - Выбирает топ-$K$ примитивов на семейство по feedback bias.
   - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_category_scanner.py`
   
2. **Spectral Primitives (Шаг 10)**:
   - Интегрированы спектральные примитивы (`dct`, `fft_filter`, `wavelet`, `spectral_mix`, `spectral_gate`).
   - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_spectral_primitives.py`
   
3. **Attention-Like Primitives (Шаг 11)**:
   - Интегрированы примитивы внимания (`qkv_gate`, `cross_attend`, `self_attend`, `key_align`, `value_mix`).
   - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_attention_primitives.py`
   
4. **Auto-Mined Atoms (Шаг 12)**:
   - Интегрированы примитивы автоматического извлечения признаков (`svd_atom_k`, `diag`, `toeplitz`, `block_mean`, `mined_gate`).
   - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_mined_atoms.py`

5. **Gradient Trace Credit Hook (Шаг 13)**:
   - Проверена работа backward hook для `Online Gradient Trace Credit Hook`.
   - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_gradient_trace_credit.py`

6. **Rank-Based Critic Loss (Шаг 14)**:
   - Интегрирован попарный ранговый лосс (`pairwise_ranking_loss`) для обучения критика.
   - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_rank_based_loss.py`

7. **Shapley-Lite Joint Credit (Шаг 15)**:
   - Интегрировано Shapley-lite распределение групповой синергии в `BoundedCounterfactualCredit`.
   - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_shapley_lite.py`

8. **CIFAR-10 Patch Classifier + Test (Шаг 16)**:
   - Добавлена нарезка изображений на 16 патчей 8х8 и классификация через `ActionMatrixModel`.
   - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_cifar10.py`

9. **Copy/Reverse Memory Task + Test (Шаг 17)**:
   - Внедрен экспорт финального состояния памяти через `"slots_out": state` в `ActionMatrixModel.forward`.
   - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_copy_reverse.py`

10. **Non-Stationary Adaptation Test (Шаг 18)**:
    - Проверена адаптация модели к принудительному отключению (абляции) примитива `diff` во время обучения.
    - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_non_stationary.py`

11. **Multi-layer Composition Credit Test (Шаг 19)**:
    - Проверено распределение кредита по цепочке слоев (Layer 0 -> Layer 1) при решении составной математической задачи.
    - Тестирование: `PYTHONPATH=. python tools/project_probe/probe_multi_layer_composition.py`

---

## Как запускать валидацию и тесты

- **Полная проверка проекта (без регрессий)**:
  ```bash
  bash commands/validate.sh
  ```
  *Критерий успешности*: Скрипт должен завершаться с `[validate] OK` и кодом 0.

- **Запуск всех 11 диагностических проб**:
  ```bash
  PYTHONPATH=. python tools/project_probe/probe_category_scanner.py && \
  PYTHONPATH=. python tools/project_probe/probe_spectral_primitives.py && \
  PYTHONPATH=. python tools/project_probe/probe_attention_primitives.py && \
  PYTHONPATH=. python tools/project_probe/probe_mined_atoms.py && \
  PYTHONPATH=. python tools/project_probe/probe_gradient_trace_credit.py && \
  PYTHONPATH=. python tools/project_probe/probe_rank_based_loss.py && \
  PYTHONPATH=. python tools/project_probe/probe_shapley_lite.py && \
  PYTHONPATH=. python tools/project_probe/probe_cifar10.py && \
  PYTHONPATH=. python tools/project_probe/probe_copy_reverse.py && \
  PYTHONPATH=. python tools/project_probe/probe_non_stationary.py && \
  PYTHONPATH=. python tools/project_probe/probe_multi_layer_composition.py
  ```

- **SpeechCommands интеграционный smoke test на CPU**:
  ```bash
  STEPS_PER_EPOCH=10 TRAIN_LIMIT=1000 VAL_LIMIT=256 TEST_LIMIT=256 WORKERS=0 SLOTS=4 LAYERS=1 EPOCHS=2 \
  python -m arch_builder.train_audio_frontend --epochs 2 --steps-per-epoch 10 --eval-steps 2 --eval-batch-size 32 \
  --batch-size 32 --out-dir agent_reports/smoke_vnext_full_audit --device cpu --enable-vnext --enable-lazy-executor \
  --enable-category-scanner --utility-category-k 2
  ```

---

## Что нужно делать дальше (План для следующего агента)

Ядро и все фазы роудмапа полностью стабилизированы. Рекомендуется перейти к этапу **эксплуатации и тюнинга гиперпараметров**:

1. **Long Run на GPU (Tesla P40)**:
   Запустить полноценное обучение SpeechCommands (например, 20-50 эпох) с поддержкой CUDA на Tesla P40 для замера финального качества модели и сравнения с baseline.
   
2. **Sweep по параметрам vNext**:
   - Попробовать различный размер пула категорий `--utility-category-k` (1, 2, 3).
   - Исследовать влияние `enable_lazy_executor` на скорость обучения на Pascal P40.
   - Сравнить работу рангового лосса критика (`pairwise_ranking_loss`) с классическим MSE.

3. **Анализ отчетов**:
   Используйте скрипт `commands/build_focused_report.sh` или анализируйте логи в `agent_reports/` для оценки метрик сходимости, энтропии выбора примитивов (`choice_entropy`) и качества кредитования (`expected_edge_recovery`).
