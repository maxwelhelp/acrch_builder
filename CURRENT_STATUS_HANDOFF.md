# CURRENT_STATUS_HANDOFF: Universal Adaptive Layer Development

Этот документ содержит контекст и инструкции для продолжения разработки и стабилизации vNext в ветке `vnext_utility_critic_diagnostic`.

---

## Текущий статус

Мы полностью закрыли **Phase 3 (Steps 9-12)**, всю **Phase 4 (Steps 13-15)**, а также первые два шага **Phase 5 (Steps 16-17)** из роудмапа [ARCHITECTURE_ROADMAP_UNIVERSAL_ADAPTIVE_LAYER.md](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/reports/ARCHITECTURE_ROADMAP_UNIVERSAL_ADAPTIVE_LAYER.md).

Все тесты и валидация (`validate.sh`, а также все старые и новые пробы) успешно проходят с кодом 0. Все изменения закоммичены в git в текущую ветку `vnext_utility_critic_diagnostic`.

---

## Что сделано в этой сессии

1. **Category-Aware Scanner (Phase 3, Step 9)**:
   - Добавлен CLI параметр `--utility-category-k` (по умолчанию 1) в оба тренера.
   - Реализован метод `category_topk` в `PrimitiveMatrix5x5` ([primitive_matrix.py](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/arch_builder/primitive_matrix.py)), выбирающий топ-$K$ примитивов для каждого семейства (строки сетки) на основе feedback bias.
   - Метод интегрирован в `HybridScanner` ([hybrid_scanner.py](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/arch_builder/hybrid_scanner.py)) и проброшен через слои и модель.
   - Написана и запущена проба [`probe_category_scanner.py`](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/tools/project_probe/probe_category_scanner.py), подтверждающая корректность выбора при cold-start (usage_score) и warm-start (gain/regret EMA).
   - Успешно проведен SpeechCommands smoke test с параметром `--utility-category-k 2`.

2. **Spectral Primitives Probe (Phase 3, Step 10)**:
   - Написана проба [`probe_spectral_primitives.py`](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/tools/project_probe/probe_spectral_primitives.py), проверяющая примитивы `dct`, `fft_filter`, `wavelet`, `spectral_mix`, `spectral_gate`.
   - Проверена корректность выходных размерностей, прохождение градиентов без NaN/Inf и математическая логика (консервация нормы вейвлета, сжатие DCT, обнуление фильтра FFT).

3. **Attention-Like Primitives Probe (Phase 3, Step 11)**:
   - Написана проба [`probe_attention_primitives.py`](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/tools/project_probe/probe_attention_primitives.py), проверяющая примитивы `qkv_gate`, `cross_attend`, `self_attend`, `key_align`, `value_mix`.
   - Проверены размерности, градиентный поток к входам и проекциям (`q_proj`, `k_proj`, `v_proj`) и математическая логика (смешивание и выравнивание косинусного расстояния).

4. **Auto-Mined Atoms Probe (Phase 3, Step 12)**:
   - Написана проба [`probe_mined_atoms.py`](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/tools/project_probe/probe_mined_atoms.py), проверяющая `svd_atom_k`, `diag`, `toeplitz`, `block_mean`, `mined_gate`.
   - Подтверждена передача градиентов на параметры авто-майнинга (`mined_u`, `mined_v`, `mined_diag` и др.) и математическая корректность.

5. **Gradient Trace Credit Proxy Probe (Phase 4, Step 13)**:
   - Написана проба [`probe_gradient_trace_credit.py`](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/tools/project_probe/probe_gradient_trace_credit.py).
   - Подтверждена работоспособность backward hook для `Online Gradient Trace Credit Hook`. Очередь `grad_credit_queue` успешно наполняется ненулевыми конечными значениями градиентного следа.

6. **Rank-Based Critic Loss Probe (Phase 4, Step 14)**:
   - Написана проба [`probe_rank_based_loss.py`](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/tools/project_probe/probe_rank_based_loss.py), проверяющая `pairwise_ranking_loss` из `credit.py`.
   - Проверена сходимость градиентов в правильном направлении, поведение на 1D/2D тензорах и все краевые случаи (равные таргеты, малая длина последовательности).

7. **Shapley-Lite Joint Credit Probe (Phase 4, Step 15)**:
   - Написана проба [`probe_shapley_lite.py`](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/tools/project_probe/probe_shapley_lite.py), проверяющая механизм Shapley-lite совместного кредикования в `BoundedCounterfactualCredit`.
   - Проверена корректность вычисления синергии группы при маскировании и равное распределение этой синергии между примитивами-участниками.

8. **CIFAR-10 Patch Classifier + Test (Phase 5, Step 16)**:
   - Создан тренировочный скрипт `arch_builder/train_cifar10.py` и диагностическая проба `tools/project_probe/probe_cifar10.py`.
   - Добавлен по-патчевый фронтенд (нарезка на 16 патчей 8х8), проецирование в скрытую размерность и классификация через встроенную голову `ActionMatrixModel`.

9. **Copy/Reverse Memory Task + Test (Phase 5, Step 17)**:
   - В `ActionMatrixModel.forward` экспортировано состояние финального слоя слоев памяти `"slots_out": state` в возвращаемый словарь `choice_info`.
   - Создан тренировочный скрипт `arch_builder/train_memory.py` и диагностическая проба `tools/project_probe/probe_copy_reverse.py`.
   - Модель обучается на оперирование памятью (копирование и разворот последовательности токенов) с использованием MSE лосса на целевых ячейках памяти.

---

## Что нужно делать дальше

Следующие шаги по роудмапу — **Phase 5: Universality (steps 18-20)**.

1. **Non-stationary adaptation test (Step 18)**:
   - Проверить способность критиков и сканера быстро адаптироваться при изменении распределения данных или принудительном отключении/замене примитивов в процессе работы.
   - Написать диагностическую пробу `tools/project_probe/probe_non_stationary.py`.
2. **Multi-layer composition credit test (Step 19)**.
3. **Full acceptance audit across all tasks (Step 20)**.

---

## Полезные команды

- Запуск всех тестов:
  ```bash
  bash commands/validate.sh
  ```
- Запуск новых проб:
  ```bash
  PYTHONPATH=. python tools/project_probe/probe_category_scanner.py
  PYTHONPATH=. python tools/project_probe/probe_spectral_primitives.py
  PYTHONPATH=. python tools/project_probe/probe_attention_primitives.py
  PYTHONPATH=. python tools/project_probe/probe_mined_atoms.py
  PYTHONPATH=. python tools/project_probe/probe_gradient_trace_credit.py
  PYTHONPATH=. python tools/project_probe/probe_rank_based_loss.py
  PYTHONPATH=. python tools/project_probe/probe_shapley_lite.py
  PYTHONPATH=. python tools/project_probe/probe_cifar10.py
  PYTHONPATH=. python tools/project_probe/probe_copy_reverse.py
  ```
