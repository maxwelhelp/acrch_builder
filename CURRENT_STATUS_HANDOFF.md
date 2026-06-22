# CURRENT_STATUS_HANDOFF: Universal Adaptive Layer Development

Этот документ содержит контекст и инструкции для продолжения разработки и стабилизации vNext в ветке `vnext_utility_critic_diagnostic`.

---

## Текущий статус

Мы полностью закрыли **Phase 3 (Steps 9-12)** и всю **Phase 4 (Steps 13-15)** из роудмапа [ARCHITECTURE_ROADMAP_UNIVERSAL_ADAPTIVE_LAYER.md](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/reports/ARCHITECTURE_ROADMAP_UNIVERSAL_ADAPTIVE_LAYER.md).

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
   - Написана проба [`probe_shapley_lite.py`](file:///home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/tools/project_probe/probe_shapley_lite.py), проверяющая механизм Shapley-lite совместного кредитования в `BoundedCounterfactualCredit`.
   - Проверена корректность вычисления синергии группы при маскировании и равное распределение этой синергии между примитивами-участниками.

---

## Что нужно делать дальше

Следующие шаги по роудмапу — **Phase 5: Universality (steps 16-20)**.

1. **CIFAR-10 patch frontend + acceptance test (Step 16)**:
   - Создать загрузчик CIFAR-10 патчей и входной фронтенд для классификатора на базе ActionMatrixModel.
   - Проверить сходимость и точность классификации (критерий: `accuracy > 20%`).
2. **Copy/reverse memory task + acceptance test (Step 17)**:
   - Проверить способность модели читать/писать в слоты памяти на задачах копирования и разворота последовательностей (критерий: `accuracy > 90%`).
3. **Non-stationary adaptation test (Step 18)**.
4. **Multi-layer composition credit test (Step 19)**.
5. **Full acceptance audit across all tasks (Step 20)**.

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
  ```
- 20-step SpeechCommands smoke-тест с категориальным сканером:
  ```bash
  STEPS_PER_EPOCH=20 TRAIN_LIMIT=1000 VAL_LIMIT=256 TEST_LIMIT=256 WORKERS=0 SLOTS=4 LAYERS=2 EPOCHS=2 python -m arch_builder.train_audio_frontend --epochs 2 --steps-per-epoch 20 --eval-steps 5 --eval-batch-size 64 --batch-size 64 --out-dir agent_reports/smoke_vnext_category --device cpu --enable-vnext --enable-lazy-executor --enable-category-scanner --utility-category-k 2
  ```
