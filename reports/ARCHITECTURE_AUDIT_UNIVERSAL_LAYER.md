# Архитектурный Аудит: Universal Layer и Causal Credit

## 1. Универсальность vs Task-Specific
**Универсальное:** Ядро `ActionMatrixLayer` (маршрутизация, soft-слоты, scanner, executor) уже абстрагировано от типа данных. Оно оперирует тензорами `[batch, slots, dim]`.
**Task-Specific:** Сейчас сильная привязка осталась в `AudioMatrixClassifier` (frontend, pooling, final_read="last"). Сама `PrimitiveMatrix5x5` содержит хардкодные базовые операции, которые могут быть недостаточно выразительны для сложных задач (например, нет attention/routing примитивов внутри самого банка).

## 2. Скрытые priors и shortcuts
*   **Инициализация гейтов (`_init_gate_priors`):** Смещение `mode_head.bias[0] = 0.6` дает сильный начальный prior на трансформацию, а не на skip-connection.
*   **Аддитивная логит-модель:** `choice_logits` суммирует `context_component`, `gain_component`, `sim_component` и `proposal_component`. Если один источник (например, single projection) выдает логит `+10`, он полностью подавляет остальные, ломая градиенты через Softmax.
*   **Diversity penalty:** Работает как shortcut, заставляя модель выбирать мусорные примитивы просто для поддержания энтропии, что снижает capacity.

## 3. Bottlenecks на Tesla P40
*   **Вычисление всех примитивов до выбора:** Если `top_k` большой или используется `all_primitive_effects`, вычисляется много лишней математики.
*   **Pair JL16 Scanner:** Даже с JL-проекцией поиск пар требует $O(\text{slots}^2 \times \text{primitives}^2)$ операций.
*   **Empirical Counterfactuals:** Измерение credit требует дополнительных forward/backward проходов или вычисления альтернативных веток на графе, что удваивает/утраивает вычислительную стоимость шага.

## 4. Реальное влияние vs Декоративность
*   **Реальное влияние:** `single_signed_projection` (очень агрессивный сигнал), `context_component` (запоминает выигрышные пути).
*   **Декоративность:** `simulator` (если `predicted_gain` болтается около 0 из-за зашумленного credit), `gain_component` (часто имеет слишком маленький scale по сравнению с `proposal_component`). `pair_jl16` в реальном train тоже оказался декоративным.

## 5. Почему single projection дал collapse?
Проекция ищет линейную корреляцию действия с псевдо-лейблом (градиентом). Если один примитив (например, `diff`) имеет высокую корреляцию на первых батчах, он получает огромный score. Это вызывает Matthew Effect (богатый богатеет): примитив часто выбирается $\rightarrow$ получает весь credit $\rightarrow$ его логит растет $\rightarrow$ exploration прекращается.

## 6. Почему diversity лечит collapse, но съедает качество?
Diversity (особенно hard) штрафует уверенность. Чтобы удовлетворить `primitive_entropy_floor`, роутер вынужден отдавать часть вероятностной массы на бесполезные примитивы (например, `min` или `smooth` там, где нужен `diff`). Это "размывает" фичи, проходящие через слои, и снижает итоговую точность классификатора.

## 7. JL16: Полезен в probe, бесполезен в train
В `probe` синтетические данные имеют идеальную структуру (например, точное умножение двух разниц). Сигнал чистый. В SpeechCommands градиент от классификатора предельно зашумлен (stochastic mini-batches, complex audio features). Умножение двух зашумленных векторов (pair interaction) возводит дисперсию шума в квадрат, и JL-проекция этот шум не спасает. Сигнал теряется.

## 8. Проблемы Credit Assignment (поздно и шумно)
Сейчас Credit измеряет глобальную полезность примитива (математическое ожидание) на небольшом counterfactual-батче.
*   **Поздно:** Обновляется с задержкой, не зависит от текущего sample.
*   **Шумно:** Оценка выигрыша от одного примитива на layer0 по отношению к Cross-Entropy loss на layer2 утопает в дисперсии данных.

## 9. Роль Simulator
Симулятор полезен только если он может предсказать *sample-specific* эффект лучше, чем глобальный `context_component`. Так как он тренируется на зашумленный глобальный `credit_gain`, он быстро выучивает предсказывать среднее (почти 0) и становится декоративным. Diagnostic Self-Delta доказывает, что в разнице `actual - sim` кроется реальный локальный сигнал.

## 10. Каких метрик не хватает
1.  **Credit SNR (Signal-to-Noise Ratio):** Отношение среднего credit gain к его стандартному отклонению (понять, есть ли вообще сигнал).
2.  **Gradient Norm per Source:** Какой из источников (grid, semantic, projection) реально получает градиенты и учится.
3.  **Active Path Variance:** Насколько сильно меняется маршрут от семпла к семплу (если дисперсия 0 — модель деградировала в статический граф).

## 11. Альтернатива дорогим counterfactuals
Использовать **Gradient Trace Decoder** (Cheaper Credit Proxy). Вместо второго forward pass, можно сохранять градиент `state` во время backward:
`sample_credit = dot(primitive_effect, grad_state) * routing_prob`.
Это аналог REINFORCE/Straight-Through Estimator, который дает first-order оценку полезности действия абсолютно бесплатно (нужен только `retain_grad` на стейтах).

## 12. Будущие примитивы (после proof)
*   **Routing/Attention:** `query_key_gate` (позволит примитивам самим решать, пропускать ли сигнал).
*   **Dynamic Norm:** `adaptive_layer_norm` (критично для глубоких универсальных сетей).
*   **Словари/Memory read:** Позволит слою работать как Turing Machine.

## 13. Оптимизации без изменения математики
*   **Lazy Executor:** Считать `executor(flat_src, flat_tgt)` только для `top_ids`, а не для всего банка.
*   **Sub-graph Caching:** Если `active` гейт равен 0, вообще не выполнять `transformed = (choice * primitive_out).sum()`.
*   **torch.compile:** Обернуть `PrimitiveMatrix5x5.forward` в `@torch.compile(mode="reduce-overhead")`.

## 14. Следующие 3 задачи
1.  **Gradient Trace / Cheaper Credit:** Заменить или дополнить эмпирический counterfactual-кредит локальным градиентным прокси. Это уберет задержку и резко ускорит обучение на P40.
2.  **Integration of Self-Delta Choice:** Так как diagnostic probe показал сильный сигнал (0.96 vs 0.56), нужно добавить `self_delta_scale` в `choice_logits` как активную компоненту (Task: включить `ENABLE_SELF_DELTA_CHOICE=1` и настроить lambda/scale).
3.  **Dynamic Projection Threshold (Anti-collapse):** Вместо тупого `projection_logit_cap` (который обрезает все), ввести динамический LayerNorm или Learnable Threshold для выхода сканнера, чтобы один примитив не мог уйти в +10 логит и убить энтропию.
