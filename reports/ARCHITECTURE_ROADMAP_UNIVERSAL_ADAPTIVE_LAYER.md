# Universal Adaptive Layer Roadmap

## 1. Цель проекта простыми словами

Построить один переиспользуемый блок, который заменяет attention/MLP/adapter в произвольной нейросети. Пользователь подключает input frontend и output head — всё остальное блок строит сам: выбирает полезные примитивы, маршрутизирует данные, оценивает кандидатов, коммитит лучшие действия и обновляет credit.

Ключевая идея: **не hardcoded architecture, а learned program per context**.

---

## 2. Текущая архитектура и что уже правильно

### Что есть

| Компонент | Файл | Статус |
|---|---|---|
| `ActionMatrixLayer` / `ActionMatrixModel` | `model.py` | Рабочее ядро: slots, context, choice, execution |
| `PrimitiveMatrix5x5` | `primitive_matrix.py` | 25 примитивов, 5 семейств, embeddings, usage EMA |
| `HybridScanner` | `hybrid_scanner.py` | 4 источника: grid, semantic, usage, random |
| `LowRankSimulator` | `simulator.py` | Low-rank preview + predicted gain |
| `ActionExecutor` | `executor.py` | Полная реализация всех 25 примитивов |
| `BoundedCounterfactualCredit` | `credit.py` | Counterfactual ablation + joint pair credit |
| `SelfDeltaCandidateField` | `self_delta_candidate_field.py` | Diagnostic: actual - sim signal |
| `ProjectionScanner` | `projection_scanner.py` | Single signed + pair JL16 |
| SpeechCommands trainer | `train_audio_frontend.py` | Real discovery + curriculum |
| Token-slot replacement | `token_slot_replacement.py` | Sequence task proof |
| Transformer plugin | `transformer_plugin.py` | Motif task proof |
| Standalone probes | `tools/project_probe/*` | 24 probes covering effect/deliberation/credit |

### Что уже правильно

1. **Slot-based state** `[B, S, D]` — модальность-агностичен.
2. **Executor отделён от Controller** — можно менять routing без пересборки операций.
3. **Credit через measured counterfactual** — честный source of truth.
4. **expected_actions запрещены в real discovery** — нет ложных PASS.
5. **Diversity через soft targets** (`top_share_target=0.65`, `entropy_floor=0.55`) — не hard.
6. **Diagnostic probes не меняют train path** — safe experimentation.

### Ключевые экспериментальные результаты

```
utility_top1 captured_gain ≈ 0.895   (vs proposal_only ≈ 0.373)
head_utility >> global_utility >> proposal >> random
effect_mmr reduces similarity 0.728→0.625 without losing gain
pool_16/24 best tradeoff; budget 1-3 sufficient
self_delta standalone: 0.88-0.96 (vs sim_only 0.55)
LowRank full-D reconstruction: weak at rank<64
scalar utility critic >> JL effect score for ranking
```

---

## 3. Главные логические дыры сейчас

### 3.1 Simulator декоративен в real train

`LowRankSimulator` (31 строка!) предсказывает `predicted_gain` по `(state, prim_emb)`. Но он не видит `tgt`, `mem`, head context. Probe v2 показал: full-context scalar utility на порядок сильнее. Текущий simulator — bottleneck качества.

### 3.2 Scanner и Critic смешаны

`choice_logits` суммирует 5+ аддитивных компонент: `context_component + gain_component + sim_component + proposal_component + primitive_pair_component`. Это **не** разделение ролей scanner/critic. Это one mixed score. Если projection даёт +10, он убивает все остальные сигналы через softmax.

### 3.3 Controller — один softmax

Нет budget allocation, нет exploration/exploitation трейдоффа, нет решения "сколько кандидатов проверять". Просто `gumbel_softmax(choice_logits)`.

### 3.4 Credit запаздывает и шумен

Credit обновляется через отдельный forward pass (counterfactual batch), с задержкой на 1+ step, на зашумлённом mini-batch. Credit SNR не отслеживается.

### 3.5 Primitive bank фиксирован

25 примитивов hardcoded. Нет механизма добавления/удаления. Нет auto-mined atoms. Нет иерархии категорий.

### 3.6 Нет sample-specific routing

Все cells `(src_i, tgt_j)` получают одинаковый набор кандидатов (одна и та же `cand_ids` для всех cells в batch). Routing не адаптируется к конкретному input.

### 3.7 Output hardcoded

`output_state = output_tape_state + 0.10 * slot_output_state + 0.02 * collector`. Коэффициенты `0.10`, `0.02` — magic numbers.

---

## 4. Правильное разделение ролей

```
┌─────────────────────────────────────────────────────────┐
│                    ADAPTIVE LAYER                        │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐       │
│  │ SCANNER  │───>│  CRITIC  │───>│  CONTROLLER   │      │
│  │ (cheap   │    │ (full    │    │ (budget +     │      │
│  │  pool)   │    │  context │    │  commit +     │      │
│  │          │    │  scalar  │    │  explore)     │      │
│  └──────────┘    │  utility)│    └──────┬───────┘      │
│       ↑          └──────────┘           │               │
│       │                                 ↓               │
│  ┌────┴─────┐                    ┌──────────────┐       │
│  │ FEEDBACK │<───────────────────│   EXECUTOR   │       │
│  │ MEMORY   │    measured gain   │ (true effect)│       │
│  │ (EMA     │<───────────────────┤              │       │
│  │  regret) │                    └──────────────┘       │
│  └──────────┘                                           │
│       ↑                                                  │
│  ┌────┴─────┐                                           │
│  │  CREDIT  │  counterfactual / gradient trace           │
│  └──────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

### Scanner (cheap, wide)

**Роль**: генерировать широкий пул кандидатов `P=16..32` из всего primitive bank.

**Входы**: `flat_context` (cheap features), `memory`, `prev_action_emb`, `usage_credit_ema`.

**Выходы**: `candidate_ids[N, P]`, `proposal_scores[N, P]`.

**Не делает**: не оценивает utility, не ранжирует по gain, не принимает решений.

**Свойства**: O(P * embed_dim), без full executor forward, feedback memory корректирует bias.

### Critic (full context, expensive per candidate)

**Роль**: оценить scalar utility каждого кандидата из пула Scanner.

**Входы**: `full_src, full_tgt, full_mem, primitive_emb, head_context_vector`.

**Выходы**: `scalar_utility[N, P]`, optional `uncertainty[N, P]`, optional `behavior_feature[N, P, F]` для MMR.

**Критично**: НЕ сжимать входы. MLP(full_context) → 1 scalar. Probe v2 доказал: head-conditioned scalar utility >> всё остальное.

### Controller (budget + commit)

**Роль**: решить сколько кандидатов проверять counterfactual, выбрать diverse subset через MMR, закоммитить лучший по measured gain.

**Входы**: `utility[N,P]`, `uncertainty[N,P]`, `behavior_features[N,P,F]`, `budget_signal`.

**Выходы**: `committed_action[N]`, `exploration_set[N, B]`.

**Свойства**: UCB-like exploration (`utility + β * uncertainty`), effect-space MMR для diversity, budget `B=1..3`.

### Executor (ground truth effect)

**Роль**: вычислить реальный эффект применения примитива. Source of truth.

**Входы**: `src, tgt, mem, primitive_id`.

**Выходы**: `effect[N, D]`.

**Свойства**: batched, no Python loops, lazy (считает только для committed + exploration set).

### Credit (delayed, measured)

**Роль**: посчитать measured counterfactual gain, обновить EMA по primitive/family/cell.

**Входы**: `full_loss`, `ablated_loss`, intervention targets.

**Выходы**: `gain_ema[layer, cell, primitive]`, `regret_ema`, feedback → Scanner.

---

## 5. Предлагаемая целевая архитектура vNext

### 5.1 Заменить LowRankSimulator на UtilityCritic

```python
class UtilityCritic(nn.Module):
    def __init__(self, dim, context_dim, prim_embed_dim, hidden=128):
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim + prim_embed_dim + dim),  # head_vec
            nn.Linear(context_dim + prim_embed_dim + dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),  # [utility, log_uncertainty]
        )

    def forward(self, flat_context, prim_emb, head_vector):
        x = torch.cat([flat_context, prim_emb, head_vector], dim=-1)
        out = self.net(x)
        return out[..., 0], F.softplus(out[..., 1])
```

Ключевое: **flat_context уже содержит full src+tgt+mem** (5*dim). Не сжимать.

### 5.2 Добавить Scanner Feedback Memory

```python
class ScannerFeedbackMemory:
    def __init__(self, num_primitives, decay=0.95):
        self.gain_ema = torch.zeros(num_primitives)
        self.regret_ema = torch.zeros(num_primitives)
        self.count = torch.zeros(num_primitives)

    def update(self, primitive_ids, measured_gains):
        # Обновляется ТОЛЬКО от measured counterfactual gain
        ...

    def proposal_bias(self):
        return self.gain_ema - 0.5 * self.regret_ema
```

### 5.3 Controller с budget allocation

```python
class AdaptiveController(nn.Module):
    def forward(self, utility, uncertainty, behavior_features, beta=0.35):
        # UCB score для exploration
        ucb = utility + beta * uncertainty

        # MMR selection для diversity
        selected = mmr_select(ucb, behavior_features, budget=3, beta_mmr=0.35)

        # Commit лучший по measured gain (после counterfactual)
        return selected, ucb
```

### 5.4 Lazy Executor

Считать executor effect **только** для committed + exploration set (budget 1-3), не для всех 25 примитивов.

---

## 6. Цикл: candidate → utility → diversity → counterfactual → commit → credit

```
Step 1: Scanner proposes pool P=16..24 candidates
         sources: grid + semantic + usage_credit + random + feedback_bias

Step 2: Critic scores each candidate
         input: full_context(5*dim) + prim_emb + head_vector
         output: scalar utility + uncertainty

Step 3: Controller selects B=1..3 diverse candidates via MMR
         mmr_score = (1-β)*utility + β*(1 - max_similarity_to_selected)
         similarity from behavior_features (optional critic side-output)

Step 4: Executor computes true effect for selected B candidates ONLY
         (lazy: not all 25 primitives)

Step 5: Controller commits best candidate
         choice = argmax(utility) among selected
         OR: measured_gain from counterfactual if available

Step 6: Credit measures counterfactual gain on separate micro-batch
         gain = CE(ablated) - CE(full)    [для ablation]
         gain = CE(full) - CE(forced)     [для forced alternative]

Step 7: Feedback updates
         credit.update(primitive_id, measured_gain)
         scanner_memory.update(primitive_id, gain, regret)
         critic trains on (predicted_utility, measured_gain) MSE
```

---

## 7. Как расширять primitive bank и категории

### 7.1 Иерархия: Category → Primitive → Params

```
Category: local_transform    → identity, diff, contrast, smooth, gated_keep
Category: learned_transform  → low_rank, channel, ctx_matrix, product, gated_add
Category: routing            → merge, split, route, edge_gate, write_gate
Category: memory             → mem_read, mem_write, forget, recall, mem_gate
Category: control            → output_write, output_mix, skip, replace, disable
Category: spectral           → dct, fft_filter, wavelet  [NEW]
Category: attention_like     → qkv_gate, cross_attend    [NEW]
Category: normalization      → adaptive_ln, rms_norm     [NEW]
Category: state_space        → s4_scan, associative_scan [NEW]
Category: mined_local        → svd_atom_k, diag, toeplitz [AUTO-MINED]
```

### 7.2 Anti-collapse при расширении

1. **Per-category scanner budget**: каждая категория гарантирует ≥1 кандидат в пуле.
2. **Usage EMA с exploration bonus**: новые примитивы получают UCB bonus пока `count < threshold`.
3. **Orthogonality loss на embeddings**: `loss += λ * (emb @ emb.T - I).pow(2).mean()`.
4. **Measured diversity**: отслеживать `primitive_entropy`, `family_coverage`, `usage_gini`.

### 7.3 Auto-mined atoms

```
target_matrix W (или learned weight)
→ extract: diagonal, row_mean, col_mean, SVD rank-1..4, toeplitz, block_mean
→ add to local primitive library
→ critic evaluates utility of mined atoms
→ counterfactual measures real gain
→ accepted atoms persist; rejected filtered out
```

Ключевое: mined atoms — это **расширение пула Scanner**, не замена fixed bank. Critic решает, полезны ли они.

---

## 8. Как избежать collapse и hidden priors

### Hidden priors в текущем коде

| Prior | Где | Риск | Решение |
|---|---|---|---|
| `mode_head.bias[0]=0.6` | model.py:125 | Сильный bias на transform vs skip | Сделать learnable без init bias |
| `0.10 * slot_output` + `0.02 * collector` | model.py:517 | Magic numbers в output | Learnable weights |
| `final_read="last"` | model.py:652 | Последний слой доминирует | Default `learned` |
| `new_memory = 0.95*old + 0.05*state` | model.py:587 | Fixed memory decay | Learnable gate |
| `source quota` (4 sources guaranteed) | model.py:314-324 | Forced diversity может быть вредна | Soft quota через proposal_bias |
| `pair_bias` all zeros init | model.py:98-101 | Нет structural prior | OK, но мониторить |

### Shortcut detection protocol

1. **Shuffle control**: `shuffle(candidate_ids)` → если accuracy не падает, routing decorative.
2. **Uniform choice control**: `choice = 1/K` → если accuracy держится, choice decorative.
3. **Frozen executor control**: freeze executor weights → если accuracy растёт, executor не нужен.
4. **Single primitive override**: force one primitive everywhere → если accuracy высокая, collapse.

---

## 9. Как тестировать универсальность

### Acceptance task suite

| Task | Input | What it tests | Pass criterion |
|---|---|---|---|
| SpeechCommands 10-class | audio 16kHz | Real data discovery | acc > 1/10 + 5%, credit_closed |
| Synthetic audio order | waveform 256 | Temporal slot routing | deploy_acc > 0.5 |
| CIFAR-10 patches | image 32×32 | Visual feature composition | acc > 20% |
| Token position | sequence tokens | Causal sequential memory | acc > 0.6 |
| Motif parity | sequence tokens | Local pattern + composition | acc > 0.6 |
| Matrix program (chain) | synthetic | Multi-layer composition credit | credit_closed, gain > 0 |
| Copy/reverse task | sequence | Pure memory read/write | acc > 0.9 |
| Non-stationary shift | any, switching | Adaptation speed | regret < fixed baseline |

### Метрики универсальности

```
1. accuracy (task-specific)
2. credit_closed (credit loop works)
3. primitive_entropy > 0.5 (no collapse)
4. primitive_top_share < 0.65 (no dominance)
5. measured_gain_snr > 1.0 (credit has signal)
6. critic_correlation > 0.3 (critic predicts gain)
7. ablation_delta > 0 (components matter)
8. source_diversity > 2 families used
9. speed: samples/sec on P40
10. shuffle_control_delta > 0.05 (routing is real)
```

---

## 10. План задач на 10–20 этапов

### Phase 1: Foundation fixes (steps 1-4)

1. **UtilityCritic standalone probe** — реализовать HeadUtilityCritic в probe, обучить на SpeechCommands, сравнить с текущим LowRankSimulator. Diagnostic only.
2. **Scanner Feedback Memory probe** — реализовать EMA regret memory в standalone probe, проверить что proposal recall растёт.
3. **Lazy Executor proof** — реализовать executor только для top-B candidates, замерить speedup на P40.
4. **Output weight learnable** — заменить `0.10`, `0.02` magic numbers на learnable parameters.

### Phase 2: Integration (steps 5-8)

5. **Replace LowRankSimulator with UtilityCritic** в real train path. Diagnostic A/B: acc, speed, credit_closed.
6. **Wire Scanner Feedback Memory** в real train. Credit → scanner bias loop.
7. **AdaptiveController с MMR** в real train. Budget 1-3, effect-space diversity.
8. **Lazy Executor integration** — считать effect только для committed set.

### Phase 3: Primitive expansion (steps 9-12)

9. **Category-aware scanner** — per-category budget в proposal pool.
10. **Add spectral primitives** (DCT, FFT filter) — diagnostic probe first.
11. **Add attention-like primitives** (QKV gate) — diagnostic probe first.
12. **Auto-mined atoms probe** — SVD/diagonal/toeplitz extraction → utility evaluation.

### Phase 4: Credit upgrade (steps 13-15)

13. **Gradient trace credit proxy** — `dot(primitive_effect, grad_state)` как cheap alternative к counterfactual.
14. **Rank-based critic loss** — ranking loss вместо MSE для critic training.
15. **Joint credit with Shapley-lite** — random subset masking для multi-primitive credit.

### Phase 5: Universality (steps 16-20)

16. **CIFAR-10 patch frontend + acceptance test**.
17. **Copy/reverse memory task + acceptance test**.
18. **Non-stationary adaptation test**.
19. **Multi-layer composition credit test** (chain product).
20. **Full acceptance audit across all tasks**.

---

## 11. Что делать первым, вторым, третьим

### Первое (неделя 1): UtilityCritic standalone probe

Почему: probe v2 уже показал что head_utility >> всего. Нужен proof в real SpeechCommands context.

```
safe diagnostic-only
не меняет real train
не меняет model.py
отдельный файл probe_utility_critic.py
```

### Второе (неделя 2): Scanner Feedback Memory probe

Почему: scanner recall — bottleneck. Pool 16 даёт только 31% recall. Feedback должен поднять recall без увеличения pool size.

```
safe diagnostic-only
отдельный файл probe_scanner_feedback.py
```

### Третье (неделя 3): Replace LowRankSimulator → UtilityCritic

Почему: это единственное место где можно получить большой качественный скачок. Текущий simulator decorative.

```
behavior-changing experimental
A/B test: old vs new
rollback plan: env flag ENABLE_UTILITY_CRITIC=0
```

---

## 12. Что НЕ делать

| Не делать | Почему |
|---|---|
| Full-D simulator rank=64 | Дорого на P40, scalar utility лучше |
| JL effect score как main ranking | utility_top1=0.895 vs JL_top1=0.136 |
| Сжимать входы critic | Probe доказал: full context >> compressed |
| Hard diversity penalty | Съедает качество, лучше MMR на candidate set |
| expected_actions в real train | Ложные PASS |
| Gradient trace как единственный credit | Noisy first-order, дополнение к counterfactual |
| Pair JL16 в real train по умолчанию | Шумный, дорогой, decorative |
| Self-delta в choice_logits без proof | Diagnostic показал сигнал, но integration не проверена |
| Расширять primitives до proof utility | Collapse risk |
| Менять executor semantics | Executor = ground truth, не трогать |

---

## 13. Оптимизация под Tesla P40

### P40 constraints

```
24GB VRAM, Pascal architecture, no tensor cores
fp16 через GradScaler (не bf16)
torch.compile может не работать (sm_61)
Triton: ограниченная поддержка
```

### Конкретные оптимизации

1. **Lazy Executor**: считать effect только для B=1..3 committed candidates, не для pool=25.
   - Текущий код: `self.executor(flat_src, flat_tgt, flat_mem, top_ids)` — top_ids может быть 25.
   - Целевой: executor только для argmax(utility) + exploration set.
   - Ожидаемый speedup: 5-8x на executor forward.

2. **Batched candidate evaluation**: UtilityCritic обрабатывает весь pool одним forward.
   - `utility = critic(context.unsqueeze(1).expand(B,P,-1), prim_emb[cand_ids])` — один matmul.

3. **Cache primitive embeddings**: `pm.emb[cand_ids]` вычислять один раз.

4. **Pool size 16-24, не 32+**: probe показал diminishing returns после P=24.

5. **Counterfactual budget B=1..3**: probe показал budget=3 captures 99% of budget=5 gain.

6. **Credit interval**: собирать counterfactual каждые 8-16 steps, не каждый step.

7. **AMP fp16**: все critic/scanner forward в fp16 через autocast.

8. **Avoid Python loops в executor**: текущий executor уже batched (put/where pattern). Сохранить.

9. **Profile dataloader**: на P40 часто bottleneck в CPU data loading, не в GPU compute.

---

## 14. Риски и как их диагностировать

### Риск 1: UtilityCritic переобучается на train distribution

**Диагностика**: `critic_train_corr` vs `critic_val_corr`. Если train >> val, overfitting.

**Митигация**: dropout в critic MLP, weight decay, critic trains on held-out micro-batch.

### Риск 2: Scanner feedback memory создаёт positive feedback loop

**Диагностика**: `primitive_entropy` падает после включения feedback. `usage_gini` растёт.

**Митигация**: exploration bonus для low-count primitives, bounded bias `clamp(-1, 1)`.

### Риск 3: MMR diversity снижает quality

**Диагностика**: `best_of_B_gain` падает при увеличении β.

**Митигация**: β sweep в probe (0.25-0.50 safe zone доказан).

### Риск 4: Lazy executor пропускает полезные primitives

**Диагностика**: `oracle_recall` при lazy vs full.

**Митигация**: critic accuracy — главная защита. Если critic хороший, top-3 достаточно.

### Риск 5: Credit noise drowns utility signal

**Диагностика**: `credit_gain_snr = mean(|gain|) / std(gain)`. Если < 0.5, сигнала нет.

**Митигация**: увеличить credit batch size, использовать rank-based loss вместо MSE.

### Риск 6: Новые примитивы не используются (cold start)

**Диагностика**: `usage_observations[new_prim] == 0` после N steps.

**Митигация**: UCB exploration bonus, guaranteed slot в scanner pool для new primitives.

---

## 15. Какие технологии применимы

| Технология | Применимость | Как использовать |
|---|---|---|
| **Contextual bandits / UCB** | ✅ Высокая | Controller: UCB(utility, uncertainty) для exploration |
| **MoE routing** | ✅ Высокая | Scanner = router, primitives = experts |
| **MMR diversity** | ✅ Доказана | Effect-space MMR для candidate set selection |
| **Counterfactual credit** | ✅ Уже есть | Улучшить: gradient trace как cheap proxy |
| **EMA regret** | ✅ Высокая | Scanner feedback memory |
| **Pairwise ranking loss** | ✅ Средняя | Critic training: rank(utility) aligned with rank(gain) |
| **Energy-based scoring** | ⚠️ Средняя | Возможно для critic, но scalar utility проще |
| **Shapley-lite** | ⚠️ Средняя | Random masking для joint credit, дорого |
| **NAS / DARTS** | ⚠️ Низкая | Наш подход уже лучше: online, sample-specific |
| **Gradient attribution** | ⚠️ Средняя | Cheap credit proxy, но noisy |
| **Thompson sampling** | ⚠️ Низкая | UCB проще и стабильнее |
| **State-space / S4** | ⚠️ Будущее | Как новый primitive, не как framework |
| **Learned optimizers** | ❌ Не сейчас | Слишком meta, нет proof of need |
| **Test-time training** | ❌ Не сейчас | Нужна стабильная base first |

---

## 16. Короткий итог

Проект `acrch_builder` находится в правильном направлении. Ядро (slots, primitives, executor, credit) работает. Главные проблемы — в **качестве critic** (текущий simulator слишком слабый), **разделении ролей** (scanner/critic/controller смешаны в одной сумме логитов), и **feedback loop** (credit → scanner bias не замкнут).

**Три ключевых шага для прорыва:**

1. Заменить `LowRankSimulator` на `UtilityCritic(full_context → scalar)` — даёт 2-5x улучшение ranking quality (доказано probe v2).
2. Замкнуть feedback: `measured_gain → EMA → scanner_proposal_bias` — поднимает pool recall без увеличения pool size.
3. Controller с budget + MMR — даёт diverse exploration за O(3) executor calls вместо O(25).

Всё остальное (расширение primitives, auto-mined atoms, новые tasks, gradient trace) — после этих трёх.

**Главный принцип: diagnostic proof → integration → validation. Никогда наоборот.**
