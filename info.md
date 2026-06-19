Ты работаешь с репозиторием `maxwelhelp/acrch_builder`.

Главный документ плана:
`docs/V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md`

Важно: мы реализуем пока только первую вертикальную часть этого большого плана, а не весь план целиком. Цель текущей стадии — доказать, что маленький `PrimitiveMatrixScanner -> ActionMatrix -> Executor -> output` умеет собирать понятную последовательную программу без collapse и без all-to-all shortcut.

Перед любой правкой:

1. Синхронизируй `main`.
2. Прочитай:

   * `docs/V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md`
   * `docs/V4_6_4_IMPLEMENTATION_PLAN.md`
   * `README.md`
   * `arch_builder/model.py`
   * `arch_builder/train_vertical_slice.py`
   * `arch_builder/synthetic_tasks.py`
   * `arch_builder/primitive_matrix.py`
   * `arch_builder/hybrid_scanner.py`
   * `arch_builder/simulator.py`
   * `arch_builder/executor.py`
   * `arch_builder/reporting.py`
3. Не переписывай весь проект. Чини текущую вертикальную петлю локально и модульно.

Текущий смысл проекта:

Мы строим универсальный механизм сборки программы/архитектуры:

```text
raw/basic input
  -> slots
  -> PrimitiveMatrix
  -> HybridScanner
  -> Top-K candidates
  -> LowRankSimulator
  -> ActionMatrix layer(s)
  -> Executor
  -> output tape / final read
  -> classifier
  -> loss
  -> gradients back into scanner/controller/simulator/executor/core
```

Что уже реализовано в текущей ветке:

1. `PrimitiveMatrix5x5`
2. `HybridScanner`
3. `Top-K candidate selection`
4. `LowRankSimulator`
5. `ActionMatrixLayer`
6. `ActionExecutor`
7. `slot_embed`
8. `input_norm=none/layernorm`
9. `final_read=last/mean/learned`
10. `PROGRAM_REPORT.md`
11. synthetic known-program tasks:

* `diff`
* `two_diff`
* `chain_diff_product`
* `semantic_rescue`

12. metrics:

* `expected_candidate_present`
* `expected_edge_choice_mass`
* `expected_edge_recovery`
* `program_recovery_rate`
* `active_cells`
* `expected_top_cells`
* `primitive_top_share`
* `active_edges_per_target`
* `final_read_mode`
* `layer_ablation_delta`
* `layer_dependency_delta`

13. generic diversity/sparse losses
14. signal-gated adaptive loss weights, not epoch schedule.

История проблем и что уже выяснено:

### 1. Первый обрыв: не было slot/address information

Раньше `TASK=diff` был почти random, потому модель не могла отличить `slot0->slot1` от других пар. Добавили `slot_embed`. После этого `expected_edge_recovery` начал работать.

### 2. Второй обрыв: `LayerNorm(dim)` убивал задачу

`TASK=diff` задавался как:

```python
y = sign(mean(x0 - x1))
```

А `LayerNorm(dim)` на входе удалял feature-mean. Поэтому для proof tasks сделали `INPUT_NORM=none`.

### 3. Третий обрыв: output path гасил правильное действие

Добавили direct output tape / cell tape, positive edge scale, отчёты по tape/gates.

### 4. Proof `diff` начал работать, но был collapse

На `TASK=diff LAYERS=1` модель достигала высокой accuracy, но фактически выбирала `diff` почти во всех cells. Это не умная программа, а primitive collapse.

### 5. Попытка `two_diff`

`TASK=two_diff` тоже не ломал collapse, потому задача всё ещё основана только на `diff`. Модель выбирала `diff` почти везде и получала высокую accuracy.

### 6. Попытка generic diversity/sparse losses

Были добавлены:

* `primitive_usage_diversity`
* `cell_choice_diversity`
* `active_budget`
* `tape_budget`
* `layer_action_diversity`

Идея взята не копированием из старого `simple_butterfly_matrix_v3`, а по смыслу: там collapse лечился не hard ban, а diversity/read/phase/slot/write budget.

Результат: collapse частично давится, но если давить с самого начала, recovery/choice умирает.

### 7. Попытка `FINAL_READ=last`

Добавили `final_read=last`, чтобы для `LAYERS=2` финал читал только последний слой. Это нужно, чтобы Layer 0 не мог напрямую решить задачу и обойти Layer 1.

### 8. Попытка signal-gated losses вместо epoch schedule

Важно: не делать календарь типа “epoch 1 одно, epoch 5 другое”. Нужно включать/выключать структурные лоссы от сигналов самой программы.

В v15 добавлены:

```text
signal_expected_choice_mass
signal_candidate_present
signal_primitive_top_share
signal_active_mean
signal_tape_mean
signal_active_cells_soft

adaptive_recovery_gate
adaptive_collapse_gate
adaptive_sparse_gate
adaptive_choice_boost
```

Логика должна быть такая:

```text
если expected_choice_mass низкий:
  не давить sparsity/diversity
  усилить expected choice / recovery

если expected_choice_mass высокий и primitive_top_share высокий:
  включить anti-collapse

если expected_choice_mass высокий и active_cells много:
  включить sparse/tape budget
```

Но свежий результат v15 на `TASK=chain_diff_product LAYERS=2 FINAL_READ=last` плохой:

```text
candidate_present = 1.000
final = last
oracle = 100%
edge_prog низкий
choice_mass низкий
val_acc около random
active_cells = 16
top_share растёт
gateR/gateC/gateS остаются низкими
train loss очень большой
```

Текущий диагноз:

1. Scanner не главный подозреваемый:
   `candidate_present = 1.000`

2. `final_read=last` включён:
   bypass через mean outputs закрыт.

3. Основная проблема сейчас:
   expected choice не становится живым на `chain_diff_product`.

4. Signal-gated losses не спасли, потому `adaptive_recovery_gate` почти закрыт, а `choice_mass` не растёт достаточно.

5. Очень большой `train_loss` показывает, что лоссы конфликтуют или scale слишком большой.

6. Возможная критическая проблема:
   проверить, что primitive set реально содержит все expected primitives для задач:

   * `diff`
   * `product`
   * `merge`

   Важно проверить `PrimitiveMatrix5x5.names` и `model.pm.name_to_id`.
   Если `TASK=chain_diff_product` требует `product`, но `product` отсутствует или плохо реализован в executor/scanner, задача будет некорректной.

Что нужно сделать дальше.

## Главная задача

Не добавлять новый hardcoded schedule. Не делать календарь по эпохам.

Нужно сделать self-regulating training loop по сигналам, но сначала починить фундамент: почему expected choice не оживает на `chain_diff_product`.

## Шаг 1. Проверить primitive availability и executor

Файлы:

* `arch_builder/primitive_matrix.py`
* `arch_builder/executor.py`
* `arch_builder/synthetic_tasks.py`

Проверить:

```text
model.pm.names содержит diff/product/merge?
name_to_id["product"] существует?
HybridScanner реально может вернуть product в candidates?
Executor реально выполняет product?
simulator знает product id?
```

Добавить invariant в startup/validate:

```python
for task in ["diff", "two_diff", "chain_diff_product"]:
    for expected_action in task.expected_actions:
        assert expected_action["primitive"] in model.pm.name_to_id
```

Если `product` отсутствует или неисполняемый — добавить его нормально как primitive, а не как заглушку.

## Шаг 2. Добавить debug по expected actions отдельно по слоям

Сейчас средние метрики скрывают, какая именно expected action сломана.

Для `chain_diff_product` нужно логировать отдельно:

```text
Layer0 edge 0->1 diff:
  candidate_present
  choice_mass
  chosen_top
  active
  tape

Layer0 edge 2->3 diff:
  candidate_present
  choice_mass
  chosen_top
  active
  tape

Layer1 edge 1->3 product:
  candidate_present
  choice_mass
  chosen_top
  active
  tape
```

Добавить в:

* `metrics.csv`
* `final_report.json`
* `PROGRAM_REPORT.md`
* `program_epoch_XXX.json`

Названия метрик:

```text
action_L0_0_1_diff_present
action_L0_0_1_diff_choice_mass
action_L0_0_1_diff_recovery
action_L0_0_1_diff_active
action_L0_0_1_diff_tape

action_L0_2_3_diff_present
...

action_L1_1_3_product_present
...
```

Главный смысл: если `Layer0 diff` живой, а `Layer1 product` мёртвый — проблема в sequential/state dependency. Если все мёртвые — проблема в choice loss/scales.

## Шаг 3. Разделить loss components в логах

Сейчас train loss большой, но не видно, какой компонент его взрывает.

В `train_vertical_slice.py` логировать по epoch:

```text
ce_loss
sim_loss
expected_choice_loss
expected_active_loss
non_expected_primitive_loss
non_expected_active_loss
non_expected_tape_loss
non_expected_transform_loss
primitive_usage_diversity_loss
cell_choice_diversity_loss
active_budget_loss
tape_budget_loss
layer_action_diversity_loss

effective_lambda_choice
effective_lambda_non_expected_primitive
effective_lambda_primitive_usage_diversity
effective_lambda_cell_choice_diversity
effective_lambda_active_budget
effective_lambda_tape_budget
effective_lambda_layer_action_diversity
```

И отдельно signal gates:

```text
signal_expected_choice_mass
signal_candidate_present
signal_primitive_top_share
signal_active_cells_soft
adaptive_recovery_gate
adaptive_collapse_gate
adaptive_sparse_gate
adaptive_choice_boost
```

Это обязательно. Без этого нельзя понять, что именно убивает обучение.

## Шаг 4. Починить expected choice без календаря

Не epoch schedule. Нужен controller от сигналов.

Текущая v15 идея хорошая, но недостаточная. Сделать более жёсткую signal logic:

```text
if candidate_present high and expected_choice_mass low:
  boost expected choice strongly
  set collapse/sparse gates almost zero

if expected_choice_mass grows above floor:
  slowly open collapse gate from signal, not epoch

if expected_edge_active low:
  boost expected_active_loss
  keep sparse gate closed

if expected_choice_mass high but active_cells high:
  open sparse gate

if expected_choice_mass high but primitive_top_share high:
  open collapse gate
```

Технически:

```python
recovery_gate = sigmoid((expected_choice_mass - floor) / sharpness)

choice_boost = 1 + boost * (1 - recovery_gate)

collapse_gate = recovery_gate * sigmoid((primitive_top_share - top_floor) / sharpness)

sparse_gate = recovery_gate * sigmoid((active_cells_soft - target_active_cells) / cell_sharpness)

expected_active_boost = sigmoid((active_floor - expected_edge_active) / sharpness)
```

Сейчас `expected_active_loss` не gated/boosted достаточно. Нужно добавить:

```text
eff_lambda_expected_active = lambda_expected_active * (1 + active_boost * expected_active_boost)
```

и логировать его.

## Шаг 5. Сделать easier chain debug task перед full chain

`chain_diff_product` может быть слишком резкий: Layer1 должен делать product над результатами Layer0, пока Layer0 ещё не стабилен.

Добавить промежуточную задачу:

```text
TASK=chain_diff_merge
Layer0:
  0->1 diff
  2->3 diff
Layer1:
  1->3 merge/add/gated_add
label:
  sign(mean(d01 + d23))
```

Это проще, чем product. Цель — проверить именно sequential dependency:

```text
Layer1 должен читать результат Layer0 и объединить два результата.
```

Если `chain_diff_merge` не работает, `chain_diff_product` рано тестировать.

Нужно убедиться, что primitive `merge` или `gated_add` реально есть в primitive matrix и executor.

## Шаг 6. Проверить, что Layer1 действительно получает Layer0 state

В `model.py` сейчас должно быть:

```python
for idx, layer in enumerate(self.layers):
    state, out, memory, tr = layer(state, memory, ...)
```

Проверить, что:

* `state` после Layer0 идёт в Layer1.
* `final_read=last` читает только output Layer1.
* Нет hidden bypass из Layer0 output в classifier.
* `outputs` Layer0 не участвует при `final_read=last`.

Добавить в `PROGRAM_REPORT.md`:

```text
Flow:
input -> Layer 0 state -> Layer 1 state -> final_read:last
```

И добавить dependency tests:

```text
layer_ablation_delta:
  zero final layer output

layer_dependency_delta:
  zero state after Layer0 and смотреть падение Layer1/final accuracy
```

Успех:
`layer_dependency_delta > 0`

## Шаг 7. Не делать hard top-k пока

Hard top-k active edges пока не добавлять. Сейчас choice не оживает стабильно. Hard mask убьёт градиент.

Пока только:

* soft budgets
* signal-gated weights
* better metrics
* expected edge protection

## Шаг 8. Acceptance tests

Сначала прогонять в таком порядке.

### Test A: single diff, sparse but alive

```bash
EPOCHS=8 \
BATCH_SIZE=128 \
DIM=64 \
DEVICE=cuda \
AMP=fp16 \
INPUT_NORM=none \
FINAL_READ=last \
TASK=diff \
LAYERS=1 \
bash commands/run_vertical_slice_push_reports.sh
```

Успех:

```text
val_acc > 85%
expected_choice_mass > 0.7
edge_prog > 0.7
active_cells < 16
primitive_top_share < 0.85
```

### Test B: two diff

```bash
EPOCHS=8 \
BATCH_SIZE=128 \
DIM=64 \
DEVICE=cuda \
AMP=fp16 \
INPUT_NORM=none \
FINAL_READ=last \
TASK=two_diff \
LAYERS=1 \
bash commands/run_vertical_slice_push_reports.sh
```

Успех:

```text
Layer0 0->1 diff alive
Layer0 2->3 diff alive
not all 16 cells top=diff
active_cells < 16
val_acc > 80%
```

### Test C: chain diff merge

Добавить задачу, потом запуск:

```bash
EPOCHS=10 \
BATCH_SIZE=128 \
DIM=64 \
DEVICE=cuda \
AMP=fp16 \
INPUT_NORM=none \
FINAL_READ=last \
TASK=chain_diff_merge \
LAYERS=2 \
bash commands/run_vertical_slice_push_reports.sh
```

Успех:

```text
Layer0 diff/diff alive
Layer1 merge/gated_add alive
final_read=last
layer_dependency_delta > 0
val_acc > random
```

### Test D: chain diff product

Только после C:

```bash
EPOCHS=12 \
BATCH_SIZE=128 \
DIM=64 \
DEVICE=cuda \
AMP=fp16 \
INPUT_NORM=none \
FINAL_READ=last \
TASK=chain_diff_product \
LAYERS=2 \
bash commands/run_vertical_slice_push_reports.sh
```

Успех:

```text
Layer0 diff/diff alive
Layer1 product alive
final_read=last
layer_dependency_delta > 0
val_acc > random
```

## Важный принцип

Не нужно “закончить весь план” одним коммитом.

Сейчас нужно закрыть первую часть большого плана:

```text
PrimitiveMatrix + Scanner + Simulator + ActionMatrix + Sequential Layers + Program Report
```

Эта часть считается рабочей только если:

```text
1. expected candidates есть
2. expected choice оживает
3. программа не collapse в один primitive
4. active cells не все 16
5. final_read=last работает
6. Layer1 зависит от Layer0
7. PROGRAM_REPORT показывает понятную последовательность
```

Только после этого переходить к следующим пунктам полного плана:

* real credit вместо synthetic expected actions
* learned simulator quality from real ablation
* honest input curriculum
* dynamic branching/slot count
* TokenSlotAdapter
* plug-in вместо слоя transformer/attention
* real tasks beyond synthetic proof

Сейчас не переходить к transformer/plugin. Сначала починить synthetic sequential program.
