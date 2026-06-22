# SRCF Graph Triangle Core

Branch: `srcf_light_closure_core`

## Что добавлено

```text
arch_builder/srcf_graph_core.py
tools/project_probe/probe_srcf_graph_triangle_core.py
```

## Главная идея

Это графовая версия SRCF-Light.

Центр теперь:

```text
R [B,N,N,C]
-> H0
-> triangle-aware closure steps
-> H*
-> task head
```

То есть вход и голова могут меняться, но центр остается одним:

```text
relation graph -> self-closing relation dynamics
```

## Почему triangle важно

В v3 benchmark правильная архитектурная правка:

```text
tri[i,j,c] = sum_k h[i,k,c] * h[k,j,c]
```

Это explicit path composition. Если мы заявляем path/closure consistency, слой должен уметь видеть, как отношение i->j поддерживается всеми промежуточными узлами k.

`SRCFGraphClosureLayer.triangle_update()` делает именно это через batched matrix multiplication по каждому hidden channel.

## Чем это отличается от старого vNext

Старый vNext выбирает primitive/candidate per cell.

Новый core не выбирает named primitive. Он учит latent graph moves:

```text
action_count = 6..8 learned moves
```

Их смысл должен возникнуть из:

```text
task loss + closure loss
```

## Замкнутый контур обучения

Probe обучает:

```text
task BCE(edge/path2/community)
+
SRCF closure loss:
  fixed
  recovery
  contract
  far_keep
  state_var
  move
```

Peer contract делается через независимое повреждение графа из той же распределенной задачи. Это не label leakage: closure loss не получает clean target.

## Probe

Запуск:

```bash
PYTHONPATH=. python tools/project_probe/probe_srcf_graph_triangle_core.py --device cuda --rel-mode raw
```

Быстрый CPU smoke:

```bash
PYTHONPATH=. python tools/project_probe/probe_srcf_graph_triangle_core.py --device cpu --steps 1 --batch 1 --eval-batches 1 --n 8 --dim 8 --hidden 16 --layers 1 --micro-steps 1 --actions 2 --graphs karate --rel-mode raw
```

Ожидаемый конец:

```text
SRCF_GRAPH_TRIANGLE_CORE_PASS
```

## Что проверять дальше

1. `--rel-mode raw`: проверяет, может ли core работать почти от одной corrupt adjacency.
2. `--rel-mode full`: сравнить с feature-rich фронтендом.
3. Ablation без triangle нужно добавить следующим файлом: `use_triangle=False`.
4. Сравнить с `closure_graph_real_benchmark_v3_triangle.py` baseline.
5. Потом добавить graph repair benchmark, где head восстанавливает не только edge/path2/comm, а полный repaired relation tensor.

## Вывод

Это ближе к нужной архитектуре, чем slot-only `srcf_light.py`:

```text
граф может быть входом,
граф может быть состоянием,
граф может быть шагом,
граф может быть архитектурой.
```

Главная формула:

```text
модель = self-organizing relation graph closure
```
