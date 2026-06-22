# SRCF Light Closure Core

Branch: `srcf_light_closure_core`

## 1. Смена центра

Эта ветка не добавляет SRCF как еще один score в старую систему выбора primitive.

Новый центр:

```text
не primitive search system,
а self-closing relation-state system
```

Главный вопрос теперь не:

```text
какой primitive выбрать?
```

А:

```text
какой learned transition делает состояние более самосогласованным и полезным для task head?
```

## 2. Добавленные файлы

```text
arch_builder/srcf_light.py
tools/project_probe/probe_srcf_light_closure_core.py
```

## 3. Контракт архитектуры

Для любой задачи меняются только:

```text
frontend
head
```

Центр остается один:

```text
slots -> learned relations -> self-organizing transitions -> descriptor
```

## 4. Почему это не task recipe

Внутри core нет заранее заданных ролей типа read/write/compare/output.

Внутри есть learned latent moves:

```text
action_count = 6..8
```

Их смысл должен возникнуть из:

```text
task loss + closure loss
```

## 5. Замкнутый контур обучения

Loop:

```text
input
-> frontend
-> slots
-> SRCFLightCore
-> descriptor
-> head
-> task loss

slots
-> closure transitions
-> fixed/recovery/contract/far_keep/move/state_var
-> closure loss

loss = task loss + closure loss
```

## 6. SRCF diagnostics

`srcf_closure_loss` считает:

```text
fixed
recovery
contract
far_keep
state_var
move
action_entropy
edge_mass
```

Смысл:

```text
task loss давит на полезность для головы;
closure loss давит на самосогласованность внутренней системы.
```

## 7. Probe

Запуск из корня репозитория:

```bash
PYTHONPATH=. python tools/project_probe/probe_srcf_light_closure_core.py
```

Ожидаемый конец:

```text
SRCF_LIGHT_CLOSURE_CORE_PASS
```

Probe проверяет:

```text
forward
closure loss
augmentation contract
backward gradients
finite metrics
short closed-loop training
```

## 8. Что дальше

Правильный следующий порядок:

```text
1. smoke stability on several seeds
2. same SRCFLightCore with different frontend/head wrappers
3. graph/relation repair task
4. compare against MLP / attention / GNN baseline
5. add optional basin memory after core is stable
```

## 9. Главный итог

`SRCF-Light` — это отдельная легкая архитектура:

```text
вход и голова меняются,
центр остается один:
slots -> relations -> closure transitions -> descriptor
```

Она исследует самоорганизацию без ручных task recipes внутри core.
