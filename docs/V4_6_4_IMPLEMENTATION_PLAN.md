# v4.6.4 implementation plan

## Stage A: clean project and vertical slice

Current repo implements this stage:

```text
1-2 layers
4 slots
PrimitiveMatrix 5x5
HybridScanner
Top-K simulator
mandatory simulator influence
ActionMatrix executor
bounded reports
```

## Stage B: synthetic known-program proof

Tasks:

```text
diff:
  expected useful primitive = diff

merge:
  expected useful primitive = merge

memory:
  expected useful primitive = memory_write / memory_read
```

The goal is not high benchmark accuracy. The goal is to prove that the program machinery chooses sensible actions and the simulator is not decorative.

## Stage C: structured matrix frontend

After proof slice:

```text
audio frame/unfold -> window matrix -> DCT/FFT-like basis -> energy/delta/onset -> state_grid
```

## Stage D: real audio

Run SpeechCommands with scaffold/honesty audits.

## Stage E: plug-in layer

Only after standalone acceptance:

```text
after Attention -> before Attention -> replace Attention
```

Replacement requires TokenSlotAdapter and incremental update.
