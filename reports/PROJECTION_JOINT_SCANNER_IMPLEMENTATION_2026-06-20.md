# Projection scanner source implementation

Scope: scanner sources, standalone probe, and report schema only. The main
SpeechCommands trainer is intentionally not wired to these sources in this
change.

Implemented:

- `single_signed_projection`: absolute magnitude ranks candidates while the
  signed score and direction-normalized feature remain available to the
  controller/simulator.
- `pair_jl16_bilinear`: scores only a bounded candidate pool in production.
  Its default pool is the top single proposals; callers may supply a combined
  active/semantic/usage/random pool.
- Exact all-pairs scoring exists only in the standalone probe.
- Projection scores are proposal evidence, never training truth. Real training
  must supply a gradient/credit signal and verify proposals using bounded
  measured delta-loss.
- Neither source reads `expected_actions`, primitive labels, or layer roles.

Run the full P40 probe:

```bash
bash commands/probe_projection_joint_scanner.sh
```

The JSON and Markdown reports are written under `reports/agent_inspector/`.
CUDA is required for a full PASS because the 1.5x speed check is not meaningful
on CPU. A CPU run can only return `SMOKE_PASS`.

This synthetic probe validates the scanner operators. It is not real discovery
acceptance and cannot prove SpeechCommands learning quality.

## Real-train stabilization

The recommended real-train mode is single signed projection only. Pair JL is
disabled unless `ENABLE_PAIR_JL_BILINEAR=1` is explicitly supplied. Projection
logit capping is disabled by default (`PROJECTION_LOGIT_CAP=0.0`). Collapse is
controlled on the final primitive distribution across all scanner sources with
a top-share target of 0.60 and normalized primitive-entropy pressure.
