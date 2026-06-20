# TASK 12 RESULT — PASS

Task: universal add-on plug-in modes.

Goal reached: a small reference Transformer with a plug-in mechanism was tested in
after-attention and before-attention positions, with attention-only and control
mechanisms as baselines.

Evidence:

- proof command: `bash commands/probe_transformer_plugin.sh`
- proof report: `reports/agent_inspector/TRANSFORMER_PLUGIN_PROOF.md`
- proof json: `reports/agent_inspector/transformer_plugin_proof.json`
- core gates inside proof: `bash commands/validate.sh`, `bash commands/inspect_with_probe.sh`

Benchmark:

- synthetic task: local motif parity over ordered token segments
- fixed seed proof: `seed=7`
- seq len: `48`
- vocab size: `64`
- model dim: `48`
- heads: `4`

Key aggregate metrics:

| setting | val acc mean | output norm | mechanism norm | latency ms | params | FLOPs | memory bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| after:learned | 0.5283 | 11.2481 | 7.6277 | 5.23 | 31683 | 1143072 | 18816 |
| after:identity | 0.5098 | 7.4502 | 0.0000 | 0.00 | 31683 | 1143072 | 18816 |
| after:random | 0.5078 | 10.4956 | 6.9303 | 0.00 | 31683 | 1143072 | 18816 |
| after:frozen | 0.5146 | 10.2611 | 6.8247 | 0.00 | 31683 | 1143072 | 18816 |
| attention_only:identity | 0.4727 | 7.3265 | 0.0000 | 0.00 | 22274 | 1128960 | 18816 |
| before:learned | 0.5293 | 11.4188 | 7.1371 | 5.23 | 31683 | 1143072 | 18816 |

Acceptance checks:

- after-attention mechanism beats attention-only — PASS
- after-attention mechanism beats identity — PASS
- after-attention mechanism beats random — PASS
- after-attention mechanism beats frozen — PASS
- before-attention mechanism beats attention-only — PASS
- shape/device/dtype smoke — PASS
- gradient/credit closure retained — PASS
- mechanism not zero / not residual-only — PASS
- core gradient/credit gates retained — PASS

Additional notes:

- `mechanism_norm` is nonzero for learned and controlled variants.
- `attention_only` is lower than both add-on placements.
- core project validate and inspector gates stayed green inside the proof.

Verdict: PASS.
Next allowed task: 13.
