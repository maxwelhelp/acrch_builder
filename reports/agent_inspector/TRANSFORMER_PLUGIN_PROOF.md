# Transformer plugin proof

- status: `PASS`
- after_mechanism_beats_attention_only: `True`
- after_mechanism_beats_identity: `True`
- after_mechanism_beats_random: `True`
- after_mechanism_beats_frozen: `True`
- before_mechanism_beats_attention_only: `True`
- shape_device_dtype_ok: `True`
- gradient_credit_closure_retained: `True`

## Aggregate table

| setting | val acc mean | val acc min | val acc max | output norm | mechanism norm | latency ms | params | FLOPs | memory bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| after:learned | 0.5283 | 0.5283 | 0.5283 | 11.2481 | 7.6277 | 5.23 | 31683 | 1143072 | 18816 |
| after:identity | 0.5098 | 0.5098 | 0.5098 | 7.4502 | 0.0000 | 0.00 | 31683 | 1143072 | 18816 |
| after:random | 0.5078 | 0.5078 | 0.5078 | 10.4956 | 6.9303 | 0.00 | 31683 | 1143072 | 18816 |
| after:frozen | 0.5146 | 0.5146 | 0.5146 | 10.2611 | 6.8247 | 0.00 | 31683 | 1143072 | 18816 |
| attention_only:identity | 0.4727 | 0.4727 | 0.4727 | 7.3265 | 0.0000 | 0.00 | 22274 | 1128960 | 18816 |
| before:learned | 0.5293 | 0.5293 | 0.5293 | 11.4188 | 7.1371 | 5.23 | 31683 | 1143072 | 18816 |
| before:identity | 0.5098 | 0.5098 | 0.5098 | 7.4502 | 0.0000 | 0.00 | 31683 | 1143072 | 18816 |
| before:random | 0.5059 | 0.5059 | 0.5059 | 11.4341 | 6.8998 | 0.00 | 31683 | 1143072 | 18816 |
| before:frozen | 0.5234 | 0.5234 | 0.5234 | 10.8900 | 6.8245 | 0.00 | 31683 | 1143072 | 18816 |

## Variants

| setting | seed | val acc | train acc | loss | output norm | mechanism norm | grad norm | params | FLOPs | memory bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| after:learned | 7 | 0.5283 | 0.5039 | 0.6922 | 11.2481 | 7.6277 | 0.3389 | 31683 | 1143072 | 18816 |
| after:identity | 7 | 0.5098 | 0.4954 | 0.6961 | 7.4502 | 0.0000 | 0.0000 | 31683 | 1143072 | 18816 |
| after:random | 7 | 0.5078 | 0.4954 | 0.7058 | 10.4956 | 6.9303 | 0.0000 | 31683 | 1143072 | 18816 |
| after:frozen | 7 | 0.5146 | 0.5046 | 0.6945 | 10.2611 | 6.8247 | 0.0000 | 31683 | 1143072 | 18816 |
| attention_only:identity | 7 | 0.4727 | 0.5078 | 0.7125 | 7.3265 | 0.0000 | 0.0000 | 22274 | 1128960 | 18816 |
| before:learned | 7 | 0.5293 | 0.5130 | 0.6915 | 11.4188 | 7.1371 | 0.5940 | 31683 | 1143072 | 18816 |
| before:identity | 7 | 0.5098 | 0.4954 | 0.6961 | 7.4502 | 0.0000 | 0.0000 | 31683 | 1143072 | 18816 |
| before:random | 7 | 0.5059 | 0.5020 | 0.7074 | 11.4341 | 6.8998 | 0.0000 | 31683 | 1143072 | 18816 |
| before:frozen | 7 | 0.5234 | 0.5104 | 0.6930 | 10.8900 | 6.8245 | 0.0000 | 31683 | 1143072 | 18816 |

## Checks
- after_mechanism_beats_attention_only: `PASS`
- after_mechanism_beats_identity: `PASS`
- after_mechanism_beats_random: `PASS`
- after_mechanism_beats_frozen: `PASS`
- before_mechanism_beats_attention_only: `PASS`
- shape_device_dtype_ok: `PASS`
- gradient_credit_closure_retained: `PASS`
- mechanism_not_zero_residual_only: `PASS`
- attention_only_baseline_reported: `PASS`
- core_gradient_credit_retained: `PASS`

## Raw report
```json
{
  "status": "PASS",
  "checks": {
    "after_mechanism_beats_attention_only": true,
    "after_mechanism_beats_identity": true,
    "after_mechanism_beats_random": true,
    "after_mechanism_beats_frozen": true,
    "before_mechanism_beats_attention_only": true,
    "shape_device_dtype_ok": true,
    "gradient_credit_closure_retained": true,
    "mechanism_not_zero_residual_only": true,
    "attention_only_baseline_reported": true,
    "core_gradient_credit_retained": true
  },
  "core_gates": [
    {
      "cmd": "bash commands/validate.sh",
      "returncode": 0,
      "stdout": "[validate] cleanup stale pycache\n[validate] py_compile\n[validate] import smoke\nActionMatrixModel ['identity', 'gated_keep', 'diff', 'contrast', 'smooth']\n[validate] cli help\n[validate] cleanup generated pycache\n[validate] forbidden files\n[validate] OK",
      "stderr": ""
    },
    {
      "cmd": "bash commands/inspect_with_probe.sh",
      "returncode": 0,
      "stdout": "[probe] wrote reports/agent_inspector/runtime_probe.json\n[probe] gradient_closed=True credit_closed=True\n[probe] recovery_loss_connected=True\n[probe] detached_enabled_losses=['branch_split_loss']\n[inspect] wrote /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/acrch_builder/reports/agent_inspector\n[inspect] gradient_closed=True bad=[]\n[inspect] credit_closed=True bad=[]\n[inspect-with-probe] summary: reports/agent_inspector/SUMMARY.md",
      "stderr": ""
    }
  ],
  "runs": [
    {
      "seed": 7,
      "placement": "after",
      "mechanism_mode": "learned",
      "task": "synthetic_motif_order",
      "train_acc": 0.50390625,
      "train_loss": 0.6940663109223048,
      "best_val_acc": 0.517578125,
      "val_acc": 0.5283203125,
      "val_loss": 0.692183181643486,
      "output_norm": 11.248091101646423,
      "pooled_norm": 6.7628538608551025,
      "logit_norm": 0.10546567291021347,
      "mechanism_norm": 7.627683758735657,
      "mechanism_abs_mean": 0.5569603815674782,
      "attn_norm": 0.997580498456955,
      "ff_norm": 2.6036113798618317,
      "grad_norm": 0.33885328378528357,
      "seconds": 12.313553248008247,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0,
      "setting": "after:learned"
    },
    {
      "seed": 7,
      "placement": "after",
      "mechanism_mode": "identity",
      "task": "synthetic_motif_order",
      "train_acc": 0.4954427083333333,
      "train_loss": 0.694045270482699,
      "best_val_acc": 0.513671875,
      "val_acc": 0.509765625,
      "val_loss": 0.6960770413279533,
      "output_norm": 7.450198113918304,
      "pooled_norm": 6.8309996128082275,
      "logit_norm": 0.4148595593869686,
      "mechanism_norm": 0.0,
      "mechanism_abs_mean": 0.0,
      "attn_norm": 1.0235988348722458,
      "ff_norm": 2.2467596530914307,
      "grad_norm": 0.0,
      "seconds": 10.634174133010674,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0,
      "setting": "after:identity"
    },
    {
      "seed": 7,
      "placement": "after",
      "mechanism_mode": "random",
      "task": "synthetic_motif_order",
      "train_acc": 0.4954427083333333,
      "train_loss": 0.6999503001570702,
      "best_val_acc": 0.5048828125,
      "val_acc": 0.5078125,
      "val_loss": 0.7057737931609154,
      "output_norm": 10.495613932609558,
      "pooled_norm": 6.700245380401611,
      "logit_norm": 0.4863472431898117,
      "mechanism_norm": 6.930290520191193,
      "mechanism_abs_mean": 0.6137838810682297,
      "attn_norm": 1.1946089267730713,
      "ff_norm": 2.959062933921814,
      "grad_norm": 0.0,
      "seconds": 12.591029678005725,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0,
      "setting": "after:random"
    },
    {
      "seed": 7,
      "placement": "after",
      "mechanism_mode": "frozen",
      "task": "synthetic_motif_order",
      "train_acc": 0.5045572916666666,
      "train_loss": 0.6930707767605782,
      "best_val_acc": 0.5263671875,
      "val_acc": 0.5146484375,
      "val_loss": 0.6944805979728699,
      "output_norm": 10.261133551597595,
      "pooled_norm": 6.772552311420441,
      "logit_norm": 0.3173903077840805,
      "mechanism_norm": 6.8247281312942505,
      "mechanism_abs_mean": 0.5519048869609833,
      "attn_norm": 1.1127199679613113,
      "ff_norm": 2.6000016033649445,
      "grad_norm": 0.0,
      "seconds": 11.557706075021997,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0,
      "setting": "after:frozen"
    },
    {
      "seed": 7,
      "placement": "attention_only",
      "mechanism_mode": "identity",
      "task": "synthetic_motif_order",
      "train_acc": 0.5078125,
      "train_loss": 0.6988863795995712,
      "best_val_acc": 0.5048828125,
      "val_acc": 0.47265625,
      "val_loss": 0.7124540731310844,
      "output_norm": 7.326477527618408,
      "pooled_norm": 6.760389924049377,
      "logit_norm": 0.4248114675283432,
      "mechanism_norm": 0.0,
      "mechanism_abs_mean": 0.0,
      "attn_norm": 1.0115223079919815,
      "ff_norm": 2.1148372888565063,
      "grad_norm": 0.0,
      "seconds": 10.999614753003698,
      "params": 22274.0,
      "flops": 1128960.0,
      "activation_bytes": 18816.0,
      "setting": "attention_only:identity"
    },
    {
      "seed": 7,
      "placement": "before",
      "mechanism_mode": "learned",
      "task": "synthetic_motif_order",
      "train_acc": 0.5130208333333334,
      "train_loss": 0.6946749240159988,
      "best_val_acc": 0.513671875,
      "val_acc": 0.529296875,
      "val_loss": 0.6914581656455994,
      "output_norm": 11.418835401535034,
      "pooled_norm": 6.813928484916687,
      "logit_norm": 0.1520352903753519,
      "mechanism_norm": 7.137139737606049,
      "mechanism_abs_mean": 0.5652009397745132,
      "attn_norm": 2.9028170108795166,
      "ff_norm": 2.119803875684738,
      "grad_norm": 0.5940220477059484,
      "seconds": 11.963246564031579,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0,
      "setting": "before:learned"
    },
    {
      "seed": 7,
      "placement": "before",
      "mechanism_mode": "identity",
      "task": "synthetic_motif_order",
      "train_acc": 0.4954427083333333,
      "train_loss": 0.694045270482699,
      "best_val_acc": 0.513671875,
      "val_acc": 0.509765625,
      "val_loss": 0.6960770413279533,
      "output_norm": 7.450198113918304,
      "pooled_norm": 6.8309996128082275,
      "logit_norm": 0.4148595593869686,
      "mechanism_norm": 0.0,
      "mechanism_abs_mean": 0.0,
      "attn_norm": 1.0235988348722458,
      "ff_norm": 2.2467596530914307,
      "grad_norm": 0.0,
      "seconds": 10.854896584001835,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0,
      "setting": "before:identity"
    },
    {
      "seed": 7,
      "placement": "before",
      "mechanism_mode": "random",
      "task": "synthetic_motif_order",
      "train_acc": 0.501953125,
      "train_loss": 0.7011939908067385,
      "best_val_acc": 0.5048828125,
      "val_acc": 0.505859375,
      "val_loss": 0.7073615789413452,
      "output_norm": 11.43410074710846,
      "pooled_norm": 6.74630069732666,
      "logit_norm": 0.3787616565823555,
      "mechanism_norm": 6.899842441082001,
      "mechanism_abs_mean": 0.6154067888855934,
      "attn_norm": 2.9143569469451904,
      "ff_norm": 2.695301353931427,
      "grad_norm": 0.0,
      "seconds": 11.555156211019494,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0,
      "setting": "before:random"
    },
    {
      "seed": 7,
      "placement": "before",
      "mechanism_mode": "frozen",
      "task": "synthetic_motif_order",
      "train_acc": 0.5104166666666666,
      "train_loss": 0.6940138339996338,
      "best_val_acc": 0.51171875,
      "val_acc": 0.5234375,
      "val_loss": 0.6930369362235069,
      "output_norm": 10.89004921913147,
      "pooled_norm": 6.831351220607758,
      "logit_norm": 0.19348313845694065,
      "mechanism_norm": 6.824453771114349,
      "mechanism_abs_mean": 0.5516106188297272,
      "attn_norm": 2.559987097978592,
      "ff_norm": 2.275031566619873,
      "grad_norm": 0.0,
      "seconds": 9.913461752003059,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0,
      "setting": "before:frozen"
    }
  ],
  "aggregates": {
    "after:learned": {
      "val_acc_mean": 0.5283203125,
      "val_acc_min": 0.5283203125,
      "val_acc_max": 0.5283203125,
      "output_norm_mean": 11.248091101646423,
      "mechanism_norm_mean": 7.627683758735657,
      "latency_ms_mean": 5.230165093962569,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0
    },
    "after:identity": {
      "val_acc_mean": 0.509765625,
      "val_acc_min": 0.509765625,
      "val_acc_max": 0.509765625,
      "output_norm_mean": 7.450198113918304,
      "mechanism_norm_mean": 0.0,
      "latency_ms_mean": 0.0,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0
    },
    "after:random": {
      "val_acc_mean": 0.5078125,
      "val_acc_min": 0.5078125,
      "val_acc_max": 0.5078125,
      "output_norm_mean": 10.495613932609558,
      "mechanism_norm_mean": 6.930290520191193,
      "latency_ms_mean": 0.0,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0
    },
    "after:frozen": {
      "val_acc_mean": 0.5146484375,
      "val_acc_min": 0.5146484375,
      "val_acc_max": 0.5146484375,
      "output_norm_mean": 10.261133551597595,
      "mechanism_norm_mean": 6.8247281312942505,
      "latency_ms_mean": 0.0,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0
    },
    "attention_only:identity": {
      "val_acc_mean": 0.47265625,
      "val_acc_min": 0.47265625,
      "val_acc_max": 0.47265625,
      "output_norm_mean": 7.326477527618408,
      "mechanism_norm_mean": 0.0,
      "latency_ms_mean": 0.0,
      "params": 22274.0,
      "flops": 1128960.0,
      "activation_bytes": 18816.0
    },
    "before:learned": {
      "val_acc_mean": 0.529296875,
      "val_acc_min": 0.529296875,
      "val_acc_max": 0.529296875,
      "output_norm_mean": 11.418835401535034,
      "mechanism_norm_mean": 7.137139737606049,
      "latency_ms_mean": 5.230165093962569,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0
    },
    "before:identity": {
      "val_acc_mean": 0.509765625,
      "val_acc_min": 0.509765625,
      "val_acc_max": 0.509765625,
      "output_norm_mean": 7.450198113918304,
      "mechanism_norm_mean": 0.0,
      "latency_ms_mean": 0.0,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0
    },
    "before:random": {
      "val_acc_mean": 0.505859375,
      "val_acc_min": 0.505859375,
      "val_acc_max": 0.505859375,
      "output_norm_mean": 11.43410074710846,
      "mechanism_norm_mean": 6.899842441082001,
      "latency_ms_mean": 0.0,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0
    },
    "before:frozen": {
      "val_acc_mean": 0.5234375,
      "val_acc_min": 0.5234375,
      "val_acc_max": 0.5234375,
      "output_norm_mean": 10.89004921913147,
      "mechanism_norm_mean": 6.824453771114349,
      "latency_ms_mean": 0.0,
      "params": 31683.0,
      "flops": 1143072.0,
      "activation_bytes": 18816.0
    }
  },
  "credit": {
    "credit_items": 2.0,
    "credit_staleness_mean": 0.5,
    "credit_age_max": 1.0
  },
  "baseline": {
    "seq_len": 48,
    "vocab_size": 64,
    "dim": 48,
    "num_heads": 4
  }
}
```
