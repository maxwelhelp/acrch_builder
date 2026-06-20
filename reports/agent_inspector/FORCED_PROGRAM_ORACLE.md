# Forced Program Oracle

- state_update_contract_pass: `True`
- raw_primitive_pass: `True`
- invariants_pass: `True`
- controller_address_choice_delta: `0.025533827021718025`
- final_read_last_layer0_output_bypass_delta: `0.0`

## Raw primitive audit

- `diff`/diff: accuracy=`1.0000` max_error=`0`
- `merge`/merge: accuracy=`1.0000` max_error=`0`
- `product`/product: accuracy=`1.0000` max_error=`0`

## Address contamination

- `diff`: raw=`1.0000` addressed=`1.0000` flip_rate=`0.0000` output_delta=`0`
- `merge`: raw=`1.0000` addressed=`1.0000` flip_rate=`0.0000` output_delta=`0`
- `product`: raw=`1.0000` addressed=`1.0000` flip_rate=`0.0000` output_delta=`0`

## Sequential state audit

- `diff`: invariants=`True` raw_path_accuracy=`1.0000` addressed_path_accuracy=`1.0000` raw_first_divergence=`None` addressed_first_divergence=`None`
  - L0 0->1 diff: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
- `merge`: invariants=`True` raw_path_accuracy=`1.0000` addressed_path_accuracy=`1.0000` raw_first_divergence=`None` addressed_first_divergence=`None`
  - L0 0->1 merge: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
- `product`: invariants=`True` raw_path_accuracy=`1.0000` addressed_path_accuracy=`1.0000` raw_first_divergence=`None` addressed_first_divergence=`None`
  - L0 0->1 product: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
- `two_diff`: invariants=`True` raw_path_accuracy=`1.0000` addressed_path_accuracy=`1.0000` raw_first_divergence=`None` addressed_first_divergence=`None`
  - L0 0->1 diff: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
  - L0 2->3 diff: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
- `chain_diff_merge`: invariants=`True` raw_path_accuracy=`1.0000` addressed_path_accuracy=`1.0000` raw_first_divergence=`None` addressed_first_divergence=`None`
  - L0 0->1 diff: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
  - L0 2->3 diff: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
  - L1 1->3 merge: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
- `chain_diff_product`: invariants=`True` raw_path_accuracy=`1.0000` addressed_path_accuracy=`1.0000` raw_first_divergence=`None` addressed_first_divergence=`None`
  - L0 0->1 diff: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
  - L0 2->3 diff: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
  - L1 1->3 product: input_src_mae=`0` input_tgt_mae=`0` primitive_mae=`0` target_after_update_mae=`0`
