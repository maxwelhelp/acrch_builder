# HybridScanner proof

- status: `PASS`
- candidate_count: `10`
- top_k_cutoff: `9`
- expected_semantic_rank: `2`
- semantic_entropy: `3.1899380683898926`
- embedding_rank: `17.0`
- semantic_grid_mismatch: `0.3333333333333333`
- global_rescue_rate: `1.0`
- semantic_candidate_real_gain: `0.486328125`
- usage_candidate_real_gain: `0.486328125`
- credited_primitive: `product`
- usage_credit_observations: `64.0`
- collapse_flag: `False`

## Paired comparisons

```json
{
  "grid": {
    "recovery": 0.0,
    "heldout_quality": 0.513671875
  },
  "grid_semantic": {
    "recovery": 1.0,
    "heldout_quality": 1.0
  },
  "grid_semantic_usage": {
    "recovery": 1.0,
    "heldout_quality": 1.0
  },
  "full_hybrid_random": {
    "recovery": 1.0,
    "heldout_quality": 1.0
  }
}
```

## Source mass / gain

```json
{
  "mass": {
    "grid": 0.026698012065025978,
    "semantic": 0.4810479949301225,
    "usage": 0.4745775171977584,
    "random": 0.01767656444280874
  },
  "gain": {
    "grid": -0.022289691943127965,
    "semantic": 0.1619400289017341,
    "usage": 0.2197265625,
    "random": -0.011932373046875
  }
}
```

## Checks

- expected_absent_grid: `PASS`
- expected_present_non_grid: `PASS`
- real_cutoff: `PASS`
- global_rescue_positive: `PASS`
- non_grid_heldout_gain_positive: `PASS`
- full_beats_grid: `PASS`
- embedding_rank_healthy: `PASS`
- semantic_entropy_healthy: `PASS`
- random_mass_bounded: `PASS`
- usage_from_credit_winner: `PASS`
- no_collapse: `PASS`
