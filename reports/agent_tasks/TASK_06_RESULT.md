# TASK 06 RESULT — PASS

Task: HybridScanner semantic and usage proof.

- Grid-only path misses the correct primitive by construction under the real cutoff.
- Non-grid semantic source rescues the correct primitive and improves held-out quality.
- Usage now comes from delayed credit, not a hardcoded stable primitive list.
- The proof keeps source IDs and source mass/gain accounting visible.

| Check | Result |
|---|---|
| expected_absent_grid | PASS |
| expected_present_non_grid | PASS |
| real_cutoff | PASS |
| global_rescue_positive | PASS |
| non_grid_heldout_gain_positive | PASS |
| full_beats_grid | PASS |
| embedding_rank_healthy | PASS |
| semantic_entropy_healthy | PASS |
| random_mass_bounded | PASS |
| usage_from_credit_winner | PASS |
| no_collapse | PASS |

Key proof values:

- `candidate_count`: 10
- `top_k_cutoff`: 9
- `expected_semantic_rank`: 2
- `semantic_entropy`: 3.1899380683898926
- `embedding_rank`: 17.0
- `global_rescue_rate`: 1.0
- `semantic_candidate_real_gain`: 0.486328125
- `usage_candidate_real_gain`: 0.486328125
- `credited_primitive`: `product`
- `usage_credit_observations`: 64.0
- `collapse_flag`: False

Paired comparisons:

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

Source mass / gain:

```json
{
  "mass": {
    "grid": 0.02690556235029362,
    "semantic": 0.4808413727078005,
    "usage": 0.4745783846228733,
    "random": 0.017674766582786106
  },
  "gain": {
    "grid": -0.02252053990610331,
    "semantic": 0.16438230994152048,
    "usage": 0.2197265625,
    "random": -0.011932373046875
  }
}
```

Verdict: PASS.
Next allowed task: 07_LAYER_LISTENING_SPECIALIZATION if PASS.

No long training was run.
