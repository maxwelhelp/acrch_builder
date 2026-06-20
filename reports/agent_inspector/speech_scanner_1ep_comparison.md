# SpeechCommands scanner 1ep comparison

| metric | old | new | delta |
|---|---:|---:|---:|
| acc | 0.367188 | 0.396094 | +0.0289063 |
| samples_per_second | 183.563 | 135.098 | -48.4652 |
| credit_closed | 1 | 1 | +0 |
| sim_disabled_delta | -0.0313355 | 0.178729 | +0.210064 |
| choice_without_sim_delta | 0.0358555 | 0.0503442 | +0.0144887 |
| single_signed_projection_usage | 0 | 0.00971818 | +0.00971818 |
| pair_jl16_usage | 0 | 0 | +0 |
| primitive_top_share | 0.490314 | 0.692105 | +0.201792 |
| active_cells | 29 | 19.3333 | -9.66667 |
| projection_logit_cap | 0 | 1 | +1 |
| projection_logit_clipped_fraction | 0 | 0.257576 | +0.257576 |

Single seed/epoch smoke; this is not acceptance.
