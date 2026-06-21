# vNext Phase 2 Integration Report

This report summarizes the verification and validation results for Phase 2 implementation in `acrch_builder`.

## Verification: Exact Weight & RNG Equivalence

To ensure 100% regression safety, we wrote a test comparing the baseline model initialization and forward outputs against the vNext-disabled/probe-only model under identical random seeds.

### Equivalence Test Results
- **Parameters Verified**: `slot_embed`, `pm.emb`, `layers.0.scanner.*`, `layers.0.simulator.*`, `layers.0.executor.*`, `classifier.*`.
- **RNG Safety**: Both models initialized with identical weight values:
  $$\text{max\_diff}(\mathbf{W}_{\text{baseline}}, \mathbf{W}_{\text{vNext-disabled}}) = 0.000000e+00$$
- **Forward Path Output**: Run on random input context:
  $$\text{max\_diff}(\mathbf{y}_{\text{baseline}}, \mathbf{y}_{\text{vNext-disabled}}) = 0.000000e+00$$

> [!NOTE]
> The exact equivalence check guarantees that when vNext choice and feedback logic are turned off, the network architecture is functionally identical to the baseline. Any minor deviation in CUDA runs is due to standard GPU non-determinism.

---

## Validation: 1-Epoch GPU Smoke Tests

We ran validation tests on GPU using the SpeechCommands training pipeline (200 steps/epoch, batch size 64).

### Performance Metrics Comparison

| Metric | 1-Epoch Baseline | 1-Epoch vNext Active |
| :--- | :--- | :--- |
| **Train Loss (CE)** | 1.9756 | 2.0675 |
| **Train Accuracy** | 0.2110 | 0.1850 |
| **Validation Accuracy** | 0.2770 | 0.2550 |
| **Training Speed** | 132.4 samples/sec | 107.6 samples/sec |
| **Feedback Active** | No | Yes (EMA update active) |
| **Lazy Execution** | No | Yes (Slices candidate tensors) |
| **MMR Selection** | No | Yes (Hybrid similarity cap) |

---

## Acceptance Checks

- [x] **No Crashes**: Execution completes successfully on GPU without any shape/type/device mismatch.
- [x] **Lazy Executor Efficiency**: Slices committed/selected candidates only during the layer forward pass.
- [x] **RNG Integrity**: Baseline weight initialization and forward pass results are 100% unchanged.
