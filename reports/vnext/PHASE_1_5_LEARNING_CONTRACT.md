# Phase 1.5 Learning Contract

## 1. Goal and Overview

This contract establishes the learning loop structure, baseline equivalence guarantees, and optimization isolation for the vNext adaptive layer (`Scanner -> UtilityCritic -> MMR/Controller -> Executor -> Credit -> Feedback Memory`).

## 2. Baseline Equivalence Guarantees

When all vNext flags are disabled (`ENABLE_VNEXT=0`, `ENABLE_UTILITY_CRITIC_PROBE=0`, etc.):
- The total parameter count must match exactly 51,058 (including frontend feature norm/projection and classifier readout) and 30,841 for the backbone alone.
- The forward pass outputs and RNG consumption must be identical to the baseline main branch.
- Spectral, attention-like, and mined primitives are not instantiated or executed, and allocate zero parameters.

## 3. Isolated Diagnostic Probe Mode

When `ENABLE_UTILITY_CRITIC_PROBE=1` but `ENABLE_VNEXT=0`:
- The `UtilityCritic` is created and evaluated.
- The critic is trained using an isolated optimizer (`critic_opt`) that only targets parameters matching `utility_critic` or `utility_logit_scale`.
- The main optimizer (`opt`) excludes all critic parameters.
- The backward trace uses detached input tensors (`ctx_gpu.detach()`, etc.) to prevent gradient leakage into the main backbone.
- The critic's training is strictly diagnostic and has zero impact on runtime choices or main model gradients.

## 4. Learning Loop Contract

### 4.1 Candidate Scanner
- Proposes candidates from: Local grid, Semantic (cosine similarity), Usage (frequency/ema prior), Random, and conditionally Feedback Memory or Category scanner.
- Source IDs: Grid (0), Semantic (1), Usage (2), Random (3), Feedback (4), Category (7).

### 4.2 UtilityCritic
- Predicts utility score and uncertainty/variance.
- Target value: Detached gradient-trace credit (counterfactual alignment target).
- Loss function: NLL (Gaussian negative log-likelihood) + Pairwise Ranking Loss.

### 4.3 MMR/Controller
- Balances candidate utility and behavior diversity.
- Modes: `effect` (primitive output similarity), `identity` (orthogonal indicators), `hybrid` (weighted combination).
- Gated by `enable_mmr_controller=1` (default off).

### 4.4 Lazy Executor
- Executes only the selected subset of candidates up to `utility_budget`.
- Reduces execution FLOPs during candidate sub-selection.

### 4.5 Credit Assignment
- Computes counterfactual credit gains: $CE(\text{ablated}) - CE(\text{full})$.
- Updates global usage score and layer-specific feedback memory.

### 4.6 Layer-Specific Feedback Memory
- Shared global feedback memory is upgraded to layer-by-primitive storage: shape `[num_layers, num_primitives]`.
- Updated only when `enable_scanner_feedback_memory=1` using layer-aligned credit observations.
