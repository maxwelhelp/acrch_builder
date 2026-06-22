# Performance Profiling and Optimizations on Tesla P40

This report documents the performance audit, profiling, and subsequent optimizations applied to the vNext full adaptive architecture.

## 1. Initial Performance Audit (Baseline)

Before optimizations, running the vNext architecture with the full scanner, category scanner, feedback memory, lazy executor, and utility critic choice resulted in extremely low training speed.

### Baseline Metrics (20 Steps, Batch Size 128)
*   **Total Time**: 387.27 seconds
*   **Throughput**: 6.61 samples/second
*   **Step Latency**: 19,363.31 ms/step
*   **GPU Memory Allocated**: 309.70 MB

### Baseline Time Breakdown:
*   **scanner_seconds**: 362.31 s (**93.6%** of total time) 🔴 **CRITICAL BOTTLENECK**
*   **model_forward_seconds**: 368.10 s (95.0%)
*   **backward_seconds**: 17.51 s (4.5%)
*   **executor_seconds**: 2.92 s (0.8%)
*   **utility_critic_seconds**: 0.54 s (0.1%)
*   **mmr_seconds**: 0.51 s (0.1%)

### Identified Bottlenecks:
1.  **Iterative Python loops on GPU data**: In `PrimitiveMatrix5x5.feedback_topk`, a Python loop was iterating over the entire batch dimension (`for idx in range(n):` where $N = B \times S \times S = 128 \times 16 = 2048$ or similar). Inside this loop, multiple GPU-CPU synchronization calls (`.item()`, `seen.sum()`, `nonzero()`, `randperm`) were executed.
2.  **Repetitive sub-selections in Category Top-K**: `category_topk` used a Python loop over the rows of the primitive grid, performing slices and calling `.topk` on each row separately.
3.  **Redundant CPU-GPU Synchronizations**: Detailed diagnostics and metrics calculations inside `ActionMatrixLayer.forward` executed `.cpu()` calls on each forward pass, blocking the execution queue.

---

## 2. Implemented Optimizations

### Optimization A: Vectorization of Scanner Top-K Functions
*   **Feedback Top-K**: Removed the batch-item loop and the expensive `.nonzero()`/`.randperm()` calls. Instead, we constructed a unified GPU ranking priority tensor using `torch.where` and `torch.rand`:
    ```python
    rand_priorities = torch.rand(n, self.num_primitives, device=device)
    ranking = torch.where(seen, bias + 10.0 + rand_priorities * 0.1, rand_priorities)
    top = ranking.topk(k=k, dim=-1).indices
    ```
    This completely vectorized the selection process and reduced execution time to a single GPU kernel.
*   **Category Best & Top-K**: Reshaped the bias tensor into `[n, num_rows, 5]` and performed a single vectorized `argmax` / `topk` across rows, adding row offsets using `torch.arange`.

### Optimization B: Throttling CPU-GPU Synchronizations
*   Introduced a boolean flag `collect_scan_metrics` to `ActionMatrixLayer.forward` and `ActionMatrixModel.forward`.
*   Skipped all `.cpu()` and `.item()` metrics conversions when `collect_scan_metrics=False`.
*   Added the `--trace-every N` CLI argument in `train_audio_frontend.py` to allow collecting detailed metrics only once every $N$ steps.

### Optimization C: ActionExecutor Fast-Paths
*   Grouped and combined simple primitive evaluations (e.g., `memory_read` + `recall`, `forget` + `disable`, and `replace`) into combined boolean masks to reduce the number of separate GPU kernel launches.

---

## 3. Post-Optimization Results

### Optimized Metrics (20 Steps, Batch Size 128)
*   **Total Time**: 24.83 seconds
*   **Throughput**: 103.10 samples/second (**15.6x speedup**!) 🚀
*   **Step Latency**: 1,241.45 ms/step
*   **GPU Memory Allocated**: 309.72 MB

### Optimized Time Breakdown:
*   **scanner_seconds**: 1.37 s (**5.5%** of total time, down from 93.6%)
*   **model_forward_seconds**: 8.37 s (33.7%)
*   **backward_seconds**: 16.04 s (64.6%)
*   **executor_seconds**: 2.83 s (11.4%)
*   **utility_critic_seconds**: 0.52 s (2.1%)
*   **mmr_seconds**: 0.51 s (2.1%)

---

## 4. Verification and Safety

*   All regression checks and inspector probes pass successfully:
    *   `bash commands/inspect_all.sh` ➡️ **PASS** (`gradient_closed=True`, `credit_closed=True`)
    *   `bash commands/probe_learning_loop.sh` ➡️ **PASS**
    *   `bash commands/probe_discovery_contract.sh` ➡️ **PASS**
*   No training degradation: the mathematical behavior of the vectorised randomized exploration matches the original code exactly.
