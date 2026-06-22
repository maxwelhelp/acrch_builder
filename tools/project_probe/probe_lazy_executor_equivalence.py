from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from arch_builder.model import ActionMatrixModel


def main() -> int:
    dim = 32
    slots = 4
    layers = 1
    top_k = 8
    classes = 2
    device = "cpu"

    torch.manual_seed(42)
    model_non_lazy = ActionMatrixModel(
        dim=dim,
        slots=slots,
        layers=layers,
        classes=classes,
        top_k=top_k,
        enable_vnext=True,
        enable_lazy_executor=False,
    ).to(device)

    model_lazy = ActionMatrixModel(
        dim=dim,
        slots=slots,
        layers=layers,
        classes=classes,
        top_k=top_k,
        enable_vnext=True,
        enable_lazy_executor=True,
        utility_budget=top_k,
    ).to(device)

    model_lazy.load_state_dict(model_non_lazy.state_dict())

    model_non_lazy.eval()
    model_lazy.eval()

    batch_size = 16

    # Forward non-lazy
    torch.manual_seed(100)
    x1 = torch.randn(batch_size, slots, dim, device=device)
    with torch.no_grad():
        out_non_lazy, tr_nl = model_non_lazy(x1)

    # Forward lazy
    torch.manual_seed(100)
    x2 = torch.randn(batch_size, slots, dim, device=device)
    with torch.no_grad():
        out_lazy, tr_l = model_lazy(x2)

    # Compute difference
    diff_logits = torch.abs(out_non_lazy - out_lazy).max().item()

    if diff_logits > 1e-5:
        print(f"FAIL: Equivalence probe failed. Max absolute difference: {diff_logits:.2e}")
        return 1

    # Now profile performance/speed
    # Lazy model with small budget (e.g. 2 instead of 8)
    model_lazy_small = ActionMatrixModel(
        dim=dim,
        slots=slots,
        layers=layers,
        classes=classes,
        top_k=top_k,
        enable_vnext=True,
        enable_lazy_executor=True,
        utility_budget=2,
    ).to(device)
    model_lazy_small.load_state_dict(model_non_lazy.state_dict())
    model_lazy_small.eval()

    # Warmup
    for _ in range(10):
        _ = model_non_lazy(x1)
        _ = model_lazy_small(x1)

    # Profile non-lazy
    t0 = time.time()
    for _ in range(100):
        _ = model_non_lazy(x1)
    non_lazy_time = time.time() - t0

    # Profile lazy with small budget
    t1 = time.time()
    for _ in range(100):
        _ = model_lazy_small(x1)
    lazy_time = time.time() - t1

    speedup = non_lazy_time / max(1e-8, lazy_time)

    out = {
        "status": "PASS",
        "max_logits_diff": float(diff_logits),
        "non_lazy_time_seconds": float(non_lazy_time),
        "lazy_time_seconds": float(lazy_time),
        "speedup_ratio": float(speedup),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
