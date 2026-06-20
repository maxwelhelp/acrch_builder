from __future__ import annotations

import json

import torch


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return float((a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False)).clamp_min(1e-8))


def main() -> None:
    torch.manual_seed(20260620)
    x = torch.randn(32768, 4, 64)
    d1 = x[:, 0] - x[:, 1]
    d2 = x[:, 2] - x[:, 3]
    joint = (d1 * d2).mean(dim=-1)
    y = (joint > 0).float()
    signed_y = y * 2.0 - 1.0

    d1_scalar = d1.mean(dim=-1)
    d2_scalar = d2.mean(dim=-1)
    result = {
        "single_diff_1_label_correlation": _corr(d1_scalar, signed_y),
        "single_diff_2_label_correlation": _corr(d2_scalar, signed_y),
        "joint_product_label_correlation": _corr(joint, signed_y),
        "single_diff_1_sign_accuracy": float(((d1_scalar > 0) == y.bool()).float().mean()),
        "single_diff_2_sign_accuracy": float(((d2_scalar > 0) == y.bool()).float().mean()),
        "joint_product_sign_accuracy": float(((joint > 0) == y.bool()).float().mean()),
    }
    result["pair_credit_required"] = bool(
        abs(result["single_diff_1_label_correlation"]) < 0.03
        and abs(result["single_diff_2_label_correlation"]) < 0.03
        and result["joint_product_sign_accuracy"] == 1.0
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pair_credit_required"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
