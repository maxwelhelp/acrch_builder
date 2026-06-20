from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from .audio_frontend import AudioMatrixClassifier, SyntheticAudioOrderTask, build_frontend
from .reporting import ensure_dir, write_json
from .train_vertical_slice import curriculum_phase


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train and audit audio frontend wrappers on a synthetic order task.")
    ap.add_argument("--variant", default="structured", choices=["raw", "conv", "structured"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=24)
    ap.add_argument("--eval-steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--dim", type=int, default=24)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--sim-rank", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", default="none", choices=["none", "fp16", "bf16"])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau-start", type=float, default=1.0)
    ap.add_argument("--tau-min", type=float, default=0.5)
    ap.add_argument("--tau-decay", type=float, default=0.95)
    ap.add_argument("--curriculum-schedule", default="phased", choices=["teacher", "phased"])
    ap.add_argument("--honesty-floor", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="agent_reports/audio_frontend_smoke")
    ap.add_argument("--latest-report", default="LATEST_AUDIO_FRONTEND_REPORT.md")
    return ap


def amp_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


@torch.no_grad()
def evaluate(
    model: AudioMatrixClassifier,
    task: SyntheticAudioOrderTask,
    steps: int,
    batch_size: int,
    device: str,
    tau: float,
    curriculum_mode: str,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for _ in range(steps):
        batch = task.sample(batch_size, device)
        logits, _ = model(batch.waveforms, tau=tau, curriculum_mode=curriculum_mode)
        loss = F.cross_entropy(logits, batch.y)
        pred = logits.argmax(dim=-1)
        correct += (pred == batch.y).sum().item()
        total += batch_size
        loss_sum += float(loss.detach().cpu()) * batch_size
    return {
        "acc": correct / max(1, total),
        "loss": loss_sum / max(1, total),
    }


def train_variant(args) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    task = SyntheticAudioOrderTask(length=args.length)
    frontend = build_frontend(args.variant, slots=args.slots, dim=args.dim)
    model = AudioMatrixClassifier(
        frontend=frontend,
        dim=args.dim,
        slots=args.slots,
        layers=args.layers,
        classes=2,
        top_k=args.top_k,
        sim_rank=args.sim_rank,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and args.amp == "fp16"))
    dtype = amp_dtype(args.amp)

    best_full = 0.0
    train_acc = 0.0
    train_loss = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        phase = curriculum_phase(epoch, args.epochs, args.curriculum_schedule)
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        steps = max(1, args.steps_per_epoch)
        for _ in range(steps):
            batch = task.sample(args.batch_size, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda") and dtype != torch.float32):
                logits, _ = model(batch.waveforms, tau=max(args.tau_min, args.tau_start * (args.tau_decay ** (epoch - 1))), curriculum_mode=phase)
                loss = F.cross_entropy(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += batch.waveforms.shape[0]
            pred = logits.argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
            loss_sum += float(loss.detach().cpu()) * batch.waveforms.shape[0]
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)
        full_eval = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="teacher")
        best_full = max(best_full, full_eval["acc"])

    full = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="teacher")
    audit = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="audit")
    deploy = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="deploy")
    honesty_score = deploy["acc"] / max(1e-8, full["acc"])

    report = {
        "variant": args.variant,
        "task": "synthetic_audio_order",
        "curriculum_schedule": args.curriculum_schedule,
        "honesty_floor": float(args.honesty_floor),
        "honesty_score": float(honesty_score),
        "train_acc": float(train_acc),
        "train_loss": float(train_loss),
        "best_full_acc": float(best_full),
        "full_acc": float(full["acc"]),
        "audit_acc": float(audit["acc"]),
        "deploy_acc": float(deploy["acc"]),
        "deploy_above_random": bool(deploy["acc"] >= 0.5),
        "deploy_above_random_margin": float(deploy["acc"] - 0.5),
        "status": "PASS" if deploy["acc"] >= 0.5 and honesty_score >= args.honesty_floor else "FAIL",
        "frontend": {
            "params": float(sum(p.numel() for p in frontend.parameters())),
            "flops": float(frontend.estimate_flops(args.length, args.slots, args.dim)),
            "activation_bytes": float(frontend.estimate_activation_bytes(args.length, args.slots, args.dim)),
        },
        "model": {
            "params": float(sum(p.numel() for p in model.parameters())),
            "approx_flops": float(model.report(args.length, args.slots, args.dim)["total_flops"]),
            "approx_activation_bytes": float(model.report(args.length, args.slots, args.dim)["total_activation_bytes"]),
        },
        "seconds": time.time() - start,
    }
    return report


def main() -> int:
    args = parser().parse_args()
    report = train_variant(args)
    out_dir = ensure_dir(Path(args.out_dir))
    write_json(out_dir / "final_report.json", report)
    (out_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# Audio frontend report",
                "",
                f"- variant: `{report['variant']}`",
                f"- status: `{report['status']}`",
                f"- honesty_score: `{report['honesty_score']}`",
                f"- full_acc: `{report['full_acc']}`",
                f"- audit_acc: `{report['audit_acc']}`",
                f"- deploy_acc: `{report['deploy_acc']}`",
                f"- frontend_params: `{report['frontend']['params']}`",
                f"- frontend_flops: `{report['frontend']['flops']}`",
                f"- frontend_activation_bytes: `{report['frontend']['activation_bytes']}`",
                f"- model_params: `{report['model']['params']}`",
                f"- model_flops: `{report['model']['approx_flops']}`",
                f"- model_activation_bytes: `{report['model']['approx_activation_bytes']}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
