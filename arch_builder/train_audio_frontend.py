from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from .audio_frontend import AudioBatch as SyntheticAudioBatch
from .audio_frontend import AudioMatrixClassifier, SyntheticAudioOrderTask, build_frontend
from .reporting import ensure_dir, write_json
from .speechcommands_data import AudioBatch as SpeechCommandsAudioBatch
from .speechcommands_data import SpeechCommandsAcceptanceTask, normalize_classes
from .train_vertical_slice import curriculum_phase


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train and audit audio frontend wrappers on synthetic or SpeechCommands data.")
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "speechcommands"])
    ap.add_argument("--variant", default="structured", choices=["raw", "conv", "structured"])
    ap.add_argument("--data-root", default="../Functional Matrix Grower/data/speechcommands")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--classes", default="yes,no,up,down,left,right,on,off,stop,go")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--test-limit", type=int, default=0)
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
    ap.add_argument("--input-norm", default="none", choices=["none", "layernorm"])
    ap.add_argument("--state-norm", default="none", choices=["none", "layernorm"])
    ap.add_argument("--final-read", default="last", choices=["last", "mean", "learned"])
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


def _sample(task, batch_size: int, device: str, split: str = "training"):
    try:
        return task.sample(batch_size, device, split=split)
    except TypeError:
        return task.sample(batch_size, device)


def _batch_x(batch) -> torch.Tensor:
    if isinstance(batch, SyntheticAudioBatch):
        return batch.waveforms
    if isinstance(batch, SpeechCommandsAudioBatch):
        return batch.x
    if hasattr(batch, "waveforms"):
        return batch.waveforms
    if hasattr(batch, "x"):
        return batch.x
    raise TypeError(f"unsupported batch type: {type(batch)!r}")


def _batch_y(batch) -> torch.Tensor:
    if hasattr(batch, "y"):
        return batch.y
    raise TypeError(f"unsupported batch type: {type(batch)!r}")


def _move_batch(batch, device: str):
    x = _batch_x(batch).to(device)
    y = _batch_y(batch).to(device)
    if isinstance(batch, SpeechCommandsAudioBatch):
        return SpeechCommandsAudioBatch(x=x, y=y)
    if isinstance(batch, SyntheticAudioBatch):
        return SyntheticAudioBatch(waveforms=x, y=y)
    return type("Batch", (), {"x": x, "waveforms": x, "y": y})()


def _run_model(model: AudioMatrixClassifier, batch, tau: float, curriculum_mode: str, device: str):
    batch = _move_batch(batch, device)
    logits, trace = model(_batch_x(batch), tau=tau, curriculum_mode=curriculum_mode)
    return batch, logits, trace


@torch.no_grad()
def evaluate(
    model: AudioMatrixClassifier,
    task,
    steps: int,
    batch_size: int,
    device: str,
    tau: float,
    curriculum_mode: str,
    split: str = "validation",
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    source_mass = defaultdict(float)
    trace_sum = defaultdict(float)
    trace_count = 0
    for _ in range(max(1, steps)):
        batch = _sample(task, batch_size, device, split=split)
        batch, logits, trace = _run_model(model, batch, tau=tau, curriculum_mode=curriculum_mode, device=device)
        loss = F.cross_entropy(logits, _batch_y(batch))
        pred = logits.argmax(dim=-1)
        correct += (pred == _batch_y(batch)).sum().item()
        total += _batch_y(batch).shape[0]
        loss_sum += float(loss.detach().cpu()) * _batch_y(batch).shape[0]
        trace_count += 1
        for layer in trace.get("layers", []):
            scan = layer.get("scan_metrics", {})
            for key in ("grid_candidate_usage", "semantic_candidate_usage", "usage_candidate_usage", "random_candidate_usage", "scanner_source_mass_sum"):
                if key in scan:
                    trace_sum[key] += float(scan[key])
        if trace.get("layers"):
            first = trace["layers"][0]
            for key in ("listen_gate", "active", "cell_output_gate"):
                if key in first and hasattr(first[key], "float"):
                    trace_sum[f"layer0_{key}_mean"] += float(first[key].float().mean().detach().cpu())
    out = {
        "acc": correct / max(1, total),
        "loss": loss_sum / max(1, total),
    }
    if trace_count:
        for key, value in trace_sum.items():
            out[key] = value / trace_count
    return out


@torch.no_grad()
def collect_ablations(model: AudioMatrixClassifier, task, batch_size: int, device: str, tau: float) -> Dict[str, float]:
    model.eval()
    batch = _sample(task, batch_size, device, split="validation")
    batch = _move_batch(batch, device)
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    usage_state = model.backbone.pm.usage_score.detach().clone()

    def run(**kwargs):
        torch.random.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.backbone.pm.usage_score.copy_(usage_state)
        features = model.frontend(_batch_x(batch))
        logits, trace = model.backbone(features, tau=tau, curriculum_mode="deploy", **kwargs)
        return logits, trace

    logits_full, trace_full = run()
    logits_no_sim, _ = run(disable_sim=True)
    logits_no_gain, _ = run(disable_gain=True)
    logits_no_result, _ = run(disable_sim_result=True)
    logits_no_slot, _ = run(disable_slot_address=True)
    logits_no_layer0, _ = run(ablate_layer_output=0)

    out = {
        "sim_disabled_delta": float((F.cross_entropy(logits_no_sim, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
        "gain_disabled_delta": float((F.cross_entropy(logits_no_gain, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
        "sim_result_disabled_delta": float((F.cross_entropy(logits_no_result, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
        "slot_disabled_delta": float((F.cross_entropy(logits_no_slot, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
        "layer0_output_disabled_delta": float((F.cross_entropy(logits_no_layer0, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
    }
    if trace_full.get("layers"):
        first = trace_full["layers"][0]
        scan = first.get("scan_metrics", {})
        for key in ("grid_candidate_usage", "semantic_candidate_usage", "usage_candidate_usage", "random_candidate_usage", "scanner_source_mass_sum"):
            if key in scan:
                out[key] = float(scan[key])
    model.backbone.pm.usage_score.copy_(usage_state)
    return out


def _build_task(args):
    if args.dataset == "speechcommands":
        return SpeechCommandsAcceptanceTask(
            root=args.data_root,
            classes=normalize_classes(args.classes),
            download=args.download,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            test_limit=args.test_limit,
            seed=args.seed,
            seconds=args.seconds,
        )
    return SyntheticAudioOrderTask(length=args.length)


def _train(args, model: AudioMatrixClassifier, task, opt, scaler, dtype, device: str):
    best_full = 0.0
    train_acc = 0.0
    train_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        phase = curriculum_phase(epoch, args.epochs, args.curriculum_schedule)
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for _ in range(max(1, args.steps_per_epoch)):
            batch = _sample(task, args.batch_size, device, split="training")
            batch = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda") and dtype != torch.float32):
                logits, _ = model(_batch_x(batch), tau=max(args.tau_min, args.tau_start * (args.tau_decay ** (epoch - 1))), curriculum_mode=phase)
                loss = F.cross_entropy(logits, _batch_y(batch))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += _batch_y(batch).shape[0]
            pred = logits.argmax(dim=-1)
            correct += (pred == _batch_y(batch)).sum().item()
            loss_sum += float(loss.detach().cpu()) * _batch_y(batch).shape[0]
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)
        full_eval = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="teacher")
        best_full = max(best_full, full_eval["acc"])
    return train_acc, train_loss, best_full


def _report_synthetic(args, model: AudioMatrixClassifier, task, train_acc: float, train_loss: float, best_full: float, device: str, start: float) -> Dict[str, object]:
    full = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="teacher")
    audit = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="audit")
    deploy = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="deploy")
    honesty_score = deploy["acc"] / max(1e-8, full["acc"])
    waveform_length = args.length
    return {
        "variant": args.variant,
        "dataset": args.dataset,
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
            "params": float(sum(p.numel() for p in model.frontend.parameters())),
            "flops": float(model.frontend.report(waveform_length, args.slots, args.dim)["flops"]),
            "activation_bytes": float(model.frontend.report(waveform_length, args.slots, args.dim)["activation_bytes"]),
        },
        "model": {
            "params": float(sum(p.numel() for p in model.parameters())),
            "approx_flops": float(model.report(waveform_length, args.slots, args.dim)["total_flops"]),
            "approx_activation_bytes": float(model.report(waveform_length, args.slots, args.dim)["total_activation_bytes"]),
        },
        "seconds": time.time() - start,
    }


def _report_real(args, model: AudioMatrixClassifier, task: SpeechCommandsAcceptanceTask, train_acc: float, train_loss: float, best_full: float, device: str, start: float) -> Dict[str, object]:
    full = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="teacher", split="validation")
    audit = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="audit", split="validation")
    deploy = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="deploy", split="validation")
    test = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="deploy", split="testing") if task.test_ds is not None else None
    ablations = collect_ablations(model, task, min(args.eval_batch_size, 32), device, tau=args.tau_min)
    chance = 1.0 / max(1, len(task.classes))
    honesty_score = deploy["acc"] / max(1e-8, full["acc"])
    waveform_length = int(args.seconds * args.sample_rate)
    report = {
        "variant": args.variant,
        "dataset": args.dataset,
        "task": "speechcommands_real_acceptance",
        "data_root": args.data_root,
        "classes": task.classes,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "curriculum_schedule": args.curriculum_schedule,
        "honesty_floor": float(args.honesty_floor),
        "honesty_score": float(honesty_score),
        "train_acc": float(train_acc),
        "train_loss": float(train_loss),
        "best_full_acc": float(best_full),
        "full_acc": float(full["acc"]),
        "audit_acc": float(audit["acc"]),
        "deploy_acc": float(deploy["acc"]),
        "test_acc": float(test["acc"]) if test is not None else None,
        "deploy_above_random": bool(deploy["acc"] >= chance),
        "deploy_above_random_margin": float(deploy["acc"] - chance),
        "status": "PASS" if deploy["acc"] >= chance and honesty_score >= args.honesty_floor else "FAIL",
        "val_metrics": full,
        "audit_metrics": audit,
        "deploy_metrics": deploy,
        "test_metrics": test or {},
        "ablations": ablations,
        "frontend": {
            "params": float(sum(p.numel() for p in model.frontend.parameters())),
            "flops": float(model.frontend.report(waveform_length, args.slots, args.dim)["flops"]),
            "activation_bytes": float(model.frontend.report(waveform_length, args.slots, args.dim)["activation_bytes"]),
        },
        "model": {
            "params": float(sum(p.numel() for p in model.parameters())),
            "approx_flops": float(model.report(waveform_length, args.slots, args.dim)["total_flops"]),
            "approx_activation_bytes": float(model.report(waveform_length, args.slots, args.dim)["total_activation_bytes"]),
        },
        "seconds": time.time() - start,
    }
    report["checks"] = {
        "deploy_above_random": report["deploy_above_random"],
        "honesty_retained": report["honesty_score"] >= args.honesty_floor,
        "simulator_ce_ablation_positive": ablations["sim_disabled_delta"] > 0,
        "non_grid_scanner_usage_positive": (
            ablations.get("semantic_candidate_usage", 0.0) + ablations.get("usage_candidate_usage", 0.0) + ablations.get("random_candidate_usage", 0.0)
        ) > 0.05,
        "layer0_ablation_positive": ablations["layer0_output_disabled_delta"] > 0,
    }
    return report


def train_variant(args) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    task = _build_task(args)
    num_classes = 2 if args.dataset == "synthetic" else len(task.classes)
    frontend = build_frontend(args.variant, slots=args.slots, dim=args.dim)
    model = AudioMatrixClassifier(
        frontend=frontend,
        dim=args.dim,
        slots=args.slots,
        layers=args.layers,
        classes=num_classes,
        top_k=args.top_k,
        sim_rank=args.sim_rank,
        input_norm=args.input_norm,
        state_norm=args.state_norm,
        final_read=args.final_read,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and args.amp == "fp16"))
    dtype = amp_dtype(args.amp)

    start = time.time()
    train_acc, train_loss, best_full = _train(args, model, task, opt, scaler, dtype, device)
    if args.dataset == "speechcommands":
        report = _report_real(args, model, task, train_acc, train_loss, best_full, device, start)
    else:
        report = _report_synthetic(args, model, task, train_acc, train_loss, best_full, device, start)
    return report


def _write_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Audio frontend report",
        "",
        f"- dataset: `{report.get('dataset')}`",
        f"- variant: `{report.get('variant')}`",
        f"- status: `{report.get('status')}`",
        f"- honesty_score: `{report.get('honesty_score')}`",
        f"- full_acc: `{report.get('full_acc')}`",
        f"- audit_acc: `{report.get('audit_acc')}`",
        f"- deploy_acc: `{report.get('deploy_acc')}`",
        f"- frontend_params: `{report['frontend']['params']}`",
        f"- frontend_flops: `{report['frontend']['flops']}`",
        f"- frontend_activation_bytes: `{report['frontend']['activation_bytes']}`",
        f"- model_params: `{report['model']['params']}`",
        f"- model_flops: `{report['model']['approx_flops']}`",
        f"- model_activation_bytes: `{report['model']['approx_activation_bytes']}`",
        "",
    ]
    if "checks" in report:
        lines.append("## Checks")
        for key, value in report["checks"].items():
            lines.append(f"- {key}: `{'PASS' if value else 'FAIL'}`")
        lines.append("")
    if "ablations" in report:
        lines.append("## Ablations")
        for key, value in report["ablations"].items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parser().parse_args()
    report = train_variant(args)
    out_dir = ensure_dir(Path(args.out_dir))
    write_json(out_dir / "final_report.json", report)
    (out_dir / "REPORT.md").write_text(_write_markdown(report), encoding="utf-8")
    latest = Path(args.latest_report)
    latest.write_text(_write_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
