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
from .reporting import append_csv, ensure_dir, write_json
from .speechcommands_data import AudioBatch as SpeechCommandsAudioBatch
from .speechcommands_data import SpeechCommandsAcceptanceTask, normalize_classes
from .credit import BoundedCounterfactualCredit, generic_discovery_health_loss


PROJECTION_METRIC_KEYS = (
    "single_signed_projection_usage",
    "single_signed_projection_candidate_count",
    "single_signed_projection_top_score",
    "single_signed_projection_signed_score_mean",
    "pair_jl16_usage",
    "pair_jl16_candidate_count",
    "pair_jl16_top_score",
    "pair_jl16_seconds",
    "pair_jl16_pairs_tested",
    "projection_logit_cap",
    "projection_logit_clipped_fraction",
)

SELF_DELTA_METRIC_KEYS = (
    "self_delta_mean",
    "self_delta_abs",
    "self_delta_score_mean",
    "self_delta_score_std",
    "self_delta_sim_norm",
    "self_delta_actual_norm",
    "self_delta_scale",
    "self_delta_enabled",
    "self_delta_choice_enabled",
)



def curriculum_phase(epoch: int, total_epochs: int, schedule: str) -> str:
    """Audio-only scaffold schedule; real discovery bypasses it entirely."""
    if schedule == "teacher":
        return "teacher"
    teacher_end = max(1, int(total_epochs * 0.33))
    audit_end = max(teacher_end + 1, int(total_epochs * 0.66))
    if epoch <= teacher_end:
        return "teacher"
    if epoch <= audit_end:
        return "audit"
    return "deploy"


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
    ap.add_argument("--discovery", action="store_true", help="oracle-free real counterfactual discovery")
    ap.add_argument("--credit-budget", type=int, default=8)
    ap.add_argument("--credit-interval", type=int, default=8)
    ap.add_argument("--credit-batch-size", type=int, default=16)
    ap.add_argument("--credit-alternative-budget", type=int, default=2)
    ap.add_argument("--lambda-credit-policy", type=float, default=0.20)
    ap.add_argument("--lambda-credit-simulator", type=float, default=0.10)
    ap.add_argument("--lambda-discovery-health", type=float, default=1.0)
    ap.add_argument("--target-active-cells", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--controller-baseline", default="learned", choices=["learned", "frozen", "random"])
    ap.add_argument("--enable-single-signed-projection", action="store_true")
    ap.add_argument("--single-proj-dim", type=int, default=32)
    ap.add_argument("--enable-pair-jl-bilinear", action="store_true")
    ap.add_argument("--pair-jl-dim", type=int, default=16)
    ap.add_argument("--pair-candidate-budget", type=int, default=64)
    ap.add_argument("--projection-logit-cap", type=float, default=0.0)
    ap.add_argument("--primitive-top-share-target", type=float, default=0.60)
    ap.add_argument("--primitive-entropy-floor", type=float, default=0.65)
    ap.add_argument("--enable-self-delta-probe", action="store_true")
    ap.add_argument("--enable-self-delta-choice", action="store_true")
    ap.add_argument("--self-delta-max-scale", type=float, default=0.25)
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
    x = _batch_x(batch).to(device, non_blocking=True)
    y = _batch_y(batch).to(device, non_blocking=True)
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
    source_trace_count = 0
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
            source_trace_count += 1
            scan = layer.get("scan_metrics", {})
            for key in (
                "grid_candidate_usage", "semantic_candidate_usage", "usage_candidate_usage", "random_candidate_usage", "global_candidate_usage",
                "grid_candidate_coverage", "semantic_candidate_coverage", "usage_candidate_coverage", "random_candidate_coverage", "global_candidate_coverage",
                "scanner_source_mass_sum",
                *PROJECTION_METRIC_KEYS,
            ):
                if key in scan:
                    trace_sum[key] += float(scan[key])

            sdm = layer.get("self_delta_metrics", {})
            for k, v in sdm.items():
                if hasattr(v, "detach"):
                    trace_sum[k] += float(v.detach().float().cpu())
                else:
                    trace_sum[k] += float(v)

            for key in ("self_delta_scale", "self_delta_enabled", "self_delta_choice_enabled"):
                if key in layer and hasattr(layer[key], "detach"):
                    trace_sum[key] += float(layer[key].detach().float().cpu())

        for layer_idx, layer in enumerate(trace.get("layers", [])):
            for key in ("listen_gate", "active", "cell_output_gate", "cell_write_mass"):
                if key in layer and hasattr(layer[key], "float"):
                    trace_sum[f"layer{layer_idx}_{key}_mean"] += float(layer[key].float().mean().detach().cpu())
            if "active" in layer:
                grid = layer["active"].view(-1, model.backbone.slots * model.backbone.slots).float().mean(dim=0)
                trace_sum[f"layer{layer_idx}_active_cells"] += float((grid > 0.05).float().sum().cpu())
            if "choice" in layer:
                choice = layer["choice"].float().clamp_min(1e-8)
                trace_sum[f"layer{layer_idx}_choice_entropy"] += float((-(choice * choice.log()).sum(dim=-1).mean()).cpu())
    out = {
        "acc": correct / max(1, total),
        "loss": loss_sum / max(1, total),
    }
    if trace_count:
        for key, value in trace_sum.items():
            divisor = source_trace_count if key.endswith("candidate_usage") or key.endswith("candidate_coverage") or key == "scanner_source_mass_sum" else trace_count
            out[key] = value / max(1, divisor)
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
    logits_no_state0, _ = run(ablate_state_after=0)

    def primitive_dist(trace):
        rows = []
        for layer in trace.get("layers", []):
            out = torch.zeros(
                layer["candidate_ids"].shape[0],
                model.backbone.pm.num_primitives,
                device=layer["choice"].device,
                dtype=layer["choice"].dtype,
            )
            out.scatter_add_(1, layer["candidate_ids"], layer["choice"])
            rows.append(out)
        return torch.cat(rows, dim=0) if rows else torch.zeros(1, 1, device=device)

    out = {
        "sim_disabled_delta": float((F.cross_entropy(logits_no_sim, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
        "gain_disabled_delta": float((F.cross_entropy(logits_no_gain, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
        "sim_result_disabled_delta": float((F.cross_entropy(logits_no_result, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
        "slot_disabled_delta": float((F.cross_entropy(logits_no_slot, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
        "layer0_output_disabled_delta": float((F.cross_entropy(logits_no_layer0, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
        "layer0_state_disabled_delta": float((F.cross_entropy(logits_no_state0, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()),
        "choice_without_sim_delta": float((primitive_dist(trace_full) - primitive_dist(run(disable_sim=True)[1])).abs().mean().cpu()),
    }

    if getattr(model.backbone, "enable_self_delta_probe", False) or any(
        getattr(layer, "enable_self_delta_probe", False) or getattr(layer, "enable_self_delta_choice", False)
        for layer in model.backbone.layers
    ):
        logits_no_self, trace_no_self = run(disable_self_delta=True)
        logits_zero_self, trace_zero_self = run(zero_self_delta=True)
        logits_shuf_self, trace_shuf_self = run(shuffle_self_delta=True)

        out["self_delta_disabled_delta"] = float(
            (F.cross_entropy(logits_no_self, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()
        )
        out["self_delta_zero_delta"] = float(
            (F.cross_entropy(logits_zero_self, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()
        )
        out["self_delta_shuffle_delta"] = float(
            (F.cross_entropy(logits_shuf_self, _batch_y(batch)) - F.cross_entropy(logits_full, _batch_y(batch))).detach().cpu()
        )
        out["choice_without_self_delta_delta"] = float(
            (primitive_dist(trace_full) - primitive_dist(trace_no_self)).abs().mean().cpu()
        )
        out["choice_zero_self_delta_delta"] = float(
            (primitive_dist(trace_full) - primitive_dist(trace_zero_self)).abs().mean().cpu()
        )
        out["choice_shuffle_self_delta_delta"] = float(
            (primitive_dist(trace_full) - primitive_dist(trace_shuf_self)).abs().mean().cpu()
        )

    if trace_full.get("layers"):
        first = trace_full["layers"][0]
        scan = first.get("scan_metrics", {})
        for key in (
            "grid_candidate_usage", "semantic_candidate_usage", "usage_candidate_usage", "random_candidate_usage", "global_candidate_usage",
            "grid_candidate_coverage", "semantic_candidate_coverage", "usage_candidate_coverage", "random_candidate_coverage", "global_candidate_coverage",
            "scanner_source_mass_sum",
            *PROJECTION_METRIC_KEYS,
        ):
            if key in scan:
                out[key] = float(scan[key])
    model.backbone.pm.usage_score.copy_(usage_state)
    return out


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _train_real_discovery(args, model, task, opt, scaler, dtype, device: str):
    """Real-data discovery with no expected actions and no layer-role schedule."""
    train_loader, _, _ = task.loaders(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        workers=args.workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )
    train_iter = iter(train_loader)
    credit = BoundedCounterfactualCredit(
        layers=model.backbone.num_layers,
        slots=model.backbone.slots,
        primitives=model.backbone.pm.num_primitives,
        budget=args.credit_budget,
        alternative_budget=args.credit_alternative_budget,
    )
    best = 0.0
    train_acc = train_loss = 0.0
    last_diag: Dict[str, float] = {}
    global_step = 0
    learned_controller = args.controller_baseline == "learned"
    sampling = "uniform" if args.controller_baseline == "random" else "softmax"
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total = correct = 0
        loss_sum = 0.0
        tau = max(args.tau_min, args.tau_start * (args.tau_decay ** (epoch - 1)))
        for step in range(max(1, args.steps_per_epoch)):
            global_step += 1
            raw_batch, train_iter = _next_batch(train_iter, train_loader)
            batch = _move_batch(raw_batch, device)
            if learned_controller:
                credit.advance(model.backbone.pm)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda") and dtype != torch.float32):
                features = model.frontend(_batch_x(batch))
                logits, trace = model.backbone(
                    features,
                    tau=tau,
                    curriculum_mode="deploy",
                    choice_sampling=sampling,
                    collect_scan_metrics=False,
                )
                ce = F.cross_entropy(logits, _batch_y(batch))
                if learned_controller:
                    policy_loss, simulator_loss, align_metrics = credit.alignment_losses(trace)
                    health_loss, health_metrics = generic_discovery_health_loss(
                        trace,
                        slots=model.backbone.slots,
                        num_primitives=model.backbone.pm.num_primitives,
                        target_active_cells=args.target_active_cells,
                        primitive_top_share_target=args.primitive_top_share_target,
                        primitive_entropy_floor=args.primitive_entropy_floor,
                    )
                else:
                    policy_loss = simulator_loss = health_loss = torch.zeros((), device=ce.device)
                    align_metrics = {"credit_alignment_items": 0.0, "sim_pred_real_corr": 0.0}
                    health_metrics = {"active_cells": 0.0, "primitive_top_share": 0.0, "primitive_entropy": 0.0, "choice_entropy": 0.0}
                loss = (
                    ce
                    + args.lambda_credit_policy * policy_loss
                    + args.lambda_credit_simulator * simulator_loss
                    + args.lambda_discovery_health * health_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if learned_controller and global_step % max(1, args.credit_interval) == 0:
                credit_raw, train_iter = _next_batch(train_iter, train_loader)
                credit_batch = _move_batch(credit_raw, device)
                credit_x = _batch_x(credit_batch)[: args.credit_batch_size]
                credit_y = _batch_y(credit_batch)[: args.credit_batch_size]
                with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda",
                    dtype=dtype,
                    enabled=device.startswith("cuda") and dtype != torch.float32,
                ):
                    credit_features = model.frontend(credit_x)
                credit.collect(model.backbone, credit_features, credit_y, tau=tau)

            n = _batch_y(batch).shape[0]
            total += n
            correct += (logits.argmax(dim=-1) == _batch_y(batch)).sum().item()
            loss_sum += float(loss.detach().cpu()) * n
            projection_diag = {key: 0.0 for key in PROJECTION_METRIC_KEYS}
            projection_layers = [
                layer.get("scan_metrics", {}) for layer in trace.get("layers", [])
            ]
            for key in PROJECTION_METRIC_KEYS:
                values = [float(scan[key]) for scan in projection_layers if key in scan]
                if values:
                    projection_diag[key] = sum(values) / len(values)
            last_diag = {
                "ce_loss": float(ce.detach().cpu()),
                "credit_policy_loss": float(policy_loss.detach().cpu()),
                "credit_simulator_loss": float(simulator_loss.detach().cpu()),
                "discovery_health_loss": float(health_loss.detach().cpu()),
                **align_metrics,
                **health_metrics,
                **credit.metrics(),
                **projection_diag,
            }
            if args.log_every > 0 and global_step % args.log_every == 0:
                print(
                    f"discovery step={global_step} ce={last_diag['ce_loss']:.4f} "
                    f"acc={correct/max(1,total):.3f} active={last_diag['active_cells']:.1f} "
                    f"top={last_diag['primitive_top_share']:.3f} "
                    f"credit={int(last_diag.get('credit_total_measurements', 0))} "
                    f"gain={last_diag.get('credit_gain_mean', 0.0):+.5f}",
                    flush=True,
                )
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)
        ev = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau, "deploy")
        best = max(best, ev["acc"])
        samples_per_second = total / max(1e-8, time.perf_counter() - epoch_start)
        last_diag["train_samples_per_second"] = float(samples_per_second)
        append_csv(
            ensure_dir(Path(args.out_dir)) / "metrics.csv",
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": ev["acc"],
                "val_loss": ev["loss"],
                "samples_per_second": samples_per_second,
                **last_diag,
            },
        )
        print(
            f"real-discovery epoch={epoch}/{args.epochs} train={train_acc:.3f} "
            f"val={ev['acc']:.3f} speed={samples_per_second:.1f}/s "
            f"credit_closed={bool(credit.metrics().get('credit_closed', 0))}",
            flush=True,
        )
    return train_acc, train_loss, best, {**last_diag, **credit.metrics()}


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
    honesty_score = deploy["acc"] / full["acc"] if full["acc"] > 0 else 0.0
    waveform_length = args.length
    return {
        "variant": args.variant,
        "dataset": args.dataset,
        "task": "synthetic_audio_order",
        "curriculum_schedule": "none" if args.discovery else args.curriculum_schedule,
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


def _report_real(args, model: AudioMatrixClassifier, task: SpeechCommandsAcceptanceTask, train_acc: float, train_loss: float, best_full: float, device: str, start: float, discovery_metrics: Dict[str, float] | None = None) -> Dict[str, object]:
    deploy = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="deploy", split="validation")
    if args.discovery:
        # Real discovery has no scaffold/hint modes. Reuse the exact same
        # measured deploy result instead of comparing different random batches.
        full = dict(deploy)
        audit = dict(deploy)
    else:
        full = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="teacher", split="validation")
        audit = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau=args.tau_min, curriculum_mode="audit", split="validation")
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
        "seed": args.seed,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "curriculum_schedule": "none" if args.discovery else args.curriculum_schedule,
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
        "supervision_mode": "real_counterfactual_discovery" if args.discovery else "ce_only",
        "expected_actions_used_for_training": False,
        "layer_role_priors_used": False,
        "controller_baseline": args.controller_baseline,
        "scanner_config": {
            "enable_single_signed_projection": bool(args.enable_single_signed_projection),
            "single_proj_dim": int(args.single_proj_dim),
            "enable_pair_jl_bilinear": bool(args.enable_pair_jl_bilinear),
            "pair_jl_dim": int(args.pair_jl_dim),
            "pair_candidate_budget": int(args.pair_candidate_budget),
            "projection_logit_cap": float(args.projection_logit_cap),
            "primitive_top_share_target": float(args.primitive_top_share_target),
            "primitive_entropy_floor": float(args.primitive_entropy_floor),
        },
        "val_metrics": full,
        "audit_metrics": audit,
        "deploy_metrics": deploy,
        "test_metrics": test or {},
        "ablations": ablations,
        "global_scan_usage": float(ablations.get("global_candidate_usage", 0.0)),
        "discovery_metrics": discovery_metrics or {},
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
    report["comparison_metrics"] = {
        "acc": float(deploy["acc"]),
        "samples_per_second": float((discovery_metrics or {}).get("train_samples_per_second", 0.0)),
        "credit_closed": float((discovery_metrics or {}).get("credit_closed", 0.0)),
        "sim_disabled_delta": float(ablations.get("sim_disabled_delta", 0.0)),
        "choice_without_sim_delta": float(ablations.get("choice_without_sim_delta", 0.0)),
        "single_signed_projection_usage": float((discovery_metrics or {}).get("single_signed_projection_usage", 0.0)),
        "pair_jl16_usage": float((discovery_metrics or {}).get("pair_jl16_usage", 0.0)),
        "primitive_top_share": float((discovery_metrics or {}).get("primitive_top_share", 0.0)),
        "active_cells": float((discovery_metrics or {}).get("active_cells", 0.0)),
        "projection_logit_cap": float((discovery_metrics or {}).get("projection_logit_cap", args.projection_logit_cap)),
        "projection_logit_clipped_fraction": float((discovery_metrics or {}).get("projection_logit_clipped_fraction", 0.0)),
        "self_delta_enabled": float(deploy.get("self_delta_enabled", 0.0)),
        "self_delta_choice_enabled": float(deploy.get("self_delta_choice_enabled", 0.0)),
        "self_delta_mean": float(deploy.get("self_delta_mean", 0.0)),
        "self_delta_abs": float(deploy.get("self_delta_abs", 0.0)),
        "self_delta_score_mean": float(deploy.get("self_delta_score_mean", 0.0)),
        "self_delta_score_std": float(deploy.get("self_delta_score_std", 0.0)),
        "self_delta_sim_norm": float(deploy.get("self_delta_sim_norm", 0.0)),
        "self_delta_actual_norm": float(deploy.get("self_delta_actual_norm", 0.0)),
        "self_delta_scale": float(deploy.get("self_delta_scale", 0.0)),
        "self_delta_disabled_delta": float(ablations.get("self_delta_disabled_delta", 0.0)),
        "self_delta_zero_delta": float(ablations.get("self_delta_zero_delta", 0.0)),
        "self_delta_shuffle_delta": float(ablations.get("self_delta_shuffle_delta", 0.0)),
        "choice_without_self_delta_delta": float(ablations.get("choice_without_self_delta_delta", 0.0)),
        "choice_zero_self_delta_delta": float(ablations.get("choice_zero_self_delta_delta", 0.0)),
        "choice_shuffle_self_delta_delta": float(ablations.get("choice_shuffle_self_delta_delta", 0.0)),
    }
    report["checks"] = {
        "deploy_above_random": deploy["acc"] >= chance + (0.02 if args.discovery else 0.0),
        "honesty_retained": report["honesty_score"] >= args.honesty_floor,
        "simulator_ce_ablation_positive": ablations["sim_disabled_delta"] > 0,
        "simulator_changes_choice": ablations.get("choice_without_sim_delta", 0.0) > 1e-5,
        "non_grid_scanner_usage_positive": (
            ablations.get("semantic_candidate_usage", 0.0)
            + ablations.get("usage_candidate_usage", 0.0)
            + ablations.get("random_candidate_usage", 0.0)
        ) > 0.05,
        "layer0_ablation_positive": (
            ablations["layer0_state_disabled_delta"] > 0
            if args.layers > 1
            else ablations["layer0_output_disabled_delta"] > 0
        ),
    }
    if args.discovery and args.controller_baseline == "learned":
        report["checks"].pop("honesty_retained", None)
        report["checks"].update({
            "credit_closed": bool((discovery_metrics or {}).get("credit_closed", 0.0)),
            "bounded_credit_budget": (
                (discovery_metrics or {}).get("credit_budget_used", args.credit_budget + args.credit_alternative_budget + 1)
                <= args.credit_budget + args.credit_alternative_budget
            ),
            "joint_credit_measured": (discovery_metrics or {}).get("credit_total_joint_measurements", 0.0) > 0,
            "random_credit_budget_nonzero": (discovery_metrics or {}).get("credit_random_targets", 0.0) > 0,
            "unchosen_candidate_credit_measured": (discovery_metrics or {}).get("credit_unchosen_measurements", 0.0) > 0,
            "no_primitive_collapse": (
                (discovery_metrics or {}).get("primitive_top_share", 1.0)
                <= args.primitive_top_share_target
            ),
            "active_path_alive": (discovery_metrics or {}).get("active_cells", 0.0) > 0,
        })
    local_ok = all(report["checks"].values())
    report["status"] = "SMOKE_PASS" if args.discovery and local_ok else ("SMOKE_FAIL" if args.discovery else ("PASS" if local_ok else "FAIL"))
    report["acceptance_status"] = "NOT_RUN" if args.discovery else report["status"]
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
        enable_single_signed_projection=args.enable_single_signed_projection,
        single_proj_dim=args.single_proj_dim,
        enable_pair_jl_bilinear=args.enable_pair_jl_bilinear,
        pair_jl_dim=args.pair_jl_dim,
        pair_candidate_budget=args.pair_candidate_budget,
        projection_logit_cap=args.projection_logit_cap,
        enable_self_delta_probe=args.enable_self_delta_probe or args.enable_self_delta_choice,
        enable_self_delta_choice=args.enable_self_delta_choice,
        self_delta_max_scale=args.self_delta_max_scale,
    ).to(device)
    model.choice_sampling = "uniform" if args.controller_baseline == "random" else "auto"
    if args.controller_baseline in {"frozen", "random"}:
        model.backbone.slot_embed.requires_grad_(False)
        for parameter in model.backbone.pm.parameters():
            parameter.requires_grad_(False)
        for layer in model.backbone.layers:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
            # Keep the operation implementations trainable; only architecture
            # selection/controller state is frozen.
            for parameter in layer.executor.parameters():
                parameter.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and args.amp == "fp16"))
    dtype = amp_dtype(args.amp)

    start = time.time()
    discovery_metrics: Dict[str, float] = {}
    if args.dataset == "speechcommands" and args.discovery:
        train_acc, train_loss, best_full, discovery_metrics = _train_real_discovery(
            args, model, task, opt, scaler, dtype, device
        )
    else:
        train_acc, train_loss, best_full = _train(args, model, task, opt, scaler, dtype, device)
    if args.dataset == "speechcommands":
        report = _report_real(args, model, task, train_acc, train_loss, best_full, device, start, discovery_metrics)
    else:
        report = _report_synthetic(args, model, task, train_acc, train_loss, best_full, device, start)
    out_dir = ensure_dir(Path(args.out_dir))
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "report_status": report.get("status"),
        },
        out_dir / "model_last.pt",
    )
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
    if report["status"] not in {"PASS", "SMOKE_PASS"}:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
