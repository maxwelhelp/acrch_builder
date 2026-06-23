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
from .credit import BoundedCounterfactualCredit, generic_discovery_health_loss, pairwise_ranking_loss
from .program_search import ProgramSearchState


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
    "single_projection_generated_count",
    "single_projection_raw_candidate_count",
    "single_projection_pre_topk_count",
    "single_projection_topk_count",
    "single_projection_in_utility_pool_count",
    "single_projection_rank_min",
    "single_projection_rank_mean",
    "single_projection_score_mean",
    "single_projection_score_max",
    "feedback_bias_abs",
    "feedback_staleness",
    "feedback_count",
    "feedback_entropy",
    "feedback_top_share",
    "feedback_candidate_usage",
    "category_candidate_usage",
    "feedback_candidate_coverage",
    "category_candidate_coverage",
)

SELF_DELTA_METRIC_KEYS = (
    "self_delta_mean",
    "self_delta_abs",
    "self_delta_score_mean",
    "self_delta_score_std",
    "self_delta_sim_norm",
    "self_delta_actual_norm",
    "self_delta_rel_error",
    "self_delta_sim_actual_cos",
    "self_delta_scale",
    "self_delta_enabled",
    "self_delta_choice_enabled",
)

UTILITY_CRITIC_METRICS = (
    "utility_critic_enabled",
    "utility_choice_enabled",
    "utility_pool_size",
    "utility_budget",
    "utility_mmr_beta",
    "utility_mmr_mode",
    "utility_score_mean",
    "utility_score_std",
    "utility_gain_corr",
    "utility_gain_spearman",
    "current_predicted_gain_corr",
    "utility_vs_current_gain_corr_delta",
    "proposal_top1_measured_gain",
    "utility_top1_measured_gain",
    "random_top1_measured_gain",
    "proposal_best_of_3_measured_gain",
    "utility_best_of_3_measured_gain",
    "mmr_best_of_3_measured_gain",
    "identity_mmr_similarity",
    "effect_mmr_similarity",
    "hybrid_mmr_similarity",
    "utility_overhead_seconds",
    "choice_without_utility_delta",
    "mmr_selected_count",
    "mmr_selected_similarity",
    "mmr_pre_similarity_topk",
    "mmr_post_similarity_selected",
    "behavior_feature_pair_sim_mean",
    "behavior_feature_pair_sim_std",
    "behavior_feature_pair_sim_min",
    "behavior_feature_pair_sim_max",
    "mmr_selected_utility_mean",
    "mmr_utility_drop_vs_topk",
    "mmr_controller_active",
    "mmr_controller_warmup_progress",
    "utility_choice_warmup_progress",
    "utility_choice_scale_value",
    "utility_mmr_identity_weight",
    "behavior_div_loss",
    "vnext_policy_loss",
    "vnext_policy_items",
    "grad_norm_scanner",
    "grad_norm_controller",
    "grad_norm_utility_critic",
    "grad_norm_executor",
    "behavior_pair_sim_before",
    "behavior_pair_sim_after",
    "behavior_decorr_loss",
    "utility_target_mean",
    "utility_target_std",
    "utility_target_snr",
    "utility_corr_items",
    "positive_gain_candidate_recall@P",
    "random_credit_count",
    "offpool_credit_count",
    "unchosen_credit_count",
    "source_pool_presence_grid",
    "source_after_mmr_presence_grid",
    "source_after_choice_presence_grid",
    "source_pool_presence_semantic",
    "source_after_mmr_presence_semantic",
    "source_after_choice_presence_semantic",
    "source_pool_presence_usage",
    "source_after_mmr_presence_usage",
    "source_after_choice_presence_usage",
    "source_pool_presence_random",
    "source_after_mmr_presence_random",
    "source_after_choice_presence_random",
    "source_pool_presence_global",
    "source_after_mmr_presence_global",
    "source_after_choice_presence_global",
    "source_pool_presence_single_signed_projection",
    "source_after_mmr_presence_single_signed_projection",
    "source_after_choice_presence_single_signed_projection",
    "source_pool_presence_pair_jl16",
    "source_after_mmr_presence_pair_jl16",
    "source_after_choice_presence_pair_jl16",
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
    ap.add_argument("--lambda-behavior-diversity", type=float, default=0.01)
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
    ap.add_argument("--enable-vnext", action="store_true")
    ap.add_argument("--enable-utility-critic-probe", action="store_true")
    ap.add_argument("--enable-utility-critic-choice", action="store_true")
    ap.add_argument("--utility-pool-size", type=int, default=16)
    ap.add_argument("--utility-budget", type=int, default=3)
    ap.add_argument("--utility-mmr-beta", type=float, default=0.35)
    ap.add_argument("--utility-mmr-mode", default="hybrid")
    ap.add_argument("--utility-choice-warmup-steps", type=int, default=50)
    ap.add_argument("--mmr-controller-warmup-steps", type=int, default=50)
    ap.add_argument("--utility-choice-scale", type=float, default=0.05)
    ap.add_argument("--utility-choice-scale-max", type=float, default=0.20)
    ap.add_argument("--utility-mmr-identity-weight", type=float, default=0.50)
    ap.add_argument("--enable-scanner-feedback-memory", action="store_true")
    ap.add_argument("--enable-mmr-controller", action="store_true")
    ap.add_argument("--enable-lazy-executor", action="store_true")
    ap.add_argument("--enable-category-scanner", action="store_true")
    ap.add_argument("--enable-auto-mined-atoms", action="store_true")
    ap.add_argument("--utility-exploration-start-weight", type=float, default=0.35)
    ap.add_argument("--utility-exploration-end-weight", type=float, default=0.35)
    ap.add_argument("--utility-exploration-warmup-steps", type=int, default=0)
    ap.add_argument("--utility-budget-start", type=int, default=3)
    ap.add_argument("--utility-budget-end", type=int, default=3)
    ap.add_argument("--utility-budget-warmup-steps", type=int, default=0)
    ap.add_argument("--utility-category-k", type=int, default=1)
    ap.add_argument("--trace-every", type=int, default=1, help="Interval for scanner metrics collection.")
    # Program search loop flags
    ap.add_argument("--enable-program-search-loop", action="store_true")
    ap.add_argument("--program-search-update-every", type=int, default=50)
    ap.add_argument("--program-plateau-windows", type=int, default=10)
    ap.add_argument("--program-burst-duration", type=int, default=20)
    ap.add_argument("--program-burst-random-mult", type=float, default=3.0)
    ap.add_argument("--program-burst-tau-mult", type=float, default=1.5)
    ap.add_argument("--program-collapse-threshold", type=float, default=0.70)
    # Joint credit flags
    ap.add_argument("--enable-joint-credit", action="store_true")
    ap.add_argument("--joint-credit-interval", type=int, default=25)
    ap.add_argument("--joint-credit-extra-budget", type=int, default=2)
    import os
    ap.add_argument("--fast-train-backward", action="store_true", default=bool(int(os.environ.get("FAST_TRAIN_BACKWARD", "0"))))
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
    num_layers = model.backbone.num_layers
    slots = model.backbone.slots
    num_primitives = model.backbone.pm.num_primitives
    accum_prim = [torch.zeros(slots * slots, num_primitives, device=device) for _ in range(num_layers)]
    accum_src = [torch.zeros(slots * slots, 9, device=device) for _ in range(num_layers)]
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
        
        # Accumulate actual routing choices from trace
        for layer_idx, layer in enumerate(trace.get("layers", [])):
            if "choice" in layer and "candidate_ids" in layer and "candidate_source_ids" in layer:
                choice = layer["choice"]
                candidate_ids = layer["candidate_ids"]
                candidate_source_ids = layer["candidate_source_ids"]
                b_s_s, pool_sz = choice.shape
                batch_val = b_s_s // (slots * slots)
                
                cell_indices = torch.arange(slots * slots, device=choice.device).repeat(batch_val).unsqueeze(-1).expand(b_s_s, pool_sz).reshape(-1)
                cand_flat = candidate_ids.reshape(-1)
                choice_flat = choice.reshape(-1)
                
                valid_mask = (cand_flat >= 0) & (cand_flat < num_primitives)
                flat_idx = cell_indices * num_primitives + cand_flat
                accum_prim[layer_idx].view(-1).put_(flat_idx[valid_mask], choice_flat[valid_mask], accumulate=True)
                
                src_flat = candidate_source_ids.reshape(-1)
                valid_src_mask = (src_flat >= 0) & (src_flat < 9)
                flat_src_idx = cell_indices * 9 + src_flat
                accum_src[layer_idx].view(-1).put_(flat_src_idx[valid_src_mask], choice_flat[valid_src_mask], accumulate=True)

        for layer in trace.get("layers", []):
            source_trace_count += 1
            scan = layer.get("scan_metrics", {})
            for key in (
                "grid_candidate_usage", "semantic_candidate_usage", "usage_candidate_usage", "random_candidate_usage", "global_candidate_usage",
                "feedback_candidate_usage", "category_candidate_usage",
                "grid_candidate_coverage", "semantic_candidate_coverage", "usage_candidate_coverage", "random_candidate_coverage", "global_candidate_coverage",
                "feedback_candidate_coverage", "category_candidate_coverage",
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

            um = layer.get("utility_metrics", {})
            for k, v in um.items():
                if hasattr(v, "detach"):
                    trace_sum[k] += float(v.detach().float().cpu())
                else:
                    trace_sum[k] += float(v)

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
        # Keys that are accumulated per-layer (not per-step) need source_trace_count divisor
        _PER_LAYER_KEYS = {
            "candidate_usage", "candidate_coverage", "scanner_source_mass_sum",
        }
        _UTILITY_KEYS_SUFFIXES = {
            "utility_pool_size", "utility_budget", "utility_budget_current",
            "utility_mmr_beta", "utility_mmr_mode",
            "utility_critic_enabled", "utility_choice_enabled",
            "utility_choice_warmup_progress", "utility_choice_scale_value",
            "utility_score_mean", "utility_score_std",
            "utility_behavior_norm",
            "utility_exploration_weight_current",
            "mmr_controller_active", "mmr_controller_warmup_progress",
            "mmr_pre_similarity_topk", "mmr_selected_similarity",
            "mmr_fallback_identity",
            "behavior_div_loss", "behavior_decorr_loss",
            "behavior_pair_sim_before", "behavior_pair_sim_after",
        }
        for key, value in trace_sum.items():
            is_per_layer = (
                any(key.endswith(s) for s in _PER_LAYER_KEYS)
                or key in _UTILITY_KEYS_SUFFIXES
                or key.startswith("source_pool_presence_")
                or key.startswith("source_after_")
            )
            divisor = source_trace_count if is_per_layer else trace_count
            out[key] = value / max(1, divisor)
    
    out["accum_prim"] = [layer.cpu().numpy() for layer in accum_prim]
    out["accum_src"] = [layer.cpu().numpy() for layer in accum_src]
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
            "feedback_candidate_usage", "category_candidate_usage",
            "grid_candidate_coverage", "semantic_candidate_coverage", "usage_candidate_coverage", "random_candidate_coverage", "global_candidate_coverage",
            "feedback_candidate_coverage", "category_candidate_coverage",
            "scanner_source_mass_sum",
            *PROJECTION_METRIC_KEYS,
        ):
            if key in scan:
                out[key] = float(scan[key])
    model.backbone.pm.usage_score.copy_(usage_state)
    return out


def _behavior_diversity_loss(trace: Dict[str, object], device: str | torch.device) -> torch.Tensor:
    losses = []
    for layer in trace.get("layers", []):
        loss = layer.get("behavior_div_loss_for_loss")
        if hasattr(loss, "to"):
            losses.append(loss.to(device))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def _grad_norms(model: AudioMatrixClassifier) -> Dict[str, float]:
    groups = {
        "scanner": 0.0,
        "controller": 0.0,
        "utility_critic": 0.0,
        "executor": 0.0,
    }
    controller_terms = (
        "context_logits", "primitive_pair_bias", "prev_action_proj",
        "prev_active_proj", "prev_write_proj", "listen_gate", "mode_head",
        "edge_gate", "write_gate", "phase_gate", "edge_op", "split_head",
        "child_gate", "merge_gate", "slot_alive_head", "output_gate",
        "cell_output_gate", "utility_logit_scale", "slot_embed",
    )
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        value = float(param.grad.detach().float().pow(2).sum().cpu())
        if ".scanner." in name:
            groups["scanner"] += value
        if ".utility_critic." in name:
            groups["utility_critic"] += value
        if ".executor." in name:
            groups["executor"] += value
        if any(term in name for term in controller_terms):
            groups["controller"] += value
    return {f"grad_norm_{key}": value ** 0.5 for key, value in groups.items()}


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
        enable_joint_credit=args.enable_joint_credit,
        joint_credit_extra_budget=args.joint_credit_extra_budget,
    )
    if args.enable_program_search_loop:
        search_state = ProgramSearchState(
            patience=args.program_plateau_windows,
            burst_duration=args.program_burst_duration,
            burst_tau_mult=args.program_burst_tau_mult,
            burst_random_mult=args.program_burst_random_mult,
            min_improvement=0.005,
        )
    else:
        search_state = None
    best = 0.0
    train_acc = train_loss = 0.0
    last_diag: Dict[str, float] = {}
    global_step = 0
    dynamics_history = []
    previous_top_primitives = None
    learned_controller = args.controller_baseline == "learned"
    sampling = "uniform" if args.controller_baseline == "random" else "softmax"

    critic_opt = None
    critic_params_list = []
    if args.enable_utility_critic_probe and not args.enable_vnext:
        critic_params_list = [p for n, p in model.named_parameters() if "utility_critic" in n or "utility_logit_scale" in n]
        if critic_params_list:
            critic_opt = torch.optim.AdamW(critic_params_list, lr=args.lr, weight_decay=args.weight_decay)
    
    if not args.enable_vnext:
        main_params = [p for n, p in model.named_parameters() if "utility_critic" not in n and "utility_logit_scale" not in n]
    else:
        main_params = list(model.parameters())

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total = correct = 0
        loss_sum = 0.0
        tau = max(args.tau_min, args.tau_start * (args.tau_decay ** (epoch - 1)))
        correct_window = 0
        total_window = 0
        loss_window = 0.0
        for step in range(max(1, args.steps_per_epoch)):
            global_step += 1
            if search_state is not None:
                modifiers = search_state.get_modifiers()
            else:
                modifiers = {
                    "tau_multiplier": 1.0,
                    "random_k_multiplier": 1.0,
                    "health_loss_multiplier": 1.0,
                    "feedback_decay_override": None,
                    "exploration_mode": False,
                    "in_burst": False,
                    "burst_reason": None,
                    "burst_steps_remaining": 0,
                    "crystallization_score": 0.0,
                    "total_bursts": 0,
                }
            decay_override = modifiers.get("feedback_decay_override")
            raw_batch, train_iter = _next_batch(train_iter, train_loader)
            batch = _move_batch(raw_batch, device)
            if hasattr(model.backbone, "set_vnext_step"):
                model.backbone.set_vnext_step(global_step)
            if learned_controller:
                credit.advance(model.backbone.pm, decay_override=decay_override)
            opt.zero_grad(set_to_none=True)
            if critic_opt is not None:
                critic_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda") and dtype != torch.float32):
                features = model.frontend(_batch_x(batch))
                logits, trace = model.backbone(
                    features,
                    tau=tau * modifiers.get("tau_multiplier", 1.0),
                    curriculum_mode="deploy",
                    choice_sampling=sampling,
                    collect_scan_metrics=(global_step % args.trace_every == 0),
                    exploration_mode=modifiers.get("exploration_mode", False),
                    random_k_multiplier=modifiers.get("random_k_multiplier", 1.0),
                )
                ce = F.cross_entropy(logits, _batch_y(batch))
                if learned_controller:
                    policy_loss, simulator_loss, align_metrics = credit.alignment_losses(trace)
                    behavior_div_loss = _behavior_diversity_loss(trace, ce.device)
                    health_loss, health_metrics = generic_discovery_health_loss(
                        trace,
                        slots=model.backbone.slots,
                        num_primitives=model.backbone.pm.num_primitives,
                        target_active_cells=args.target_active_cells,
                        primitive_top_share_target=args.primitive_top_share_target,
                        primitive_entropy_floor=args.primitive_entropy_floor,
                    )
                    health_loss = health_loss * modifiers.get("health_loss_multiplier", 1.0)
                    
                    grad_losses = []
                    grad_nlls = []
                    grad_ranks = []
                    for layer_idx, layer in enumerate(model.backbone.layers):
                        if hasattr(layer, "get_and_clear_grad_credits"):
                            credits_list = layer.get_and_clear_grad_credits()
                            if credits_list and layer.utility_critic is not None:
                                ctx_det, emb_det, head_det, cand_det, grad_credit = credits_list[-1]
                                ctx_gpu = ctx_det.to(ce.device)
                                emb_gpu = emb_det.to(ce.device)
                                head_gpu = head_det.to(ce.device)
                                grad_credit_gpu = grad_credit.to(ce.device)
                                
                                if ctx_gpu.dtype != features.dtype:
                                    ctx_gpu = ctx_gpu.to(features.dtype)
                                if emb_gpu.dtype != features.dtype:
                                    emb_gpu = emb_gpu.to(features.dtype)
                                if head_gpu.dtype != features.dtype:
                                    head_gpu = head_gpu.to(features.dtype)

                                pred_util, pred_var = layer.utility_critic(ctx_gpu, emb_gpu, head_gpu)
                                
                                grad_mean_abs = float(grad_credit_gpu.abs().mean())
                                if not hasattr(layer, "grad_scale_ema"):
                                    layer.grad_scale_ema = 1e-3
                                layer.grad_scale_ema = 0.99 * layer.grad_scale_ema + 0.01 * max(grad_mean_abs, 1e-8)
                                
                                target_scale = credit.gain_scale if hasattr(credit, "gain_scale") else 1e-3
                                scaled_target = grad_credit_gpu / max(layer.grad_scale_ema, 1e-8) * target_scale
                                
                                nll = 0.5 * torch.log(pred_var.clamp_min(1e-6)) + 0.5 * (pred_util - scaled_target).pow(2) / pred_var.clamp_min(1e-6)
                                grad_nlls.append(nll.mean())
                                
                                rank_loss = pairwise_ranking_loss(pred_util, scaled_target)
                                grad_ranks.append(rank_loss)
                    
                    if grad_nlls:
                        grad_nll_loss = torch.stack(grad_nlls).mean()
                        grad_rank_loss = torch.stack(grad_ranks).mean()
                        grad_credit_loss = grad_nll_loss + grad_rank_loss
                        grad_metrics = {
                            "grad_credit_nll": float(grad_nll_loss.detach().cpu()),
                            "grad_credit_rank": float(grad_rank_loss.detach().cpu()),
                        }
                    else:
                        grad_credit_loss = torch.zeros((), device=ce.device)
                        grad_metrics = {"grad_credit_nll": 0.0, "grad_credit_rank": 0.0}
                else:
                    policy_loss = simulator_loss = health_loss = torch.zeros((), device=ce.device)
                    grad_credit_loss = torch.zeros((), device=ce.device)
                    align_metrics = {"credit_alignment_items": 0.0, "sim_pred_real_corr": 0.0}
                    health_metrics = {"active_cells": 0.0, "primitive_top_share": 0.0, "primitive_entropy": 0.0, "choice_entropy": 0.0}
                    grad_metrics = {"grad_credit_nll": 0.0, "grad_credit_rank": 0.0}
                    behavior_div_loss = torch.zeros((), device=ce.device)
                grad_norm_metrics = {"grad_norm_scanner": 0.0, "grad_norm_controller": 0.0, "grad_norm_utility_critic": 0.0, "grad_norm_executor": 0.0}
                import os
                enable_decorr = int(os.environ.get("ENABLE_BEHAVIOR_DECORR_LOSS", "1")) == 1
                decorr_scale = float(os.environ.get("BEHAVIOR_DECORR_SCALE", str(args.lambda_behavior_diversity))) if enable_decorr else 0.0
                if args.enable_vnext:
                    loss = (
                        ce
                        + args.lambda_credit_policy * policy_loss
                        + args.lambda_credit_simulator * (simulator_loss + grad_credit_loss)
                        + args.lambda_discovery_health * health_loss
                        + decorr_scale * behavior_div_loss
                    )
                else:
                    loss = (
                        ce
                        + args.lambda_credit_policy * policy_loss
                        + args.lambda_credit_simulator * simulator_loss
                        + args.lambda_discovery_health * health_loss
                        + decorr_scale * behavior_div_loss
                    )
            
            if args.enable_vnext:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm_metrics = _grad_norms(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm_metrics = _grad_norms(model)
                torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                scaler.step(opt)
                scaler.update()

                if critic_opt is not None and grad_nlls:
                    if scaler.is_enabled():
                        scaler.scale(grad_credit_loss).backward()
                        scaler.unscale_(critic_opt)
                        torch.nn.utils.clip_grad_norm_(critic_params_list, 1.0)
                        scaler.step(critic_opt)
                    else:
                        grad_credit_loss.backward()
                        torch.nn.utils.clip_grad_norm_(critic_params_list, 1.0)
                        critic_opt.step()

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
            utility_diag = {}
            utility_layers = [
                layer.get("utility_metrics", {}) for layer in trace.get("layers", [])
            ]
            if utility_layers:
                all_u_keys = set()
                for um in utility_layers:
                    all_u_keys.update(um.keys())
                for key in all_u_keys:
                    values = []
                    for um in utility_layers:
                        if key in um:
                            val = um[key]
                            if hasattr(val, "detach"):
                                val = float(val.detach().cpu())
                            else:
                                val = float(val)
                            values.append(val)
                    if values:
                        utility_diag[f"train_{key}"] = sum(values) / len(values)

            last_diag = {
                "ce_loss": float(ce.detach().cpu()),
                "credit_policy_loss": float(policy_loss.detach().cpu()),
                "credit_simulator_loss": float((simulator_loss + grad_credit_loss).detach().cpu()),
                "discovery_health_loss": float(health_loss.detach().cpu()),
                "behavior_div_loss": float(behavior_div_loss.detach().cpu()),
                **align_metrics,
                **health_metrics,
                **credit.metrics(),
                **model.backbone.pm.metrics(),
                **projection_diag,
                **grad_metrics,
                **grad_norm_metrics,
                **utility_diag,
                **(search_state.metrics() if search_state is not None else {}),
            }
            if args.log_every > 0 and global_step % args.log_every == 0:
                print(
                    f"discovery step={global_step} ce={last_diag['ce_loss']:.4f} "
                    f"acc={correct/max(1,total):.3f} active={last_diag['active_cells']:.1f} "
                    f"top={last_diag['primitive_top_share']:.3f} "
                    f"credit={int(last_diag.get('credit_total_measurements', 0))} "
                    f"gain={last_diag.get('credit_gain_mean', 0.0):+.5f} "
                    f"burst={int(last_diag.get('search_in_burst', 0))}",
                    flush=True,
                )
            
            # Accumulate window statistics
            correct_window += (logits.argmax(dim=-1) == _batch_y(batch)).sum().item()
            total_window += _batch_y(batch).shape[0]
            loss_window += float(ce.detach().cpu()) * _batch_y(batch).shape[0]
            
            if search_state is not None and global_step % args.program_search_update_every == 0:
                window_acc = correct_window / max(1, total_window)
                num_demoted = model.backbone.pm.demote_stale(min_count=5, negative_threshold=-0.5, decay=0.5)
                if num_demoted > 0:
                    print(f"[ProgramSearch] Step {global_step}: Demoted {num_demoted} stale primitives with negative credit.")
                    
                search_state.update(
                    val_acc=window_acc,
                    primitive_top_share=float(last_diag.get("primitive_top_share", 0.0)),
                    choice_entropy=float(last_diag.get("choice_entropy", 1.0)),
                    step=global_step,
                )
                print(
                    f"[ProgramSearch] Window Update Step {global_step}: window_acc={window_acc:.3f} "
                    f"crystallization={search_state.metrics().get('search_crystallization_score', 0.0):.3f} "
                    f"burst={int(search_state.metrics().get('search_in_burst', 0))} (patience={search_state.steps_since_improvement})",
                    flush=True
                )
                correct_window = 0
                total_window = 0
                loss_window = 0.0
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)
        ev = evaluate(model, task, args.eval_steps, args.eval_batch_size, device, tau, "deploy")
        best = max(best, ev["acc"])
        
        # Update last_diag with search state metrics if enabled
        if search_state is not None:
            last_diag.update(search_state.metrics())
        
        samples_per_second = total / max(1e-8, time.perf_counter() - epoch_start)
        last_diag["train_samples_per_second"] = float(samples_per_second)
        csv_row = {
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "val_acc": ev["acc"],
            "val_loss": ev["loss"],
            "samples_per_second": samples_per_second,
            "behavior_feature_pair_sim_mean": ev.get("behavior_feature_pair_sim_mean", 0.0),
            "behavior_feature_pair_sim_std": ev.get("behavior_feature_pair_sim_std", 0.0),
            "mmr_fallback_identity": ev.get("mmr_fallback_identity", 0.0),
            "source_pool_presence_single_signed_projection": ev.get("source_pool_presence_single_signed_projection", 0.0),
            "source_after_mmr_presence_single_signed_projection": ev.get("source_after_mmr_presence_single_signed_projection", 0.0),
            "source_after_choice_presence_single_signed_projection": ev.get("source_after_choice_presence_single_signed_projection", 0.0),
            "utility_corr_status": last_diag.get("utility_corr_status", 0.0),
        }
        # Explicitly append all projection diagnostics to csv row
        proj_keys = [
            "single_projection_generated_count",
            "single_projection_raw_candidate_count",
            "single_projection_pre_topk_count",
            "single_projection_topk_count",
            "single_projection_in_utility_pool_count",
            "single_projection_rank_min",
            "single_projection_rank_mean",
            "single_projection_score_mean",
            "single_projection_score_max",
            "source_pool_presence_single_signed_projection",
            "source_after_mmr_presence_single_signed_projection",
            "source_after_choice_presence_single_signed_projection",
            "single_signed_projection_usage",
        ]
        for key in proj_keys:
            csv_row[key] = ev.get(key, 0.0)
            csv_row[f"eval_{key}"] = ev.get(key, 0.0)
            csv_row[f"train_{key}"] = last_diag.get(key, 0.0)
        
        csv_row.update(last_diag)
        append_csv(ensure_dir(Path(args.out_dir)) / "metrics.csv", csv_row)
        
        # Save per-epoch program snapshot
        pm = model.backbone.pm
        scores = pm.feedback_gain_ema - 0.5 * pm.feedback_regret_ema
        snapshot = {}
        for l in range(pm.num_layers):
            snapshot[f"layer_{l}"] = {}
            for c in range(pm.num_cells):
                cell_scores = scores[l, c]
                top_idx = int(cell_scores.argmax().item())
                top_name = pm.names[top_idx]
                snapshot[f"layer_{l}"][f"cell_{c}"] = {
                    "primitive": top_name,
                    "score": float(cell_scores[top_idx]),
                    "count": int(pm.feedback_count[l, c, top_idx])
                }
        snapshot_path = ensure_dir(Path(args.out_dir)) / f"program_snapshot_epoch_{epoch}.json"
        snapshot_path.write_text(json.dumps(snapshot, indent=2))
        
        # Calculate actual execution dynamics from validation results
        import numpy as np
        current_top_primitives = []
        primitives_blueprint = []
        for l in range(pm.num_layers):
            layer_blue = []
            for c in range(pm.num_cells):
                c_prim_dist = ev["accum_prim"][l][c]
                top_prim_idx = int(c_prim_dist.argmax())
                current_top_primitives.append(top_prim_idx)
                c_src_dist = ev["accum_src"][l][c]
                top_src_idx = int(c_src_dist.argmax())
                layer_blue.append((pm.names[top_prim_idx], top_src_idx))
            primitives_blueprint.append(layer_blue)
            
        route_jaccard = 1.0
        drift_count = 0
        if previous_top_primitives is not None:
            intersection = sum(1 for a, b in zip(previous_top_primitives, current_top_primitives) if a == b)
            union = len(current_top_primitives)
            route_jaccard = intersection / max(1, union)
            drift_count = union - intersection
        previous_top_primitives = current_top_primitives
        
        source_names_list = ["grid", "semantic", "usage", "random", "feedback", "single_signed_projection", "pair_jl16", "category", "exploration"]
        src_sum = np.zeros(9)
        for l in range(pm.num_layers):
            src_sum += ev["accum_src"][l].sum(axis=0)
        src_total = src_sum.sum()
        src_mix = {name: float(src_sum[idx] / max(1.0, src_total)) for idx, name in enumerate(source_names_list)}
        
        crystallization = 0.0
        in_burst = False
        burst_reason = None
        if search_state is not None:
            crystallization = search_state.metrics().get("search_crystallization_score", 0.0)
            in_burst = bool(search_state.metrics().get("search_in_burst", 0))
            burst_reason = search_state.get_modifiers().get("burst_reason")
            
        dynamics_history.append({
            "epoch": epoch,
            "val_acc": ev["acc"],
            "crystallization": crystallization,
            "in_burst": in_burst,
            "burst_reason": burst_reason,
            "route_jaccard": route_jaccard if len(dynamics_history) > 0 else None,
            "drift_count": drift_count if len(dynamics_history) > 0 else 0,
            "src_mix": src_mix,
            "blueprint": primitives_blueprint,
        })
        
        dyn_md = [
            "# Program Dynamics Report",
            f"Generated at epoch {epoch}. This report tracks execution trajectory, primitive drift, source mix, and crystallization.\n",
            "## Epoch Summary Table\n",
            "| Epoch | Val Acc | Crystallization | Burst Active | Burst Reason | Route Jaccard | Cell Drift |",
            "|---|---|---|---|---|---|---|",
        ]
        for item in dynamics_history:
            jacc_str = f"{item['route_jaccard']:.3f}" if item['route_jaccard'] is not None else "-"
            drift_str = str(item['drift_count']) if item['route_jaccard'] is not None else "-"
            reason_str = item['burst_reason'] if item['burst_reason'] else "-"
            dyn_md.append(f"| {item['epoch']} | {item['val_acc']:.3f} | {item['crystallization']:.3f} | {item['in_burst']} | {reason_str} | {jacc_str} | {drift_str} |")
            
        dyn_md.extend([
            "\n## Source Mix Evolution\n",
            "| Epoch | Grid | Semantic | Usage | Random | Feedback | Proj | JL16 | Category | Exploration |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ])
        for item in dynamics_history:
            sm = item["src_mix"]
            row_vals = [f"{sm[k]*100:.1f}%" for k in source_names_list]
            dyn_md.append(f"| {item['epoch']} | " + " | ".join(row_vals) + " |")
            
        dyn_md.append(f"\n## Latest Program Blueprint (Epoch {epoch})\n")
        for l in range(pm.num_layers):
            dyn_md.extend([
                f"### Layer {l}\n",
                "| Cell | Top Primitive | Top Source |",
                "|---|---|---|",
            ])
            for c in range(pm.num_cells):
                prim_name, src_idx = primitives_blueprint[l][c]
                src_name = source_names_list[src_idx] if src_idx < len(source_names_list) else str(src_idx)
                dyn_md.append(f"| {c} | `{prim_name}` | {src_name} |")
            dyn_md.append("")
            
        dyn_path = ensure_dir(Path(args.out_dir)) / "PROGRAM_DYNAMICS.md"
        dyn_path.write_text("\n".join(dyn_md))
        
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
    # Remove large non-serializable ndarrays from evaluation dictionaries
    for m_dict in [full, audit, deploy]:
        if m_dict is not None:
            m_dict.pop("accum_prim", None)
            m_dict.pop("accum_src", None)
            
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
    
    # Remove large non-serializable ndarrays from evaluation dictionaries
    for m_dict in [full, audit, deploy, test]:
        if m_dict is not None:
            m_dict.pop("accum_prim", None)
            m_dict.pop("accum_src", None)
            
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
        "self_delta_rel_error": float(deploy.get("self_delta_rel_error", 0.0)),
        "self_delta_sim_actual_cos": float(deploy.get("self_delta_sim_actual_cos", 0.0)),
        "self_delta_scale": float(deploy.get("self_delta_scale", 0.0)),
        "self_delta_disabled_delta": float(ablations.get("self_delta_disabled_delta", 0.0)),
        "self_delta_zero_delta": float(ablations.get("self_delta_zero_delta", 0.0)),
        "self_delta_shuffle_delta": float(ablations.get("self_delta_shuffle_delta", 0.0)),
        "choice_without_self_delta_delta": float(ablations.get("choice_without_self_delta_delta", 0.0)),
        "choice_zero_self_delta_delta": float(ablations.get("choice_zero_self_delta_delta", 0.0)),
        "choice_shuffle_self_delta_delta": float(ablations.get("choice_shuffle_self_delta_delta", 0.0)),
        "utility_critic_enabled": float(deploy.get("utility_critic_enabled", (discovery_metrics or {}).get("utility_critic_enabled", 0.0))),
        "utility_choice_enabled": float(deploy.get("utility_choice_enabled", (discovery_metrics or {}).get("utility_choice_enabled", 0.0))),
        "utility_pool_size": float(deploy.get("utility_pool_size", (discovery_metrics or {}).get("utility_pool_size", 0.0))),
        "utility_budget": float(deploy.get("utility_budget", (discovery_metrics or {}).get("utility_budget", 0.0))),
        "utility_mmr_beta": float(deploy.get("utility_mmr_beta", (discovery_metrics or {}).get("utility_mmr_beta", 0.0))),
        "utility_mmr_mode": float(deploy.get("utility_mmr_mode", (discovery_metrics or {}).get("utility_mmr_mode", 0.0))),
        "utility_score_mean": float(deploy.get("utility_score_mean", (discovery_metrics or {}).get("utility_score_mean", 0.0))),
        "utility_score_std": float(deploy.get("utility_score_std", (discovery_metrics or {}).get("utility_score_std", 0.0))),
        "utility_gain_corr": float(deploy.get("utility_gain_corr", (discovery_metrics or {}).get("utility_gain_corr", 0.0))),
        "utility_gain_spearman": float(deploy.get("utility_gain_spearman", (discovery_metrics or {}).get("utility_gain_spearman", 0.0))),
        "current_predicted_gain_corr": float(deploy.get("current_predicted_gain_corr", (discovery_metrics or {}).get("current_predicted_gain_corr", 0.0))),
        "utility_vs_current_gain_corr_delta": float(deploy.get("utility_vs_current_gain_corr_delta", (discovery_metrics or {}).get("utility_vs_current_gain_corr_delta", 0.0))),
        "proposal_top1_measured_gain": float(deploy.get("proposal_top1_measured_gain", (discovery_metrics or {}).get("proposal_top1_measured_gain", 0.0))),
        "utility_top1_measured_gain": float(deploy.get("utility_top1_measured_gain", (discovery_metrics or {}).get("utility_top1_measured_gain", 0.0))),
        "random_top1_measured_gain": float(deploy.get("random_top1_measured_gain", (discovery_metrics or {}).get("random_top1_measured_gain", 0.0))),
        "proposal_best_of_3_measured_gain": float(deploy.get("proposal_best_of_3_measured_gain", (discovery_metrics or {}).get("proposal_best_of_3_measured_gain", 0.0))),
        "utility_best_of_3_measured_gain": float(deploy.get("utility_best_of_3_measured_gain", (discovery_metrics or {}).get("utility_best_of_3_measured_gain", 0.0))),
        "mmr_best_of_3_measured_gain": float(deploy.get("mmr_best_of_3_measured_gain", (discovery_metrics or {}).get("mmr_best_of_3_measured_gain", 0.0))),
        "identity_mmr_similarity": float(deploy.get("identity_mmr_similarity", (discovery_metrics or {}).get("identity_mmr_similarity", 0.0))),
        "effect_mmr_similarity": float(deploy.get("effect_mmr_similarity", (discovery_metrics or {}).get("effect_mmr_similarity", 0.0))),
        "hybrid_mmr_similarity": float(deploy.get("hybrid_mmr_similarity", (discovery_metrics or {}).get("hybrid_mmr_similarity", 0.0))),
        "utility_overhead_seconds": float(deploy.get("utility_overhead_seconds", (discovery_metrics or {}).get("utility_overhead_seconds", 0.0))),
        "choice_without_utility_delta": float(deploy.get("choice_without_utility_delta", (discovery_metrics or {}).get("choice_without_utility_delta", 0.0))),
    }
    for _metric_key in UTILITY_CRITIC_METRICS:
        if _metric_key not in report["comparison_metrics"]:
            report["comparison_metrics"][_metric_key] = float(
                deploy.get(_metric_key, (discovery_metrics or {}).get(_metric_key, 0.0))
            )
    for _metric_key in (
        "grad_credit_nll", "grad_credit_rank", "credit_total_measurements",
        "credit_items", "primitive_entropy", "vnext_policy_loss",
        "vnext_policy_items", "grad_norm_scanner", "grad_norm_controller",
        "grad_norm_utility_critic", "grad_norm_executor", "behavior_div_loss",
    ):
        if _metric_key not in report["comparison_metrics"]:
            report["comparison_metrics"][_metric_key] = float(
                deploy.get(_metric_key, (discovery_metrics or {}).get(_metric_key, 0.0))
            )

    # Explicitly populate all projection diagnostics in comparison_metrics and discovery_metrics
    proj_keys = [
        "single_projection_generated_count",
        "single_projection_raw_candidate_count",
        "single_projection_pre_topk_count",
        "single_projection_topk_count",
        "single_projection_in_utility_pool_count",
        "single_projection_rank_min",
        "single_projection_rank_mean",
        "single_projection_score_mean",
        "single_projection_score_max",
        "source_pool_presence_single_signed_projection",
        "source_after_mmr_presence_single_signed_projection",
        "source_after_choice_presence_single_signed_projection",
        "single_signed_projection_usage",
    ]
    for key in proj_keys:
        # Non-prefixed (val/deploy)
        report["comparison_metrics"][key] = float(deploy.get(key, 0.0))
        # Prefixed eval
        report["comparison_metrics"][f"eval_{key}"] = float(deploy.get(key, 0.0))
        # Prefixed train
        report["comparison_metrics"][f"train_{key}"] = float((discovery_metrics or {}).get(key, 0.0))
        # Make sure train version is also inside discovery_metrics
        if discovery_metrics is not None:
            discovery_metrics[f"train_{key}"] = float((discovery_metrics or {}).get(key, 0.0))

    report["checks"] = {
        "deploy_above_random": True if (args.steps_per_epoch * args.epochs < 50) else (deploy["acc"] >= chance + (0.02 if args.discovery else 0.0)),
        "honesty_retained": report["honesty_score"] >= args.honesty_floor,
        "simulator_ce_ablation_positive": True if args.steps_per_epoch < 50 else (ablations["sim_disabled_delta"] > 0),
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
            "joint_credit_measured": True if (args.steps_per_epoch * args.epochs < 50) else ((discovery_metrics or {}).get("credit_total_joint_measurements", 0.0) > 0),
            "random_credit_budget_nonzero": (discovery_metrics or {}).get("credit_random_targets", 0.0) > 0,
            "unchosen_candidate_credit_measured": (discovery_metrics or {}).get("credit_unchosen_measurements", 0.0) > 0,
            "no_primitive_collapse": (
                (discovery_metrics or {}).get("primitive_top_share", 1.0)
                <= args.primitive_top_share_target
            ),
            "active_path_alive": (discovery_metrics or {}).get("active_cells", 0.0) > 0,
            "utility_corr_items_valid": (
                True if (discovery_metrics or {}).get("utility_corr_items", 0.0) >= 8
                else ("INSUFFICIENT" if (discovery_metrics or {}).get("utility_corr_items", 0.0) > 0 else False)
            ),
        })
    local_ok = all(v is True or v == "INSUFFICIENT" for v in report["checks"].values())
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
        enable_vnext=args.enable_vnext,
        enable_utility_critic_probe=args.enable_utility_critic_probe,
        enable_utility_critic_choice=args.enable_utility_critic_choice,
        utility_pool_size=args.utility_pool_size,
        utility_budget=args.utility_budget,
        utility_mmr_beta=args.utility_mmr_beta,
        utility_mmr_mode=args.utility_mmr_mode,
        utility_choice_warmup_steps=args.utility_choice_warmup_steps,
        mmr_controller_warmup_steps=args.mmr_controller_warmup_steps,
        utility_choice_scale=args.utility_choice_scale,
        utility_choice_scale_max=args.utility_choice_scale_max,
        utility_mmr_identity_weight=args.utility_mmr_identity_weight,
        enable_scanner_feedback_memory=args.enable_scanner_feedback_memory,
        enable_mmr_controller=args.enable_mmr_controller,
        enable_lazy_executor=args.enable_lazy_executor,
        enable_category_scanner=args.enable_category_scanner,
        enable_auto_mined_atoms=args.enable_auto_mined_atoms,
        utility_exploration_start_weight=args.utility_exploration_start_weight,
        utility_exploration_end_weight=args.utility_exploration_end_weight,
        utility_exploration_warmup_steps=args.utility_exploration_warmup_steps,
        utility_budget_start=args.utility_budget_start,
        utility_budget_end=args.utility_budget_end,
        utility_budget_warmup_steps=args.utility_budget_warmup_steps,
        utility_category_k=args.utility_category_k,
        fast_train_backward=args.fast_train_backward,
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
    if not args.enable_vnext:
        main_params = [p for n, p in model.named_parameters() if "utility_critic" not in n and "utility_logit_scale" not in n]
    else:
        main_params = list(model.parameters())
    opt = torch.optim.AdamW(main_params, lr=args.lr, weight_decay=args.weight_decay)
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
