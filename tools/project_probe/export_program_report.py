#!/usr/bin/env python3
"""Program Export Tool — Human-Readable Program Report Exporter.

Loads a trained model checkpoint, extracts the learned routing decisions,
and prints/saves a beautiful Markdown report detailing the top primitives
selected for each layer/cell, gate statuses, and crystallization scores.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

from arch_builder.audio_frontend import AudioMatrixClassifier, build_frontend
from arch_builder.train_audio_frontend import _build_task


def load_model(checkpoint_path: str, device: str = "cpu") -> AudioMatrixClassifier:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Check if this is a full checkpoint or state_dict
    state_dict = ckpt.get("state_dict", ckpt)
    
    # We need args to build task and model
    args_data = ckpt.get("args")
    if args_data is not None:
        if isinstance(args_data, dict):
            args = argparse.Namespace(**args_data)
        else:
            args = args_data
    else:
        # Fallback defaults if args not stored in checkpoint
        class FallbackArgs:
            dataset = "speechcommands"
            data_root = "data/speechcommands"
            sample_rate = 16000
            seconds = 1.0
            classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
            dim = 32
            slots = 4
            layers = 3
            top_k = 8
            sim_rank = 8
            enable_vnext = True
            enable_scanner_feedback_memory = True
            enable_category_scanner = True
            enable_utility_critic_choice = True
            enable_mmr_controller = True
            enable_lazy_executor = True
            enable_single_signed_projection = True
            single_proj_dim = 32
            enable_pair_jl_bilinear = True
            pair_jl_dim = 16
            pair_candidate_budget = 4
            projection_logit_cap = 2.0
            enable_self_delta_probe = True
            enable_self_delta_choice = True
            self_delta_max_scale = 1.0
        args = FallbackArgs()
        
    task = _build_task(args)
    variant = args.variant if hasattr(args, "variant") else "identity"
    frontend = build_frontend(variant, slots=args.slots, dim=args.dim)
    num_classes = 2 if args.dataset == "synthetic" else len(task.classes)
    
    # Build model
    model = AudioMatrixClassifier(
        frontend=frontend,
        classes=num_classes,
        dim=args.dim,
        slots=args.slots,
        layers=args.layers,
        top_k=args.top_k,
        sim_rank=args.sim_rank,
        enable_vnext=args.enable_vnext,
        enable_scanner_feedback_memory=args.enable_scanner_feedback_memory,
        enable_category_scanner=args.enable_category_scanner,
        enable_utility_critic_choice=args.enable_utility_critic_choice,
        enable_mmr_controller=args.enable_mmr_controller,
        enable_lazy_executor=args.enable_lazy_executor,
        enable_single_signed_projection=args.enable_single_signed_projection,
        single_proj_dim=args.single_proj_dim,
        enable_pair_jl_bilinear=args.enable_pair_jl_bilinear,
        pair_jl_dim=args.pair_jl_dim,
        pair_candidate_budget=args.pair_candidate_budget,
        projection_logit_cap=args.projection_logit_cap,
        enable_self_delta_probe=args.enable_self_delta_probe,
        enable_self_delta_choice=args.enable_self_delta_choice,
        self_delta_max_scale=args.self_delta_max_scale,
    )
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def generate_markdown_report(model: AudioMatrixClassifier, output_path: str | None = None) -> str:
    """Generate Markdown report from model's PrimitiveMatrix."""
    pm = model.backbone.pm
    num_layers = pm.num_layers
    num_cells = pm.num_cells
    names = pm.names
    
    # Compute scores per layer, cell, primitive
    # score = gain_ema - 0.5 * regret_ema
    scores = pm.feedback_gain_ema - 0.5 * pm.feedback_regret_ema
    counts = pm.feedback_count
    
    lines = []
    lines.append("# Synthesized Neural Program Report")
    lines.append("")
    lines.append("Analysis of the learned routing and selected primitives across the network topology.")
    lines.append("")
    
    # Overall usage summary
    lines.append("## Global Primitive Usage Score")
    lines.append("| Rank | Primitive | Usage Score | Observations |")
    lines.append("|------|-----------|-------------|--------------|")
    usage_indices = torch.argsort(pm.usage_score, descending=True)
    for rank, idx in enumerate(usage_indices.tolist()[:10], 1):
        name = names[idx]
        score_val = float(pm.usage_score[idx])
        obs_val = int(pm.usage_observations[idx])
        lines.append(f"| {rank} | `{name}` | {score_val:.4f} | {obs_val} |")
    lines.append("")

    # Detailed layers breakdown
    for l in range(num_layers):
        lines.append(f"## Layer {l} Program Blueprint")
        lines.append("| Cell | Top Primitive | Score | Counts | 2nd Primitive | Score | 3rd Primitive | Score |")
        lines.append("|------|---------------|-------|--------|---------------|-------|---------------|-------|")
        
        for c in range(num_cells):
            cell_scores = scores[l, c]
            cell_counts = counts[l, c]
            topk = torch.topk(cell_scores, k=3)
            
            p1_idx, p2_idx, p3_idx = topk.indices.tolist()
            p1_score, p2_score, p3_score = topk.values.tolist()
            
            p1_name = names[p1_idx]
            p1_count = int(cell_counts[p1_idx])
            
            p2_name = names[p2_idx]
            p3_name = names[p3_idx]
            
            lines.append(
                f"| {l},{c} | `{p1_name}` | {p1_score:+.4f} | {p1_count} "
                f"| `{p2_name}` | {p2_score:+.4f} | `{p3_name}` | {p3_score:+.4f} |"
            )
        lines.append("")
        
    # Program stats
    lines.append("## Program Execution Parameters")
    metrics = pm.metrics()
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    
    report_content = "\n".join(lines)
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report_content)
        print(f"Report successfully saved to: {out_p}")
        
    return report_content


def main():
    parser = argparse.ArgumentParser(description="Export learned neural program report.")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--out", type=str, default="reports/neural_program_report.md", help="Output markdown path")
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: checkpoint file {args.checkpoint} does not exist.", file=sys.stderr)
        sys.exit(1)
        
    model = load_model(args.checkpoint)
    report = generate_markdown_report(model, args.out)
    print(report[:1000])
    print("...")


if __name__ == "__main__":
    main()
