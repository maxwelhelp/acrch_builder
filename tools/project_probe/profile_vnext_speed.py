#!/usr/bin/env python3
import argparse
import time
import os
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F

from arch_builder.train_audio_frontend import (
    parser as frontend_parser,
    _build_task,
    _next_batch,
    _move_batch,
    _batch_x,
    _batch_y,
    _behavior_diversity_loss,
    generic_discovery_health_loss,
    pairwise_ranking_loss,
)
from arch_builder.audio_frontend import AudioMatrixClassifier
from arch_builder.credit import BoundedCounterfactualCredit
import arch_builder.model
import arch_builder.vnext_controller

def make_parser():
    p = frontend_parser()
    p.add_argument("--profile-light", action="store_true", help="Use perf_counter with GPU sync")
    p.add_argument("--profile-torch", action="store_true", help="Use torch.profiler")
    p.add_argument("--profile-steps", type=int, default=20, help="Number of profiling steps")
    p.add_argument("--credit-mode", type=str, default="counterfactual", choices=["counterfactual", "grad_trace", "hybrid_fast"])
    p.add_argument("--diag-every", type=int, default=1)
    return p

def main():
    args = make_parser().parse_args()
    torch.manual_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    
    print(f"Initializing profile_vnext_speed on {device}...")
    task = _build_task(args)
    num_classes = 2 if args.dataset == "synthetic" else len(task.classes)
    
    # Initialize Model
    from arch_builder.train_audio_frontend import build_frontend
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
    ).to(device)
    
    model.train()
    
    # Optimizer & Scaler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from arch_builder.train_audio_frontend import amp_dtype
    scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and args.amp == "fp16"))
    dtype = amp_dtype(args.amp)
    
    credit = BoundedCounterfactualCredit(
        layers=model.backbone.num_layers,
        slots=model.backbone.slots,
        primitives=model.backbone.pm.num_primitives,
        budget=args.credit_budget,
        alternative_budget=args.credit_alternative_budget,
    )
    
    # Dataloader / Sampler setup
    if args.dataset == "speechcommands":
        train_loader, _, _ = task.loaders(
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            workers=args.workers,
            pin_memory=device.startswith("cuda"),
            drop_last=True,
        )
        train_iter = iter(train_loader)
        def get_batch_fn():
            nonlocal train_iter
            raw, train_iter = _next_batch(train_iter, train_loader)
            return _move_batch(raw, device)
            
        def get_credit_batch_fn():
            nonlocal train_iter
            raw, train_iter = _next_batch(train_iter, train_loader)
            return _move_batch(raw, device)
    else:
        from arch_builder.train_audio_frontend import _sample
        def get_batch_fn():
            return _sample(task, args.batch_size, device, split="training")
            
        def get_credit_batch_fn():
            return _sample(task, args.credit_batch_size, device, split="training")
            
    # Timing breakdown dict
    timings = defaultdict(float)
    
    # Monkey-patch wrappers
    def get_sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            
    def timed_wrapper(obj, attr_name, metric_name):
        original_fn = getattr(obj, attr_name)
        def wrapper(*args, **kwargs):
            get_sync()
            t0 = time.perf_counter()
            res = original_fn(*args, **kwargs)
            get_sync()
            t1 = time.perf_counter()
            timings[metric_name] += (t1 - t0)
            return res
        setattr(obj, attr_name, wrapper)
        return original_fn

    # Patch Model forward parts
    orig_frontend = timed_wrapper(model.frontend, "forward", "frontend_seconds")
    orig_backbone = timed_wrapper(model.backbone, "forward", "model_forward_seconds")
    
    # Patch layers internals
    orig_scanners = []
    orig_critics = []
    orig_executors = []
    
    for i, layer in enumerate(model.backbone.layers):
        orig_scanners.append(timed_wrapper(layer.scanner, "forward", "scanner_seconds"))
        if layer.utility_critic is not None:
            orig_critics.append(timed_wrapper(layer.utility_critic, "forward", "utility_critic_seconds"))
        # Executor uses __call__ which routes to forward, so wrap forward
        orig_executors.append(timed_wrapper(layer.executor, "forward", "executor_seconds"))
        
    # Patch MMR controller
    orig_mmr = timed_wrapper(arch_builder.model, "vnext_mmr_select", "mmr_seconds")
    
    # Patch credit
    orig_align = timed_wrapper(credit, "alignment_losses", "credit_plan_seconds")
    orig_collect = timed_wrapper(credit, "collect", "counterfactual_seconds")
    
    # Warmup loop (2 steps)
    print("Running warmup steps...")
    for _ in range(2):
        batch = get_batch_fn()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda") and dtype != torch.float32):
            features = model.frontend(_batch_x(batch))
            logits, trace = model.backbone(features, tau=1.0, curriculum_mode="deploy", collect_scan_metrics=False)
            ce = F.cross_entropy(logits, _batch_y(batch))
            policy_loss, simulator_loss, _ = credit.alignment_losses(trace)
            loss = ce + policy_loss + simulator_loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
    # Reset timings after warmup
    timings.clear()
    
    # Prepare PyTorch Profiler if requested
    prof = None
    if args.profile_torch:
        out_trace_dir = Path(args.out_dir)
        out_trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"Setting up torch.profiler, output trace to {out_trace_dir}...")
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_trace_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()
        
    print(f"Starting profiling for {args.profile_steps} steps...")
    total_start = time.perf_counter()
    
    for step in range(args.profile_steps):
        t_data_0 = time.perf_counter()
        batch = get_batch_fn()
        get_sync()
        t_data_1 = time.perf_counter()
        timings["data_load_seconds"] += (t_data_1 - t_data_0)
        
        opt.zero_grad(set_to_none=True)
        
        t_forward_0 = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda") and dtype != torch.float32):
            features = model.frontend(_batch_x(batch))
            logits, trace = model.backbone(
                features,
                tau=1.0,
                curriculum_mode="deploy",
                collect_scan_metrics=(step % args.trace_every == 0),
            )
            ce = F.cross_entropy(logits, _batch_y(batch))
            
            # Credit Alignment
            t_credit_0 = time.perf_counter()
            policy_loss, simulator_loss, _ = credit.alignment_losses(trace)
            get_sync()
            t_credit_1 = time.perf_counter()
            # credit_plan_seconds will be tracked by wrapper, but we check here too
            
            loss = ce + policy_loss + simulator_loss
            
        get_sync()
        t_forward_1 = time.perf_counter()
        timings["total_forward_seconds"] += (t_forward_1 - t_forward_0)
        
        # Backward Pass
        t_back_0 = time.perf_counter()
        scaler.scale(loss).backward()
        get_sync()
        t_back_1 = time.perf_counter()
        timings["backward_seconds"] += (t_back_1 - t_back_0)
        
        # Optimizer Step
        t_opt_0 = time.perf_counter()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        get_sync()
        t_opt_1 = time.perf_counter()
        timings["optimizer_seconds"] += (t_opt_1 - t_opt_0)
        
        # Counterfactual Credit
        t_cf_0 = time.perf_counter()
        if (step + 1) % max(1, args.credit_interval) == 0:
            credit_batch = get_credit_batch_fn()
            credit_x = _batch_x(credit_batch)[: args.credit_batch_size]
            credit_y = _batch_y(credit_batch)[: args.credit_batch_size]
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda") and dtype != torch.float32):
                credit_features = model.frontend(credit_x)
            credit.collect(model.backbone, credit_features, credit_y, tau=1.0)
        get_sync()
        t_cf_1 = time.perf_counter()
        # counterfactual_seconds tracked by collect wrapper
        
        if prof is not None:
            prof.step()
            
    if prof is not None:
        prof.stop()
        
    total_end = time.perf_counter()
    total_duration = total_end - total_start
    
    # Calculate averages
    steps = args.profile_steps
    samples = steps * args.batch_size
    samples_per_sec = samples / total_duration
    
    gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.startswith("cuda") else 0.0
    gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2) if device.startswith("cuda") else 0.0
    
    print("\n=== PROFILING RESULTS ===")
    print(f"Total steps: {steps}")
    print(f"Total time: {total_duration:.4f} s")
    print(f"Throughput: {samples_per_sec:.2f} samples/second")
    print(f"Step latency: {total_duration / steps * 1000:.2f} ms/step")
    print(f"GPU Mem Allocated: {gpu_mem_allocated:.2f} MB")
    print(f"GPU Mem Reserved: {gpu_mem_reserved:.2f} MB")
    print("\nBreakdown (seconds):")
    
    # Print metrics
    metrics = [
        "data_load_seconds",
        "frontend_seconds",
        "model_forward_seconds",
        "scanner_seconds",
        "utility_critic_seconds",
        "mmr_seconds",
        "executor_seconds",
        "credit_plan_seconds",
        "counterfactual_seconds",
        "backward_seconds",
        "optimizer_seconds",
    ]
    
    for m in metrics:
        v = timings.get(m, 0.0)
        pct = (v / total_duration) * 100
        print(f"  {m:<25}: {v:8.4f} s ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
