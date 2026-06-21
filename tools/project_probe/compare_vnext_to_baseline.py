#!/usr/bin/env python3
import json
import sys
import os
import numpy as np

# Reference baseline values (3-seed mean)
BASELINE = {
    "acc": 0.5919,
    "test": 0.5794,
    "speed": 136.3,
    "sim_delta": 0.3193,
    "choice_sim": 0.0407,
    "top_share": 0.4799,
    "credit_closed": 1.0
}

def load_report(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_metrics(data):
    # Accuracies
    acc = data.get("val_metrics", {}).get("acc", data.get("train_acc", 0.0))
    test = data.get("test_metrics", {}).get("acc", data.get("test_acc", 0.0))
    
    # Speed (samples/sec)
    speed = data.get("discovery_metrics", {}).get("train_samples_per_second", 0.0)
    
    # Ablations
    # sim_disabled_delta is in comparison_metrics
    sim_delta = data.get("comparison_metrics", {}).get("sim_disabled_delta", 0.0)
    # choice_without_sim_delta is in comparison_metrics
    choice_sim = data.get("comparison_metrics", {}).get("choice_without_sim_delta", 0.0)
    
    # Primitive dominance
    top_share = data.get("discovery_metrics", {}).get("primitive_top_share", 0.0)
    
    # Credit closed loop
    credit_closed = data.get("discovery_metrics", {}).get("credit_closed", 0.0)
    
    return {
        "acc": acc,
        "test": test,
        "speed": speed,
        "sim_delta": sim_delta,
        "choice_sim": choice_sim,
        "top_share": top_share,
        "credit_closed": credit_closed
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: compare_vnext_to_baseline.py <path_to_final_report.json> ...")
        sys.exit(1)
        
    paths = sys.argv[1:]
    reports_data = []
    
    for p in paths:
        if not os.path.exists(p):
            print(f"Error: file not found: {p}")
            sys.exit(1)
        try:
            reports_data.append(load_report(p))
        except Exception as e:
            print(f"Error reading {p}: {e}")
            sys.exit(1)
            
    extracted = [extract_metrics(r) for r in reports_data]
    
    # Compute mean across reports
    aggregated = {}
    for key in BASELINE.keys():
        vals = [item[key] for item in extracted]
        aggregated[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "raw": vals
        }
        
    print("=" * 70)
    print(f"vNext vs Baseline Comparison (N={len(paths)} runs)")
    print("=" * 70)
    print(f"{'Metric':<18} | {'Baseline':<10} | {'vNext Mean':<10} | {'Delta':<10} | {'vNext Std':<8}")
    print("-" * 70)
    
    for key, base_val in BASELINE.items():
        vnext_val = aggregated[key]["mean"]
        delta = vnext_val - base_val
        vnext_std = aggregated[key]["std"]
        
        # Color coding or positive/negative sign representation
        delta_str = f"{delta:+.4f}" if key != "speed" else f"{delta:+.1f}"
        base_str = f"{base_val:.4f}" if key != "speed" else f"{base_val:.1f}"
        vnext_str = f"{vnext_val:.4f}" if key != "speed" else f"{vnext_val:.1f}"
        vnext_std_str = f"{vnext_std:.4f}" if key != "speed" else f"{vnext_std:.1f}"
        
        print(f"{key:<18} | {base_str:<10} | {vnext_str:<10} | {delta_str:<10} | {vnext_std_str:<8}")
        
    print("=" * 70)
    
    # Print extra vNext diagnostic metrics if they are present in the first report
    first_report = reports_data[0]
    comparison = first_report.get("comparison_metrics", {})
    
    utility_keys = [
        "utility_score_mean", "utility_score_std", "utility_gain_corr",
        "utility_gain_spearman", "current_predicted_gain_corr",
        "utility_vs_current_gain_corr_delta", "proposal_top1_measured_gain",
        "utility_top1_measured_gain", "random_top1_measured_gain",
        "proposal_best_of_3_measured_gain", "utility_best_of_3_measured_gain",
        "mmr_best_of_3_measured_gain", "identity_mmr_similarity",
        "effect_mmr_similarity", "hybrid_mmr_similarity",
        "utility_overhead_seconds", "choice_without_utility_delta"
    ]
    
    vnext_diagnostics = {}
    for key in utility_keys:
        # Check in comparison_metrics, val_metrics, or top-level of reports
        for rep in reports_data:
            # We check val_metrics first, then comparison_metrics, then top-level
            v = rep.get("comparison_metrics", {}).get(key)
            if v is None:
                v = rep.get("val_metrics", {}).get(key)
            if v is None:
                v = rep.get(key)
            if v is not None:
                if key not in vnext_diagnostics:
                    vnext_diagnostics[key] = []
                vnext_diagnostics[key].append(float(v))
                
    if vnext_diagnostics:
        print("\nvNext Phase 1 Diagnostic Metrics:")
        print("-" * 50)
        for key, vals in vnext_diagnostics.items():
            mean_v = np.mean(vals)
            std_v = np.std(vals) if len(vals) > 1 else 0.0
            print(f"{key:<35} : {mean_v:.4f} (std={std_v:.4f})")
        print("=" * 70)

if __name__ == "__main__":
    main()
