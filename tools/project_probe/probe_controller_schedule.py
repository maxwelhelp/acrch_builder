#!/usr/bin/env python3
import torch
import torch.nn as nn
from arch_builder.model import ActionMatrixModel

def test_controller_schedule():
    print("--- Running test_controller_schedule ---")
    
    # 1. Initialize model with active schedules
    dim = 16
    slots = 2
    layers = 1
    
    model = ActionMatrixModel(
        dim=dim,
        slots=slots,
        layers=layers,
        enable_vnext=True,
        enable_utility_critic_choice=True,
        enable_mmr_controller=True,
        enable_lazy_executor=True,
        utility_exploration_start_weight=2.0,
        utility_exploration_end_weight=0.5,
        utility_exploration_warmup_steps=10,
        utility_budget_start=4,
        utility_budget_end=1,
        utility_budget_warmup_steps=10,
    )
    
    model.train()
    
    # Target values over steps
    # Formula for budget: round(start + progress * (end - start))
    # Formula for exploration: start + progress * (end - start)
    expected_budgets = []
    expected_weights = []
    for step in range(16):
        progress = max(0.0, min(1.0, float(step) / 10.0))
        budget = int(round(4.0 + progress * (1.0 - 4.0)))
        weight = 2.0 + progress * (0.5 - 2.0)
        expected_budgets.append(budget)
        expected_weights.append(weight)
        
    x = torch.randn(2, slots, dim)
    
    for step in range(16):
        model.set_vnext_step(step)
        
        # Forward pass to produce trace and metrics
        logits, trace = model(x)
        
        # Extract utility metrics from trace
        layer_trace = trace["layers"][0]
        utility_metrics = layer_trace.get("utility_metrics", {})
        
        budget_current = utility_metrics.get("utility_budget_current")
        weight_current = utility_metrics.get("utility_exploration_weight_current")
        
        print(f"Step {step}: expected_budget={expected_budgets[step]}, got={budget_current} | expected_weight={expected_weights[step]:.4f}, got={weight_current:.4f}")
        
        assert budget_current is not None, "budget_current missing from metrics"
        assert weight_current is not None, "weight_current missing from metrics"
        
        assert int(budget_current) == expected_budgets[step], f"Budget mismatch at step {step}: expected {expected_budgets[step]}, got {budget_current}"
        assert abs(float(weight_current) - expected_weights[step]) < 1e-5, f"Weight mismatch at step {step}: expected {expected_weights[step]}, got {weight_current}"

    # 2. Test evaluation mode (should freeze schedule and use static defaults)
    model.eval()
    model.set_vnext_step(0)
    logits, trace = model(x)
    utility_metrics = trace["layers"][0].get("utility_metrics", {})
    assert int(utility_metrics["utility_budget_current"]) == 3, f"Eval mode should use default budget (3), got {utility_metrics['utility_budget_current']}"
    assert abs(float(utility_metrics["utility_exploration_weight_current"]) - 0.5) < 1e-5, f"Eval mode should use end weight (0.5), got {utility_metrics['utility_exploration_weight_current']}"

    print("SUCCESS: test_controller_schedule passed!")

if __name__ == "__main__":
    test_controller_schedule()
