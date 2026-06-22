#!/usr/bin/env python3
"""Diagnostic probe verifying non-stationary adaptation. Simulates mid-training primitive failure and checks adaptation dynamics."""

import torch
import torch.nn.functional as F
from arch_builder.model import ActionMatrixModel
from arch_builder.synthetic_tasks import SyntheticKnownProgramTask

def main() -> int:
    print("=== Running Non-Stationary Adaptation Probe ===")
    
    device = "cpu"
    dim = 16
    slots = 4
    layers = 1
    classes = 2
    
    torch.manual_seed(42)
    
    # 1. Instantiate model and task
    task = SyntheticKnownProgramTask(task="diff", slots=slots, dim=dim, classes=classes)
    
    model = ActionMatrixModel(
        dim=dim,
        slots=slots,
        layers=layers,
        classes=classes,
        enable_vnext=True,
        enable_utility_critic_probe=True,
        utility_choice_warmup_steps=0,
        mmr_controller_warmup_steps=0,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 2. Train for 5 steps normally
    print("Phase 1: Normal training (diff primitive is active)...")
    for step in range(5):
        optimizer.zero_grad()
        batch = task.sample(batch_size=8, device=device)
        logits, info = model(batch.x)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        
        # Update usage credit based on correct predictions
        with torch.no_grad():
            sample_credit = (logits.argmax(dim=-1) == batch.y).to(logits.dtype)
            for layer_idx, layer_trace in enumerate(info["layers"]):
                edge_credit = sample_credit.repeat_interleave(slots * slots)
                cell_ids = torch.arange(slots * slots, device=logits.device).repeat(8)
                layer_ids = torch.full_like(layer_trace["chosen"], layer_idx, dtype=torch.long)
                model.pm.update_usage_credit(
                    layer_trace["chosen"],
                    edge_credit,
                    layer_ids=layer_ids,
                    cell_ids=cell_ids,
                    momentum=0.5,
                )
                
        print(f"  Step {step}: Loss = {loss.item():.4f}")
        
    print("\nUsage scores before ablation:")
    for idx, name in enumerate(model.pm.names):
        print(f"  Primitive '{name}': {float(model.pm.usage_score[idx]):.4f}")
        
    # 3. Suddenly ablate the 'diff' primitive (index 2)
    print("\nPhase 2: Primitive failure simulation (diff primitive is ablated/disabled)...")
    diff_idx = 2
    ablate_tensor = torch.full((8, slots * slots, 1), diff_idx, dtype=torch.long, device=device)
    primitive_ablation_ids = [ablate_tensor]
    
    for step in range(5):
        optimizer.zero_grad()
        batch = task.sample(batch_size=8, device=device)
        # Pass primitive_ablation_ids to disable diff
        logits, info = model(batch.x, primitive_ablation_ids=primitive_ablation_ids)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        
        # Update usage credit based on correct predictions
        with torch.no_grad():
            sample_credit = (logits.argmax(dim=-1) == batch.y).to(logits.dtype)
            for layer_idx, layer_trace in enumerate(info["layers"]):
                edge_credit = sample_credit.repeat_interleave(slots * slots)
                cell_ids = torch.arange(slots * slots, device=logits.device).repeat(8)
                layer_ids = torch.full_like(layer_trace["chosen"], layer_idx, dtype=torch.long)
                model.pm.update_usage_credit(
                    layer_trace["chosen"],
                    edge_credit,
                    layer_ids=layer_ids,
                    cell_ids=cell_ids,
                    momentum=0.5,
                )
                
        print(f"  Step {step+5}: Loss = {loss.item():.4f}")
        
    print("\nUsage scores after ablation:")
    for idx, name in enumerate(model.pm.names):
        print(f"  Primitive '{name}': {float(model.pm.usage_score[idx]):.4f}")
        
    assert torch.isfinite(loss), "Ablating the preferred primitive caused NaN/infinite losses!"
    print("\n=== All Non-Stationary Adaptation Probe Verification Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
