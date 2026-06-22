#!/usr/bin/env python3
"""Diagnostic probe verifying multi-layer composition credit assignment. Trains a 2-layer model and checks that dependent primitives receive credit."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from arch_builder.model import ActionMatrixModel

class CompositionTask:
    """Synthetic composite task requiring two steps:
    1. diff = S0 - S1
    2. product = diff * S1
    Target is sign of product.
    """
    def __init__(self, slots: int = 4, dim: int = 16, classes: int = 2):
        self.slots = slots
        self.dim = dim
        self.classes = classes

    def sample(self, batch_size: int = 8, device: str = "cpu"):
        x = torch.zeros(batch_size, self.slots, self.dim, device=device)
        # Fill slots 0 and 1 with random features
        x[:, 0] = torch.randn(batch_size, self.dim, device=device)
        x[:, 1] = torch.randn(batch_size, self.dim, device=device)
        
        diff = x[:, 0] - x[:, 1]
        prod = diff * x[:, 1]
        
        # Target: 1 if dot product of prod is positive, 0 otherwise
        val = prod.sum(dim=-1)
        y = (val > 0.0).long()
        
        # Wrap in a simple Namespace-like class
        class Batch:
            pass
        b = Batch()
        b.x = x
        b.y = y
        return b


def main() -> int:
    print("=== Running Multi-Layer Composition Credit Probe ===")
    
    device = "cpu"
    dim = 16
    slots = 4
    layers = 2
    classes = 2
    
    torch.manual_seed(42)
    
    # 1. Instantiate model and task
    task = CompositionTask(slots=slots, dim=dim, classes=classes)
    
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
    
    # 2. Train for 10 steps
    print("Training 2-layer model on composite task (10 steps)...")
    initial_loss = None
    for step in range(10):
        optimizer.zero_grad()
        batch = task.sample(batch_size=8, device=device)
        logits, info = model(batch.x)
        loss = F.cross_entropy(logits, batch.y)
        
        if initial_loss is None:
            initial_loss = loss.item()
            
        loss.backward()
        optimizer.step()
        
        # Update usage credit across both layers
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
        
    print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {loss.item():.4f}")
    assert loss.item() < initial_loss, f"Loss did not decrease! Initial: {initial_loss}, Final: {loss.item()}"
    
    # 3. Print usage scores of key primitives: 'diff' (index 2) and 'product' (index 8)
    diff_idx = 2
    prod_idx = 8
    diff_score = float(model.pm.usage_score[diff_idx])
    prod_score = float(model.pm.usage_score[prod_idx])
    
    print(f"\nKey Primitives Usage Credit:")
    print(f"  Primitive 'diff' (Layer 0 candidate): {diff_score:.4f}")
    print(f"  Primitive 'product' (Layer 1 candidate): {prod_score:.4f}")
    
    # Assert that both primitives received credit (usage score > 0)
    assert diff_score > 0.0, f"Expected diff to have positive credit, got {diff_score}"
    assert prod_score > 0.0, f"Expected product to have positive credit, got {prod_score}"
    print("\nMulti-layer credit propagation: PASS")
    
    print("\n=== All Multi-Layer Composition Credit Probe Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
