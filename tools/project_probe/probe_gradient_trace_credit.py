#!/usr/bin/env script
"""Diagnostic probe verifying the correctness and activation of the Gradient Trace Credit Hook."""

import torch
import torch.nn as nn
from arch_builder.model import ActionMatrixModel

def main() -> int:
    print("=== Running Gradient Trace Credit Probe ===")
    
    device = "cpu"
    dim = 32
    slots = 4
    layers = 1
    classes = 2
    
    torch.manual_seed(42)
    
    # 1. Initialize ActionMatrixModel in vnext mode with utility critic enabled
    # We must set self.training = True to trigger the hook,
    # and enable_utility_critic_probe=True to instantiate the utility critic.
    model = ActionMatrixModel(
        dim=dim,
        slots=slots,
        layers=layers,
        classes=classes,
        enable_vnext=True,
        enable_utility_critic_probe=True,
        utility_choice_warmup_steps=0,  # Ensure warmup doesn't bypass normal flows
        mmr_controller_warmup_steps=0,
    ).to(device)
    
    model.train()
    
    # Enable gradients on model inputs
    inputs = torch.randn(2, slots, dim, requires_grad=True, device=device)
    
    # 2. Perform forward pass
    out, choice_info = model(inputs)
    
    # Since we set classes=2, out has shape (2, dim) or (2, classes)?
    # Let's inspect out.shape
    print("Model output shape:", out.shape)
    
    # Define a simple loss
    loss = out.sum()
    
    # Clear any prior queue states
    for layer in model.layers:
        layer.grad_credit_queue.clear()
        
    # 3. Perform backward pass to trigger hooks
    print("Running backward pass to trigger hooks...")
    loss.backward()
    
    # 4. Verify that hook triggered and populated grad_credit_queue
    for i, layer in enumerate(model.layers):
        print(f"\nChecking Layer {i} grad_credit_queue...")
        queue = layer.grad_credit_queue
        print("Queue length:", len(queue))
        
        # Hook should be triggered on backward pass, appending to queue
        assert len(queue) > 0, f"Layer {i} grad_credit_queue is empty! Hook was not triggered."
        
        # Verify queue contents
        for item_idx, item in enumerate(queue):
            assert len(item) == 5, f"Expected 5 elements in queue item, got {len(item)}"
            c_det, e_det, h_det, cand_det, grad_credit = item
            
            # Print shapes
            print(f"Queue Item {item_idx}:")
            print("  c_det (flat_context) shape:", c_det.shape)
            print("  e_det (prim_emb) shape:", e_det.shape)
            print("  h_det (flat_target_address) shape:", h_det.shape)
            print("  cand_det (chosen_ids) shape:", cand_det.shape)
            print("  grad_credit shape:", grad_credit.shape)
            
            # Check for NaNs/Infs
            assert torch.isfinite(c_det).all(), "flat_context contains NaN/Inf!"
            assert torch.isfinite(e_det).all(), "prim_emb contains NaN/Inf!"
            assert torch.isfinite(h_det).all(), "flat_target_address contains NaN/Inf!"
            assert torch.isfinite(cand_det).all(), "chosen_ids contains NaN/Inf!"
            assert torch.isfinite(grad_credit).all(), "grad_credit contains NaN/Inf!"
            
            # Assert grad_credit has non-trivial values (non-zero since gradients are active)
            print("  grad_credit values:", grad_credit)
            assert (grad_credit.abs().sum() > 0.0).item(), "grad_credit values are all zero! Expected active gradients."
            
    print("\n=== All Gradient Trace Credit Hook Verification Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
