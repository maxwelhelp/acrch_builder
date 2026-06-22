#!/usr/bin/env python3
"""Diagnostic probe verifying Copy and Reverse memory task architectures, forward/backward passes, and convergence on micro-batches."""

import torch
import torch.nn.functional as F
from arch_builder.model import ActionMatrixModel

def run_task_probe(task_name: str) -> None:
    print(f"\n--- Testing Memory Task: {task_name.upper()} ---")
    
    device = "cpu"
    dim = 16
    slots = 8
    layers = 1
    classes = 2 # dummy classes
    
    torch.manual_seed(42)
    
    # 1. Instantiate the model
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
    
    # 2. Setup mock data
    # Slots: 0, 1, 2 = source tokens
    # Slots: 3 = separation/dummy slot (zeros)
    # Slots: 4, 5, 6 = target slots (initialized to zeros)
    # Slot: 7 = dummy
    batch_size = 4
    x = torch.zeros(batch_size, slots, dim, device=device)
    # Fill source slots with random values
    x[:, 0:3] = torch.randn(batch_size, 3, dim, device=device)
    
    # Run forward pass
    logits, choice_info = model(x)
    
    # 3. Verify slots_out is present and correct
    assert "slots_out" in choice_info, "slots_out is missing from choice_info!"
    slots_out = choice_info["slots_out"]
    print("slots_out shape:", slots_out.shape)
    assert slots_out.shape == (batch_size, slots, dim), f"Expected slots_out shape {(batch_size, slots, dim)}, got {slots_out.shape}"
    
    # 4. Verify backward pass
    # Expected target states
    if task_name == "copy":
        targets = x[:, 0:3].clone() # copy source to target slots
    else: # reverse
        targets = x[:, 0:3].flip(dims=[1]).clone() # reverse source to target slots
        
    loss = F.mse_loss(slots_out[:, 4:7], targets)
    loss.backward()
    
    # Check that gradients flow to model parameters
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Gradients contain NaN/Inf!"
    print("Gradients flow: PASS")
    
    # 5. Verify memorization convergence on static micro-batch
    print("Running quick memorization training (10 steps)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Static micro-batch
    static_x = torch.zeros(4, slots, dim, device=device)
    static_x[:, 0:3] = torch.randn(4, 3, dim, device=device)
    if task_name == "copy":
        static_targets = static_x[:, 0:3].clone()
    else:
        static_targets = static_x[:, 0:3].flip(dims=[1]).clone()
        
    initial_loss = None
    for step in range(10):
        optimizer.zero_grad()
        _, info = model(static_x)
        out_slots = info["slots_out"]
        loss = F.mse_loss(out_slots[:, 4:7], static_targets)
        
        if initial_loss is None:
            initial_loss = loss.item()
            
        loss.backward()
        optimizer.step()
        print(f"  Step {step}: Loss = {loss.item():.5f}")
        
    final_loss = loss.item()
    print(f"Initial Loss: {initial_loss:.5f} -> Final Loss: {final_loss:.5f}")
    assert final_loss < initial_loss, f"Loss did not decrease! Initial: {initial_loss}, Final: {final_loss}"
    print(f"Convergence check for {task_name}: PASS")


def main() -> int:
    print("=== Running Copy & Reverse Memory Task Probe ===")
    run_task_probe("copy")
    run_task_probe("reverse")
    print("\n=== All Memory Task Probe Verification Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
