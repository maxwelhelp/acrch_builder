#!/usr/bin/env python3
"""Diagnostic probe verifying CIFAR-10 patch classifier architecture, forward/backward passes, and training convergence on a mock micro-batch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from arch_builder.model import ActionMatrixModel

class CIFAR10PatchClassifier(nn.Module):
    """Classifier slicing 32x32 RGB images into 16 non-overlapping 8x8 patches,
    routing them through ActionMatrixModel, and classifying into 10 categories.
    """
    def __init__(self, dim: int = 64, slots: int = 16, layers: int = 1, classes: int = 10, enable_vnext: bool = True):
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.layers = layers
        self.classes = classes
        
        # Patch projection layer: 8*8*3 = 192 inputs
        self.patch_proj = nn.Linear(192, dim)
        
        # Backbone ActionMatrixModel
        # This model has a built-in classification head self.classifier mapping from dim -> classes
        self.backbone = ActionMatrixModel(
            dim=dim,
            slots=slots,
            layers=layers,
            classes=classes,
            enable_vnext=enable_vnext,
            enable_utility_critic_probe=True,
            utility_choice_warmup_steps=0,
            mmr_controller_warmup_steps=0,
        )

    def forward(self, x: torch.Tensor):
        # x shape: [B, 3, 32, 32]
        batch_size = x.shape[0]
        
        # Slice into 16 patches of 8x8
        # x.unfold(2, 8, 8) -> [B, 3, 4, 32, 8]
        # x.unfold(...) -> [B, 3, 4, 4, 8, 8]
        patches = x.unfold(2, 8, 8).unfold(3, 8, 8)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() # [B, 4, 4, 3, 8, 8]
        patches = patches.view(batch_size, 16, 192) # [B, 16, 192]
        
        # Project patches to model dimension
        slots_in = self.patch_proj(patches) # [B, 16, dim]
        
        # Route through ActionMatrixModel, which directly returns 10-class logits
        logits, choice_info = self.backbone(slots_in) # logits shape: [B, 10]
        return logits, choice_info

def main() -> int:
    print("=== Running CIFAR-10 Patch Classifier Probe ===")
    
    device = "cpu"
    dim = 32
    slots = 16
    layers = 1
    classes = 10
    
    torch.manual_seed(42)
    
    # 1. Instantiate the patch classifier
    model = CIFAR10PatchClassifier(
        dim=dim,
        slots=slots,
        layers=layers,
        classes=classes,
        enable_vnext=True
    ).to(device)
    
    # 2. Verify forward pass shapes
    print("Verifying forward pass shapes...")
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, classes, (batch_size,), device=device)
    
    logits, choice_info = model(x)
    print("Logits shape:", logits.shape)
    assert logits.shape == (batch_size, classes), f"Expected shape {(batch_size, classes)}, got {logits.shape}"
    
    # 3. Verify backward pass and gradient flow
    print("Verifying backward pass...")
    loss = F.cross_entropy(logits, y)
    loss.backward()
    
    # Check that gradients flowed to patch projection and backbone
    assert model.patch_proj.weight.grad is not None, "Gradients did not flow to patch projection layer!"
    assert torch.isfinite(model.patch_proj.weight.grad).all(), "Gradients in patch projection contain NaN/Inf!"
    print("Gradients flow: PASS")
    
    # 4. Verify quick learning convergence on mock micro-batch
    print("\nVerifying learning convergence on mock micro-batch (5 steps)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Use static batch to test memorization
    static_x = torch.randn(8, 3, 32, 32, device=device)
    static_y = torch.randint(0, classes, (8,), device=device)
    
    initial_loss = None
    for step in range(5):
        optimizer.zero_grad()
        logits, _ = model(static_x)
        loss = F.cross_entropy(logits, static_y)
        
        if initial_loss is None:
            initial_loss = loss.item()
            
        loss.backward()
        optimizer.step()
        print(f"  Step {step}: Loss = {loss.item():.4f}")
        
    final_loss = loss.item()
    print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss, f"Loss did not decrease! Initial: {initial_loss}, Final: {final_loss}"
    print("Mock micro-batch convergence: PASS")
    
    print("\n=== All CIFAR-10 Patch Classifier Probe Verification Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
