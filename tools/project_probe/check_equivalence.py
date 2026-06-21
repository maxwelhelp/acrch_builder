import torch
import numpy as np
import sys
from arch_builder.audio_frontend import AudioMatrixClassifier, build_frontend
from arch_builder.primitive_matrix import PrimitiveMatrix5x5

def test_equivalence():
    print("Testing equivalence of parameter counts and forward outputs...")

    dim = 24
    slots = 4
    top_k = 8
    sim_rank = 8
    
    # 1. Baseline Model (enable_vnext=False)
    torch.manual_seed(42)
    frontend_base = build_frontend("structured", slots=slots, dim=dim)
    model_base = AudioMatrixClassifier(
        frontend=frontend_base,
        dim=dim,
        slots=slots,
        layers=1,
        top_k=top_k,
        sim_rank=sim_rank,
        enable_vnext=False,
        enable_utility_critic_probe=False,
        enable_utility_critic_choice=False,
    )
        
    base_params = sum(p.numel() for p in model_base.parameters())
    print(f"Baseline parameter count: {base_params}")
    
    # Let's inspect the parameter names in model_base
    print("\nBaseline parameter names:")
    for name, param in model_base.named_parameters():
        print(f"  {name}: {param.shape}")

if __name__ == "__main__":
    test_equivalence()
