#!/usr/bin/env script
"""Diagnostic probe verifying the mathematical correctness and gradient flow of attention-like primitives."""

import torch
import torch.nn as nn
from arch_builder.primitive_matrix import PrimitiveMatrix5x5
from arch_builder.executor import ActionExecutor

def main() -> int:
    print("=== Running Attention-Like Primitives Probe ===")
    
    device = "cpu"
    dim = 32
    torch.manual_seed(42)
    
    # 1. Initialize PrimitiveMatrix5x5 and ActionExecutor in vnext mode
    pm = PrimitiveMatrix5x5(
        embed_dim=16,
        enable_vnext=True,
        num_layers=1,
        enable_scanner_feedback_memory=False,
        slots=4
    ).to(device)
    
    executor = ActionExecutor(dim=dim, primitive_matrix=pm, enable_vnext=True).to(device)
    
    # Attention-like primitives to test
    attn_names = ["qkv_gate", "cross_attend", "self_attend", "key_align", "value_mix"]
    for name in attn_names:
        assert name in pm.name_to_id, f"Primitive {name} not found in primitive matrix!"
        
    # 2. Test each attention-like primitive individually
    batch_size = 4
    num_candidates = 1
    
    for name in attn_names:
        print(f"\nTesting primitive: {name}")
        
        # Prepare inputs with gradients enabled
        src = torch.randn(batch_size, dim, requires_grad=True, device=device)
        tgt = torch.randn(batch_size, dim, requires_grad=True, device=device)
        memory = torch.randn(batch_size, dim, requires_grad=True, device=device)
        
        pid = pm.name_to_id[name]
        candidate_ids = torch.full((batch_size, num_candidates), pid, dtype=torch.long, device=device)
        
        # Forward pass
        out = executor(src, tgt, memory, candidate_ids)
        
        # Verify output shape
        expected_shape = (batch_size, num_candidates, dim)
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"
        
        # Verify output is finite
        assert torch.isfinite(out).all(), f"Output of {name} contains NaN or Inf!"
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        
        # Verify gradients flow back to inputs without NaNs
        assert src.grad is not None, f"Gradient did not flow back to src for {name}!"
        assert torch.isfinite(src.grad).all(), f"src gradient of {name} contains NaN/Inf!"
        
        if name in {"cross_attend", "value_mix", "key_align"}:
            assert tgt.grad is not None, f"Gradient did not flow back to tgt for {name}!"
            assert torch.isfinite(tgt.grad).all(), f"tgt gradient of {name} contains NaN/Inf!"
            
        if name in {"self_attend"}:
            assert memory.grad is not None, f"Gradient did not flow back to memory for {name}!"
            assert torch.isfinite(memory.grad).all(), f"memory gradient of {name} contains NaN/Inf!"
            
        # Verify gradients flow back to executor projection parameters
        assert executor.q_proj.grad is not None, "Gradient did not flow back to q_proj!"
        assert torch.isfinite(executor.q_proj.grad).all(), "q_proj gradient contains NaN/Inf!"
        
        assert executor.k_proj.grad is not None, "Gradient did not flow back to k_proj!"
        assert torch.isfinite(executor.k_proj.grad).all(), "k_proj gradient contains NaN/Inf!"
        
        assert executor.v_proj.grad is not None, "Gradient did not flow back to v_proj!"
        assert torch.isfinite(executor.v_proj.grad).all(), "v_proj gradient contains NaN/Inf!"
        
        print(f"Primitive {name} Forward & Backward: PASS")
        
    # 3. Test mathematical specifics
    with torch.no_grad():
        # key_align specific check:
        # If Q and K are perfectly aligned (cosine similarity 1.0), output should be exactly src.
        print("\nVerifying key_align mathematical correctness...")
        pid_ka = pm.name_to_id["key_align"]
        ka_cand = torch.full((1, 1), pid_ka, dtype=torch.long, device=device)
        
        # Set q_proj and k_proj to identity matrices, so q = src and k = tgt
        executor.q_proj[pid_ka].copy_(torch.eye(dim))
        executor.k_proj[pid_ka].copy_(torch.eye(dim))
        
        test_src = torch.randn(1, dim, device=device)
        # Normalize test_src so Q and K are aligned unit vectors
        test_src = test_src / torch.norm(test_src)
        test_tgt = test_src.clone()  # tgt identical to src -> cosine sim is 1.0
        
        ka_out = executor(test_src, test_tgt, test_src, ka_cand).squeeze()
        print("Cosine similarity should be 1.0, key_align output matches src:", torch.allclose(ka_out, test_src))
        assert torch.allclose(ka_out, test_src, atol=1e-5), "key_align output should match src when Q & K are identical!"
        
        test_tgt_opp = -test_src.clone()  # opposite direction -> cosine sim is -1.0
        ka_out_opp = executor(test_src, test_tgt_opp, test_src, ka_cand).squeeze()
        assert torch.allclose(ka_out_opp, -test_src, atol=1e-5), "key_align output should be -src when Q & K are opposite!"
        print("key_align mathematical verification: PASS")
        
        # value_mix specific check:
        # If scalar product of Q and K is 0, weight should be sigmoid(0) = 0.5, yielding 0.5 * src + 0.5 * tgt.
        print("\nVerifying value_mix mathematical correctness...")
        pid_vm = pm.name_to_id["value_mix"]
        vm_cand = torch.full((1, 1), pid_vm, dtype=torch.long, device=device)
        
        # Set projections to zero so q = 0, k = 0, weight = sigmoid(0) = 0.5
        executor.q_proj[pid_vm].fill_(0.0)
        executor.k_proj[pid_vm].fill_(0.0)
        # Set v_proj to identity so v = tgt
        executor.v_proj[pid_vm].copy_(torch.eye(dim))
        
        test_src_vm = torch.randn(1, dim, device=device)
        test_tgt_vm = torch.randn(1, dim, device=device)
        vm_out = executor(test_src_vm, test_tgt_vm, test_src_vm, vm_cand).squeeze()
        expected_vm = 0.5 * test_src_vm + 0.5 * test_tgt_vm
        assert torch.allclose(vm_out, expected_vm, atol=1e-5), "value_mix output should be average when weight is 0.5!"
        print("value_mix mathematical verification: PASS")
        
    print("\n=== All Attention-Like Primitives Verification Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
