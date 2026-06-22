#!/usr/bin/env python3
"""Diagnostic probe verifying the mathematical correctness and gradient flow of mined local primitives."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from arch_builder.primitive_matrix import PrimitiveMatrix5x5
from arch_builder.executor import ActionExecutor

def main() -> int:
    print("=== Running Mined Atoms Primitives Probe ===")
    
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
    
    # Mined local primitives to test
    mined_names = ["svd_atom_k", "diag", "toeplitz", "block_mean", "mined_gate"]
    for name in mined_names:
        assert name in pm.name_to_id, f"Primitive {name} not found in primitive matrix!"
        
    # 2. Test each mined local primitive individually
    batch_size = 4
    num_candidates = 1
    
    for name in mined_names:
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
        
        # Verify gradients flow back to executor's mined parameters
        if name == "svd_atom_k":
            assert executor.mined_u.grad is not None, "Gradient did not flow back to mined_u!"
            assert torch.isfinite(executor.mined_u.grad).all(), "mined_u gradient contains NaN/Inf!"
            assert executor.mined_v.grad is not None, "Gradient did not flow back to mined_v!"
            assert torch.isfinite(executor.mined_v.grad).all(), "mined_v gradient contains NaN/Inf!"
            
        elif name == "diag":
            assert executor.mined_diag.grad is not None, "Gradient did not flow back to mined_diag!"
            assert torch.isfinite(executor.mined_diag.grad).all(), "mined_diag gradient contains NaN/Inf!"
            
        elif name == "toeplitz":
            assert executor.mined_toeplitz_filter.grad is not None, "Gradient did not flow back to mined_toeplitz_filter!"
            assert torch.isfinite(executor.mined_toeplitz_filter.grad).all(), "mined_toeplitz_filter gradient contains NaN/Inf!"
            
        elif name == "mined_gate":
            assert executor.mined_source_matrix.grad is not None, "Gradient did not flow back to mined_source_matrix!"
            assert torch.isfinite(executor.mined_source_matrix.grad).all(), "mined_source_matrix gradient contains NaN/Inf!"
            
        print(f"Primitive {name} Forward & Backward: PASS")
        
    # 3. Test mathematical specifics
    with torch.no_grad():
        # diag specific check:
        print("\nVerifying diag mathematical correctness...")
        pid_diag = pm.name_to_id["diag"]
        diag_cand = torch.full((1, 1), pid_diag, dtype=torch.long, device=device)
        executor.mined_diag.fill_(3.0)
        test_src = torch.randn(1, dim, device=device)
        diag_out = executor(test_src, test_src, test_src, diag_cand).squeeze()
        assert torch.allclose(diag_out, 3.0 * test_src, atol=1e-5), "diag output should be exactly 3.0 * src!"
        print("diag mathematical verification: PASS")
        
        # block_mean specific check:
        print("\nVerifying block_mean mathematical correctness...")
        pid_bm = pm.name_to_id["block_mean"]
        bm_cand = torch.full((1, 1), pid_bm, dtype=torch.long, device=device)
        
        # Create test_src where each of 4 blocks of size 8 has a constant value
        # e.g., block 0 has 1s, block 1 has 2s, block 2 has 3s, block 3 has 4s
        test_src_bm = torch.cat([
            torch.full((1, 8), 1.0),
            torch.full((1, 8), 2.0),
            torch.full((1, 8), 3.0),
            torch.full((1, 8), 4.0),
        ], dim=-1).to(device)
        
        bm_out = executor(test_src_bm, test_src_bm, test_src_bm, bm_cand).squeeze()
        print("block_mean output matches expected constant blocks:", torch.allclose(bm_out, test_src_bm))
        assert torch.allclose(bm_out, test_src_bm, atol=1e-5), "block_mean should keep constant blocks constant!"
        
        # Create test_src with non-constant values, check that output is average
        test_src_rand = torch.randn(1, dim, device=device)
        bm_out_rand = executor(test_src_rand, test_src_rand, test_src_rand, bm_cand).squeeze()
        for i in range(4):
            avg_val = test_src_rand[0, i*8 : (i+1)*8].mean().item()
            assert (bm_out_rand[i*8 : (i+1)*8] - avg_val).abs().max() < 1e-5, f"Block {i} elements should equal average {avg_val}!"
        print("block_mean mathematical verification: PASS")
        
        # toeplitz specific check:
        print("\nVerifying toeplitz mathematical correctness...")
        pid_tp = pm.name_to_id["toeplitz"]
        tp_cand = torch.full((1, 1), pid_tp, dtype=torch.long, device=device)
        
        # Set filter to [0, 0, 1, 0, 0] (center element is 1.0, others are 0.0)
        # Filter is 1D convolution kernel with shape [out_channels, in_channels, kernel_width] -> [1, 1, 5]
        custom_filter = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], device=device).view(1, 1, 5)
        executor.mined_toeplitz_filter.copy_(custom_filter)
        
        test_src_tp = torch.randn(1, dim, device=device)
        tp_out = executor(test_src_tp, test_src_tp, test_src_tp, tp_cand).squeeze()
        assert torch.allclose(tp_out, test_src_tp, atol=1e-5), "toeplitz with identity filter should return src!"
        print("toeplitz mathematical verification: PASS")
        
        # mined_gate specific check:
        print("\nVerifying mined_gate mathematical correctness...")
        pid_mg = pm.name_to_id["mined_gate"]
        mg_cand = torch.full((1, 1), pid_mg, dtype=torch.long, device=device)
        
        # Set source matrix to zero, so sigmoid(0) = 0.5 -> output should be 0.5 * src
        executor.mined_source_matrix.fill_(0.0)
        test_src_mg = torch.randn(1, dim, device=device)
        mg_out = executor(test_src_mg, test_src_mg, test_src_mg, mg_cand).squeeze()
        assert torch.allclose(mg_out, 0.5 * test_src_mg, atol=1e-5), "mined_gate output should be 0.5 * src when matrix is zero!"
        print("mined_gate mathematical verification: PASS")
        
    print("\n=== All Mined Atoms Primitives Verification Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
