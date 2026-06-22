#!/usr/bin/env python3
"""Diagnostic probe verifying the mathematical correctness and gradient flow of spectral primitives."""

import torch
import torch.nn as nn
from arch_builder.primitive_matrix import PrimitiveMatrix5x5
from arch_builder.executor import ActionExecutor

def main() -> int:
    print("=== Running Spectral Primitives Probe ===")
    
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
    
    # Primitives to test
    spectral_names = ["dct", "fft_filter", "wavelet", "spectral_mix", "spectral_gate"]
    for name in spectral_names:
        assert name in pm.name_to_id, f"Primitive {name} not found in primitive matrix!"
        
    # 2. Test each spectral primitive individually
    batch_size = 4
    num_candidates = 1
    
    for name in spectral_names:
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
        
        # If binary/memory/etc, verify respective grads
        if name in {"spectral_mix"}:
            assert tgt.grad is not None, f"Gradient did not flow back to tgt for {name}!"
            assert torch.isfinite(tgt.grad).all(), f"tgt gradient of {name} contains NaN/Inf!"
            
        # Verify gradients flow back to executor parameters
        if name == "fft_filter":
            assert executor.fft_filter_weight.grad is not None, "Gradient did not flow back to fft_filter_weight!"
            assert torch.isfinite(executor.fft_filter_weight.grad).all(), "fft_filter_weight gradient contains NaN/Inf!"
            
        print(f"Primitive {name} Forward & Backward: PASS")
        
    # 3. Test mathematical specifics
    # DCT specific check: check that applying DCT to a flat signal yields energy concentrated at frequency 0
    print("\nVerifying DCT mathematical correctness...")
    flat_signal = torch.ones(1, dim, device=device)
    pid_dct = pm.name_to_id["dct"]
    dct_cand = torch.full((1, 1), pid_dct, dtype=torch.long, device=device)
    dct_out = executor(flat_signal, flat_signal, flat_signal, dct_cand).squeeze()
    
    # In DCT-II, flat signal should have non-zero value only at index 0 (DC coefficient)
    # let's assert that index 0 is non-zero, and all other indexes are close to 0
    print("DCT output for constant signal:", dct_out.cpu().numpy())
    assert abs(dct_out[0].item()) > 1.0
    assert (dct_out[1:].abs() < 1e-5).all(), f"DCT coefficients other than DC should be zero for constant signal, got {dct_out}"
    print("DCT mathematical verification: PASS")
    
    # FFT filter specific check: setting filter weights to 0 should zero out the signal
    print("\nVerifying FFT filter mathematical correctness...")
    with torch.no_grad():
        pid_fft = pm.name_to_id["fft_filter"]
        fft_cand = torch.full((1, 1), pid_fft, dtype=torch.long, device=device)
        
        # Zero out filter weights for this primitive id
        executor.fft_filter_weight[pid_fft].fill_(0.0)
        fft_out = executor(src[:1], tgt[:1], memory[:1], fft_cand).squeeze()
        print("FFT output with zero weight:", fft_out.cpu().numpy())
        assert (fft_out.abs() < 1e-5).all(), "FFT filtered output should be zero when filter weights are zero!"
    print("FFT filter mathematical verification: PASS")
    
    # Wavelet check: wavelet output should preserve signal norm (it's an orthogonal transform)
    print("\nVerifying Wavelet norm conservation...")
    pid_wv = pm.name_to_id["wavelet"]
    wv_cand = torch.full((1, 1), pid_wv, dtype=torch.long, device=device)
    test_src = torch.randn(1, dim, device=device)
    wv_out = executor(test_src, test_src, test_src, wv_cand).squeeze()
    norm_in = torch.norm(test_src).item()
    norm_out = torch.norm(wv_out).item()
    print(f"Wavelet input norm: {norm_in:.4f}, output norm: {norm_out:.4f}")
    assert abs(norm_in - norm_out) < 1e-4, "Wavelet transform should preserve L2 norm of the signal!"
    print("Wavelet mathematical verification: PASS")
    
    print("\n=== All Spectral Primitives Verification Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
