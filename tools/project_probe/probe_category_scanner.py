#!/usr/bin/env python3
"""Diagnostic probe verifying the correctness of category_topk and Category-Aware Scanner integration."""

import torch
import numpy as np
from arch_builder.primitive_matrix import PrimitiveMatrix5x5
from arch_builder.hybrid_scanner import HybridScanner

def main() -> int:
    print("=== Running Category-Aware Scanner Probe ===")
    
    # 1. Initialize PrimitiveMatrix5x5 in vnext mode (8 families, 40 primitives)
    torch.manual_seed(42)
    device = "cpu"
    pm = PrimitiveMatrix5x5(
        embed_dim=32,
        enable_vnext=True,
        num_layers=1,
        enable_scanner_feedback_memory=True,
        slots=4
    ).to(device)
    
    # Cold start check: usage_score prior
    # Let's populate usage_score with some distinct values
    usage_scores = torch.linspace(0.1, 1.0, steps=40)
    pm.usage_score.copy_(usage_scores)
    # Set usage observations to simulate some measurements
    pm.usage_observations.copy_(torch.ones(40))
    
    anchor_ids = torch.zeros(2, dtype=torch.long)  # Batch size 2
    
    # Run category_topk with k=2 in cold start
    # No feedback observations yet, so feedback_count is all 0, should fall back to usage_score
    print("Testing category_topk in cold start (falling back to usage_score)...")
    res_cold = pm.category_topk(anchor_ids, k=2, layer_idx=0)
    print("Cold start category_topk(k=2) output shape:", res_cold.shape)
    assert res_cold.shape == (2, 8 * 2), f"Expected shape (2, 16), got {res_cold.shape}"
    
    # Validate that indices correspond to the best usage_score within each row
    for b in range(2):
        row_indices = res_cold[b].cpu().numpy().reshape(8, 2)
        for r in range(8):
            row_start = r * 5
            row_vals = pm.usage_score[row_start : row_start + 5].cpu().numpy()
            expected_best_in_row = np.argsort(row_vals)[-2:][::-1] + row_start
            actual_in_row = row_indices[r]
            assert set(actual_in_row) == set(expected_best_in_row), \
                f"Row {r} expected {expected_best_in_row}, got {actual_in_row}"
    print("Cold start category_topk validation: PASS")
    
    # 2. Warm start check: feedback_gain_ema and feedback_regret_ema
    # Setup mock feedback counts to mark them as seen
    pm.feedback_count[0, 0].fill_(1.0)
    pm.feedback_count[0, 1].fill_(1.0)
    
    # Populate mock gain/regret
    # Let's make the second primitive in each row (row_start + 1) have high gain,
    # and the fourth primitive in each row (row_start + 3) have medium gain
    mock_gain = torch.zeros(1, 16, 40)
    mock_regret = torch.zeros(1, 16, 40)
    
    for r in range(8):
        row_start = r * 5
        mock_gain[0, :, row_start + 1] = 2.0  # Best
        mock_gain[0, :, row_start + 3] = 1.0  # Second best
        # Let's add some regret to row_start + 2 to make it low bias
        mock_gain[0, :, row_start + 2] = 1.5
        mock_regret[0, :, row_start + 2] = 2.0  # bias = 1.5 - 1.0 = 0.5
        
    pm.feedback_gain_ema.copy_(mock_gain)
    pm.feedback_regret_ema.copy_(mock_regret)
    
    print("Testing category_topk in warm start (with feedback)...")
    cell_ids = torch.tensor([0, 1])
    res_warm = pm.category_topk(anchor_ids, k=2, layer_idx=0, cell_ids=cell_ids)
    assert res_warm.shape == (2, 16)
    
    for b in range(2):
        row_indices = res_warm[b].cpu().numpy().reshape(8, 2)
        for r in range(8):
            row_start = r * 5
            # For row r, best should be row_start + 1 and row_start + 3
            actual_in_row = row_indices[r]
            expected_in_row = [row_start + 1, row_start + 3]
            assert set(actual_in_row) == set(expected_in_row), \
                f"Row {r} warm expected {expected_in_row}, got {actual_in_row}"
    print("Warm start category_topk validation: PASS")
    
    # Compare with category_best
    res_best = pm.category_best(anchor_ids, layer_idx=0, cell_ids=cell_ids)
    assert res_best.shape == (2, 8)
    for b in range(2):
        row_indices = res_best[b].cpu().numpy().reshape(8, 1)
        for r in range(8):
            row_start = r * 5
            assert row_indices[r, 0] == row_start + 1, \
                f"Expected top-1 row {r} to be {row_start+1}, got {row_indices[r, 0]}"
    print("Equivalence with category_best validation: PASS")
    
    # 3. Test HybridScanner forward pass shapes and integration
    print("Testing HybridScanner forward integration...")
    context = torch.randn(2, 32 * 5)
    memory = torch.randn(2, 32)
    
    scanner = HybridScanner(
        dim=32,
        context_dim=32 * 5,
        prim_embed_dim=32,
        local_k=9,
        semantic_k=4,
        usage_k=5,
        random_k=1,
        enable_scanner_feedback_memory=True,
        enable_category_scanner=True,
        num_primitives=40,
        layer_idx=0,
        category_k=2
    ).to(device)
    
    # Check that category_k is correctly stored
    assert scanner.category_k == 2
    
    candidate_ids, proposal_logits, source_ids, metrics = scanner(
        context=context,
        memory=memory,
        primitive_matrix=pm,
        cell_ids=cell_ids
    )
    
    # Sources configuration:
    # local_k=9, semantic_k=4, usage_k=5, random_k=1 -> 19 candidates
    # plus feedback_k=3 -> 22 candidates
    # plus category_k=2 per row (8 rows) -> 16 candidates
    # Total candidates = 19 + 3 + 16 = 38 candidates
    expected_candidates = 9 + 4 + 5 + 1 + 3 + (8 * 2)
    print(f"HybridScanner returned {candidate_ids.shape[1]} candidates, expected {expected_candidates}")
    assert candidate_ids.shape == (2, expected_candidates), \
        f"Expected shape {(2, expected_candidates)}, got {candidate_ids.shape}"
        
    print("HybridScanner outputs integration validation: PASS")
    print("=== All Category-Aware Scanner Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
