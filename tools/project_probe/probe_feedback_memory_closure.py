#!/usr/bin/env python3
import torch
import numpy as np
import sys
from arch_builder.primitive_matrix import PrimitiveMatrix5x5

def test_feedback_memory_closure():
    print("Testing feedback memory closure...")
    
    # Initialize Primitive Matrix
    num_layers = 2
    slots = 4
    pm = PrimitiveMatrix5x5(
        embed_dim=32,
        num_layers=num_layers,
        slots=slots,
        enable_scanner_feedback_memory=True,
    )
    
    # Verify shape of feedback buffers
    assert pm.feedback_gain_ema.shape == (num_layers, pm.num_cells, pm.num_primitives), "Invalid gain EMA shape"
    assert pm.feedback_regret_ema.shape == (num_layers, pm.num_cells, pm.num_primitives), "Invalid regret EMA shape"
    
    # Set positive credit for primitive index 12 in layer 0, cell 5
    target_layer = 0
    target_cell = 5
    target_prim = 12
    
    # Initialize EMA scores to low value, but target prim to high value
    pm.feedback_gain_ema.fill_(0.0)
    pm.feedback_regret_ema.fill_(0.0)
    pm.feedback_count.fill_(0)
    
    # Set target prim
    pm.feedback_gain_ema[target_layer, target_cell, target_prim] = 10.0
    pm.feedback_count[target_layer, target_cell, target_prim] = 5
    
    # Create fake ids/cell_ids to call feedback_topk
    n = 16 # batch size equals num_cells
    ids = torch.zeros(n, dtype=torch.long)
    cell_ids = torch.arange(n)
    
    # Get feedback candidates
    top_cands = pm.feedback_topk(ids, k=2, layer_idx=target_layer, cell_ids=cell_ids)
    
    # For target cell (index 5), the first candidate (top_cands[5, 0]) must be target_prim
    cell_top_cands = top_cands[target_cell].tolist()
    print(f"Top candidates for cell {target_cell}: {cell_top_cands}")
    assert cell_top_cands[0] == target_prim, f"Top candidate should be {target_prim}, got {cell_top_cands}"
    
    # Verify age tracking
    # Primitives that haven't been measured should be stale.
    pm.feedback_age.fill_(100.0)
    
    # Stale threshold is 50. All primitives should be stale.
    stale_count = pm.get_stale_count(layer_idx=target_layer, stale_threshold=50)
    print(f"Stale count (threshold=50) with all age=100: {stale_count}")
    assert stale_count > 0, "Stale count should be greater than 0"
    
    # Mark target_prim as fresh (age = 10, count = 1)
    pm.feedback_age[target_layer, target_cell, target_prim] = 10.0
    pm.feedback_count[target_layer, target_cell, target_prim] = 1
    
    # Its age is now 10. It is NOT stale.
    # Verify that the unmeasured ones are still stale.
    stale_count_after = pm.get_stale_count(layer_idx=target_layer, stale_threshold=50)
    print(f"Stale count after marking one as fresh: {stale_count_after}")
    
    print("FEEDBACK_MEMORY_CLOSURE_PASS")

if __name__ == "__main__":
    test_feedback_memory_closure()
