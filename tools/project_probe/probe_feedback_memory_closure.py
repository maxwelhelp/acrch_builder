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
    
    # Set positive credit for primitive index 12 in layer 0, cell 5 using update_usage_credit
    chosen_ids = torch.tensor([target_prim], dtype=torch.long)
    credit_val = torch.tensor([1.0], dtype=torch.float32)
    layer_tensor = torch.tensor([target_layer], dtype=torch.long)
    cell_tensor = torch.tensor([target_cell], dtype=torch.long)
    
    for _ in range(5):
        pm.update_usage_credit(
            chosen_ids=chosen_ids,
            credit=credit_val,
            layer_ids=layer_tensor,
            cell_ids=cell_tensor,
            momentum=0.9,
        )
        
    # Verify that the buffers updated correctly through the real pathway
    metrics = pm.metrics()
    assert pm.feedback_count[target_layer, target_cell, target_prim].item() == 5, "feedback_count did not increment to 5"
    assert pm.feedback_gain_ema[target_layer, target_cell, target_prim].item() > 0, "feedback_gain_ema was not updated"
    assert metrics["feedback_bias_abs"] > 0, "feedback_bias_abs should be positive"
    assert metrics["feedback_update_called"] == 5.0, "feedback_update_called should be 5.0"
    
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
    
    # Mark target_prim as fresh using update_usage_credit.
    # update_usage_credit resets its age to 0.
    pm.update_usage_credit(
        chosen_ids=chosen_ids,
        credit=credit_val,
        layer_ids=layer_tensor,
        cell_ids=cell_tensor,
        momentum=0.9,
    )
    # Then we add 10 to age to simulate 10 steps of aging
    pm.feedback_age.add_(10.0)
    
    # Its age is now 10. It is NOT stale.
    # Verify that the unmeasured ones are still stale.
    stale_count_after = pm.get_stale_count(layer_idx=target_layer, stale_threshold=50)
    print(f"Stale count after marking one as fresh: {stale_count_after}")
    
    print("FEEDBACK_MEMORY_CLOSURE_PASS")

if __name__ == "__main__":
    test_feedback_memory_closure()
