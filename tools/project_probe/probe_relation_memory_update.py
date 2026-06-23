#!/usr/bin/env python3
import torch
import sys
from arch_builder.primitive_matrix import PrimitiveMatrix5x5
from arch_builder.relation_consistency import RelationMemory

def test_relation_memory_update():
    print("Testing relation memory update...")

    # Initialize
    num_primitives = 40
    rm = RelationMemory(num_primitives)

    # Verify initial state
    assert rm.co_occurrence_pos.shape == (num_primitives, num_primitives)
    assert rm.co_occurrence_neg.shape == (num_primitives, num_primitives)
    assert rm.conflict.shape == (num_primitives, num_primitives)
    assert rm.pair_count.shape == (num_primitives, num_primitives)

    assert rm.co_occurrence_pos.sum().item() == 0.0
    assert rm.co_occurrence_neg.sum().item() == 0.0
    assert rm.conflict.sum().item() == 0.0
    assert rm.pair_count.sum().item() == 0.0

    # 1. Update with positive gain
    chosen_pos = torch.tensor([5, 12, 18], dtype=torch.long)
    rm.update(chosen_pos, measured_gain=1.5, momentum=0.9)

    # Verify pairs (5,12), (12,5), (5,18), (18,5), (12,18), (18,12)
    expected_pairs = [(5, 12), (12, 5), (5, 18), (18, 5), (12, 18), (18, 12)]
    for i, j in expected_pairs:
        assert rm.pair_count[i, j].item() == 1.0
        assert rm.co_occurrence_pos[i, j].item() > 0.0
        assert rm.co_occurrence_neg[i, j].item() == 0.0
        assert rm.conflict[i, j].item() == 0.0

    # Diagonal must be zero
    for idx in range(num_primitives):
        assert rm.pair_count[idx, idx].item() == 0.0
        assert rm.co_occurrence_pos[idx, idx].item() == 0.0

    # 2. Update with negative gain
    chosen_neg = torch.tensor([12, 25], dtype=torch.long)
    rm.update(chosen_neg, measured_gain=-2.0, momentum=0.9)

    # Verify (12, 25) and (25, 12)
    neg_pairs = [(12, 25), (25, 12)]
    for i, j in neg_pairs:
        assert rm.pair_count[i, j].item() == 1.0
        assert rm.co_occurrence_pos[i, j].item() == 0.0
        assert rm.co_occurrence_neg[i, j].item() > 0.0
        assert rm.conflict[i, j].item() > 0.0

    # 3. Test single-item update (should do nothing since it's not a pair)
    rm.update(torch.tensor([8], dtype=torch.long), measured_gain=3.0)
    assert rm.pair_count[8, 8].item() == 0.0
    assert rm.co_occurrence_pos[8, 8].item() == 0.0

    # 4. Check metrics
    m = rm.metrics()
    print("Metrics:", m)
    assert m["relation_update_called"] == 2.0
    assert m["relation_update_pairs"] == 8.0 # 6 (from pos) + 2 (from neg)
    assert m["relation_update_positive"] == 1.0
    assert m["relation_update_negative"] == 1.0
    assert m["relation_pair_count_total"] == 8.0
    assert m["relation_observed_pairs"] == 8.0
    assert m["relation_co_pos_mean"] > 0.0
    assert m["relation_conflict_mean"] > 0.0

    # 5. Wire verification using PrimitiveMatrix5x5
    pm = PrimitiveMatrix5x5(
        embed_dim=32,
        num_layers=2,
        slots=4,
        enable_relation_memory=True,
        enable_vnext=True,
    )
    assert pm.relation_memory is not None, "relation_memory not created"

    # Call update_usage_credit with joint ids through credit advance simulation
    # Here we simulate record updates
    pm.relation_memory.update(torch.tensor([2, 3], dtype=torch.long), measured_gain=0.5, momentum=0.9)
    pm_metrics = pm.metrics()
    assert pm_metrics["relation_pair_count_total"] == 2.0
    assert pm_metrics["relation_co_pos_mean"] > 0.0

    print("RELATION_MEMORY_UPDATE_PASS")

if __name__ == "__main__":
    test_relation_memory_update()
