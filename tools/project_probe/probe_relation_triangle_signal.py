#!/usr/bin/env python3
import torch
import sys
from arch_builder.primitive_matrix import PrimitiveMatrix5x5
from arch_builder.relation_consistency import TriangleRelationScorer

def test_relation_triangle_signal():
    print("Testing relation triangle signal scorer...")

    num_layers = 1
    slots = 4
    pm = PrimitiveMatrix5x5(
        embed_dim=32,
        num_layers=num_layers,
        slots=slots,
        enable_relation_memory=True,
        enable_vnext=True,
    )

    # Setup simulated feedback in PrimitiveMatrix:
    # Cell 0 has positive gain on primitive 5
    pm.feedback_gain_ema[0, 0, 5] = 1.0
    # Cell 0 has regret on primitive 10
    pm.feedback_regret_ema[0, 0, 10] = 1.0

    # Setup simulated co-occurrences in relation_memory:
    # Primitive 5 works well with primitive 12 (positive co-occurrence)
    pm.relation_memory.co_occurrence_pos[5, 12] = 0.8
    pm.relation_memory.pair_count[5, 12] = 1.0

    # Primitive 5 conflicts with primitive 25 (conflict)
    pm.relation_memory.conflict[5, 25] = 0.9
    pm.relation_memory.pair_count[5, 25] = 1.0

    # Primitive 10 co-occurs negatively with primitive 30
    pm.relation_memory.co_occurrence_neg[10, 30] = 0.5
    pm.relation_memory.pair_count[10, 30] = 1.0

    # Create candidates:
    # Candidate pool: 12 (good partner), 25 (conflicting partner), 30 (conflict from regretful prim), 15 (neutral)
    candidate_ids = torch.tensor([[12, 25, 30, 15]], dtype=torch.long)
    cell_ids = torch.tensor([0], dtype=torch.long)

    scorer = TriangleRelationScorer()
    res = scorer.compute(
        candidate_ids=candidate_ids,
        layer_idx=0,
        cell_ids=cell_ids,
        pm=pm,
        program_entropy_norm=0.5,
    )

    print("Scorer results:")
    print("Support:", res["support"].tolist())
    print("Contradiction:", res["contradiction"].tolist())
    print("Novelty:", res["novelty"].tolist())
    print("Score:", res["score"].tolist())

    support = res["support"][0]
    contradiction = res["contradiction"][0]
    score = res["score"][0]

    # Verify support: primitive 12 should have higher support than others
    assert support[0] > support[1], "Primitive 12 should have higher support than 25"
    assert support[0] > support[3], "Primitive 12 should have higher support than neutral 15"

    # Verify contradiction: primitive 25 and 30 should have higher contradiction than others
    assert contradiction[1] > contradiction[0], "Primitive 25 should have higher contradiction than 12"
    assert contradiction[2] > contradiction[3], "Primitive 30 should have higher contradiction than neutral 15"

    # Verify final score: primitive 12 should have the highest score
    scores = score.tolist()
    assert scores[0] == max(scores), f"Primitive 12 should have highest score, got {scores}"
    assert scores[1] == min(scores), f"Primitive 25 should have lowest score, got {scores}"

    # Verify with program_entropy gating
    res_high_entropy = scorer.compute(
        candidate_ids=candidate_ids,
        layer_idx=0,
        cell_ids=cell_ids,
        pm=pm,
        program_entropy_norm=0.9, # very high entropy -> novelty gate should be smaller
    )
    res_low_entropy = scorer.compute(
        candidate_ids=candidate_ids,
        layer_idx=0,
        cell_ids=cell_ids,
        pm=pm,
        program_entropy_norm=0.1, # low entropy -> novelty gate should be larger
    )
    
    # Primitives have 0 feedback count, so novelty is high.
    # Low entropy (0.1) should boost novelty more than high entropy (0.9)
    score_low = res_low_entropy["score"][0, 3].item() # neutral prim 15
    score_high = res_high_entropy["score"][0, 3].item()
    print(f"Neutral prim score low entropy: {score_low}, high entropy: {score_high}")
    assert score_low > score_high, "Low entropy should increase novelty score boost"

    print("RELATION_TRIANGLE_SIGNAL_PASS")

if __name__ == "__main__":
    test_relation_triangle_signal()
