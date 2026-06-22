#!/usr/bin/env python3
"""Diagnostic probe verifying the mathematical correctness of Shapley-lite joint credit synergy calculations and updates."""

import torch
from arch_builder.credit import BoundedCounterfactualCredit, CounterfactualTarget, CounterfactualRecord

def main() -> int:
    print("=== Running Shapley-Lite Joint Credit Probe ===")
    
    device = "cpu"
    slots = 4
    layers = 1
    num_primitives = 25
    
    torch.manual_seed(42)
    
    # 1. Initialize BoundedCounterfactualCredit
    credit = BoundedCounterfactualCredit(
        slots=slots,
        layers=layers,
        primitives=num_primitives,
        budget=8,
        ema_decay=0.9,  # ema = decay * old + (1 - decay) * share
    )
    
    # 2. Setup mock targets
    t1 = CounterfactualTarget(layer=0, cell=5, primitive=2, force=False)
    t2 = CounterfactualTarget(layer=0, cell=5, primitive=4, force=False)
    
    # We will simulate the behavior of measure_loss_interventions.
    # In measure_loss_interventions, we have:
    # 1. Singles interventions: [(t1,), (t2,)]
    # 2. Pairs interventions: [(t1, t2)]
    # Suppose:
    #   - full_loss.mean() = 1.0
    #   - counterfactual_loss(t1) = 1.2  -> gain_t1 = 1.2 - 1.0 = 0.2
    #   - counterfactual_loss(t2) = 1.3  -> gain_t2 = 1.3 - 1.0 = 0.3
    #   - counterfactual_loss(t1, t2) = 1.8  -> gain_joint = 1.8 - 1.0 = 0.8
    #
    # Then:
    #   - single_gain = {t1: 0.2, t2: 0.3}
    #   - joint_synergy = 0.8 - (0.2 + 0.3) = 0.3
    #
    # In the code, for len(targets) > 1:
    #   synergy = gain - sum(single_gain.get(target, 0.0) for target in targets)
    #   gain = synergy = 0.3
    
    single_gain_t1 = 0.2
    single_gain_t2 = 0.3
    joint_gain = 0.8
    synergy = joint_gain - (single_gain_t1 + single_gain_t2)
    
    print(f"Mocked gains: t1={single_gain_t1}, t2={single_gain_t2}, joint={joint_gain}")
    print(f"Calculated synergy: {synergy:.4f}")
    assert abs(synergy - 0.3) < 1e-5
    
    # 3. Populate credit.pending directly with these calculated gains
    credit.pending = [
        CounterfactualRecord(targets=(t1,), gain=single_gain_t1, measured_step=1),
        CounterfactualRecord(targets=(t2,), gain=single_gain_t2, measured_step=1),
        CounterfactualRecord(targets=(t1, t2), gain=synergy, measured_step=1),
    ]
    
    # 4. Advance ledger
    credit.advance(primitive_matrix=None)
    
    # 5. Verify updates
    # Let's check EMA values.
    # Initial EMA for all values was 0.0.
    # For t1:
    #   First record (t1,):
    #     share = 0.2 / 1 = 0.2
    #     ema_t1 = 0.9 * 0.0 + 0.1 * 0.2 = 0.02
    #   Second record (t1, t2):
    #     share = synergy / 2 = 0.3 / 2 = 0.15
    #     ema_t1 = 0.9 * 0.02 + 0.1 * 0.15 = 0.018 + 0.015 = 0.033
    #
    # For t2:
    #   First record (t2,):
    #     share = 0.3 / 1 = 0.3
    #     ema_t2 = 0.9 * 0.0 + 0.1 * 0.3 = 0.03
    #   Second record (t1, t2):
    #     share = synergy / 2 = 0.3 / 2 = 0.15
    #     ema_t2 = 0.9 * 0.03 + 0.1 * 0.15 = 0.027 + 0.015 = 0.042
    
    ema_t1_expected = 0.033
    ema_t2_expected = 0.042
    
    ema_t1_actual = float(credit.ema[t1.layer, t1.cell, t1.primitive])
    ema_t2_actual = float(credit.ema[t2.layer, t2.cell, t2.primitive])
    
    print(f"EMA t1: expected={ema_t1_expected:.4f}, actual={ema_t1_actual:.4f}")
    print(f"EMA t2: expected={ema_t2_expected:.4f}, actual={ema_t2_actual:.4f}")
    
    assert abs(ema_t1_actual - ema_t1_expected) < 1e-5, "t1 EMA update does not match expected value!"
    assert abs(ema_t2_actual - ema_t2_expected) < 1e-5, "t2 EMA update does not match expected value!"
    
    # 6. Verify ledger counters
    count_t1 = float(credit.count[t1.layer, t1.cell, t1.primitive])
    count_t2 = float(credit.count[t2.layer, t2.cell, t2.primitive])
    age_t1 = float(credit.age[t1.layer, t1.cell, t1.primitive])
    age_t2 = float(credit.age[t2.layer, t2.cell, t2.primitive])
    
    print(f"Counts: t1={count_t1}, t2={count_t2}")
    print(f"Ages: t1={age_t1}, t2={age_t2}")
    
    # Both targets are present in 2 records
    assert count_t1 == 2.0, f"Expected count_t1 to be 2.0, got {count_t1}"
    assert count_t2 == 2.0, f"Expected count_t2 to be 2.0, got {count_t2}"
    assert age_t1 == 0.0, f"Expected age_t1 to be reset to 0.0, got {age_t1}"
    assert age_t2 == 0.0, f"Expected age_t2 to be reset to 0.0, got {age_t2}"
    
    print("\n=== All Shapley-Lite Joint Credit Verification Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
