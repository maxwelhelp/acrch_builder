#!/usr/bin/env python3
import sys
from arch_builder.program_search import ProgramSearchState

def test_program_search_loop():
    print("Testing Program Search Loop state machine...")
    
    # Initialize Program Search State with short plateau/burst durations for testing
    patience = 2
    burst_duration = 3
    search = ProgramSearchState(
        patience=patience,
        burst_duration=burst_duration,
        burst_tau_mult=2.0,
        burst_random_mult=3.0,
        min_improvement=0.01,
    )
    
    # Verify initial modifiers
    mods = search.get_modifiers()
    print("Initial modifiers:", mods)
    assert mods["tau_multiplier"] == 1.0
    assert mods["random_k_multiplier"] == 1.0
    assert not mods["exploration_mode"]
    assert not mods["in_burst"]
    
    # 1. Update with initial score
    search.update(val_acc=0.50, primitive_top_share=0.5, choice_entropy=1.5, step=1)
    mods = search.get_modifiers()
    assert not mods["in_burst"], "Should not be in burst after first update"
    
    # 2. Update with no improvement (patience step 1)
    search.update(val_acc=0.50, primitive_top_share=0.5, choice_entropy=1.5, step=2)
    mods = search.get_modifiers()
    assert not mods["in_burst"]
    
    # 3. Update with no improvement (patience step 2 -> triggers burst!)
    search.update(val_acc=0.50, primitive_top_share=0.5, choice_entropy=1.5, step=3)
    mods = search.get_modifiers()
    print("Modifiers after triggering burst:", mods)
    assert mods["in_burst"], "Should be in burst after reaching plateau patience limit"
    assert mods["exploration_mode"], "Exploration mode must be active during burst"
    assert mods["tau_multiplier"] == 2.0, "Tau multiplier should match burst_tau_mult"
    assert mods["random_k_multiplier"] == 3.0, "Random K multiplier should match burst_random_mult"
    assert mods["burst_steps_remaining"] == 2, "Should have 2 steps remaining (duration 3 - 1 tick)"
    
    # 4. Burst steps remaining: should decrease with updates/steps
    # Step 1 of burst (will tick down remaining to 1)
    search.update(val_acc=0.50, primitive_top_share=0.5, choice_entropy=1.5, step=4)
    mods = search.get_modifiers()
    print("Burst step 1 mods:", mods)
    assert mods["in_burst"]
    assert mods["burst_steps_remaining"] == 1
    
    # Step 2 of burst (will tick down remaining to 0 and terminate burst)
    search.update(val_acc=0.50, primitive_top_share=0.5, choice_entropy=1.5, step=5)
    mods = search.get_modifiers()
    print("Mods after burst finished:", mods)
    assert not mods["in_burst"], "Burst should have ended"
    assert mods["tau_multiplier"] == 1.0, "Modifiers must return to baseline after burst"
    assert mods["random_k_multiplier"] == 1.0
    assert not mods["exploration_mode"]
    
    print("PROGRAM_SEARCH_LOOP_PASS")

if __name__ == "__main__":
    test_program_search_loop()
