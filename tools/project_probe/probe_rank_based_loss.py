#!/usr/bin/env python3
"""Diagnostic probe verifying the correctness, edge cases, and gradients of pairwise_ranking_loss."""

import torch
from arch_builder.credit import pairwise_ranking_loss

def main() -> int:
    print("=== Running Rank-Based Critic Loss Probe ===")
    
    device = "cpu"
    torch.manual_seed(42)
    
    # 1. Test 1D Case
    print("\nTesting 1D pairwise_ranking_loss...")
    
    # Perfect order: pred_utility matches target order
    targets_1d = torch.tensor([1.0, 2.0, 3.0], device=device)
    preds_perfect = torch.tensor([1.5, 2.5, 3.5], device=device)
    loss_perfect = pairwise_ranking_loss(preds_perfect, targets_1d)
    print("Perfect order loss:", loss_perfect.item())
    assert abs(loss_perfect.item()) < 1e-5, f"Expected perfect loss to be 0, got {loss_perfect.item()}"
    
    # Inverse order: pred_utility is in opposite direction
    preds_inverse = torch.tensor([3.5, 2.5, 1.5], requires_grad=True, device=device)
    loss_inverse = pairwise_ranking_loss(preds_inverse, targets_1d)
    print("Inverse order loss:", loss_inverse.item())
    assert loss_inverse.item() > 0.0, "Expected positive loss for inverse order!"
    
    # Backward pass and gradient direction check
    loss_inverse.backward()
    grads = preds_inverse.grad
    print("Inverse order grads:", grads.cpu().numpy())
    
    # We want utility values to align with targets (which are [1.0, 2.0, 3.0])
    # The gradient of loss w.r.t preds represents how the loss changes when preds increase.
    # To decrease loss, we should step in the direction of negative gradient.
    # Therefore, negative gradient (-grad) should be positive for higher target elements
    # and negative for lower target elements.
    # Let's verify that -grad is ascending.
    neg_grads = -grads
    assert neg_grads[0] < neg_grads[1] < neg_grads[2], \
        f"Gradients do not push predictions in the right direction! -grads: {neg_grads.cpu().numpy()}"
    print("1D gradient direction check: PASS")
    
    # 2. Test 2D Case
    print("\nTesting 2D pairwise_ranking_loss...")
    
    # Batch size 2, P=3
    targets_2d = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 1.0, 2.0]
    ], device=device)
    
    # Perfect predictions for both batch elements
    preds_2d_perfect = torch.tensor([
        [1.5, 2.5, 3.5],
        [3.5, 1.5, 2.5]
    ], device=device)
    
    loss_2d_perfect = pairwise_ranking_loss(preds_2d_perfect, targets_2d)
    print("2D perfect loss:", loss_2d_perfect.item())
    assert abs(loss_2d_perfect.item()) < 1e-5, f"Expected 2D perfect loss to be 0, got {loss_2d_perfect.item()}"
    
    # Misaligned predictions
    preds_2d_mixed = torch.tensor([
        [3.0, 2.0, 1.0],  # Inverse
        [3.5, 1.5, 2.5]   # Perfect
    ], requires_grad=True, device=device)
    
    loss_2d_mixed = pairwise_ranking_loss(preds_2d_mixed, targets_2d)
    print("2D mixed loss:", loss_2d_mixed.item())
    assert loss_2d_mixed.item() > 0.0
    
    loss_2d_mixed.backward()
    print("2D mixed grads:\n", preds_2d_mixed.grad.cpu().numpy())
    
    # Batch element 1 (perfect) should have zero gradients
    assert (preds_2d_mixed.grad[1] == 0.0).all(), "Perfect batch element should have 0 gradients!"
    # Batch element 0 (inverse) should have non-zero gradients in the correct direction
    neg_grads_0 = -preds_2d_mixed.grad[0]
    assert neg_grads_0[0] < neg_grads_0[1] < neg_grads_0[2], \
        f"Incorrect gradient direction on batch element 0! -grads: {neg_grads_0.cpu().numpy()}"
    print("2D mixed gradient direction check: PASS")
    
    # 3. Test Edge Cases
    print("\nTesting Edge Cases...")
    
    # 3.1 All equal targets (no pairs possible)
    equal_targets = torch.tensor([2.0, 2.0, 2.0], device=device)
    equal_loss = pairwise_ranking_loss(preds_perfect, equal_targets)
    assert equal_loss.item() == 0.0, f"Expected 0 loss for all-equal targets, got {equal_loss.item()}"
    
    # 3.2 List/Sequence targets input
    list_targets = [1.0, 2.0, 3.0]
    list_loss = pairwise_ranking_loss(preds_perfect, list_targets)
    assert abs(list_loss.item()) < 1e-5
    
    # 3.3 Tiny sequence length (len < 2)
    tiny_targets = torch.tensor([1.0], device=device)
    tiny_preds = torch.tensor([1.5], device=device)
    tiny_loss = pairwise_ranking_loss(tiny_preds, tiny_targets)
    assert tiny_loss.item() == 0.0, f"Expected 0 loss for length < 2, got {tiny_loss.item()}"
    
    # 3.4 2D with one item in batch having equal targets
    mixed_targets_2d = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 2.0]  # All equal
    ], device=device)
    mixed_preds_2d = torch.tensor([
        [3.0, 2.0, 1.0],
        [1.5, 2.5, 3.5]
    ], device=device)
    
    mixed_loss_2d = pairwise_ranking_loss(mixed_preds_2d, mixed_targets_2d)
    assert mixed_loss_2d.item() > 0.0, "Expected positive loss for mixed batch"
    assert torch.isfinite(mixed_loss_2d), "Loss should be finite"
    
    print("Edge cases checks: PASS")
    print("\n=== All Rank-Based Loss Verification Tests Passed Successfully! ===")
    return 0

if __name__ == "__main__":
    exit(main())
