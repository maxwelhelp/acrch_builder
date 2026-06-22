#!/usr/bin/env python3
"""Training script for Copy and Reverse Memory Tasks utilizing ActionMatrixModel."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from arch_builder.model import ActionMatrixModel

class MemoryTaskDataset(Dataset):
    """Dataset generating synthetic examples for copy/reverse memory tasks."""
    def __init__(self, size: int = 1000, slots: int = 8, dim: int = 16, source_len: int = 3, task: str = "copy"):
        super().__init__()
        self.size = size
        self.slots = slots
        self.dim = dim
        self.source_len = source_len
        self.task = task
        
        # Pre-generate data for reproducibility
        torch.manual_seed(42)
        
        self.x = torch.zeros(size, slots, dim)
        self.x[:, 0:source_len] = torch.randn(size, source_len, dim)
        
        if task == "copy":
            self.y = self.x[:, 0:source_len].clone()
        else: # reverse
            self.y = self.x[:, 0:source_len].flip(dims=[1]).clone()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="Train ActionMatrixModel on memory copy/reverse tasks.")
    ap.add_argument("--task", default="copy", choices=["copy", "reverse"], help="Memory task type.")
    ap.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    ap.add_argument("--steps-per-epoch", type=int, default=20, help="Number of training steps per epoch.")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    ap.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    ap.add_argument("--dim", type=int, default=16, help="Hidden dimension size.")
    ap.add_argument("--layers", type=int, default=1, help="Number of layers.")
    ap.add_argument("--slots", type=int, default=8, help="Number of memory slots.")
    ap.add_argument("--source-len", type=int, default=3, help="Length of the source sequence in slots.")
    ap.add_argument("--target-start", type=int, default=4, help="Starting slot index of target output.")
    ap.add_argument("--device", default="cpu", help="Device to use (cpu/cuda).")
    ap.add_argument("--threshold", type=float, default=0.05, help="MSE threshold to consider a slot successfully matched.")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Configuring Memory Task: {args.task.upper()}")
    print(f"Slots structure: source [0:{args.source_len}], target [{args.target_start}:{args.target_start+args.source_len}]")
    
    # 1. Load data
    train_dataset = MemoryTaskDataset(
        size=1000, slots=args.slots, dim=args.dim, source_len=args.source_len, task=args.task
    )
    val_dataset = MemoryTaskDataset(
        size=256, slots=args.slots, dim=args.dim, source_len=args.source_len, task=args.task
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Build ActionMatrixModel
    model = ActionMatrixModel(
        dim=args.dim,
        slots=args.slots,
        layers=args.layers,
        classes=2, # dummy
        enable_vnext=True,
        enable_utility_critic_probe=True,
        utility_choice_warmup_steps=0,
        mmr_controller_warmup_steps=0,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 3. Training Loop
    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        correct_slots = 0
        total_slots = 0
        
        start_time = time.time()
        
        steps = 0
        for x, y in train_loader:
            if steps >= args.steps_per_epoch:
                break
                
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            _, info = model(x)
            slots_out = info["slots_out"]
            
            # Predict only on target slots
            tgt_out = slots_out[:, args.target_start : args.target_start + args.source_len]
            loss = F.mse_loss(tgt_out, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy metric: fraction of individual slots with MSE below threshold
            # Element-wise MSE shape: [B, source_len, dim]
            slot_mse = (tgt_out - y).pow(2).mean(dim=-1) # [B, source_len]
            correct_slots += (slot_mse < args.threshold).sum().item()
            total_slots += y.size(0) * args.source_len
            
            steps += 1
            
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / steps if steps > 0 else 0.0
        acc = correct_slots / total_slots if total_slots > 0 else 0.0
        
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.5f} - Slot Acc: {acc:.4f} ({correct_slots}/{total_slots}) - Time: {elapsed:.2f}s")
        
    # 4. Evaluation Loop
    print("\nStarting evaluation...")
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        eval_steps = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, info = model(x)
            slots_out = info["slots_out"]
            tgt_out = slots_out[:, args.target_start : args.target_start + args.source_len]
            loss = F.mse_loss(tgt_out, y)
            val_loss += loss.item()
            
            slot_mse = (tgt_out - y).pow(2).mean(dim=-1)
            val_correct += (slot_mse < args.threshold).sum().item()
            val_total += y.size(0) * args.source_len
            eval_steps += 1
            
    val_avg_loss = val_loss / eval_steps if eval_steps > 0 else 0.0
    val_acc = val_correct / val_total if val_total > 0 else 0.0
    print(f"Validation Loss: {val_avg_loss:.5f} - Slot Acc: {val_acc:.4f} ({val_correct}/{val_total})")
    
    print("\nMemory task training completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
