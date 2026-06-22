#!/usr/bin/env python3
"""Training script for CIFAR-10 image patch classification utilizing ActionMatrixModel."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

from arch_builder.model import ActionMatrixModel

class MockCIFAR10Dataset(Dataset):
    """Fallback Dataset generating random mock 32x32 images when offline or torchvision fails."""
    def __init__(self, size: int = 1000):
        super().__init__()
        self.size = size
        # Generate stable pseudo-random tensors
        torch.manual_seed(42)
        self.data = torch.randn(size, 3, 32, 32)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], int(self.targets[idx])


class CIFAR10PatchClassifier(nn.Module):
    """Classifier slicing 32x32 RGB images into 16 non-overlapping 8x8 patches,
    routing them through ActionMatrixModel, and classifying into 10 categories.
    """
    def __init__(self, dim: int = 64, slots: int = 16, layers: int = 1, classes: int = 10, enable_vnext: bool = True):
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.layers = layers
        self.classes = classes
        
        # Patch projection layer: 8*8*3 = 192 inputs
        self.patch_proj = nn.Linear(192, dim)
        
        # Backbone ActionMatrixModel
        self.backbone = ActionMatrixModel(
            dim=dim,
            slots=slots,
            layers=layers,
            classes=classes,
            enable_vnext=enable_vnext,
            enable_utility_critic_probe=True,
            utility_choice_warmup_steps=0,
            mmr_controller_warmup_steps=0,
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        
        # Slice into 16 patches of 8x8
        patches = x.unfold(2, 8, 8).unfold(3, 8, 8)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, 16, 192)
        
        # Project patches to model dimension
        slots_in = self.patch_proj(patches)
        
        # Route through ActionMatrixModel, which directly returns 10-class logits
        logits, choice_info = self.backbone(slots_in)
        return logits, choice_info


def get_cifar10_data(data_root: str, download: bool) -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 dataset, falling back to Mock dataset on download failure."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        print(f"Attempting to load CIFAR-10 from '{data_root}'...")
        train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=download, transform=transform)
        val_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=download, transform=transform)
        print("Successfully loaded official CIFAR-10 dataset.")
        return train_set, val_set
    except Exception as e:
        print(f"\n[Warning] Failed to load official CIFAR-10 dataset: {e}")
        print("Falling back to mock CIFAR-10 dataset generation...")
        return MockCIFAR10Dataset(size=1200), MockCIFAR10Dataset(size=300)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train ActionMatrixModel on image patches (CIFAR-10).")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--download", action="store_true", default=False)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps-per-epoch", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--disable-vnext", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. Load data
    train_set, val_set = get_cifar10_data(args.data_root, args.download)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    # 2. Build model
    model = CIFAR10PatchClassifier(
        dim=args.dim,
        slots=16,
        layers=args.layers,
        classes=10,
        enable_vnext=not args.disable_vnext
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 3. Training Loop
    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        steps = 0
        for x, y in train_loader:
            if steps >= args.steps_per_epoch:
                break
                
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, preds = logits.max(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            steps += 1
            
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / steps if steps > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f} ({correct}/{total}) - Time: {elapsed:.2f}s")
        
    # 4. Evaluation Loop
    print("\nStarting evaluation...")
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        eval_steps = 0
        for x, y in val_loader:
            if eval_steps >= 5: # Limit evaluation steps for smoke testing
                break
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            _, preds = logits.max(dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)
            eval_steps += 1
            
    val_acc = val_correct / val_total if val_total > 0 else 0.0
    print(f"Validation Acc: {val_acc:.4f} ({val_correct}/{val_total})")
    
    print("\nCIFAR-10 training completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
