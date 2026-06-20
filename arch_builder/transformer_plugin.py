from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .credit import CreditBuffer


PlacementMode = Literal["after", "before", "attention_only"]
MechanismMode = Literal["learned", "identity", "random", "frozen"]


@dataclass
class SequenceBatch:
    x: torch.Tensor
    y: torch.Tensor


class SyntheticMotifTask:
    """Synthetic sequence task that rewards local pattern extraction."""

    def __init__(
        self,
        seq_len: int = 48,
        vocab_size: int = 64,
        motif_len: int = 3,
        site_count: int = 4,
    ) -> None:
        if seq_len % site_count != 0:
            raise ValueError("seq_len must be divisible by site_count for the parity task")
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.motif_len = motif_len
        self.site_count = site_count
        self.site_stride = seq_len // site_count

    def _insert_motif(self, tokens: torch.Tensor, bits: torch.Tensor) -> None:
        batch = tokens.shape[0]
        for i in range(batch):
            for site in range(self.site_count):
                base = int(torch.randint(2, self.vocab_size - self.motif_len - 2, (1,), device=tokens.device).item())
                region_start = site * self.site_stride
                offset = int(torch.randint(0, self.site_stride - self.motif_len + 1, (1,), device=tokens.device).item())
                start = region_start + offset
                if int(bits[i, site].item()) == 1:
                    motif = torch.arange(base, base + self.motif_len, device=tokens.device)
                else:
                    motif = torch.tensor([base, base + 2, base + 1], device=tokens.device)
                tokens[i, start : start + self.motif_len] = motif
                distract_offset = (offset + 2) % max(1, self.site_stride - self.motif_len + 1)
                distract_start = region_start + distract_offset
                distract = torch.tensor([base + 1, base, base + 2], device=tokens.device)
                tokens[i, distract_start : distract_start + self.motif_len] = distract

    def sample(self, batch_size: int, device: str | torch.device) -> SequenceBatch:
        tokens = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=device)
        bits = torch.randint(0, 2, (batch_size, self.site_count), device=device)
        label = (bits.sum(dim=-1) % 2).long()
        self._insert_motif(tokens, bits)
        return SequenceBatch(x=tokens, y=label)


class LocalMechanism(nn.Module):
    def __init__(self, dim: int, mode: MechanismMode = "learned", kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.dim = dim
        self.mode = mode
        self.kernel_size = kernel_size
        self.proj1 = nn.Linear(dim * 3, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.75))
        if mode in {"identity"}:
            self.freeze()
        elif mode == "frozen":
            self.freeze()
        elif mode == "random":
            self.freeze()
        elif mode == "learned":
            pass
        else:
            raise ValueError(f"unknown mechanism mode: {mode!r}")

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.mode == "identity":
            delta = torch.zeros_like(x)
        else:
            left = torch.cat([x[:, :1], x[:, :-1]], dim=1)
            right = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
            y = torch.cat([left, x, right], dim=-1)
            y = self.proj2(F.relu(self.proj1(y)))
            y = self.norm(y)
            scale = torch.sigmoid(self.scale) * 1.95 + 0.05
            delta = F.relu(y) * scale
            if self.mode == "random":
                noise = 0.15 * torch.randn_like(delta)
                delta = delta + noise
        stats = {
            "mechanism_norm": float(delta.detach().norm(dim=-1).mean().cpu()),
            "mechanism_abs_mean": float(delta.detach().abs().mean().cpu()),
        }
        return delta, stats


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLayerWithMechanism(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        placement: PlacementMode = "after",
        mechanism_mode: MechanismMode = "learned",
    ) -> None:
        super().__init__()
        if placement not in {"after", "before", "attention_only"}:
            raise ValueError(f"unknown placement: {placement!r}")
        self.placement = placement
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.ff = FeedForward(dim)
        self.mechanism = None if placement == "attention_only" else LocalMechanism(dim, mode=mechanism_mode)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        trace: Dict[str, float] = {}
        if self.placement == "before":
            mech_out, mech_stats = self.mechanism(self.norm1(x)) if self.mechanism is not None else (torch.zeros_like(x), {})
            x = x + mech_out
            attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask, need_weights=False)
            x = x + attn_out
        else:
            attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask, need_weights=False)
            x = x + attn_out
            mech_out, mech_stats = self.mechanism(self.norm1(x)) if self.mechanism is not None else (torch.zeros_like(x), {})
            x = x + mech_out
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        trace["attn_norm"] = float(attn_out.detach().norm(dim=-1).mean().cpu())
        trace["ff_norm"] = float(ff_out.detach().norm(dim=-1).mean().cpu())
        trace["output_norm"] = float(x.detach().norm(dim=-1).mean().cpu())
        trace["mechanism_norm"] = float(mech_out.detach().norm(dim=-1).mean().cpu()) if self.mechanism is not None else 0.0
        trace["mechanism_abs_mean"] = float(mech_out.detach().abs().mean().cpu()) if self.mechanism is not None else 0.0
        trace.update(mech_stats if self.mechanism is not None else {})
        return x, trace


class ReferenceTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        dim: int = 48,
        num_heads: int = 4,
        placement: PlacementMode = "after",
        mechanism_mode: MechanismMode = "learned",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.placement = placement
        self.mechanism_mode = mechanism_mode
        self.embed = nn.Embedding(vocab_size, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.layer = TransformerLayerWithMechanism(dim, num_heads=num_heads, placement=placement, mechanism_mode=mechanism_mode)
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if tokens.ndim != 2:
            raise ValueError(f"expected tokens shaped [batch, seq], got {tuple(tokens.shape)}")
        x = self.embed(tokens)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x, trace = self.layer(x)
        pooled = self.final_norm(x[:, 0])
        logits = self.head(pooled)
        trace["pooled_norm"] = float(pooled.detach().norm(dim=-1).mean().cpu())
        trace["logit_norm"] = float(logits.detach().norm(dim=-1).mean().cpu())
        return logits, trace

    def report(self, seq_len: int | None = None) -> Dict[str, float]:
        seq = seq_len or self.seq_len
        params = float(sum(p.numel() for p in self.parameters()))
        dim = float(self.dim)
        attn_cost = float((seq + 1) * (self.dim * self.dim * 4 + self.dim * self.dim * 2))
        mech_cost = 0.0 if self.placement == "attention_only" else float((seq + 1) * self.dim * self.layer.mechanism.kernel_size * 2)
        ff_cost = float((seq + 1) * self.dim * self.dim * 4)
        activation = float((seq + 1) * self.dim * 8)
        return {
            "params": params,
            "flops": attn_cost + mech_cost + ff_cost,
            "activation_bytes": activation,
        }


def set_trainable_mode(model: ReferenceTransformerClassifier, mechanism_mode: MechanismMode) -> None:
    mech = model.layer.mechanism
    if mech is None:
        return
    for p in mech.parameters():
        p.requires_grad = mechanism_mode == "learned"
    if mechanism_mode == "identity":
        mech.mode = "identity"
    elif mechanism_mode == "random":
        mech.mode = "random"
    elif mechanism_mode == "frozen":
        mech.mode = "frozen"
    elif mechanism_mode == "learned":
        mech.mode = "learned"
    else:
        raise ValueError(mechanism_mode)


@torch.no_grad()
def evaluate(model: ReferenceTransformerClassifier, task: SyntheticMotifTask, steps: int, batch_size: int, device: str) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    trace_sum: Dict[str, float] = {}
    for _ in range(max(1, steps)):
        batch = task.sample(batch_size, device)
        logits, trace = model(batch.x)
        loss = F.cross_entropy(logits, batch.y)
        pred = logits.argmax(dim=-1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch.y.shape[0]
        for key, value in trace.items():
            trace_sum[key] = trace_sum.get(key, 0.0) + float(value)
    out = {
        "acc": correct / max(1, total),
        "loss": loss_sum / max(1, total),
    }
    for key, value in trace_sum.items():
        out[key] = value / max(1, steps)
    return out


def train_variant(
    *,
    seed: int,
    placement: PlacementMode,
    mechanism_mode: MechanismMode,
    seq_len: int = 32,
    vocab_size: int = 64,
    dim: int = 48,
    num_heads: int = 4,
    epochs: int = 4,
    steps_per_epoch: int = 24,
    batch_size: int = 64,
    eval_steps: int = 8,
    eval_batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> Dict[str, object]:
    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    task = SyntheticMotifTask(seq_len=seq_len, vocab_size=vocab_size)
    model = ReferenceTransformerClassifier(
        vocab_size=vocab_size,
        seq_len=seq_len,
        dim=dim,
        num_heads=num_heads,
        placement=placement,
        mechanism_mode=mechanism_mode,
    ).to(device)
    set_trainable_mode(model, mechanism_mode)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    train_acc = 0.0
    train_loss = 0.0
    best_val = 0.0
    grad_norm = 0.0
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for _ in range(max(1, steps_per_epoch)):
            batch = task.sample(batch_size, device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(batch.x)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            if mechanism_mode == "learned" and model.layer.mechanism is not None:
                total_grad = 0.0
                for p in model.layer.mechanism.parameters():
                    if p.grad is not None:
                        total_grad += float(p.grad.detach().norm().cpu())
                grad_norm = total_grad
            opt.step()
            pred = logits.argmax(dim=-1)
            total += batch.y.shape[0]
            correct += (pred == batch.y).sum().item()
            loss_sum += float(loss.detach().cpu()) * batch.y.shape[0]
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)
        val = evaluate(model, task, eval_steps, eval_batch_size, device)
        best_val = max(best_val, val["acc"])

    val = evaluate(model, task, eval_steps, eval_batch_size, device)
    elapsed = time.perf_counter() - start
    report = {
        "seed": seed,
        "placement": placement,
        "mechanism_mode": mechanism_mode,
        "task": "synthetic_motif_order",
        "train_acc": float(train_acc),
        "train_loss": float(train_loss),
        "best_val_acc": float(best_val),
        "val_acc": float(val["acc"]),
        "val_loss": float(val["loss"]),
        "output_norm": float(val.get("output_norm", 0.0)),
        "pooled_norm": float(val.get("pooled_norm", 0.0)),
        "logit_norm": float(val.get("logit_norm", 0.0)),
        "mechanism_norm": float(val.get("mechanism_norm", 0.0)),
        "mechanism_abs_mean": float(val.get("mechanism_abs_mean", 0.0)),
        "attn_norm": float(val.get("attn_norm", 0.0)),
        "ff_norm": float(val.get("ff_norm", 0.0)),
        "grad_norm": float(grad_norm),
        "seconds": float(elapsed),
        "model": model,
        "task_obj": task,
    }
    report.update(model.report(seq_len=seq_len))
    return report


def collect_credit(rows: Iterable[Dict[str, object]]) -> Dict[str, float]:
    credit = CreditBuffer()
    for row in rows:
        credit.update(
            {
                f"{row['placement']}_{row['mechanism_mode']}": float(row["val_acc"]),
            }
        )
    return credit.metrics()
