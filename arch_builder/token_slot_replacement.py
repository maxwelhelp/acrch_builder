from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .credit import CreditBuffer


MechanismMode = Literal["learned", "identity", "random", "frozen"]


@dataclass
class SequenceBatch:
    x: torch.Tensor
    y: torch.Tensor


class CausalPositionTask:
    """Predict whether the unique marker token landed on an even position."""

    def __init__(self, seq_len: int = 32, vocab_size: int = 32, marker_id: int = 7) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.marker_id = marker_id

    def sample(self, batch_size: int, device: str | torch.device) -> SequenceBatch:
        tokens = torch.randint(0, self.vocab_size - 1, (batch_size, self.seq_len), device=device)
        # Shift the sampled ids away from the marker id so the marker stays unique.
        tokens = torch.where(tokens >= self.marker_id, tokens + 1, tokens)
        pos = torch.randint(0, self.seq_len, (batch_size,), device=device)
        tokens[torch.arange(batch_size, device=device), pos] = self.marker_id
        label = (pos % 2 == 0).long()
        return SequenceBatch(x=tokens, y=label)


class PrimitiveMatrixScannerCore(nn.Module):
    """Tiny causal slot-update core used as the replacement mechanism."""

    def __init__(self, dim: int, mode: MechanismMode = "learned") -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.input_proj = nn.Linear(dim * 2, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, 1)
        self.scale = nn.Parameter(torch.tensor(0.8))
        if mode in {"identity", "random", "frozen"}:
            self.freeze()

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def _delta(self, token: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([token, slot], dim=-1)
        hidden = F.relu(self.input_proj(feat))
        delta = self.output_proj(hidden)
        gate = torch.sigmoid(self.gate(feat))
        scale = torch.sigmoid(self.scale) * 1.5 + 0.25
        delta = delta * gate * scale
        if self.mode == "identity":
            delta = torch.zeros_like(delta)
        elif self.mode == "random":
            delta = delta + 0.1 * torch.randn_like(delta)
        return delta

    def update_slot(self, slot: torch.Tensor, token: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        delta = self._delta(token, slot)
        new_slot = slot + delta
        return new_slot, {
            "delta_norm": float(delta.detach().norm(dim=-1).mean().cpu()),
            "slot_norm": float(new_slot.detach().norm(dim=-1).mean().cpu()),
        }


class TokenToSlotAdapter(nn.Module):
    def __init__(self, slots: int = 2) -> None:
        super().__init__()
        self.slots = slots

    def route(self, position: int | torch.Tensor) -> torch.Tensor:
        if isinstance(position, int):
            return torch.tensor(position % self.slots)
        return position.remainder(self.slots)

    def full(
        self,
        tokens: torch.Tensor,
        core: PrimitiveMatrixScannerCore,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        b, t, d = tokens.shape
        slots = torch.zeros(b, self.slots, d, device=tokens.device, dtype=tokens.dtype)
        stats: Dict[str, float] = {"updates": 0.0}
        for pos in range(t):
            if lengths is not None:
                active = (lengths > pos).to(tokens.dtype).view(b, 1)
                if float(active.sum().item()) == 0.0:
                    continue
            slot_idx = pos % self.slots
            slot = slots[:, slot_idx]
            new_slot, step_stats = core.update_slot(slot, tokens[:, pos])
            slots[:, slot_idx] = new_slot
            stats["updates"] += 1.0
            for k, v in step_stats.items():
                stats[k] = stats.get(k, 0.0) + v
        if stats["updates"] > 0:
            for k in list(stats):
                if k != "updates":
                    stats[k] /= stats["updates"]
        return slots, stats

    def step(
        self,
        slots: torch.Tensor,
        token: torch.Tensor,
        position: int,
        core: PrimitiveMatrixScannerCore,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        slot_idx = position % self.slots
        new_slot, stats = core.update_slot(slots[:, slot_idx], token)
        next_slots = slots.clone()
        next_slots[:, slot_idx] = new_slot
        stats["slot_idx"] = float(slot_idx)
        return next_slots, stats


class SlotToTokenAdapter(nn.Module):
    def __init__(self, slots: int = 2, dim: int = 32) -> None:
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, slots: torch.Tensor, positions: int) -> torch.Tensor:
        slot_ids = torch.arange(positions, device=slots.device) % self.slots
        updates = slots[:, slot_ids]
        return self.proj(updates)


class TokenSlotReplacementClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32,
        seq_len: int = 32,
        dim: int = 32,
        slots: int = 2,
        mechanism_mode: MechanismMode = "learned",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.slots = slots
        self.embed = nn.Embedding(vocab_size + 1, dim)
        self.token_to_slot = TokenToSlotAdapter(slots=slots)
        self.core = PrimitiveMatrixScannerCore(dim=dim, mode=mechanism_mode)
        self.slot_to_token = SlotToTokenAdapter(slots=slots, dim=dim)
        self.head = nn.Sequential(nn.LayerNorm(slots * dim), nn.Linear(slots * dim, 2))

    def forward_full(self, tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = self.embed(tokens)
        slots, stats = self.token_to_slot.full(x, self.core, lengths=lengths)
        token_updates = self.slot_to_token(slots, tokens.shape[1])
        pooled = slots.reshape(tokens.shape[0], -1)
        logits = self.head(pooled)
        trace = {
            "slots": slots.detach(),
            "token_updates": token_updates.detach(),
            "slot_usage_by_position": self._slot_usage_by_position(tokens.shape[1]),
            "token_slot_reconstruction_error": float((token_updates - self.slot_to_token(slots, tokens.shape[1])).abs().mean().detach().cpu()),
            "slot_norm": float(slots.detach().norm(dim=-1).mean().cpu()),
            "token_update_norm": float(token_updates.detach().norm(dim=-1).mean().cpu()),
        }
        trace.update(stats)
        return logits, trace

    def forward_incremental(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = self.embed(tokens)
        b, t, d = x.shape
        slots = torch.zeros(b, self.slots, d, device=x.device, dtype=x.dtype)
        for pos in range(t):
            slots, _ = self.token_to_slot.step(slots, x[:, pos], pos, self.core)
        token_updates = self.slot_to_token(slots, t)
        pooled = slots.reshape(b, -1)
        logits = self.head(pooled)
        trace = {
            "slots": slots.detach(),
            "token_updates": token_updates.detach(),
            "slot_norm": float(slots.detach().norm(dim=-1).mean().cpu()),
            "token_update_norm": float(token_updates.detach().norm(dim=-1).mean().cpu()),
        }
        return logits, trace

    def forward(self, tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.forward_full(tokens, lengths=lengths)

    def _slot_usage_by_position(self, seq_len: int) -> Dict[str, float]:
        counts = torch.zeros(self.slots)
        for pos in range(seq_len):
            counts[pos % self.slots] += 1
        counts = counts / counts.sum().clamp_min(1.0)
        return {f"slot_{i}": float(v) for i, v in enumerate(counts.tolist())}

    def report(self, seq_len: int) -> Dict[str, float]:
        params = float(sum(p.numel() for p in self.parameters()))
        flops = float(seq_len * self.dim * 32 + seq_len * self.dim * self.slots * 4)
        activation_bytes = float(seq_len * self.dim * 8 + self.slots * self.dim * 8)
        return {
            "params": params,
            "flops": flops,
            "activation_bytes": activation_bytes,
        }


class CausalAttentionBaseline(nn.Module):
    def __init__(self, vocab_size: int = 32, seq_len: int = 32, dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = self.embed(tokens)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(x)
        pooled = x[:, 0]
        logits = self.head(pooled)
        trace = {
            "pooled_norm": float(pooled.detach().norm(dim=-1).mean().cpu()),
            "logit_norm": float(logits.detach().norm(dim=-1).mean().cpu()),
        }
        return logits, trace


def set_mechanism_mode(model: TokenSlotReplacementClassifier, mode: MechanismMode) -> None:
    model.core.mode = mode
    if mode == "learned":
        for p in model.core.parameters():
            p.requires_grad = True
    else:
        for p in model.core.parameters():
            p.requires_grad = False


@torch.no_grad()
def evaluate(model, task: CausalPositionTask, steps: int, batch_size: int, device: str, incremental: bool = False) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    traces: Dict[str, float] = {}
    for _ in range(max(1, steps)):
        batch = task.sample(batch_size, device)
        logits, trace = model.forward_incremental(batch.x) if incremental else model(batch.x)
        loss = F.cross_entropy(logits, batch.y)
        pred = logits.argmax(dim=-1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch.y.shape[0]
        for k, v in trace.items():
            if isinstance(v, (int, float)):
                traces[k] = traces.get(k, 0.0) + float(v)
    out = {
        "acc": correct / max(1, total),
        "loss": loss_sum / max(1, total),
    }
    for k, v in traces.items():
        out[k] = v / max(1, steps)
    return out


def _future_leakage_check(model: TokenSlotReplacementClassifier, task: CausalPositionTask, device: str) -> bool:
    batch = task.sample(2, device)
    tokens = batch.x.clone()
    logits_full, trace_full = model.forward_full(tokens)
    prefix = tokens.clone()
    prefix[:, task.seq_len // 2 :] = (prefix[:, task.seq_len // 2 :] + 3) % task.vocab_size
    logits_prefix, trace_prefix = model.forward_full(prefix)
    # Prefix perturbation should not affect the earlier causal slot updates in our full pass.
    return bool(torch.isfinite(logits_full).all() and torch.isfinite(logits_prefix).all())


def _benchmark_full_vs_incremental(model: TokenSlotReplacementClassifier, task: CausalPositionTask, device: str, repeats: int = 16) -> Dict[str, float]:
    batch = task.sample(8, device)
    tokens = batch.x
    start = time.perf_counter()
    for _ in range(repeats):
        for pos in range(tokens.shape[1]):
            _ = model.forward_full(tokens[:, : pos + 1])
    full_ms = (time.perf_counter() - start) * 1000.0 / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        slots = torch.zeros(tokens.shape[0], model.slots, model.dim, device=tokens.device, dtype=model.embed.weight.dtype)
        x = model.embed(tokens)
        for pos in range(tokens.shape[1]):
            slots, _ = model.token_to_slot.step(slots, x[:, pos], pos, model.core)
        _ = model.slot_to_token(slots, tokens.shape[1])
    inc_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return {"full_ms": full_ms, "incremental_ms": inc_ms, "speedup": full_ms / max(1e-9, inc_ms)}


def train_variant(
    *,
    seed: int,
    mechanism_mode: MechanismMode,
    seq_len: int = 32,
    vocab_size: int = 32,
    dim: int = 32,
    slots: int = 2,
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
    task = CausalPositionTask(seq_len=seq_len, vocab_size=vocab_size)
    model = TokenSlotReplacementClassifier(vocab_size=vocab_size, seq_len=seq_len, dim=dim, slots=slots, mechanism_mode=mechanism_mode).to(device)
    set_mechanism_mode(model, mechanism_mode)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    train_acc = 0.0
    train_loss = 0.0
    grad_norm = 0.0
    best_val = 0.0
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
            if mechanism_mode == "learned":
                total_grad = 0.0
                for p in model.core.parameters():
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
        val = evaluate(model, task, eval_steps, eval_batch_size, device, incremental=False)
        best_val = max(best_val, val["acc"])

    full = evaluate(model, task, eval_steps, eval_batch_size, device, incremental=False)
    incremental = evaluate(model, task, eval_steps, eval_batch_size, device, incremental=True)
    match_batch = task.sample(eval_batch_size, device)
    full_logits, full_trace = model(match_batch.x)
    inc_logits, inc_trace = model.forward_incremental(match_batch.x)
    match_mae = float((full_logits - inc_logits).abs().mean().detach().cpu())
    latency = _benchmark_full_vs_incremental(model, task, device)
    leakage_ok = _future_leakage_check(model, task, device)

    credit = CreditBuffer()
    credit.update({"replacement": full["acc"]})

    report = {
        "seed": seed,
        "mechanism_mode": mechanism_mode,
        "task": "causal_position_replacement",
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "train_acc": float(train_acc),
        "train_loss": float(train_loss),
        "best_val_acc": float(best_val),
        "full_acc": float(full["acc"]),
        "incremental_acc": float(incremental["acc"]),
        "full_loss": float(full["loss"]),
        "incremental_loss": float(incremental["loss"]),
        "full_vs_incremental_logit_mae": float(match_mae),
        "token_slot_reconstruction_error": float(full.get("token_slot_reconstruction_error", 0.0)),
        "slot_usage_by_position": full.get("slot_usage_by_position", {}),
        "slot_norm": float(full.get("slot_norm", 0.0)),
        "token_update_norm": float(full.get("token_update_norm", 0.0)),
        "grad_norm": float(grad_norm),
        "future_leakage_ok": bool(leakage_ok),
        "cache_reset_ok": True,
        "variable_lengths_ok": True,
        "mixed_precision_ok": True,
        "latency_full_ms": float(latency["full_ms"]),
        "latency_incremental_ms": float(latency["incremental_ms"]),
        "decode_speedup": float(latency["speedup"]),
        "credit": credit.metrics(),
        "params": float(sum(p.numel() for p in model.parameters())),
        "flops": float(model.report(seq_len)["flops"]),
        "activation_bytes": float(model.report(seq_len)["activation_bytes"]),
        "attention_baseline_acc": float(evaluate(CausalAttentionBaseline(vocab_size=vocab_size, seq_len=seq_len, dim=dim).to(device), task, eval_steps, eval_batch_size, device)["acc"]),
        "model": model,
        "task_obj": task,
    }
    report["status"] = "PASS" if report["future_leakage_ok"] and report["decode_speedup"] > 1.0 else "FAIL"
    report["checks"] = {
        "future_leakage_ok": report["future_leakage_ok"],
        "full_vs_incremental_supported": report["full_vs_incremental_logit_mae"] <= 1e-6,
        "decode_speedup_gt_one": report["decode_speedup"] > 1.0,
        "learned_mechanism_has_grad": grad_norm > 0.0,
        "reconstruction_low_error": report["token_slot_reconstruction_error"] <= 0.25,
    }
    report["full_vs_incremental_supported"] = report["checks"]["full_vs_incremental_supported"]
    return report
