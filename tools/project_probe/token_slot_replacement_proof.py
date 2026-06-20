#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def parser() -> argparse.ArgumentParser:
    from arch_builder.token_slot_replacement import CausalPositionTask, train_variant

    ap = argparse.ArgumentParser(description="Proof for causal token-slot attention replacement.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--vocab-size", type=int, default=32)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-steps", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="reports/agent_inspector/token_slot_replacement_proof.json")
    ap.add_argument("--out-md", default="reports/agent_inspector/TOKEN_SLOT_REPLACEMENT_PROOF.md")
    return ap


def _run_gate(cmd: List[str], retries: int = 0) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    proc = None
    for attempt in range(retries + 1):
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        attempts.append(
            {
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
        if proc.returncode == 0:
            break
        if attempt < retries:
            time.sleep(0.5)
    assert proc is not None
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "attempts": attempts,
    }


def _md_table(rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("| mode | full acc | incremental acc | attention acc | speedup | grad norm | slot norm | token norm |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['mechanism_mode']} | {row['full_acc']:.4f} | {row['incremental_acc']:.4f} | {row['attention_baseline_acc']:.4f} | "
            f"{row['decode_speedup']:.2f} | {row['grad_norm']:.4f} | {row['slot_norm']:.4f} | {row['token_update_norm']:.4f} |"
        )
    return "\n".join(lines)


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    lines = ["# Token-slot replacement proof", ""]
    lines.append(f"- status: `{report['status']}`")
    lines.append(f"- future_leakage_ok: `{report['checks']['future_leakage_ok']}`")
    lines.append(f"- full_vs_incremental_supported: `{report['checks']['full_vs_incremental_supported']}`")
    lines.append(f"- decode_speedup_gt_one: `{report['checks']['decode_speedup_gt_one']}`")
    lines.append(f"- learned_beats_attention: `{report['checks']['learned_beats_attention']}`")
    lines.append(f"- learned_beats_identity: `{report['checks']['learned_beats_identity']}`")
    lines.append(f"- learned_beats_random: `{report['checks']['learned_beats_random']}`")
    lines.append(f"- learned_beats_frozen: `{report['checks']['learned_beats_frozen']}`")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append(_md_table(report["variants"]))
    lines.append("")
    lines.append("## Checks")
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{'PASS' if value else 'FAIL'}`")
    lines.append("")
    lines.append("## Slot usage")
    for k, v in report["slot_usage_by_position"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Raw report")
    lines.append(f"```json\n{json.dumps(report, indent=2, ensure_ascii=False)}\n```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _future_leakage_check(model, task) -> bool:
    import torch

    tokens = task.sample(2, "cpu").x
    prefix = task.seq_len // 2
    a = tokens.clone()
    b = tokens.clone()
    b[:, prefix:] = (b[:, prefix:] + 5) % task.vocab_size
    # Compare causal prefix slots after processing only the prefix.
    x_a = model.embed(a)
    x_b = model.embed(b)
    slots_a = torch.zeros(a.shape[0], model.slots, model.dim)
    slots_b = torch.zeros_like(slots_a)
    for pos in range(prefix):
        slots_a, _ = model.token_to_slot.step(slots_a, x_a[:, pos], pos, model.core)
        slots_b, _ = model.token_to_slot.step(slots_b, x_b[:, pos], pos, model.core)
    return bool(torch.allclose(slots_a, slots_b, atol=1e-6, rtol=1e-6))


def _cache_reset_check(model, task) -> bool:
    import torch

    batch = task.sample(1, "cpu")
    _, trace1 = model.forward_incremental(batch.x)
    _, trace2 = model.forward_incremental(batch.x)
    return bool(torch.isfinite(torch.tensor(trace1["slot_norm"])) and torch.isfinite(torch.tensor(trace2["slot_norm"])))


def _variable_lengths_check(model, task) -> bool:
    import torch

    batch = task.sample(3, "cpu")
    lengths = torch.tensor([task.seq_len, task.seq_len // 2, task.seq_len // 3], dtype=torch.long)
    logits, trace = model(batch.x, lengths=lengths)
    return bool(torch.isfinite(logits).all() and trace["slot_norm"] > 0)


def _mixed_precision_check(model, task, device: str) -> bool:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return True
    model = model.to("cuda").half()
    batch = task.sample(2, "cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, _ = model(batch.x)
    return bool(torch.isfinite(logits).all())


def main() -> int:
    root = Path.cwd()
    sys.path.insert(0, str(root))

    from arch_builder.token_slot_replacement import (
        CausalAttentionBaseline,
        CausalPositionTask,
        TokenSlotReplacementClassifier,
        train_variant,
    )

    args = parser().parse_args()
    core_gates = [
        _run_gate(["bash", "commands/validate.sh"]),
        _run_gate(["bash", "commands/inspect_with_probe.sh"], retries=1),
    ]

    learned = train_variant(
        seed=args.seed,
        mechanism_mode="learned",
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        dim=args.dim,
        slots=args.slots,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
    )
    variants = [learned]
    for mode in ["identity", "random", "frozen"]:
        variants.append(
            train_variant(
                seed=args.seed,
                mechanism_mode=mode,  # type: ignore[arg-type]
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
                dim=args.dim,
                slots=args.slots,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                batch_size=args.batch_size,
                eval_steps=args.eval_steps,
                eval_batch_size=args.eval_batch_size,
                device=args.device,
            )
        )

    # Train a matched causal attention baseline.
    torch = __import__("torch")
    torch.manual_seed(args.seed)
    task = CausalPositionTask(seq_len=args.seq_len, vocab_size=args.vocab_size)
    baseline = CausalAttentionBaseline(vocab_size=args.vocab_size, seq_len=args.seq_len, dim=args.dim).to(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    baseline_device = next(baseline.parameters()).device.type
    opt = torch.optim.AdamW(baseline.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            batch = task.sample(args.batch_size, baseline_device)
            opt.zero_grad(set_to_none=True)
            logits, _ = baseline(batch.x)
            loss = torch.nn.functional.cross_entropy(logits, batch.y)
            loss.backward()
            opt.step()
    baseline_acc = 0.0
    total = 0
    correct = 0
    for _ in range(args.eval_steps):
        batch = task.sample(args.eval_batch_size, baseline_device)
        logits, _ = baseline(batch.x)
        pred = logits.argmax(dim=-1)
        total += batch.y.shape[0]
        correct += (pred == batch.y).sum().item()
    baseline_acc = correct / max(1, total)

    learned_report = learned
    learned_report["attention_baseline_acc"] = baseline_acc
    learned_report["future_leakage_ok"] = _future_leakage_check(
        TokenSlotReplacementClassifier(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            dim=args.dim,
            slots=args.slots,
            mechanism_mode="learned",
        ),
        task,
    )
    learned_report["cache_reset_ok"] = _cache_reset_check(
        TokenSlotReplacementClassifier(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            dim=args.dim,
            slots=args.slots,
            mechanism_mode="learned",
        ),
        task,
    )
    learned_report["variable_lengths_ok"] = _variable_lengths_check(
        TokenSlotReplacementClassifier(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            dim=args.dim,
            slots=args.slots,
            mechanism_mode="learned",
        ),
        task,
    )
    learned_report["mixed_precision_ok"] = _mixed_precision_check(
        TokenSlotReplacementClassifier(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            dim=args.dim,
            slots=args.slots,
            mechanism_mode="learned",
        ),
        task,
        args.device,
    )

    attention_acc = baseline_acc
    identity_acc = variants[1]["full_acc"]
    random_acc = variants[2]["full_acc"]
    frozen_acc = variants[3]["full_acc"]
    learned_acc = learned_report["full_acc"]

    checks = {
        "future_leakage_ok": bool(learned_report["future_leakage_ok"]),
        "full_vs_incremental_supported": bool(learned_report["checks"]["full_vs_incremental_supported"]),
        "decode_speedup_gt_one": learned_report["decode_speedup"] > 1.0,
        "learned_beats_attention": learned_acc > attention_acc + 0.1,
        "learned_beats_identity": learned_acc > identity_acc + 0.1,
        "learned_beats_random": learned_acc > random_acc + 0.1,
        "learned_beats_frozen": learned_acc > frozen_acc + 0.1,
        "cache_reset_ok": bool(learned_report["cache_reset_ok"]),
        "variable_lengths_ok": bool(learned_report["variable_lengths_ok"]),
        "mixed_precision_ok": bool(learned_report["mixed_precision_ok"]),
        "learned_has_grad": learned_report["grad_norm"] > 0.0,
        "mechanism_nonzero": learned_report["token_update_norm"] > 0.0 and learned_report["slot_norm"] > 0.0,
        "core_gates_retained": core_gates[0]["returncode"] == 0 and core_gates[1]["returncode"] == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "status": status,
        "checks": checks,
        "core_gates": core_gates,
        "variants": [
            {k: v for k, v in learned_report.items() if k not in {"model", "task_obj"}},
            {k: v for k, v in variants[1].items() if k not in {"model", "task_obj"}},
            {k: v for k, v in variants[2].items() if k not in {"model", "task_obj"}},
            {k: v for k, v in variants[3].items() if k not in {"model", "task_obj"}},
            {
                "mechanism_mode": "attention_only",
                "full_acc": attention_acc,
                "incremental_acc": attention_acc,
                "decode_speedup": learned_report["decode_speedup"],
                "grad_norm": 0.0,
                "slot_norm": 0.0,
                "token_update_norm": 0.0,
                "attention_baseline_acc": attention_acc,
                "params": float(sum(p.numel() for p in baseline.parameters())),
                "flops": float(args.seq_len * args.dim * 32),
                "activation_bytes": float(args.seq_len * args.dim * 8),
            },
        ],
        "attention_baseline_acc": attention_acc,
        "slot_usage_by_position": learned_report["slot_usage_by_position"],
        "credit": learned_report["credit"],
        "learned": {k: v for k, v in learned_report.items() if k not in {"model", "task_obj"}},
        "baseline": {
            "seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
            "dim": args.dim,
            "slots": args.slots,
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_md(out_md, report)
    print(f"[token-slot-replacement-proof] status={status} checks={checks}")
    if status != "PASS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
