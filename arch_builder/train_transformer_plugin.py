from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reporting import ensure_dir, write_json
from .transformer_plugin import MechanismMode, PlacementMode, train_variant


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a tiny reference Transformer with add-on plug-in modes.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--placement", default="after", choices=["after", "before", "attention_only"])
    ap.add_argument("--mechanism-mode", default="learned", choices=["learned", "identity", "random", "frozen"])
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--vocab-size", type=int, default=64)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-steps", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="agent_reports/transformer_plugin_smoke")
    ap.add_argument("--latest-report", default="LATEST_TRANSFORMER_PLUGIN_REPORT.md")
    return ap


def main() -> int:
    args = parser().parse_args()
    report = train_variant(
        seed=args.seed,
        placement=args.placement,
        mechanism_mode=args.mechanism_mode,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_heads=args.num_heads,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    report = {k: v for k, v in report.items() if k not in {"model", "task_obj"}}
    report["placement"] = args.placement
    report["mechanism_mode"] = args.mechanism_mode
    out_dir = ensure_dir(Path(args.out_dir))
    write_json(out_dir / "final_report.json", report)
    summary = [
        "# Transformer plugin report",
        "",
        f"- placement: `{report['placement']}`",
        f"- mechanism_mode: `{report['mechanism_mode']}`",
        f"- val_acc: `{report['val_acc']}`",
        f"- val_loss: `{report['val_loss']}`",
        f"- output_norm: `{report['output_norm']}`",
        f"- mechanism_norm: `{report['mechanism_norm']}`",
        f"- params: `{report['params']}`",
        f"- flops: `{report['flops']}`",
        f"- activation_bytes: `{report['activation_bytes']}`",
        f"- grad_norm: `{report['grad_norm']}`",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(summary), encoding="utf-8")
    Path(args.latest_report).write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
