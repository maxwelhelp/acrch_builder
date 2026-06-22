from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception as exc:
    raise SystemExit("scikit-learn is required") from exc

from arch_builder.srcf_graph_core import (
    SRCFGraphConfig,
    SRCFGraphCore,
    SRCFGraphLossWeights,
    srcf_graph_closure_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def upper_mask(n: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


def metric_pack(target: torch.Tensor, score: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    y = target[mask].detach().cpu().numpy().astype(np.int32)
    s = score[mask].detach().cpu().numpy().astype(np.float64)
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "acc": float("nan")}
    pred = (s >= 0.5).astype(np.int32)
    return {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "acc": float((pred == y).mean()),
    }


def balance_labels(n: int, communities: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.arange(n) % communities
    rng.shuffle(labels)
    return labels.astype(np.int64)


def generate_clean_program_graph(args, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a hard graph from a hidden relation program.

    The model does NOT receive labels/hubs. It receives only corrupted adjacency
    by default. Targets require edge repair, 2-hop path closure, community basin,
    and bridge-basin relation.
    """
    n = args.n
    k = args.communities
    labels = balance_labels(n, k, rng)
    hubs = []
    for c in range(k):
        nodes = np.where(labels == c)[0]
        hubs.append(int(rng.choice(nodes)))
    hubs = np.array(hubs, dtype=np.int64)

    a = np.zeros((n, n), dtype=np.float32)

    # sparse intra-community structure
    for i in range(n):
        for j in range(i + 1, n):
            p = args.p_in if labels[i] == labels[j] else args.p_out
            if rng.random() < p:
                a[i, j] = a[j, i] = 1.0

    # force local motifs inside communities: triangles and chains
    for c in range(k):
        nodes = np.where(labels == c)[0]
        if len(nodes) >= 3:
            for _ in range(args.motifs_per_comm):
                tri = rng.choice(nodes, size=3, replace=False)
                a[tri[0], tri[1]] = a[tri[1], tri[0]] = 1.0
                a[tri[1], tri[2]] = a[tri[2], tri[1]] = 1.0
                if rng.random() < 0.60:
                    a[tri[0], tri[2]] = a[tri[2], tri[0]] = 1.0
        # connect hub to a few local nodes
        h = hubs[c]
        others = [x for x in nodes if x != h]
        rng.shuffle(others)
        for v in others[: max(1, min(len(others), args.hub_degree))]:
            a[h, v] = a[v, h] = 1.0

    # ring of community hubs creates cross-community operational basins
    for c in range(k):
        h1 = hubs[c]
        h2 = hubs[(c + 1) % k]
        a[h1, h2] = a[h2, h1] = 1.0
        if rng.random() < args.extra_bridge_p:
            h3 = hubs[(c + 2) % k]
            a[h1, h3] = a[h3, h1] = 1.0

    np.fill_diagonal(a, 0.0)
    return a, labels, hubs


def targets_from_program(a: np.ndarray, labels: np.ndarray, hubs: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    edge = (a > 0).astype(np.float32)
    path2 = ((edge @ edge) > 0).astype(np.float32)
    path2 = np.maximum(path2, edge)
    comm = (labels[:, None] == labels[None, :]).astype(np.float32)

    # Bridge-basin target: same community OR adjacent hub-connected communities.
    k = int(labels.max()) + 1
    adjacent_comm = np.zeros((k, k), dtype=np.float32)
    for c in range(k):
        adjacent_comm[c, c] = 1.0
        adjacent_comm[c, (c + 1) % k] = 1.0
        adjacent_comm[(c + 1) % k, c] = 1.0
    basin = adjacent_comm[labels[:, None], labels[None, :]].astype(np.float32)
    # make it less trivial: require either path support or same community
    basin = basin * np.maximum(path2, comm)

    for x in (edge, path2, comm, basin):
        np.fill_diagonal(x, 0.0)
    return np.stack([edge, path2, comm, basin], axis=-1).astype(np.float32)


def corrupt(a: np.ndarray, args, rng: np.random.Generator, hard: bool = False) -> np.ndarray:
    drop_p = args.hard_drop_p if hard else args.drop_p
    add_p = args.hard_add_p if hard else args.add_p
    n = a.shape[0]
    c = a.copy()
    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j] > 0.5 and rng.random() < drop_p:
                c[i, j] = c[j, i] = 0.0
            elif a[i, j] < 0.5 and rng.random() < add_p:
                c[i, j] = c[j, i] = 1.0
    np.fill_diagonal(c, 0.0)
    return c


def relation_features(c: np.ndarray, mode: str) -> np.ndarray:
    n = c.shape[0]
    c = (c > 0).astype(np.float32)
    if mode == "raw":
        return c[..., None].astype(np.float32)
    deg = c.sum(-1) / max(n - 1, 1)
    cn = (c @ c) / max(n, 1)
    two = (cn > 0).astype(np.float32)
    jac = cn / (deg[:, None] + deg[None, :] - cn + 1e-6)
    deg_diff = np.abs(deg[:, None] - deg[None, :])
    deg_prod = deg[:, None] * deg[None, :]
    row = c.mean(-1)[:, None].repeat(n, axis=1)
    miss = (1.0 - c) * cn
    return np.stack([c, cn, jac, two, deg_diff, deg_prod, row, miss], axis=-1).astype(np.float32)


@dataclass
class Batch:
    rel: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor


def make_pair_batch(args, rng: np.random.Generator, device: torch.device, hard: bool = False) -> Tuple[Batch, Batch]:
    rel_a, rel_b, targets = [], [], []
    for _ in range(args.batch):
        clean, labels, hubs = generate_clean_program_graph(args, rng)
        target = targets_from_program(clean, labels, hubs)
        ca = corrupt(clean, args, rng, hard=hard)
        cb = corrupt(clean, args, rng, hard=hard)
        rel_a.append(relation_features(ca, args.rel_mode))
        rel_b.append(relation_features(cb, args.rel_mode))
        targets.append(target)
    rel_a_t = torch.tensor(np.stack(rel_a), device=device)
    rel_b_t = torch.tensor(np.stack(rel_b), device=device)
    target_t = torch.tensor(np.stack(targets), device=device)
    mask = upper_mask(args.n, device)[None].expand(args.batch, -1, -1)
    return Batch(rel_a_t, target_t, mask), Batch(rel_b_t, target_t, mask)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = logits.new_tensor([1.0, 1.0, 0.75, 0.75])
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights
    return loss[mask].mean()


@torch.no_grad()
def evaluate(model: SRCFGraphCore, args, rng: np.random.Generator, device: torch.device, hard: bool = False) -> Dict[str, float]:
    model.eval()
    names = ["edge", "path2", "comm", "basin"]
    vals: Dict[str, List[float]] = {}
    for _ in range(args.eval_batches):
        b, _ = make_pair_batch(args, rng, device, hard=hard)
        out = model(b.rel, active_mask=b.mask)
        score = torch.sigmoid(out["logits"])
        for i, name in enumerate(names):
            m = metric_pack(b.target[..., i], score[..., i], b.mask)
            for mk, mv in m.items():
                vals.setdefault(f"{name}_{mk}", []).append(mv)
    model.train()
    return {k: float(np.nanmean(v)) for k, v in vals.items()}


def train_one(model: SRCFGraphCore, opt, b: Batch, peer: Batch, weights: SRCFGraphLossWeights, closure_w: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    out = model(b.rel, active_mask=b.mask)
    out_peer = model(peer.rel, active_mask=peer.mask)
    task = masked_bce(out["logits"], b.target, b.mask)
    closs, cm = srcf_graph_closure_loss(out, weights, peer_output=out_peer)
    loss = task + closure_w * closs
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad):
        raise RuntimeError(f"bad grad={grad}")
    opt.step()
    cm["task_loss"] = float(task.detach().cpu())
    cm["total_loss"] = float(loss.detach().cpu())
    return loss, cm


@torch.no_grad()
def inspect_program(model: SRCFGraphCore, args, rng: np.random.Generator, device: torch.device, title: str) -> None:
    model.eval()
    b, _ = make_pair_batch(args, rng, device, hard=False)
    h = model.enc(b.rel)
    print(f"PROGRAM_INSPECT {title}")
    for li, layer in enumerate(model.layers):
        ctx = layer.context_norm(layer.build_context(h, b.rel))
        edge = torch.sigmoid(layer.edge_head(ctx))
        ap = torch.softmax(layer.action_head(ctx), dim=-1)
        usage = ap.mean(dim=(0, 1, 2))
        topv, topi = torch.topk(usage, k=min(4, usage.numel()))
        entropy = (-(ap.clamp_min(1e-8) * ap.clamp_min(1e-8).log()).sum(dim=-1)).mean()
        print(
            f"  layer={li} edge_mass={float(edge.mean().cpu()):.4f} "
            f"entropy={float(entropy.cpu()):.4f} "
            f"top_actions=" + ",".join([f"a{int(i)}:{float(v):.3f}" for v, i in zip(topv.cpu(), topi.cpu())])
        )
        h, d = layer(h, b.rel)
    model.train()


def fmt(res: Dict[str, float], task: str) -> str:
    return f"{task}=auc:{res[f'{task}_auc']:.3f} ap:{res[f'{task}_ap']:.3f} acc:{res[f'{task}_acc']:.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--communities", type=int, default=4)
    p.add_argument("--p-in", type=float, default=0.18)
    p.add_argument("--p-out", type=float, default=0.015)
    p.add_argument("--motifs-per-comm", type=int, default=3)
    p.add_argument("--hub-degree", type=int, default=3)
    p.add_argument("--extra-bridge-p", type=float, default=0.35)
    p.add_argument("--rel-mode", choices=["raw", "full"], default="raw")
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--micro-steps", type=int, default=3)
    p.add_argument("--actions", type=int, default=8)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=30)
    p.add_argument("--drop-p", type=float, default=0.30)
    p.add_argument("--add-p", type=float, default=0.08)
    p.add_argument("--hard-drop-p", type=float, default=0.55)
    p.add_argument("--hard-add-p", type=float, default=0.18)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--closure-w", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    rel_dim = 1 if args.rel_mode == "raw" else 8

    def make_model(use_triangle: bool) -> SRCFGraphCore:
        cfg = SRCFGraphConfig(
            rel_dim=rel_dim,
            out_dim=4,
            dim=args.dim,
            hidden=args.hidden,
            layers=args.layers,
            micro_steps=args.micro_steps,
            action_count=args.actions,
            use_triangle=use_triangle,
            use_rel_skip=True,
            noise_std=0.02,
        )
        return SRCFGraphCore(cfg).to(device)

    tri = make_model(True)
    no_tri = make_model(False)
    opt_tri = torch.optim.AdamW(tri.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_no = torch.optim.AdamW(no_tri.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = SRCFGraphLossWeights(
        fixed=0.04,
        recovery=0.04,
        contract=0.05,
        far_keep=0.02,
        state_var=0.02,
        move_band=0.02,
        far_margin=0.20,
        state_var_floor=0.003,
        move_min=0.003,
        move_max=2.5,
    )

    print("SRCF_GRAPH_SYNTHETIC_PROGRAM_PROBE")
    print(f"device={device} rel_mode={args.rel_mode} n={args.n} params_tri={sum(p.numel() for p in tri.parameters() if p.requires_grad):,} params_no_tri={sum(p.numel() for p in no_tri.parameters() if p.requires_grad):,}")

    last_tri: Dict[str, float] = {}
    last_no: Dict[str, float] = {}
    for step in range(1, args.steps + 1):
        b, peer = make_pair_batch(args, rng, device, hard=False)
        loss_tri, last_tri = train_one(tri, opt_tri, b, peer, weights, args.closure_w)
        loss_no, last_no = train_one(no_tri, opt_no, b, peer, weights, args.closure_w)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            tri_in = evaluate(tri, args, rng, device, hard=False)
            no_in = evaluate(no_tri, args, rng, device, hard=False)
            tri_ood = evaluate(tri, args, rng, device, hard=True)
            no_ood = evaluate(no_tri, args, rng, device, hard=True)
            print(f"step={step:04d} tri_loss={last_tri['total_loss']:.4f} no_tri_loss={last_no['total_loss']:.4f} tri_srcf={last_tri['srcf_graph_loss']:.4f} no_srcf={last_no['srcf_graph_loss']:.4f}")
            for task in ["edge", "path2", "comm", "basin"]:
                din = tri_in[f"{task}_auc"] - no_in[f"{task}_auc"]
                dood = tri_ood[f"{task}_auc"] - no_ood[f"{task}_auc"]
                print(f"  IN  {fmt(tri_in, task)} | no_tri_auc={no_in[f'{task}_auc']:.3f} Δauc={din:+.3f}")
                print(f"  OOD {fmt(tri_ood, task)} | no_tri_auc={no_ood[f'{task}_auc']:.3f} Δauc={dood:+.3f}")
            print(
                f"  diag tri: fixed={last_tri['srcf_graph_fixed']:.4f} rec={last_tri['srcf_graph_recovery']:.4f} "
                f"move={last_tri['srcf_graph_move']:.4f} var={last_tri['srcf_graph_state_var']:.4f} curve={last_tri['srcf_graph_curve_ratio']:.4f}"
            )

    inspect_program(tri, args, rng, device, "TRIANGLE")
    inspect_program(no_tri, args, rng, device, "NO_TRIANGLE")
    for d in [last_tri, last_no]:
        for k, v in d.items():
            if not math.isfinite(float(v)):
                raise RuntimeError(f"bad metric {k}={v}")
    print("SRCF_GRAPH_SYNTHETIC_PROGRAM_PASS")


if __name__ == "__main__":
    main()
