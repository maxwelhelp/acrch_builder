#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_builder.primitive_matrix import PrimitiveMatrix5x5
from arch_builder.executor import ActionExecutor


class CheapProposal(nn.Module):
    """
    Scanner-like cheap scorer:
    sees src + primitive id only.
    """
    def __init__(self, dim: int, num_primitives: int, hidden: int = 64, embed_dim: int = 32):
        super().__init__()
        self.prim_emb = nn.Embedding(num_primitives, embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim + embed_dim),
            nn.Linear(dim + embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, src, ids):
        emb = self.prim_emb(ids)
        x = torch.cat([src, emb], dim=-1)
        return self.net(x).squeeze(-1)


class GlobalUtilityCritic(nn.Module):
    """
    Global utility critic:
    sees src+tgt+mem+primitive id but NO head_vector.
    """
    def __init__(self, dim: int, num_primitives: int, hidden: int = 128, embed_dim: int = 32):
        super().__init__()
        self.prim_emb = nn.Embedding(num_primitives, embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(3 * dim + embed_dim),
            nn.Linear(3 * dim + embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, src, tgt, mem, ids):
        emb = self.prim_emb(ids)
        x = torch.cat([src, tgt, mem, emb], dim=-1)
        return self.net(x).squeeze(-1)


class HeadUtilityCritic(nn.Module):
    """
    Head-conditioned utility critic:
    sees src+tgt+mem+primitive id + active head_vector.
    """
    def __init__(self, dim: int, num_primitives: int, hidden: int = 128, embed_dim: int = 32):
        super().__init__()
        self.prim_emb = nn.Embedding(num_primitives, embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(4 * dim + embed_dim),
            nn.Linear(4 * dim + embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, src, tgt, mem, ids, head_vectors):
        b, k = ids.shape
        d = src.shape[-1]
        emb = self.prim_emb(ids)
        head_e = head_vectors.unsqueeze(1).expand(b, k, d)
        x = torch.cat([src, tgt, mem, head_e, emb], dim=-1)
        return self.net(x).squeeze(-1)


class JLTaskSim(nn.Module):
    """
    Predicts low-dimensional task-space effect.
    """
    def __init__(self, dim: int, proj_dim: int, num_primitives: int, rank: int = 16, embed_dim: int = 32):
        super().__init__()
        self.prim_emb = nn.Embedding(num_primitives, embed_dim)
        self.to_rank = nn.Linear(3 * dim + embed_dim, rank)
        self.from_rank = nn.Linear(rank, proj_dim)

    def forward(self, src, tgt, mem, ids):
        emb = self.prim_emb(ids)
        x = torch.cat([src, tgt, mem, emb], dim=-1)
        return self.from_rank(torch.tanh(self.to_rank(x)))


def norm_logits(x):
    x = x.float()
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp_min(1e-6)


def make_R(dim, proj_dim, device):
    return torch.randn(dim, proj_dim, device=device) / math.sqrt(dim)


def primitive_similarity(pm: PrimitiveMatrix5x5, device):
    emb = F.normalize(pm.emb.detach().float().to(device), dim=-1)
    sim_emb = emb @ emb.t()

    ids = torch.arange(pm.num_primitives, device=device)
    row = ids // 5
    same_family = (row[:, None] == row[None, :]).float()
    return 0.7 * sim_emb + 0.3 * same_family


def mmr_select(score, sim, m: int, beta: float):
    """
    score: [B, K]
    sim: [K, K] or [B, K, K]
    returns selected indices [B, M]
    """
    b, k = score.shape
    selected = []
    available = torch.ones(b, k, dtype=torch.bool, device=score.device)

    for step in range(m):
        if step == 0:
            penalized = score.masked_fill(~available, -1e9)
        else:
            prev = torch.stack(selected, dim=1)  # [B, S]
            if sim.dim() == 2:
                sim_to_prev = sim[torch.arange(k, device=score.device)[None, :, None], prev[:, None, :]]
                max_sim = sim_to_prev.max(dim=-1).values
            else:
                sim_to_prev = sim.gather(2, prev[:, None, :].expand(b, k, prev.shape[1]))
                max_sim = sim_to_prev.max(dim=-1).values
            penalized = (score - beta * max_sim).masked_fill(~available, -1e9)

        idx = penalized.argmax(dim=-1)
        selected.append(idx)
        available.scatter_(1, idx[:, None], False)

    return torch.stack(selected, dim=1)


def top1_stats(pred_score, true_score):
    pred_idx = pred_score.argmax(dim=-1)
    true_idx = true_score.argmax(dim=-1)
    chosen_true = true_score.gather(1, pred_idx[:, None]).squeeze(1)
    oracle = true_score.max(dim=-1).values
    random_base = true_score.mean(dim=-1)

    regret = (oracle - chosen_true).mean()
    denom = (oracle - random_base).mean().clamp_min(1e-6)
    captured = 1.0 - regret / denom

    return {
        "top1_acc": float((pred_idx == true_idx).float().mean().cpu()),
        "regret": float(regret.cpu()),
        "captured_oracle_gain": float(captured.cpu()),
    }


def set_stats(selected, true_score, prim_sim, effect_sim=None, prim_ids=None):
    b, m = selected.shape
    vals = true_score.gather(1, selected)
    best = vals.max(dim=-1).values
    oracle = true_score.max(dim=-1).values
    random_base = true_score.mean(dim=-1)
    regret = (oracle - best).mean()
    denom = (oracle - random_base).mean().clamp_min(1e-6)
    captured = 1.0 - regret / denom

    if prim_ids is not None:
        selected_prim_ids = prim_ids.gather(1, selected)
    else:
        selected_prim_ids = selected

    # identity similarity
    if m <= 1:
        avg_id_sim = torch.zeros((), device=true_score.device)
    else:
        sims = []
        for i in range(m):
            for j in range(i + 1, m):
                sims.append(prim_sim[selected_prim_ids[:, i], selected_prim_ids[:, j]])
        avg_id_sim = torch.stack(sims, dim=0).mean()

    # effect similarity
    if m <= 1 or effect_sim is None:
        avg_eff_sim = torch.zeros((), device=true_score.device)
    else:
        sims = []
        b_idx = torch.arange(b, device=true_score.device)
        for i in range(m):
            for j in range(i + 1, m):
                sims.append(effect_sim[b_idx, selected[:, i], selected[:, j]])
        avg_eff_sim = torch.stack(sims, dim=0).mean()

    families = selected_prim_ids // 5
    unique_family_count = []
    for row in families.detach().cpu().tolist():
        unique_family_count.append(len(set(row)))

    return {
        "best_of_m_captured_gain": float(captured.cpu()),
        "best_of_m_regret": float(regret.cpu()),
        "avg_selected_identity_similarity": float(avg_id_sim.cpu()),
        "avg_selected_effect_similarity": float(avg_eff_sim.cpu()),
        "unique_families_mean": float(sum(unique_family_count) / max(1, len(unique_family_count))),
    }


def evaluate_policies(scores, true_score, prim_sim, effect_sim, hybrid_sim, prim_ids, beta):
    out = {}
    b, k = true_score.shape

    random_score = torch.randn_like(true_score)
    out["random_top1"] = top1_stats(random_score, true_score)

    for name, score in scores.items():
        out[f"{name}_top1"] = top1_stats(score, true_score)

        top3 = score.topk(k=min(3, k), dim=-1).indices
        top5 = score.topk(k=min(5, k), dim=-1).indices
        out[f"{name}_top3_set"] = set_stats(top3, true_score, prim_sim, effect_sim, prim_ids)
        out[f"{name}_top5_set"] = set_stats(top5, true_score, prim_sim, effect_sim, prim_ids)

        # 1. Identity MMR
        identity_sim = prim_sim[prim_ids.unsqueeze(-1), prim_ids.unsqueeze(-2)]
        immr3 = mmr_select(score, identity_sim, m=min(3, k), beta=beta)
        immr5 = mmr_select(score, identity_sim, m=min(5, k), beta=beta)
        out[f"{name}_identity_mmr3_set"] = set_stats(immr3, true_score, prim_sim, effect_sim, prim_ids)
        out[f"{name}_identity_mmr5_set"] = set_stats(immr5, true_score, prim_sim, effect_sim, prim_ids)

        # 2. Effect MMR
        emmr3 = mmr_select(score, effect_sim, m=min(3, k), beta=beta)
        emmr5 = mmr_select(score, effect_sim, m=min(5, k), beta=beta)
        out[f"{name}_effect_mmr3_set"] = set_stats(emmr3, true_score, prim_sim, effect_sim, prim_ids)
        out[f"{name}_effect_mmr5_set"] = set_stats(emmr5, true_score, prim_sim, effect_sim, prim_ids)

        # 3. Hybrid MMR
        hmmr3 = mmr_select(score, hybrid_sim, m=min(3, k), beta=beta)
        hmmr5 = mmr_select(score, hybrid_sim, m=min(5, k), beta=beta)
        out[f"{name}_hybrid_mmr3_set"] = set_stats(hmmr3, true_score, prim_sim, effect_sim, prim_ids)
        out[f"{name}_hybrid_mmr5_set"] = set_stats(hmmr5, true_score, prim_sim, effect_sim, prim_ids)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--proj-dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--mmr-beta", type=float, default=0.35)
    ap.add_argument("--num-cells", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--K", type=int, default=64)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)

    pm = PrimitiveMatrix5x5(embed_dim=32).to(args.device)
    executor = ActionExecutor(dim=args.dim, primitive_matrix=pm).to(args.device)
    for p in pm.parameters():
        p.requires_grad_(False)
    for p in executor.parameters():
        p.requires_grad_(False)

    total_cands = args.num_cells * pm.num_primitives

    proposal = CheapProposal(args.dim, pm.num_primitives, hidden=64, embed_dim=32).to(args.device)
    global_utility = GlobalUtilityCritic(args.dim, pm.num_primitives, hidden=args.hidden, embed_dim=32).to(args.device)
    head_utility = HeadUtilityCritic(args.dim, pm.num_primitives, hidden=args.hidden, embed_dim=32).to(args.device)
    jl = JLTaskSim(args.dim, args.proj_dim, pm.num_primitives, rank=args.rank, embed_dim=32).to(args.device)

    R = make_R(args.dim, args.proj_dim, args.device)
    
    head_vectors = torch.randn(args.num_heads, args.dim, device=args.device)
    head_vectors = F.normalize(head_vectors, dim=-1)

    opt = torch.optim.AdamW(
        list(proposal.parameters()) + list(global_utility.parameters()) + list(head_utility.parameters()) + list(jl.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    prim_sim = primitive_similarity(pm, args.device)
    history = []

    for step in range(1, args.steps + 1):
        src_base = torch.randn(args.batch, args.num_cells, args.dim, device=args.device)
        tgt_base = torch.randn(args.batch, args.num_cells, args.dim, device=args.device)
        mem_base = torch.randn(args.batch, args.num_cells, args.dim, device=args.device)

        cand_indices = torch.stack([torch.randperm(total_cands, device=args.device)[:args.K] for _ in range(args.batch)], dim=0)
        cell_ids = cand_indices // pm.num_primitives
        prim_ids = cand_indices % pm.num_primitives

        b_idx = torch.arange(args.batch, device=args.device).unsqueeze(1).expand(-1, args.K)
        src_gathered = src_base[b_idx, cell_ids]
        tgt_gathered = tgt_base[b_idx, cell_ids]
        mem_gathered = mem_base[b_idx, cell_ids]

        head_id = torch.randint(0, args.num_heads, [args.batch], device=args.device)
        head_vectors_batch = head_vectors[head_id]

        with torch.no_grad():
            flat_src = src_gathered.view(args.batch * args.K, args.dim)
            flat_tgt = tgt_gathered.view(args.batch * args.K, args.dim)
            flat_mem = mem_gathered.view(args.batch * args.K, args.dim)
            flat_prim = prim_ids.view(args.batch * args.K, 1)

            flat_actual = executor(flat_src, flat_tgt, flat_mem, flat_prim).float()
            actual = flat_actual.view(args.batch, args.K, args.dim)
            
            true_score = (actual * head_vectors_batch.unsqueeze(1)).sum(dim=-1)
            target_jl = actual @ R

        proposal_score = proposal(src_gathered, prim_ids)
        global_utility_score = global_utility(src_gathered, tgt_gathered, mem_gathered, prim_ids)
        head_utility_score = head_utility(src_gathered, tgt_gathered, mem_gathered, prim_ids, head_vectors_batch)
        
        jl_pred = jl(src_gathered, tgt_gathered, mem_gathered, prim_ids)
        h_proj = head_vectors_batch @ R
        jl_score = (jl_pred * h_proj.unsqueeze(1)).sum(dim=-1)

        loss = (
            F.mse_loss(proposal_score.float(), true_score.float()) * 0.5
            + F.mse_loss(global_utility_score.float(), true_score.float())
            + F.mse_loss(head_utility_score.float(), true_score.float())
            + F.mse_loss(jl_pred.float(), target_jl.float())
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % max(1, args.steps // 5) == 0:
            with torch.no_grad():
                jl_pred_norm = F.normalize(jl_pred.float(), dim=-1)
                effect_sim = torch.bmm(jl_pred_norm, jl_pred_norm.transpose(1, 2))
                identity_sim = prim_sim[prim_ids.unsqueeze(-1), prim_ids.unsqueeze(-2)]
                hybrid_sim = 0.2 * identity_sim + 0.8 * effect_sim

                scores = {
                    "proposal": norm_logits(proposal_score),
                    "global_utility": norm_logits(global_utility_score),
                    "head_utility": norm_logits(head_utility_score),
                    "jl_task": norm_logits(jl_score),
                    "proposal_plus_head_utility": norm_logits(proposal_score) + norm_logits(head_utility_score),
                }
                pol = evaluate_policies(
                    scores, true_score, prim_sim, effect_sim, hybrid_sim, prim_ids, beta=args.mmr_beta
                )
                history.append({
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "loss_proposal": float(F.mse_loss(proposal_score.float(), true_score.float()).detach().cpu()),
                    "loss_global_utility": float(F.mse_loss(global_utility_score.float(), true_score.float()).detach().cpu()),
                    "loss_head_utility": float(F.mse_loss(head_utility_score.float(), true_score.float()).detach().cpu()),
                    "loss_jl": float(F.mse_loss(jl_pred.float(), target_jl.float()).detach().cpu()),
                    "policies": pol,
                })

    # Final Validation Evaluation
    with torch.no_grad():
        val_batch = args.batch
        src_base = torch.randn(val_batch, args.num_cells, args.dim, device=args.device)
        tgt_base = torch.randn(val_batch, args.num_cells, args.dim, device=args.device)
        mem_base = torch.randn(val_batch, args.num_cells, args.dim, device=args.device)

        cand_indices = torch.stack([torch.randperm(total_cands, device=args.device)[:args.K] for _ in range(val_batch)], dim=0)
        cell_ids = cand_indices // pm.num_primitives
        prim_ids = cand_indices % pm.num_primitives

        b_idx = torch.arange(val_batch, device=args.device).unsqueeze(1).expand(-1, args.K)
        src_gathered = src_base[b_idx, cell_ids]
        tgt_gathered = tgt_base[b_idx, cell_ids]
        mem_gathered = mem_base[b_idx, cell_ids]

        head_id = torch.randint(0, args.num_heads, [val_batch], device=args.device)
        head_vectors_batch = head_vectors[head_id]

        flat_src = src_gathered.view(val_batch * args.K, args.dim)
        flat_tgt = tgt_gathered.view(val_batch * args.K, args.dim)
        flat_mem = mem_gathered.view(val_batch * args.K, args.dim)
        flat_prim = prim_ids.view(val_batch * args.K, 1)

        flat_actual = executor(flat_src, flat_tgt, flat_mem, flat_prim).float()
        actual = flat_actual.view(val_batch, args.K, args.dim)
        
        true_score = (actual * head_vectors_batch.unsqueeze(1)).sum(dim=-1)
        oracle = true_score.max(dim=-1).values
        random_base = true_score.mean(dim=-1)
        denom = (oracle - random_base).mean().clamp_min(1e-6)

        proposal_score = proposal(src_gathered, prim_ids)
        global_utility_score = global_utility(src_gathered, tgt_gathered, mem_gathered, prim_ids)
        head_utility_score = head_utility(src_gathered, tgt_gathered, mem_gathered, prim_ids, head_vectors_batch)
        
        jl_pred = jl(src_gathered, tgt_gathered, mem_gathered, prim_ids)

        # Similarity matrices
        jl_pred_norm = F.normalize(jl_pred.float(), dim=-1)
        effect_sim = torch.bmm(jl_pred_norm, jl_pred_norm.transpose(1, 2))
        identity_sim = prim_sim[prim_ids.unsqueeze(-1), prim_ids.unsqueeze(-2)]
        hybrid_sim = 0.2 * identity_sim + 0.8 * effect_sim

        # 1. MMR beta sweep
        beta_sweep_results = {}
        for b_val in [0.0, 0.1, 0.25, 0.35, 0.5, 0.75]:
            beta_results = {}
            for sim_name, sim_mat in {
                "identity": identity_sim,
                "effect": effect_sim,
                "hybrid": hybrid_sim
            }.items():
                mmr3 = mmr_select(head_utility_score, sim_mat, m=3, beta=b_val)
                mmr5 = mmr_select(head_utility_score, sim_mat, m=5, beta=b_val)
                stats3 = set_stats(mmr3, true_score, prim_sim, effect_sim, prim_ids)
                stats5 = set_stats(mmr5, true_score, prim_sim, effect_sim, prim_ids)
                beta_results[sim_name] = {
                    "best_of_3_gain": stats3["best_of_m_captured_gain"],
                    "best_of_5_gain": stats5["best_of_m_captured_gain"],
                    "avg_identity_similarity_3": stats3["avg_selected_identity_similarity"],
                    "avg_effect_similarity_3": stats3["avg_selected_effect_similarity"],
                    "unique_families_3": stats3["unique_families_mean"],
                    "unique_families_5": stats5["unique_families_mean"],
                    "regret_3": stats3["best_of_m_regret"],
                    "regret_5": stats5["best_of_m_regret"],
                }
            beta_sweep_results[str(b_val)] = beta_results

        # 2. Scanner Pool Size vs Controller Budget Grid Analysis
        pool_budget_results = {}
        prop_sort_indices = proposal_score.argsort(dim=-1, descending=True) # [B, K]

        for p_val in [8, 16, 24, 32]:
            pool_results = {}
            pool_indices = prop_sort_indices[:, :p_val] # [B, P]
            
            oracle_best_idx = true_score.argmax(dim=-1)
            is_in_pool = (pool_indices == oracle_best_idx.unsqueeze(1)).any(dim=-1).float()
            recall = float(is_in_pool.mean().cpu())

            pool_critic_score = head_utility_score.gather(1, pool_indices)
            pool_true_score = true_score.gather(1, pool_indices)

            pool_critic_sort = pool_critic_score.argsort(dim=-1, descending=True)

            for b_val in [1, 3, 5]:
                actual_b = min(b_val, p_val)
                top_b_indices = pool_critic_sort[:, :actual_b]
                top_b_true_scores = pool_true_score.gather(1, top_b_indices)
                best_true_score = top_b_true_scores.max(dim=-1).values
                
                regret = (oracle - best_true_score).mean()
                captured_gain = 1.0 - regret / denom

                pool_results[f"budget_{b_val}"] = {
                    "captured_gain": float(captured_gain.cpu()),
                    "regret": float(regret.cpu()),
                }
            pool_results["oracle_recall"] = recall
            pool_results["cost_proxy"] = p_val
            pool_budget_results[f"pool_{p_val}"] = pool_results

    last = history[-1]["policies"]

    checks = {
        "head_utility_beats_global_utility_top1": (
            last["head_utility_top1"]["captured_oracle_gain"]
            > last["global_utility_top1"]["captured_oracle_gain"] + 0.02
        ),
        "head_utility_beats_proposal_top1": (
            last["head_utility_top1"]["captured_oracle_gain"]
            > last["proposal_top1"]["captured_oracle_gain"] + 0.05
        ),
        "mmr3_reduces_effect_similarity": (
            last["proposal_plus_head_utility_effect_mmr3_set"]["avg_selected_effect_similarity"]
            < last["proposal_plus_head_utility_top3_set"]["avg_selected_effect_similarity"]
        ),
    }

    out = {
        "device": args.device,
        "config": vars(args),
        "primitive_count": pm.num_primitives,
        "history": history,
        "last": history[-1],
        "beta_sweep": beta_sweep_results,
        "pool_budget": pool_budget_results,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "meaning": "Tests whether simulator/utility is useful beyond scanner proposal and whether MMR gives diverse high-value candidate sets.",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
