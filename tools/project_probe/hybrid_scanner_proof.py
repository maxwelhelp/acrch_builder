from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from arch_builder.executor import ActionExecutor
from arch_builder.hybrid_scanner import HybridScanner
from arch_builder.primitive_matrix import PrimitiveMatrix5x5


SOURCE_NAMES = {0: "grid", 1: "semantic", 2: "usage", 3: "random"}


@torch.no_grad()
def primitive_quality(
    executor: ActionExecutor,
    pm: PrimitiveMatrix5x5,
    src: torch.Tensor,
    tgt: torch.Tensor,
) -> torch.Tensor:
    batch = src.shape[0]
    ids = torch.arange(pm.num_primitives).view(1, -1).expand(batch, -1)
    memory = torch.zeros_like(src)
    output = executor(src, tgt, memory, ids)
    labels = ((src * tgt).mean(dim=-1) > 0).long()
    pred = (output.mean(dim=-1) > 0).long()
    return (pred == labels[:, None]).float().mean(dim=0)


def paired_result(
    candidate_ids: torch.Tensor,
    source_ids: torch.Tensor,
    quality: torch.Tensor,
    allowed_sources: Iterable[int],
    expected_id: int,
    cutoff: int,
) -> Dict[str, float]:
    allowed = torch.zeros_like(source_ids, dtype=torch.bool)
    for source in allowed_sources:
        allowed |= source_ids == source
    recoveries, best_quality = [], []
    for row in range(candidate_ids.shape[0]):
        ids = candidate_ids[row][allowed[row]]
        scores = quality[ids]
        keep = scores.topk(k=min(cutoff, ids.numel())).indices
        ids, scores = ids[keep], scores[keep]
        winner = ids[scores.argmax()]
        recoveries.append(float(winner == expected_id))
        best_quality.append(float(scores.max()))
    return {
        "recovery": sum(recoveries) / len(recoveries),
        "heldout_quality": sum(best_quality) / len(best_quality),
    }


@torch.no_grad()
def main() -> None:
    torch.manual_seed(606)
    dim = 32
    pm = PrimitiveMatrix5x5(embed_dim=32, noise_std=0.0)
    expected_id = pm.name_to_id["product"]
    anchor_id = pm.name_to_id["identity"]

    # The proof fixture creates a semantic relation, not a usage/task list:
    # product is a near semantic neighbour of identity but outside its local grid.
    anchor_vector = F.normalize(pm.emb[anchor_id].detach(), dim=-1)
    perturbation = torch.zeros_like(anchor_vector)
    perturbation[-1] = 0.04
    pm.emb[expected_id].copy_(F.normalize(anchor_vector + perturbation, dim=-1))

    executor = ActionExecutor(dim, pm)
    calibration_src = torch.randn(512, dim)
    calibration_tgt = torch.randn(512, dim)
    calibration_quality = primitive_quality(executor, pm, calibration_src, calibration_tgt)
    credited_id = int(calibration_quality.argmax())
    pm.update_usage_credit(
        torch.full((64,), credited_id, dtype=torch.long),
        torch.full((64,), float(calibration_quality[credited_id])),
        momentum=0.0,
    )

    test_src = torch.randn(1024, dim)
    test_tgt = torch.randn(1024, dim)
    heldout_quality = primitive_quality(executor, pm, test_src, test_tgt)

    scanner = HybridScanner(
        dim=dim,
        context_dim=dim * 5,
        prim_embed_dim=pm.emb.shape[-1],
        local_k=4,
        semantic_k=3,
        usage_k=2,
        random_k=1,
    )
    scanner.anchor.weight.zero_()
    scanner.anchor.bias.fill_(-10.0)
    scanner.anchor.bias[anchor_id] = 10.0
    context = torch.zeros(64, dim * 5)
    memory = torch.zeros(64, dim)
    candidate_ids, _, source_ids, scanner_metrics = scanner(context, memory, pm)

    grid_ids = candidate_ids[source_ids == 0].unique()
    semantic_ids = candidate_ids[source_ids == 1].unique()
    usage_ids = candidate_ids[source_ids == 2].unique()
    expected_absent_grid = not bool((grid_ids == expected_id).any())
    expected_semantic_rank = int(
        ((F.normalize(pm.emb, dim=-1) @ F.normalize(pm.emb[anchor_id], dim=-1))
         .argsort(descending=True) == expected_id).nonzero(as_tuple=False)[0]
    ) + 1

    cutoff = candidate_ids.shape[1] - 1
    comparisons = {
        "grid": paired_result(candidate_ids, source_ids, heldout_quality, [0], expected_id, cutoff),
        "grid_semantic": paired_result(candidate_ids, source_ids, heldout_quality, [0, 1], expected_id, cutoff),
        "grid_semantic_usage": paired_result(candidate_ids, source_ids, heldout_quality, [0, 1, 2], expected_id, cutoff),
        "full_hybrid_random": paired_result(candidate_ids, source_ids, heldout_quality, [0, 1, 2, 3], expected_id, cutoff),
    }

    grid_best = float(heldout_quality[grid_ids].max())
    semantic_best = float(heldout_quality[semantic_ids].max())
    usage_best = float(heldout_quality[usage_ids].max())
    semantic_gain = semantic_best - grid_best
    usage_gain = usage_best - grid_best

    # Real held-out quality drives the cutoff and the following source masses.
    source_mass = {name: 0.0 for name in SOURCE_NAMES.values()}
    source_quality = {name: [] for name in SOURCE_NAMES.values()}
    for row in range(candidate_ids.shape[0]):
        ids = candidate_ids[row]
        qualities = heldout_quality[ids]
        keep = qualities.topk(k=cutoff).indices
        mass = torch.softmax(8.0 * qualities[keep], dim=0)
        for pos, weight in zip(keep.tolist(), mass.tolist()):
            name = SOURCE_NAMES[int(source_ids[row, pos])]
            source_mass[name] += weight / candidate_ids.shape[0]
            source_quality[name].append(float(qualities[pos]))
    source_gain = {
        name: (sum(values) / len(values) - grid_best if values else 0.0)
        for name, values in source_quality.items()
    }

    pm_metrics = pm.metrics()
    semantic_entropy = float(scanner_metrics["semantic_neighbor_entropy"])
    embedding_rank = float(pm_metrics["primitive_embedding_rank"])
    collapse_flag = embedding_rank < 8 or semantic_entropy < 0.5
    global_rescue_rate = float(
        expected_absent_grid
        and semantic_gain > 0
        and comparisons["full_hybrid_random"]["recovery"] > comparisons["grid"]["recovery"]
    )

    checks = {
        "expected_absent_grid": expected_absent_grid,
        "expected_present_non_grid": bool(((candidate_ids == expected_id) & (source_ids != 0)).any()),
        "real_cutoff": cutoff < candidate_ids.shape[1] and cutoff < pm.num_primitives,
        "global_rescue_positive": global_rescue_rate > 0,
        "non_grid_heldout_gain_positive": max(semantic_gain, usage_gain) > 0,
        "full_beats_grid": comparisons["full_hybrid_random"]["recovery"] > comparisons["grid"]["recovery"],
        "embedding_rank_healthy": embedding_rank >= 8,
        "semantic_entropy_healthy": semantic_entropy >= 0.5,
        "random_mass_bounded": 0 < source_mass["random"] < 0.25,
        "usage_from_credit_winner": credited_id == expected_id and int(pm.usage_topk(torch.tensor([anchor_id]), 1).item()) == credited_id,
        "no_collapse": not collapse_flag,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "task": "semantic_rescue_product",
        "anchor": pm.names[anchor_id],
        "expected": pm.names[expected_id],
        "candidate_count": int(candidate_ids.shape[1]),
        "top_k_cutoff": cutoff,
        "expected_semantic_rank": expected_semantic_rank,
        "semantic_entropy": semantic_entropy,
        "embedding_rank": embedding_rank,
        "semantic_grid_mismatch": scanner_metrics["semantic_grid_mismatch"],
        "global_rescue_rate": global_rescue_rate,
        "semantic_candidate_quality": semantic_best,
        "semantic_candidate_real_gain": semantic_gain,
        "usage_candidate_quality": usage_best,
        "usage_candidate_real_gain": usage_gain,
        "credited_primitive": pm.names[credited_id],
        "usage_credit_observations": pm_metrics["usage_credit_observations"],
        "source_choice_mass": source_mass,
        "source_heldout_gain": source_gain,
        "collapse_flag": collapse_flag,
        "paired_comparisons": comparisons,
        "checks": checks,
    }

    out_dir = Path("reports/agent_inspector")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hybrid_scanner_proof.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["# HybridScanner proof", "", f"- status: `{report['status']}`"]
    for key in [
        "candidate_count", "top_k_cutoff", "expected_semantic_rank", "semantic_entropy",
        "embedding_rank", "semantic_grid_mismatch", "global_rescue_rate",
        "semantic_candidate_real_gain", "usage_candidate_real_gain", "credited_primitive",
        "usage_credit_observations", "collapse_flag",
    ]:
        lines.append(f"- {key}: `{report[key]}`")
    lines += ["", "## Paired comparisons", "", f"```json\n{json.dumps(comparisons, indent=2)}\n```",
              "", "## Source mass / gain", "", f"```json\n{json.dumps({'mass': source_mass, 'gain': source_gain}, indent=2)}\n```",
              "", "## Checks", ""]
    lines.extend(f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in checks.items())
    (out_dir / "HYBRID_SCANNER_PROOF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[hybrid-scanner-proof] status={report['status']} checks={checks}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
