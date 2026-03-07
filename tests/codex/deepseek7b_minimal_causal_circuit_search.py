#!/usr/bin/env python
"""
任务2：最小因果回路搜索（从统计相关转向可干预、可复现）。

离线方法：
- 以 noun signature 神经元集合为基础，定义可干预目标函数：
  target_selectivity = target_coverage - off_target_overlap
- 用贪心搜索找到达到指定保真阈值的最小子集（minimal causal subset）。
- 多 seed 重复抽样负样本，评估子集复现性（pairwise jaccard）。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from agi_research_result_schema import build_result_payload, write_result_bundle


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(xs: Sequence[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def safe_std(xs: Sequence[float]) -> float:
    return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def load_signatures(mass_json: Path, top_k: int) -> Dict[str, Set[int]]:
    obj = read_json(mass_json)
    out: Dict[str, Set[int]] = {}
    for row in (obj.get("noun_records") or []):
        noun = str(row.get("noun", "")).strip().lower()
        sig = [int(x) for x in (row.get("signature_top_indices") or [])[:top_k]]
        if noun and sig:
            out[noun] = set(sig)
    return out


def objective(subset: Set[int], target_sig: Set[int], negatives: Sequence[Set[int]]) -> float:
    if not subset:
        return 0.0
    cov = float(len(subset & target_sig) / max(1, len(target_sig)))
    off = safe_mean([float(len(subset & ns) / max(1, len(subset))) for ns in negatives]) if negatives else 0.0
    return float(cov - off)


def greedy_minimal_subset(
    target_sig: Set[int],
    negatives: Sequence[Set[int]],
    fidelity_ratio: float,
    max_steps: int,
) -> Dict[str, object]:
    full_score = objective(set(target_sig), target_sig, negatives)
    threshold = float(full_score * fidelity_ratio)
    chosen: Set[int] = set()
    candidates = set(target_sig)

    best_trace: List[float] = []
    for _ in range(min(max_steps, len(candidates))):
        best_item = None
        best_score = -1e9
        for x in candidates:
            tmp = set(chosen)
            tmp.add(x)
            sc = objective(tmp, target_sig, negatives)
            if sc > best_score:
                best_score = sc
                best_item = x
        if best_item is None:
            break
        chosen.add(int(best_item))
        candidates.remove(best_item)
        best_trace.append(float(best_score))
        if best_score >= threshold:
            break

    # 干预验证：移除 minimal subset 后，目标函数应显著下降
    removed = set(target_sig) - set(chosen)
    drop = float(full_score - objective(removed, target_sig, negatives))
    return {
        "minimal_subset": sorted(int(x) for x in chosen),
        "minimal_size": int(len(chosen)),
        "full_size": int(len(target_sig)),
        "full_score": float(full_score),
        "subset_score": float(objective(chosen, target_sig, negatives)),
        "fidelity_ratio": float(objective(chosen, target_sig, negatives) / (full_score + 1e-12)),
        "intervention_drop_after_remove_subset": drop,
        "score_trace": best_trace,
    }


def pick_targets(sig_map: Dict[str, Set[int]], target_nouns: Sequence[str], n_auto: int) -> List[str]:
    out = [x.strip().lower() for x in target_nouns if x.strip().lower() in sig_map]
    if out:
        return sorted(set(out))
    # 自动选取 signature 较大的概念
    rows = sorted(sig_map.items(), key=lambda kv: len(kv[1]), reverse=True)
    return [k for k, _ in rows[: max(1, n_auto)]]


def run_multiseed(
    sig_map: Dict[str, Set[int]],
    targets: Sequence[str],
    seeds: Sequence[int],
    neg_sample_size: int,
    fidelity_ratio: float,
    max_steps: int,
) -> Dict[str, object]:
    nouns = sorted(sig_map.keys())
    per_target: Dict[str, Dict[str, object]] = {}
    summary_min_sizes: List[float] = []
    summary_fidelity: List[float] = []
    summary_drop: List[float] = []
    summary_jaccards: List[float] = []

    for t in targets:
        t_sig = sig_map[t]
        runs = []
        subsets = []
        for sd in seeds:
            rng = random.Random(sd + abs(hash(t)) % 100000)
            neg_names = [x for x in nouns if x != t]
            if not neg_names:
                continue
            picked = rng.sample(neg_names, k=min(len(neg_names), neg_sample_size))
            negatives = [sig_map[x] for x in picked]
            row = greedy_minimal_subset(
                target_sig=t_sig,
                negatives=negatives,
                fidelity_ratio=fidelity_ratio,
                max_steps=max_steps,
            )
            row["seed"] = int(sd)
            row["negatives"] = picked
            runs.append(row)
            subsets.append(set(int(x) for x in row["minimal_subset"]))

        pair_j = [jaccard(a, b) for a, b in combinations(subsets, 2)] if len(subsets) >= 2 else []
        for r in runs:
            summary_min_sizes.append(float(r["minimal_size"]))
            summary_fidelity.append(float(r["fidelity_ratio"]))
            summary_drop.append(float(r["intervention_drop_after_remove_subset"]))
        summary_jaccards.extend(pair_j)

        per_target[t] = {
            "runs": runs,
            "stability_pairwise_jaccard_mean": safe_mean(pair_j),
            "stability_pairwise_jaccard_std": safe_std(pair_j),
            "minimal_size_mean": safe_mean([float(x["minimal_size"]) for x in runs]),
            "fidelity_ratio_mean": safe_mean([float(x["fidelity_ratio"]) for x in runs]),
            "intervention_drop_mean": safe_mean([float(x["intervention_drop_after_remove_subset"]) for x in runs]),
        }

    return {
        "targets": per_target,
        "global": {
            "target_count": len(per_target),
            "min_subset_size_mean": safe_mean(summary_min_sizes),
            "min_subset_size_std": safe_std(summary_min_sizes),
            "fidelity_mean": safe_mean(summary_fidelity),
            "fidelity_std": safe_std(summary_fidelity),
            "intervention_drop_mean": safe_mean(summary_drop),
            "intervention_drop_std": safe_std(summary_drop),
            "reproducibility_jaccard_mean": safe_mean(summary_jaccards),
            "reproducibility_jaccard_std": safe_std(summary_jaccards),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal causal circuit search (offline)")
    ap.add_argument(
        "--mass-json",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    )
    ap.add_argument("--top-k", type=int, default=120)
    ap.add_argument("--target-nouns", default="apple,banana,cat,dog")
    ap.add_argument("--auto-targets", type=int, default=4)
    ap.add_argument("--seeds", default="101,202,303,404,505")
    ap.add_argument("--neg-sample-size", type=int, default=24)
    ap.add_argument("--fidelity-ratio", type=float, default=0.90)
    ap.add_argument("--max-steps", type=int, default=48)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    sig_map = load_signatures(Path(args.mass_json), top_k=args.top_k)
    if len(sig_map) < 20:
        raise RuntimeError("signature 样本不足，无法执行最小因果回路搜索。")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    target_nouns = [x.strip() for x in args.target_nouns.split(",")]
    targets = pick_targets(sig_map, target_nouns, n_auto=args.auto_targets)

    result = run_multiseed(
        sig_map=sig_map,
        targets=targets,
        seeds=seeds,
        neg_sample_size=args.neg_sample_size,
        fidelity_ratio=args.fidelity_ratio,
        max_steps=args.max_steps,
    )

    g = result["global"]
    payload = build_result_payload(
        experiment_id="minimal_causal_circuit_search_v1",
        title="最小因果回路搜索（可干预/可复现）",
        config={
            "mass_json": args.mass_json,
            "top_k": args.top_k,
            "targets": targets,
            "seeds": seeds,
            "neg_sample_size": args.neg_sample_size,
            "fidelity_ratio": args.fidelity_ratio,
            "max_steps": args.max_steps,
        },
        metrics=result,
        hypotheses=[
            {
                "id": "H_minimal_subset_exists",
                "rule": "min_subset_size_mean < 0.5 * top_k",
                "pass": bool(g["min_subset_size_mean"] < 0.5 * args.top_k),
            },
            {
                "id": "H_intervention_effective",
                "rule": "intervention_drop_mean > 0.05",
                "pass": bool(g["intervention_drop_mean"] > 0.05),
            },
            {
                "id": "H_reproducible",
                "rule": "reproducibility_jaccard_mean >= 0.30",
                "pass": bool(g["reproducibility_jaccard_mean"] >= 0.30),
            },
        ],
        notes=[
            "该实验通过显式子集保留/移除实现干预，不是纯相关性统计。",
            "多 seed 复现实验用于验证最小回路稳定性。",
        ],
    )

    lines = [
        "# 最小因果回路搜索报告",
        "",
        "## 全局指标",
        f"- min_subset_size_mean: {g['min_subset_size_mean']:.4f}",
        f"- fidelity_mean: {g['fidelity_mean']:.4f}",
        f"- intervention_drop_mean: {g['intervention_drop_mean']:.4f}",
        f"- reproducibility_jaccard_mean: {g['reproducibility_jaccard_mean']:.4f}",
        "",
        "## 目标概念摘要",
    ]
    for t, row in result["targets"].items():
        lines.append(
            f"- {t}: size_mean={row['minimal_size_mean']:.2f}, fidelity_mean={row['fidelity_ratio_mean']:.4f}, "
            f"drop_mean={row['intervention_drop_mean']:.4f}, stability_jaccard={row['stability_pairwise_jaccard_mean']:.4f}"
        )
    lines.append("")
    lines.append("## 结论")
    lines.append("- 若 intervention_drop 与 reproducibility 同时较高，可支持“可干预 + 可复现”的最小回路证据。")
    report = "\n".join(lines) + "\n"

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_minimal_causal_circuit_{ts}")
    paths = write_result_bundle(
        out_dir=out_dir,
        base_name="minimal_causal_circuit_search",
        payload=payload,
        report_md=report,
    )
    print(f"[OK] {paths['json']}")
    print(f"[OK] {paths['md']}")


if __name__ == "__main__":
    main()

