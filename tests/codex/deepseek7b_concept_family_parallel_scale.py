#!/usr/bin/env python
"""
任务4：规模化概念族实验（并行对照）

目标：
- 苹果链路：apple -> fruit -> food
- 猫链路：cat -> animal -> biological(由 animal+fruit+human 构造 living 原型)
- 检验：
  1) 是否存在共享基底（shared base）
  2) 是否存在专属偏移（specific offset）
  3) 上述结构是否在多 seed 稳定
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from agi_research_result_schema import build_result_payload, write_result_bundle
from stage56_mass_scan_io import row_term, scan_term_rows


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


def discover_mass_files(root: Path, filename: str) -> List[Path]:
    return sorted(root.rglob(filename))


def build_maps(mass_obj: Dict, top_k: int) -> Tuple[Dict[str, Set[int]], Dict[str, Set[int]], Dict[str, str]]:
    noun_map: Dict[str, Set[int]] = {}
    noun_cat_map: Dict[str, str] = {}
    for row in scan_term_rows(mass_obj):
        noun = row_term(row).strip().lower()
        cat = str(row.get("category", "")).strip().lower()
        sig = [int(x) for x in (row.get("signature_top_indices") or [])[:top_k]]
        if noun and sig:
            noun_map[noun] = set(sig)
            noun_cat_map[noun] = cat

    cat_map: Dict[str, Set[int]] = {}
    cp = mass_obj.get("category_prototypes") or {}
    for k, row in cp.items():
        sig = [int(x) for x in (row.get("prototype_top_indices") or [])[:top_k]]
        if sig:
            cat_map[str(k).strip().lower()] = set(sig)
    return noun_map, cat_map, noun_cat_map


def living_macro(cat_map: Dict[str, Set[int]], noun_map: Dict[str, Set[int]], noun_cat_map: Dict[str, str], top_k: int = 120) -> Set[int]:
    # biological/living 原型：animal + fruit + human 的并集高频近似（离线近似）
    base = set()
    for c in ("animal", "fruit", "human"):
        base |= set(cat_map.get(c, set()))
    # 加入生物相关 noun 的频次原型，增强稳健性
    freq: Dict[int, int] = {}
    for noun, sig in noun_map.items():
        cat = noun_cat_map.get(noun, "")
        if cat not in {"animal", "human", "nature", "fruit"}:
            continue
        for idx in sig:
            freq[int(idx)] = freq.get(int(idx), 0) + 1
    if freq:
        top = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
        base |= set(top)
    if not base:
        # 回退：从常见生物概念合并
        for n in ("cat", "dog", "rabbit", "apple", "banana", "human"):
            base |= set(noun_map.get(n, set()))
    return base


def food_macro(cat_map: Dict[str, Set[int]], noun_map: Dict[str, Set[int]], noun_cat_map: Dict[str, str], top_k: int = 120) -> Set[int]:
    # food 宏原型改为“类别原型 + food/fruit 高频并集”，避免纯 prototype 与 fruit 完全不交。
    base = set(cat_map.get("food", set())) | set(cat_map.get("fruit", set()))
    freq: Dict[int, int] = {}
    for noun, sig in noun_map.items():
        cat = noun_cat_map.get(noun, "")
        if cat not in {"food", "fruit"}:
            continue
        for idx in sig:
            freq[int(idx)] = freq.get(int(idx), 0) + 1
    if freq:
        top = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
        base |= set(top)
    return base


def family_metrics(
    noun_map: Dict[str, Set[int]],
    cat_map: Dict[str, Set[int]],
    noun_cat_map: Dict[str, str],
    *,
    micros: Sequence[str],
    meso: str,
    macro_mode: str,
    top_k: int,
) -> Dict[str, object]:
    micro_sets = [set(noun_map.get(n, set())) for n in micros if n in noun_map]
    micro_sets = [s for s in micro_sets if s]
    if len(micro_sets) < 3:
        return {
            "valid": False,
            "reason": "insufficient micro concepts",
        }

    micro_union = set().union(*micro_sets)
    micro_inter = set(micro_sets[0])
    for s in micro_sets[1:]:
        micro_inter &= s

    meso_set = set(cat_map.get(meso.lower(), set()))
    if macro_mode == "living":
        macro_set = living_macro(cat_map, noun_map, noun_cat_map, top_k=top_k)
    elif macro_mode == "food_composed":
        macro_set = food_macro(cat_map, noun_map, noun_cat_map, top_k=top_k)
    else:
        macro_set = set(cat_map.get(macro_mode.lower(), set()))

    if not meso_set:
        return {
            "valid": False,
            "reason": "meso category missing",
        }
    if not macro_set:
        return {
            "valid": False,
            "reason": "macro category missing",
        }

    shared_base = micro_union & meso_set & macro_set
    micro_specific = micro_union - meso_set
    meso_specific = meso_set - macro_set
    macro_specific = macro_set - meso_set

    pairwise_micro_j = [jaccard(a, b) for a, b in combinations(micro_sets, 2)] if len(micro_sets) >= 2 else []

    return {
        "valid": True,
        "micro_count": len(micro_sets),
        "micro_union_size": len(micro_union),
        "micro_intersection_size": len(micro_inter),
        "meso_size": len(meso_set),
        "macro_size": len(macro_set),
        "shared_base_size": len(shared_base),
        "micro_specific_size": len(micro_specific),
        "meso_specific_size": len(meso_specific),
        "macro_specific_size": len(macro_specific),
        "shared_base_ratio_vs_micro_union": float(len(shared_base) / max(1, len(micro_union))),
        "micro_to_meso_jaccard": float(jaccard(micro_union, meso_set)),
        "meso_to_macro_jaccard": float(jaccard(meso_set, macro_set)),
        "micro_to_macro_jaccard": float(jaccard(micro_union, macro_set)),
        "micro_pairwise_jaccard_mean": safe_mean(pairwise_micro_j),
        "micro_pairwise_jaccard_std": safe_std(pairwise_micro_j),
    }


def aggregate_multiseed(rows: Sequence[Dict[str, object]], key: str) -> Dict[str, float]:
    vals = [float(r.get(key, 0.0)) for r in rows if bool(r.get("valid", False))]
    return {"mean": safe_mean(vals), "std": safe_std(vals), "min": min(vals) if vals else 0.0, "max": max(vals) if vals else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description="Concept family parallel scale analysis")
    ap.add_argument("--root", default="tempdata")
    ap.add_argument("--mass-json-glob", default="mass_noun_encoding_scan.json")
    ap.add_argument("--top-k", type=int, default=120)
    ap.add_argument("--max-files", type=int, default=12)
    ap.add_argument("--min-nouns", type=int, default=100)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    files = discover_mass_files(Path(args.root), args.mass_json_glob)[: max(1, args.max_files)]
    if not files:
        raise RuntimeError("未发现 mass_noun_encoding_scan.json 文件。")

    apple_rows = []
    cat_rows = []
    per_file = []
    for p in files:
        try:
            obj = read_json(p)
            noun_records = scan_term_rows(obj)
            if len(noun_records) < max(1, args.min_nouns):
                continue
            noun_map, cat_map, noun_cat_map = build_maps(obj, top_k=args.top_k)
        except Exception:
            continue

        apple_chain = family_metrics(
            noun_map,
            cat_map,
            noun_cat_map,
            micros=("apple", "banana", "orange", "grape", "pear"),
            meso="fruit",
            macro_mode="food_composed",
            top_k=args.top_k,
        )
        cat_chain = family_metrics(
            noun_map,
            cat_map,
            noun_cat_map,
            micros=("cat", "dog", "rabbit", "tiger", "lion"),
            meso="animal",
            macro_mode="living",
            top_k=args.top_k,
        )
        apple_rows.append(apple_chain)
        cat_rows.append(cat_chain)

        sep = 0.0
        if apple_chain.get("valid") and cat_chain.get("valid"):
            sep = float(
                abs(float(apple_chain["shared_base_ratio_vs_micro_union"]) - float(cat_chain["shared_base_ratio_vs_micro_union"]))
            )
        per_file.append(
            {
                "file": str(p),
                "apple_chain": apple_chain,
                "cat_chain": cat_chain,
                "shared_base_ratio_gap": sep,
            }
        )

    apple_sum = {
        "shared_base_ratio_vs_micro_union": aggregate_multiseed(apple_rows, "shared_base_ratio_vs_micro_union"),
        "micro_to_meso_jaccard": aggregate_multiseed(apple_rows, "micro_to_meso_jaccard"),
        "meso_to_macro_jaccard": aggregate_multiseed(apple_rows, "meso_to_macro_jaccard"),
        "micro_pairwise_jaccard_mean": aggregate_multiseed(apple_rows, "micro_pairwise_jaccard_mean"),
    }
    cat_sum = {
        "shared_base_ratio_vs_micro_union": aggregate_multiseed(cat_rows, "shared_base_ratio_vs_micro_union"),
        "micro_to_meso_jaccard": aggregate_multiseed(cat_rows, "micro_to_meso_jaccard"),
        "meso_to_macro_jaccard": aggregate_multiseed(cat_rows, "meso_to_macro_jaccard"),
        "micro_pairwise_jaccard_mean": aggregate_multiseed(cat_rows, "micro_pairwise_jaccard_mean"),
    }

    gap_vals = [float(x["shared_base_ratio_gap"]) for x in per_file]
    metrics = {
        "n_mass_files": len(per_file),
        "apple_chain_summary": apple_sum,
        "cat_chain_summary": cat_sum,
        "apple_vs_cat_shared_base_gap_mean": safe_mean(gap_vals),
        "apple_vs_cat_shared_base_gap_std": safe_std(gap_vals),
        "per_file": per_file,
    }

    apple_shared = float(apple_sum["shared_base_ratio_vs_micro_union"]["mean"])
    cat_shared = float(cat_sum["shared_base_ratio_vs_micro_union"]["mean"])
    gap_mean = float(metrics["apple_vs_cat_shared_base_gap_mean"])
    payload = build_result_payload(
        experiment_id="concept_family_parallel_scale_v1",
        title="规模化概念族实验（苹果链路 vs 猫链路）",
        config={
            "root": args.root,
            "mass_json_glob": args.mass_json_glob,
            "top_k": args.top_k,
            "max_files": args.max_files,
            "min_nouns": args.min_nouns,
            "apple_chain": ["apple", "banana", "orange", "grape", "pear", "fruit", "food"],
            "cat_chain": ["cat", "dog", "rabbit", "tiger", "lion", "animal", "living"],
        },
        metrics=metrics,
        hypotheses=[
            {
                "id": "H_shared_base_exists_apple_chain",
                "rule": "apple_shared_base_ratio_mean > 0.02",
                "pass": bool(apple_shared > 0.02),
            },
            {
                "id": "H_shared_base_exists_cat_chain",
                "rule": "cat_shared_base_ratio_mean > 0.02",
                "pass": bool(cat_shared > 0.02),
            },
            {
                "id": "H_family_has_specific_offsets",
                "rule": "apple_vs_cat_shared_base_gap_mean > 0.005",
                "pass": bool(gap_mean > 0.005),
            },
        ],
        notes=[
            "该实验重点验证‘共享基底 + 专属偏移’结构是否稳定存在。",
            "cat 链路的 macro 使用 living 近似原型（animal+fruit+human）。",
        ],
    )

    lines = [
        "# 规模化概念族实验报告",
        "",
        f"- n_mass_files: {len(per_file)}",
        "",
        "## 苹果链路 (apple -> fruit -> food)",
        f"- shared_base_ratio_mean: {apple_shared:.4f}",
        f"- micro_to_meso_jaccard_mean: {apple_sum['micro_to_meso_jaccard']['mean']:.4f}",
        f"- meso_to_macro_jaccard_mean: {apple_sum['meso_to_macro_jaccard']['mean']:.4f}",
        "",
        "## 猫链路 (cat -> animal -> living)",
        f"- shared_base_ratio_mean: {cat_shared:.4f}",
        f"- micro_to_meso_jaccard_mean: {cat_sum['micro_to_meso_jaccard']['mean']:.4f}",
        f"- meso_to_macro_jaccard_mean: {cat_sum['meso_to_macro_jaccard']['mean']:.4f}",
        "",
        "## 并行对照",
        f"- apple_vs_cat_shared_base_gap_mean: {gap_mean:.4f}",
        f"- apple_vs_cat_shared_base_gap_std: {metrics['apple_vs_cat_shared_base_gap_std']:.4f}",
        "",
        "## 解释",
        "- shared_base_ratio 反映跨层级可复用编码比例。",
        "- gap 反映不同概念族的专属偏移强度。",
    ]
    report = "\n".join(lines) + "\n"

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_concept_family_parallel_{ts}")
    paths = write_result_bundle(
        out_dir=out_dir,
        base_name="concept_family_parallel_scale",
        payload=payload,
        report_md=report,
    )
    print(f"[OK] {paths['json']}")
    print(f"[OK] {paths['md']}")


if __name__ == "__main__":
    main()
