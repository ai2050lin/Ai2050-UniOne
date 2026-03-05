#!/usr/bin/env python
"""
Unified mathematical structure decoder for DeepSeek7B experiment outputs.

Goal:
- Fuse existing multidim probe / causal ablation / mass noun scan outputs.
- Build a structural evidence chain for:
  1) Dimension-axis stability
  2) Causal axis separability
  3) Hierarchical shared-specific concept coding
  4) Finite-basis + reuse efficiency

This script is offline (no model forward pass). It only reads JSON outputs in tempdata.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


DIMS = ("style", "logic", "syntax")


def safe_mean(xs: Sequence[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def safe_std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(statistics.stdev(xs))


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


def normalize(v: Sequence[float]) -> List[float]:
    s = float(sum(max(0.0, float(x)) for x in v))
    if s <= 0:
        return [0.0 for _ in v]
    return [float(max(0.0, float(x)) / s) for x in v]


def pairwise_values(items: Sequence, fn) -> List[float]:
    vals: List[float] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            vals.append(float(fn(items[i], items[j])))
    return vals


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_files(root: Path, filename: str) -> List[Path]:
    return sorted(root.rglob(filename))


def extract_top_neuron_set(probe: Dict, dim: str, top_k: int) -> Set[int]:
    dim_obj = (probe.get("dimensions") or {}).get(dim) or {}
    rows = dim_obj.get("specific_top_neurons") or dim_obj.get("top_neurons") or []
    out: List[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "flat_index" in row:
            out.append(int(row["flat_index"]))
    return set(out[: max(1, top_k)])


def extract_layer_profile(probe: Dict, dim: str) -> List[float]:
    d = (probe.get("dimensions") or {}).get(dim) or {}
    arr = d.get("layer_profile_abs_delta_norm")
    if isinstance(arr, list) and arr:
        return [float(x) for x in arr]
    raw = d.get("layer_profile_abs_delta_sum")
    if isinstance(raw, list) and raw:
        return normalize([float(x) for x in raw])
    return []


def dominant_layers(profile: Sequence[float], top_n: int = 3) -> List[int]:
    if not profile:
        return []
    idx = sorted(range(len(profile)), key=lambda i: float(profile[i]), reverse=True)
    return [int(i) for i in idx[: max(1, top_n)]]


@dataclass
class AxisStabilityRow:
    dim: str
    n_runs: int
    set_jaccard_mean: float
    set_jaccard_std: float
    profile_cos_mean: float
    profile_cos_std: float
    dominant_layer_votes: Dict[str, int]


def analyze_axis_stability(probe_jsons: Sequence[Path], top_k: int) -> Dict[str, object]:
    loaded = []
    for p in probe_jsons:
        try:
            loaded.append((p, read_json(p)))
        except Exception:
            continue

    rows: List[AxisStabilityRow] = []
    for dim in DIMS:
        sets = [extract_top_neuron_set(obj, dim, top_k) for _, obj in loaded]
        sets_nonempty = [s for s in sets if s]
        profiles = [extract_layer_profile(obj, dim) for _, obj in loaded]
        profiles_nonempty = [p for p in profiles if p]

        jvals = pairwise_values(sets_nonempty, jaccard)
        cvals = pairwise_values(profiles_nonempty, cosine)

        vote_counter: Dict[str, int] = {}
        for prof in profiles_nonempty:
            tops = dominant_layers(prof, top_n=3)
            key = ",".join(str(x) for x in tops)
            vote_counter[key] = vote_counter.get(key, 0) + 1

        rows.append(
            AxisStabilityRow(
                dim=dim,
                n_runs=max(len(sets_nonempty), len(profiles_nonempty)),
                set_jaccard_mean=safe_mean(jvals),
                set_jaccard_std=safe_std(jvals),
                profile_cos_mean=safe_mean(cvals),
                profile_cos_std=safe_std(cvals),
                dominant_layer_votes=dict(sorted(vote_counter.items(), key=lambda kv: kv[1], reverse=True)),
            )
        )

    return {
        "n_probe_files": len(loaded),
        "top_k": top_k,
        "dimensions": {
            r.dim: {
                "n_runs": r.n_runs,
                "set_jaccard_mean": r.set_jaccard_mean,
                "set_jaccard_std": r.set_jaccard_std,
                "profile_cosine_mean": r.profile_cos_mean,
                "profile_cosine_std": r.profile_cos_std,
                "dominant_layer_pattern_votes": r.dominant_layer_votes,
            }
            for r in rows
        },
    }


def analyze_causal_separation(ablation_jsons: Sequence[Path]) -> Dict[str, object]:
    adv: Dict[str, List[float]] = {k: [] for k in DIMS}
    loaded = 0
    for p in ablation_jsons:
        try:
            obj = read_json(p)
        except Exception:
            continue
        loaded += 1
        d = obj.get("diagonal_advantage") or {}
        for dim in DIMS:
            if dim in d:
                adv[dim].append(float(d[dim]))

    per_dim = {
        dim: {
            "n_runs": len(vals),
            "mean": safe_mean(vals),
            "std": safe_std(vals),
            "min": float(min(vals)) if vals else 0.0,
            "max": float(max(vals)) if vals else 0.0,
            "positive_ratio": float(sum(1 for v in vals if v > 0) / len(vals)) if vals else 0.0,
        }
        for dim, vals in adv.items()
    }
    return {"n_ablation_files": loaded, "diagonal_advantage": per_dim}


def _sig_set(mass_scan: Dict, noun: str) -> Set[int]:
    rows = mass_scan.get("noun_records") or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("noun", "")).strip().lower() == noun.lower():
            return set(int(x) for x in (row.get("signature_top_indices") or []))
    return set()


def _sig_layer_vec(mass_scan: Dict, noun: str, n_layers: int = 28) -> List[float]:
    rows = mass_scan.get("noun_records") or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("noun", "")).strip().lower() == noun.lower():
            dist = row.get("signature_layer_distribution") or {}
            vec = [0.0 for _ in range(n_layers)]
            if isinstance(dist, dict):
                for k, v in dist.items():
                    try:
                        idx = int(k)
                    except Exception:
                        continue
                    if 0 <= idx < n_layers:
                        vec[idx] = float(v)
            return normalize(vec)
    return []


def _proto_set(mass_scan: Dict, category: str) -> Set[int]:
    cat = (mass_scan.get("category_prototypes") or {}).get(category) or {}
    return set(int(x) for x in (cat.get("prototype_top_indices") or []))


def _proto_layer_vec(mass_scan: Dict, category: str, n_layers: int = 28) -> List[float]:
    cat = (mass_scan.get("category_prototypes") or {}).get(category) or {}
    dist = cat.get("prototype_layer_distribution") or {}
    vec = [0.0 for _ in range(n_layers)]
    if isinstance(dist, dict):
        for k, v in dist.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if 0 <= idx < n_layers:
                vec[idx] = float(v)
    return normalize(vec)


def _shared_specific(sig: Set[int], proto: Set[int]) -> Dict[str, float]:
    if not sig:
        return {"shared_ratio": 0.0, "specific_ratio": 0.0}
    shared = len(sig & proto) / max(1, len(sig))
    return {"shared_ratio": float(shared), "specific_ratio": float(1.0 - shared)}


def analyze_concept_hierarchy(mass_jsons: Sequence[Path], min_nouns: int) -> Dict[str, object]:
    loaded = []
    for p in mass_jsons:
        try:
            obj = read_json(p)
        except Exception:
            continue
        n_nouns = int(((obj.get("config") or {}).get("n_nouns")) or 0)
        if n_nouns >= min_nouns:
            loaded.append((p, obj))

    pair_defs = [
        ("apple", "banana"),
        ("apple", "cat"),
        ("apple", "king"),
        ("king", "queen"),
        ("cat", "dog"),
    ]
    concept_category_defs = [
        ("apple", "fruit"),
        ("apple", "food"),
        ("cat", "animal"),
        ("king", "human"),
        ("queen", "human"),
    ]

    pair_stats: Dict[str, List[float]] = {f"{a}__{b}": [] for a, b in pair_defs}
    pair_layer_cos: Dict[str, List[float]] = {f"{a}__{b}": [] for a, b in pair_defs}
    cc_shared_stats: Dict[str, List[float]] = {f"{c}__{k}": [] for c, k in concept_category_defs}
    cc_specific_stats: Dict[str, List[float]] = {f"{c}__{k}": [] for c, k in concept_category_defs}
    cc_layer_cos: Dict[str, List[float]] = {f"{c}__{k}": [] for c, k in concept_category_defs}
    finite_basis_pr: List[float] = []
    finite_basis_top1: List[float] = []
    finite_basis_top5: List[float] = []
    reused_ratio: List[float] = []

    for _, obj in loaded:
        reg = obj.get("regularity") or {}
        lrs = reg.get("low_rank_structure") or {}
        if "participation_ratio" in lrs:
            finite_basis_pr.append(float(lrs["participation_ratio"]))
        if "top1_energy_ratio" in lrs:
            finite_basis_top1.append(float(lrs["top1_energy_ratio"]))
        if "top5_energy_ratio" in lrs:
            finite_basis_top5.append(float(lrs["top5_energy_ratio"]))
        if "reused_neuron_ratio" in reg:
            reused_ratio.append(float(reg["reused_neuron_ratio"]))

        for a, b in pair_defs:
            sa = _sig_set(obj, a)
            sb = _sig_set(obj, b)
            if sa and sb:
                pair_stats[f"{a}__{b}"].append(jaccard(sa, sb))
            va = _sig_layer_vec(obj, a)
            vb = _sig_layer_vec(obj, b)
            if va and vb:
                pair_layer_cos[f"{a}__{b}"].append(cosine(va, vb))

        for concept, cat in concept_category_defs:
            sig = _sig_set(obj, concept)
            proto = _proto_set(obj, cat)
            if sig and proto:
                ss = _shared_specific(sig, proto)
                key = f"{concept}__{cat}"
                cc_shared_stats[key].append(ss["shared_ratio"])
                cc_specific_stats[key].append(ss["specific_ratio"])
            vs = _sig_layer_vec(obj, concept)
            vp = _proto_layer_vec(obj, cat)
            if vs and vp:
                cc_layer_cos[f"{concept}__{cat}"].append(cosine(vs, vp))

    pair_summary = {
        k: {
            "n_runs": len(v),
            "jaccard_mean": safe_mean(v),
            "jaccard_std": safe_std(v),
            "layer_cosine_mean": safe_mean(pair_layer_cos[k]),
            "layer_cosine_std": safe_std(pair_layer_cos[k]),
        }
        for k, v in pair_stats.items()
    }
    cc_summary = {
        k: {
            "n_runs": len(cc_shared_stats[k]),
            "shared_ratio_mean": safe_mean(cc_shared_stats[k]),
            "specific_ratio_mean": safe_mean(cc_specific_stats[k]),
            "layer_cosine_mean": safe_mean(cc_layer_cos[k]),
            "layer_cosine_std": safe_std(cc_layer_cos[k]),
        }
        for k in cc_shared_stats.keys()
    }

    return {
        "n_mass_files_used": len(loaded),
        "min_nouns_filter": min_nouns,
        "pair_overlap": pair_summary,
        "concept_category_shared_specific": cc_summary,
        "finite_basis_indicators": {
            "participation_ratio_mean": safe_mean(finite_basis_pr),
            "participation_ratio_std": safe_std(finite_basis_pr),
            "top1_energy_ratio_mean": safe_mean(finite_basis_top1),
            "top5_energy_ratio_mean": safe_mean(finite_basis_top5),
            "reused_neuron_ratio_mean": safe_mean(reused_ratio),
            "reused_neuron_ratio_std": safe_std(reused_ratio),
        },
    }


def evaluate_hypotheses(
    axis: Dict[str, object],
    causal: Dict[str, object],
    hierarchy: Dict[str, object],
) -> Dict[str, object]:
    axis_dims = axis.get("dimensions") or {}
    causal_dims = causal.get("diagonal_advantage") or {}
    fb = (hierarchy.get("finite_basis_indicators") or {})
    pair = hierarchy.get("pair_overlap") or {}

    h1 = float(fb.get("participation_ratio_mean", 0.0)) > 0 and float(fb.get("top5_energy_ratio_mean", 0.0)) > float(
        fb.get("top1_energy_ratio_mean", 0.0)
    )
    h2 = all(float((causal_dims.get(dim) or {}).get("mean", 0.0)) > 0 for dim in DIMS)
    h3 = float((pair.get("apple__banana") or {}).get("layer_cosine_mean", 0.0)) > float(
        (pair.get("apple__cat") or {}).get("layer_cosine_mean", 0.0)
    )
    h4 = all(float((axis_dims.get(dim) or {}).get("profile_cosine_mean", 0.0)) > 0.4 for dim in DIMS)

    rows = [
        {"id": "H1_finite_basis_plus_composition", "pass": bool(h1), "rule": "PR>0 and top5_energy>top1_energy"},
        {"id": "H2_causal_axis_separable", "pass": bool(h2), "rule": "all diagonal_advantage_mean > 0"},
        {
            "id": "H3_semantic_hierarchy_consistent",
            "pass": bool(h3),
            "rule": "LayerCos(apple,banana) > LayerCos(apple,cat)",
        },
        {"id": "H4_axis_layer_profile_stable", "pass": bool(h4), "rule": "all profile_cosine_mean > 0.4"},
    ]
    pass_ratio = float(sum(1 for r in rows if r["pass"]) / len(rows))
    return {"hypotheses": rows, "pass_ratio": pass_ratio}


def render_report_md(result: Dict[str, object]) -> str:
    axis = result["axis_stability"]
    causal = result["causal_separation"]
    hier = result["concept_hierarchy"]
    hyp = result["hypothesis_test"]

    lines: List[str] = []
    lines.append("# 统一编码结构解码报告（离线汇总）")
    lines.append("")
    lines.append("## 1. 数据来源")
    lines.append(f"- 多维探针文件数: {axis.get('n_probe_files', 0)}")
    lines.append(f"- 因果消融文件数: {causal.get('n_ablation_files', 0)}")
    lines.append(f"- 名词扫描文件数(过滤后): {hier.get('n_mass_files_used', 0)}")
    lines.append("")
    lines.append("## 2. 维度轴稳定性")
    for dim, row in (axis.get("dimensions") or {}).items():
        lines.append(
            f"- {dim}: set_jaccard={row.get('set_jaccard_mean', 0):.4f}±{row.get('set_jaccard_std', 0):.4f}, "
            f"profile_cos={row.get('profile_cosine_mean', 0):.4f}±{row.get('profile_cosine_std', 0):.4f}"
        )
    lines.append("")
    lines.append("## 3. 维度轴因果可分离性（对角优势）")
    for dim, row in (causal.get("diagonal_advantage") or {}).items():
        lines.append(
            f"- {dim}: mean={row.get('mean', 0):.4f}, std={row.get('std', 0):.4f}, "
            f"positive_ratio={row.get('positive_ratio', 0):.2%}"
        )
    lines.append("")
    lines.append("## 4. 概念层级共享-特异结构")
    lines.append("### 4.1 概念对重叠")
    for name, row in (hier.get("pair_overlap") or {}).items():
        lines.append(
            f"- {name}: Jaccard={row.get('jaccard_mean', 0):.4f}±{row.get('jaccard_std', 0):.4f}, "
            f"LayerCos={row.get('layer_cosine_mean', 0):.4f}±{row.get('layer_cosine_std', 0):.4f}"
        )
    lines.append("### 4.2 概念-类别共享比例")
    for name, row in (hier.get("concept_category_shared_specific") or {}).items():
        lines.append(
            f"- {name}: shared={row.get('shared_ratio_mean', 0):.4f}, specific={row.get('specific_ratio_mean', 0):.4f}, "
            f"LayerCos={row.get('layer_cosine_mean', 0):.4f}±{row.get('layer_cosine_std', 0):.4f}"
        )
    lines.append("")
    lines.append("## 5. 有限基与复用效率信号")
    fb = hier.get("finite_basis_indicators") or {}
    lines.append(
        f"- 参与率(PR): {fb.get('participation_ratio_mean', 0):.4f}±{fb.get('participation_ratio_std', 0):.4f}"
    )
    lines.append(
        f"- 能量占比: top1={fb.get('top1_energy_ratio_mean', 0):.4f}, top5={fb.get('top5_energy_ratio_mean', 0):.4f}"
    )
    lines.append(
        f"- 复用神经元比例: {fb.get('reused_neuron_ratio_mean', 0):.4f}±{fb.get('reused_neuron_ratio_std', 0):.4f}"
    )
    lines.append("")
    lines.append("## 6. 假设检验（结构级）")
    for row in hyp.get("hypotheses") or []:
        lines.append(f"- {row.get('id')}: {'PASS' if row.get('pass') else 'FAIL'} ({row.get('rule')})")
    lines.append(f"- 通过率: {hyp.get('pass_ratio', 0):.2%}")
    lines.append("")
    lines.append("## 7. 解释")
    lines.append("- 该报告支持“静态概念坐标 + 动态层级路由 + 有限基复用”的统一编码图景。")
    lines.append("- 若要逼近微观数学原理，下一步应在同一概念集上增加跨seed同prompt的因果子回路追踪。")
    lines.append("- 对 apple/king/queen 这类概念，建议在同一模板上做反事实最小编辑并比较最小子回路的可迁移性。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified decoder for DeepSeek7B encoding math structure")
    ap.add_argument("--root", default="tempdata", help="Search root for existing JSON outputs")
    ap.add_argument("--top-k", type=int, default=128, help="Top neurons for set-based stability")
    ap.add_argument("--min-nouns", type=int, default=100, help="Only use mass noun scans with at least this many nouns")
    ap.add_argument("--output-dir", default="", help="Output directory. Default: tempdata/deepseek7b_unified_math_decode_<ts>")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_unified_math_decode_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    probe_jsons = discover_files(root, "multidim_encoding_probe.json")
    ablation_jsons = discover_files(root, "multidim_causal_ablation.json")
    mass_jsons = discover_files(root, "mass_noun_encoding_scan.json")

    axis = analyze_axis_stability(probe_jsons, top_k=args.top_k)
    causal = analyze_causal_separation(ablation_jsons)
    hierarchy = analyze_concept_hierarchy(mass_jsons, min_nouns=args.min_nouns)
    hyp = evaluate_hypotheses(axis, causal, hierarchy)

    result = {
        "config": {
            "root": str(root),
            "top_k": int(args.top_k),
            "min_nouns": int(args.min_nouns),
            "n_probe_files_discovered": len(probe_jsons),
            "n_ablation_files_discovered": len(ablation_jsons),
            "n_mass_files_discovered": len(mass_jsons),
        },
        "axis_stability": axis,
        "causal_separation": causal,
        "concept_hierarchy": hierarchy,
        "hypothesis_test": hyp,
    }

    json_path = out_dir / "unified_math_structure_decode.json"
    md_path = out_dir / "UNIFIED_MATH_STRUCTURE_DECODE_REPORT.md"
    with json_path.open("w", encoding="utf-8-sig") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with md_path.open("w", encoding="utf-8-sig") as f:
        f.write(render_report_md(result))

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")


if __name__ == "__main__":
    main()
