#!/usr/bin/env python
"""
任务3：统一坐标系测试。

目标：
- 将 style / logic / syntax / concept 放入同一可解码空间。
- 计算正交性（orthogonality）与耦合项（coupling）。

数据来源：
- tempdata 下已有 multidim probe / multidim causal ablation / mass_noun_encoding_scan 结果。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from agi_research_result_schema import build_result_payload, write_result_bundle


DIMS = ("style", "logic", "syntax")


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(xs: Sequence[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def safe_std(xs: Sequence[float]) -> float:
    return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


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


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def discover_files(root: Path, filename: str) -> List[Path]:
    return sorted(root.rglob(filename))


def extract_probe_stats(probe_files: Sequence[Path], top_k: int) -> Dict[str, object]:
    layer_profiles: Dict[str, List[List[float]]] = {d: [] for d in DIMS}
    top_sets: Dict[str, List[List[int]]] = {d: [] for d in DIMS}
    loaded = 0
    for p in probe_files:
        try:
            obj = read_json(p)
        except Exception:
            continue
        dims = obj.get("dimensions") or {}
        if not isinstance(dims, dict):
            continue
        loaded += 1
        for d in DIMS:
            dd = dims.get(d) or {}
            prof = dd.get("layer_profile_abs_delta_norm") or dd.get("layer_profile_abs_delta_sum") or []
            if isinstance(prof, list) and prof:
                layer_profiles[d].append(normalize([float(x) for x in prof]))
            rows = dd.get("specific_top_neurons") or dd.get("generic_top_neurons") or dd.get("top_neurons") or []
            flat = []
            for r in rows:
                if isinstance(r, dict) and "flat_index" in r:
                    flat.append(int(r["flat_index"]))
            if flat:
                top_sets[d].append(flat[:top_k])

    # 聚合 profile：按维度求均值
    agg_profiles: Dict[str, List[float]] = {}
    for d in DIMS:
        arr = layer_profiles[d]
        if not arr:
            agg_profiles[d] = []
            continue
        n = min(len(x) for x in arr)
        vec = []
        for i in range(n):
            vec.append(float(sum(float(x[i]) for x in arr) / len(arr)))
        agg_profiles[d] = normalize(vec)

    # 聚合 top set：按出现频次排序
    agg_top: Dict[str, List[int]] = {}
    for d in DIMS:
        freq: Dict[int, int] = {}
        for s in top_sets[d]:
            for x in s:
                freq[int(x)] = freq.get(int(x), 0) + 1
        ordered = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
        agg_top[d] = [int(k) for k, _ in ordered[:top_k]]

    # 正交性矩阵
    layer_cos = {a: {} for a in DIMS}
    neuron_jac = {a: {} for a in DIMS}
    offdiag_abs = []
    for a in DIMS:
        for b in DIMS:
            c = cosine(agg_profiles.get(a, []), agg_profiles.get(b, []))
            j = jaccard(agg_top.get(a, []), agg_top.get(b, []))
            layer_cos[a][b] = float(c)
            neuron_jac[a][b] = float(j)
            if a != b:
                offdiag_abs.append(abs(float(c)))
                offdiag_abs.append(abs(float(j)))

    orthogonality_index = float(1.0 - safe_mean(offdiag_abs))
    return {
        "n_probe_files": loaded,
        "aggregated_layer_profiles": agg_profiles,
        "aggregated_top_neurons": agg_top,
        "layer_cosine_matrix": layer_cos,
        "neuron_jaccard_matrix": neuron_jac,
        "orthogonality_index": orthogonality_index,
    }


def extract_ablation_coupling(ablation_files: Sequence[Path]) -> Dict[str, object]:
    diag_vals: List[float] = []
    off_vals: List[float] = []
    loaded = 0
    per_run = []

    for p in ablation_files:
        try:
            obj = read_json(p)
        except Exception:
            continue
        sm = obj.get("suppression_matrix_mean") or {}
        if not isinstance(sm, dict):
            continue
        loaded += 1
        run_diag = []
        run_off = []
        for a in DIMS:
            row = sm.get(a) or {}
            for b in DIMS:
                v = float((row.get(b) if isinstance(row, dict) else 0.0) or 0.0)
                if a == b:
                    run_diag.append(v)
                else:
                    run_off.append(v)
        diag_vals.extend(run_diag)
        off_vals.extend(run_off)
        coupling_ratio = float(safe_mean([abs(x) for x in run_off]) / (safe_mean([abs(x) for x in run_diag]) + 1e-12))
        per_run.append(
            {
                "file": str(p),
                "diag_mean": safe_mean(run_diag),
                "off_mean": safe_mean(run_off),
                "coupling_ratio": coupling_ratio,
            }
        )

    diag_abs = safe_mean([abs(x) for x in diag_vals])
    off_abs = safe_mean([abs(x) for x in off_vals])
    coupling_ratio = float(off_abs / (diag_abs + 1e-12))
    decoupling_score = float(max(0.0, 1.0 - coupling_ratio))
    return {
        "n_ablation_files": loaded,
        "diag_abs_mean": diag_abs,
        "offdiag_abs_mean": off_abs,
        "coupling_ratio": coupling_ratio,
        "decoupling_score": decoupling_score,
        "per_run": per_run,
    }


def concept_axes_from_mass(mass_json: Path) -> Dict[str, List[float]]:
    obj = read_json(mass_json)
    cp = obj.get("category_prototypes") or {}
    cats = {}
    for name in ("fruit", "animal", "food", "object", "human", "tech"):
        row = cp.get(name) or {}
        dist = row.get("prototype_layer_distribution") or {}
        if not isinstance(dist, dict):
            continue
        n_layers = 28
        vec = [0.0 for _ in range(n_layers)]
        for k, v in dist.items():
            try:
                i = int(k)
            except Exception:
                continue
            if 0 <= i < n_layers:
                vec[i] = float(v)
        cats[name] = normalize(vec)

    axes = {}
    if "fruit" in cats and "animal" in cats:
        axes["concept_fruit_vs_animal"] = [float(a - b) for a, b in zip(cats["fruit"], cats["animal"])]
    if "food" in cats and "object" in cats:
        axes["concept_food_vs_object"] = [float(a - b) for a, b in zip(cats["food"], cats["object"])]
    if "human" in cats and "tech" in cats:
        axes["concept_human_vs_tech"] = [float(a - b) for a, b in zip(cats["human"], cats["tech"])]
    return axes


def concept_coupling_with_dims(concept_axes: Dict[str, List[float]], dim_profiles: Dict[str, List[float]]) -> Dict[str, object]:
    out = {}
    vals = []
    for an, av in concept_axes.items():
        row = {}
        for d in DIMS:
            c = float(cosine(av, dim_profiles.get(d, [])))
            row[d] = c
            vals.append(abs(c))
        out[an] = row
    return {
        "axis_dim_cosine": out,
        "concept_dim_coupling_abs_mean": safe_mean(vals),
        "concept_dim_coupling_abs_std": safe_std(vals),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified coordinate system test")
    ap.add_argument("--root", default="tempdata")
    ap.add_argument(
        "--mass-json",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    )
    ap.add_argument("--top-k", type=int, default=128)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    root = Path(args.root)
    probe_files = discover_files(root, "multidim_encoding_probe.json")
    ablation_files = discover_files(root, "multidim_causal_ablation.json")
    probe_stats = extract_probe_stats(probe_files, top_k=args.top_k)
    ablation_stats = extract_ablation_coupling(ablation_files)
    concept_axes = concept_axes_from_mass(Path(args.mass_json))
    concept_stats = concept_coupling_with_dims(
        concept_axes=concept_axes,
        dim_profiles=probe_stats.get("aggregated_layer_profiles") or {},
    )

    metrics = {
        "probe_orthogonality": probe_stats,
        "ablation_coupling": ablation_stats,
        "concept_dim_alignment": concept_stats,
    }

    orth = float(probe_stats.get("orthogonality_index", 0.0))
    dec = float(ablation_stats.get("decoupling_score", 0.0))
    cpl = float(concept_stats.get("concept_dim_coupling_abs_mean", 0.0))
    unified_score = float((orth + dec + (1.0 - min(1.0, cpl))) / 3.0)
    metrics["unified_coordinate_score"] = unified_score

    payload = build_result_payload(
        experiment_id="unified_coordinate_system_test_v1",
        title="统一坐标系测试（风格/逻辑/语法/概念）",
        config={
            "root": str(root),
            "mass_json": args.mass_json,
            "top_k": args.top_k,
            "probe_files": len(probe_files),
            "ablation_files": len(ablation_files),
        },
        metrics=metrics,
        hypotheses=[
            {
                "id": "H_axis_orthogonal",
                "rule": "orthogonality_index > 0.50",
                "pass": bool(orth > 0.50),
            },
            {
                "id": "H_low_cross_coupling",
                "rule": "decoupling_score > 0.50",
                "pass": bool(dec > 0.50),
            },
            {
                "id": "H_concept_axis_not_entangled",
                "rule": "concept_dim_coupling_abs_mean < 0.35",
                "pass": bool(cpl < 0.35),
            },
        ],
        notes=[
            "若正交性低但可解码，则说明维度更偏‘可分离但耦合’，不是严格正交。",
            "统一坐标系用于后续将语言/逻辑/语法/概念联立建模。",
        ],
    )

    lines = [
        "# 统一坐标系测试报告",
        "",
        "## 关键分数",
        f"- orthogonality_index: {orth:.4f}",
        f"- decoupling_score: {dec:.4f}",
        f"- concept_dim_coupling_abs_mean: {cpl:.4f}",
        f"- unified_coordinate_score: {unified_score:.4f}",
        "",
        "## 解释",
        "- orthogonality_index 越高，维度在神经元集合与层分布上越接近正交。",
        "- decoupling_score 越高，因果干预的对角优势越强，跨维干扰越弱。",
        "- concept_dim_coupling_abs_mean 越低，概念轴与 style/logic/syntax 耦合越弱。",
    ]
    report = "\n".join(lines) + "\n"

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_unified_coordinate_{ts}")
    paths = write_result_bundle(
        out_dir=out_dir,
        base_name="unified_coordinate_system_test",
        payload=payload,
        report_md=report,
    )
    print(f"[OK] {paths['json']}")
    print(f"[OK] {paths['md']}")


if __name__ == "__main__":
    main()

