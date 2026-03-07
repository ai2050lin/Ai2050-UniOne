#!/usr/bin/env python3
"""Build an apple-centric encoding-law dossier from existing DeepSeek result artifacts.

This script does NOT rerun the large model. It aggregates already-produced outputs:
- multidim encoding probe (style/logic/syntax)
- concept family parallel scale (micro/meso/macro chain)
- triplet probe (apple/king/queen)
- mass noun scan (apple layer profile)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def normalize_layer_profile(dist: Dict[str, float], n_layers: int = 28) -> List[float]:
    vec = [0.0 for _ in range(n_layers)]
    for k, v in (dist or {}).items():
        try:
            idx = int(k)
        except Exception:
            continue
        if 0 <= idx < n_layers:
            vec[idx] = float(v)
    s = sum(vec)
    if s <= 0:
        return vec
    return [x / s for x in vec]


def layer_peak_band(profile: List[float]) -> str:
    if not profile:
        return "unknown"
    idx = max(range(len(profile)), key=lambda i: profile[i])
    n = len(profile)
    if idx < n // 3:
        return "early"
    if idx < (2 * n) // 3:
        return "middle"
    return "late"


def main() -> None:
    ap = argparse.ArgumentParser(description="Apple encoding law dossier from existing artifacts")
    ap.add_argument(
        "--multidim-json",
        default="tempdata/deepseek7b_multidim_encoding_probe_v2_specific/multidim_encoding_probe.json",
    )
    ap.add_argument(
        "--concept-family-json",
        default="tempdata/deepseek7b_concept_family_parallel_latest/concept_family_parallel_scale.json",
    )
    ap.add_argument(
        "--triplet-json",
        default="tempdata/deepseek7b_triplet_probe_20260306_150637/apple_king_queen_triplet_probe.json",
    )
    ap.add_argument(
        "--mass-json",
        default="tempdata/deepseek7b_mass_noun_scan_n120_mech_v4fix_seed101/mass_noun_encoding_scan.json",
    )
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    multidim = read_json(Path(args.multidim_json))
    concept = read_json(Path(args.concept_family_json))
    triplet = read_json(Path(args.triplet_json))
    mass = read_json(Path(args.mass_json))

    dims = multidim.get("dimensions", {})
    cross = multidim.get("cross_dimension", {})

    pair_abs = [float((dims.get(d) or {}).get("mean_pair_delta_abs", 0.0)) for d in ("style", "logic", "syntax")]
    style_logic_syntax_signal = safe_mean(pair_abs)

    jaccards = [float(v.get("top_neuron_jaccard", 0.0)) for v in cross.values() if isinstance(v, dict)]
    layer_corrs = [float(v.get("layer_profile_corr", 0.0)) for v in cross.values() if isinstance(v, dict)]

    mean_j = safe_mean(jaccards)
    mean_c = safe_mean(layer_corrs)
    decoupling_index = float((1.0 - mean_j) * 0.55 + (1.0 - mean_c) * 0.45)

    apple_chain = (((concept.get("metrics") or {}).get("apple_chain_summary") or {}))
    micro_to_meso = float((((apple_chain.get("micro_to_meso_jaccard") or {}).get("mean")) or 0.0))
    meso_to_macro = float((((apple_chain.get("meso_to_macro_jaccard") or {}).get("mean")) or 0.0))
    shared_base = float((((apple_chain.get("shared_base_ratio_vs_micro_union") or {}).get("mean")) or 0.0))

    tm = triplet.get("metrics", {})
    triplet_sep = float(tm.get("triplet_separability_index", 0.0))
    axis_spec = float(tm.get("axis_specificity_index", 0.0))
    kq_j = float(tm.get("king_queen_jaccard", 0.0))
    ak_j = float(tm.get("apple_king_jaccard", 0.0))

    noun_records = mass.get("noun_records", [])
    apple_rec = next((x for x in noun_records if str(x.get("noun", "")).lower() == "apple"), None)
    apple_layer_profile = normalize_layer_profile((apple_rec or {}).get("signature_layer_distribution") or {}, n_layers=mass.get("config", {}).get("n_layers", 28))
    apple_peak_band = layer_peak_band(apple_layer_profile)

    hypotheses = {
        "H1_parallel_axes_exist": bool(style_logic_syntax_signal > 0.5),
        "H2_axes_not_collapsed": bool(decoupling_index > 0.25),
        "H3_apple_hierarchy_closure": bool(meso_to_macro > micro_to_meso and shared_base > 0.0),
        "H4_local_linearity_triplet": bool(axis_spec > 0 and kq_j > ak_j),
        "H5_apple_has_layer_anchor": bool(max(apple_layer_profile) > 0.08),
    }

    payload = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "apple_multiaxis_encoding_law_dossier_v1",
        "title": "Apple 多轴编码规律档案（真实结果聚合）",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "multidim_json": args.multidim_json,
            "concept_family_json": args.concept_family_json,
            "triplet_json": args.triplet_json,
            "mass_json": args.mass_json,
        },
        "metrics": {
            "style_logic_syntax_signal": style_logic_syntax_signal,
            "cross_dim_mean_top_neuron_jaccard": mean_j,
            "cross_dim_mean_layer_profile_corr": mean_c,
            "cross_dim_decoupling_index": decoupling_index,
            "apple_micro_to_meso_jaccard_mean": micro_to_meso,
            "apple_meso_to_macro_jaccard_mean": meso_to_macro,
            "apple_shared_base_ratio_mean": shared_base,
            "triplet_separability_index": triplet_sep,
            "axis_specificity_index": axis_spec,
            "king_queen_jaccard": kq_j,
            "apple_king_jaccard": ak_j,
            "apple_layer_peak_band": apple_peak_band,
            "apple_layer_peak_value": max(apple_layer_profile) if apple_layer_profile else 0.0,
        },
        "hypotheses": [{"id": k, "pass": bool(v)} for k, v in hypotheses.items()],
        "notes": [
            "这是聚合分析，不是新一轮模型前向/消融；用于快速锁定下一步实验方向。",
            "若 H2 或 H5 失败，应优先补做 layer-wise 干预与多种子稳健性验证。",
        ],
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_apple_encoding_law_dossier_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "apple_multiaxis_encoding_law_dossier.json"
    out_md = out_dir / "APPLE_MULTIAXIS_ENCODING_LAW_DOSSIER_REPORT.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Apple 多轴编码规律档案",
        "",
        "## 核心指标",
        f"- style/logic/syntax 信号强度: {style_logic_syntax_signal:.4f}",
        f"- 交叉维度解耦指数: {decoupling_index:.4f}",
        f"- apple micro->meso jaccard: {micro_to_meso:.4f}",
        f"- apple meso->macro jaccard: {meso_to_macro:.4f}",
        f"- apple shared_base_ratio: {shared_base:.4f}",
        f"- triplet 可分离指数: {triplet_sep:.4f}",
        f"- 轴特异性指数: {axis_spec:.4f}",
        f"- king_queen vs apple_king jaccard: {kq_j:.4f} vs {ak_j:.4f}",
        f"- apple 层峰值区段: {apple_peak_band} (peak={max(apple_layer_profile) if apple_layer_profile else 0.0:.4f})",
        "",
        "## 规律判定",
    ]
    for h in payload["hypotheses"]:
        lines.append(f"- {h['id']}: {'PASS' if h['pass'] else 'FAIL'}")

    lines += [
        "",
        "## 解读",
        "- 若 meso_to_macro > micro_to_meso，说明苹果编码从属性层到实体层再到系统层形成层级闭包。",
        "- 若 axis_specificity 与 triplet_separability 同时为正，说明关系轴与实体轴可并存而不完全塌缩。",
    ]

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(out_dir),
        "json": str(out_json),
        "report": str(out_md),
        "key_metrics": payload["metrics"],
        "hypotheses": payload["hypotheses"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
