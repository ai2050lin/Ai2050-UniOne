from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v2_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v2_summary() -> dict:
    direct = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_native_direct_measure_20260320" / "summary.json"
    )
    native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_nativeization_20260320" / "summary.json"
    )
    structure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_refinement_20260320" / "summary.json"
    )
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_contextual_trainable_prototype_20260320" / "summary.json"
    )
    expanded = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_object_attribute_structure_prototype_20260320" / "summary.json"
    )
    sparse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
    )

    hd = direct["headline_metrics"]
    hn = native["headline_metrics"]
    hs = structure["headline_metrics"]
    hp = proto["headline_metrics"]
    he = expanded["headline_metrics"]
    hsp = sparse["headline_metrics"]

    direct_origin_measure_v2 = _clip01((hd["direct_origin_measure"] + hn["origin_nativeization"] + hsp["sparse_seed_activation"]) / 3.0)
    direct_feature_measure_v2 = _clip01((hd["direct_feature_measure"] + hp["shared_red_consistency"] + hn["feature_nativeization"]) / 3.0)
    direct_structure_measure_v2 = _clip01((hd["direct_structure_measure"] + hs["structure_direct_confidence_v3"] + hp["route_split_consistency"]) / 3.0)
    direct_route_measure_v2 = _clip01((hd["direct_route_measure"] + hp["context_split_consistency"] + he["object_route_split"]) / 3.0)
    direct_brain_measure_v2 = (
        direct_origin_measure_v2
        + direct_feature_measure_v2
        + direct_structure_measure_v2
        + direct_route_measure_v2
    ) / 4.0
    direct_brain_gap_v2 = 1.0 - direct_brain_measure_v2
    structure_route_balance_v2 = (direct_structure_measure_v2 + direct_route_measure_v2) / 2.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v2": direct_origin_measure_v2,
            "direct_feature_measure_v2": direct_feature_measure_v2,
            "direct_structure_measure_v2": direct_structure_measure_v2,
            "direct_route_measure_v2": direct_route_measure_v2,
            "direct_brain_measure_v2": direct_brain_measure_v2,
            "direct_brain_gap_v2": direct_brain_gap_v2,
            "structure_route_balance_v2": structure_route_balance_v2,
        },
        "direct_equation_v2": {
            "origin_term": "D_origin_v2 = mean(D_origin, N_origin_native, A_seed)",
            "feature_term": "D_feature_v2 = mean(D_feature, H_red_ctx, N_feature_native)",
            "structure_term": "D_structure_v2 = mean(D_structure, C_struct_v3, route_split_consistency)",
            "route_term": "D_route_v2 = mean(D_route, context_split_consistency, object_route_split)",
            "system_term": "M_brain_direct_v2 = mean(D_origin_v2, D_feature_v2, D_structure_v2, D_route_v2)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测强化第二版把结构层和路线层重新并入原型路由一致性之后，整体近直测强度有所提升，但结构层仍然是偏弱环节。",
            "next_question": "下一步要把结构层和路线层继续推进到更接近原生回路直测，而不是继续停在跨对象的一致性聚合上。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第二版报告",
        "",
        f"- direct_origin_measure_v2: {hm['direct_origin_measure_v2']:.6f}",
        f"- direct_feature_measure_v2: {hm['direct_feature_measure_v2']:.6f}",
        f"- direct_structure_measure_v2: {hm['direct_structure_measure_v2']:.6f}",
        f"- direct_route_measure_v2: {hm['direct_route_measure_v2']:.6f}",
        f"- direct_brain_measure_v2: {hm['direct_brain_measure_v2']:.6f}",
        f"- direct_brain_gap_v2: {hm['direct_brain_gap_v2']:.6f}",
        f"- structure_route_balance_v2: {hm['structure_route_balance_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v2_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
