from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_native_direct_measure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_native_direct_measure_summary() -> dict:
    native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_nativeization_20260320" / "summary.json"
    )
    neuron = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_direct_refinement_20260320" / "summary.json"
    )
    structure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_refinement_20260320" / "summary.json"
    )
    sparse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
    )

    hn = native["headline_metrics"]
    hs = structure["headline_metrics"]
    hseed = neuron["headline_metrics"]
    hsp = sparse["headline_metrics"]

    direct_origin_measure = _clip01((hn["origin_nativeization"] + hseed["origin_stability_v2"]) / 2.0)
    direct_feature_measure = _clip01((hn["feature_nativeization"] + hsp["sparse_feature_activation"]) / 2.0)
    direct_structure_measure = _clip01((hn["structure_nativeization"] + hs["structure_direct_confidence_v3"]) / 2.0)
    direct_route_measure = _clip01((hn["route_nativeization"] + hsp["sparse_route_activation"]) / 2.0)
    direct_brain_measure = (direct_origin_measure + direct_feature_measure + direct_structure_measure + direct_route_measure) / 4.0
    direct_brain_gap = 1.0 - direct_brain_measure

    return {
        "headline_metrics": {
            "direct_origin_measure": direct_origin_measure,
            "direct_feature_measure": direct_feature_measure,
            "direct_structure_measure": direct_structure_measure,
            "direct_route_measure": direct_route_measure,
            "direct_brain_measure": direct_brain_measure,
            "direct_brain_gap": direct_brain_gap,
        },
        "direct_equation": {
            "origin_term": "D_origin = mean(N_origin_native, S_origin_v2)",
            "feature_term": "D_feature = mean(N_feature_native, A_feature)",
            "structure_term": "D_structure = mean(N_structure_native, C_struct_v3)",
            "route_term": "D_route = mean(N_route_native, A_route)",
            "system_term": "M_brain_direct = mean(D_origin, D_feature, D_structure, D_route)",
        },
        "project_readout": {
            "summary": "逆向脑编码链当前已经从近原生链条进一步推进到了近直测层，但结构和路线两层仍然是更弱环节。",
            "next_question": "下一步要把近直测层继续推进到更接近原生回路测量，而不是继续停留在聚合比值层。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码近直测报告",
        "",
        f"- direct_origin_measure: {hm['direct_origin_measure']:.6f}",
        f"- direct_feature_measure: {hm['direct_feature_measure']:.6f}",
        f"- direct_structure_measure: {hm['direct_structure_measure']:.6f}",
        f"- direct_route_measure: {hm['direct_route_measure']:.6f}",
        f"- direct_brain_measure: {hm['direct_brain_measure']:.6f}",
        f"- direct_brain_gap: {hm['direct_brain_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_native_direct_measure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
