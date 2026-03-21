from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_reverse_analysis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_reverse_summary() -> dict:
    neuron = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_direct_refinement_20260320" / "summary.json"
    )
    structure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_refinement_20260320" / "summary.json"
    )
    sparse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
    )
    color = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_color_fiber_nativeization_20260320" / "summary.json"
    )
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_object_attribute_coupling_prototype_20260320" / "summary.json"
    )

    hn = neuron["headline_metrics"]
    hs = structure["headline_metrics"]
    hsp = sparse["headline_metrics"]
    hc = color["headline_metrics"]
    hp = proto["headline_metrics"]

    origin_recovery = _clip01(hn["origin_stability_v2"])
    feature_recovery = _clip01((hsp["sparse_feature_activation"] + hc["native_color_binding"]) / 2.0)
    structure_recovery = _clip01((hs["structure_direct_confidence_v3"] + hsp["sparse_structure_activation"]) / 2.0)
    route_recovery = _clip01((hp["same_attribute_different_route"] + hsp["sparse_route_activation"]) / 2.0)
    reverse_chain_strength = (origin_recovery + feature_recovery + structure_recovery + route_recovery) / 4.0
    reverse_chain_gap = 1.0 - reverse_chain_strength

    return {
        "headline_metrics": {
            "origin_recovery": origin_recovery,
            "feature_recovery": feature_recovery,
            "structure_recovery": structure_recovery,
            "route_recovery": route_recovery,
            "reverse_chain_strength": reverse_chain_strength,
            "reverse_chain_gap": reverse_chain_gap,
        },
        "reverse_equation": {
            "origin_term": "R_origin = S_origin_v2",
            "feature_term": "R_feature = mean(A_feature, B_red)",
            "structure_term": "R_structure = mean(C_struct_v3, A_structure)",
            "route_term": "R_route = mean(R_split, A_route)",
            "system_term": "M_brain_reverse = mean(R_origin, R_feature, R_structure, R_route)",
        },
        "project_readout": {
            "summary": "逆向大脑编码机制当前已经能比较稳定地分成起点恢复、特征恢复、结构恢复和路线恢复四层。",
            "next_question": "下一步要把这四层推进到更接近原生回路直测，而不是只停在近原生结构量。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向大脑编码机制报告",
        "",
        f"- origin_recovery: {hm['origin_recovery']:.6f}",
        f"- feature_recovery: {hm['feature_recovery']:.6f}",
        f"- structure_recovery: {hm['structure_recovery']:.6f}",
        f"- route_recovery: {hm['route_recovery']:.6f}",
        f"- reverse_chain_strength: {hm['reverse_chain_strength']:.6f}",
        f"- reverse_chain_gap: {hm['reverse_chain_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_reverse_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
