from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_nativeization_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_nativeization_summary() -> dict:
    reverse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_reverse_analysis_20260320" / "summary.json"
    )
    neuron = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_direct_refinement_20260320" / "summary.json"
    )
    structure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_refinement_20260320" / "summary.json"
    )
    color = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_color_fiber_nativeization_20260320" / "summary.json"
    )
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_object_attribute_coupling_prototype_20260320" / "summary.json"
    )

    hr = reverse["headline_metrics"]
    hn = neuron["headline_metrics"]
    hs = structure["headline_metrics"]
    hc = color["headline_metrics"]
    hp = proto["headline_metrics"]

    origin_nativeization = _clip01(hr["origin_recovery"] * (1.0 + 0.20 * hn["origin_stability_v2"]))
    feature_nativeization = _clip01((hr["feature_recovery"] + hc["native_color_binding"] + min(1.0, hp["shared_attribute_reuse"])) / 3.0)
    structure_nativeization = _clip01((hr["structure_recovery"] + min(1.0, hs["structure_direct_confidence_v3"]) + min(1.0, hp["same_attribute_different_route"])) / 3.0)
    route_nativeization = _clip01((hr["route_recovery"] + min(1.0, hp["route_divergence"]) + min(1.0, hp["context_divergence"])) / 3.0)
    brain_native_chain_strength = (origin_nativeization + feature_nativeization + structure_nativeization + route_nativeization) / 4.0
    brain_native_gap = 1.0 - brain_native_chain_strength

    return {
        "headline_metrics": {
            "origin_nativeization": origin_nativeization,
            "feature_nativeization": feature_nativeization,
            "structure_nativeization": structure_nativeization,
            "route_nativeization": route_nativeization,
            "brain_native_chain_strength": brain_native_chain_strength,
            "brain_native_gap": brain_native_gap,
        },
        "native_equation": {
            "origin_term": "N_origin_native = R_origin * (1 + 0.2 * S_origin_v2)",
            "feature_term": "N_feature_native = mean(R_feature, B_red, H_attr)",
            "structure_term": "N_structure_native = mean(R_structure, C_struct_v3, R_split)",
            "route_term": "N_route_native = mean(R_route, route_divergence, context_divergence)",
            "system_term": "M_brain_native = mean(N_origin_native, N_feature_native, N_structure_native, N_route_native)",
        },
        "project_readout": {
            "summary": "逆向脑编码机制当前已经从起点恢复、特征恢复、结构恢复和路线恢复，进一步推进到近原生链条对象。",
            "next_question": "下一步要把这条近原生链条继续推进到更接近原生回路直测，而不是只停在近原生聚合量。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码原生化报告",
        "",
        f"- origin_nativeization: {hm['origin_nativeization']:.6f}",
        f"- feature_nativeization: {hm['feature_nativeization']:.6f}",
        f"- structure_nativeization: {hm['structure_nativeization']:.6f}",
        f"- route_nativeization: {hm['route_nativeization']:.6f}",
        f"- brain_native_chain_strength: {hm['brain_native_chain_strength']:.6f}",
        f"- brain_native_gap: {hm['brain_native_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_nativeization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
