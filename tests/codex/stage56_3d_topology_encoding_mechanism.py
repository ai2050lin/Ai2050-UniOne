from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_3d_topology_encoding_mechanism_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_3d_topology_encoding_mechanism_summary() -> dict:
    language = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_total_analysis_20260320" / "summary.json"
    )
    brain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v3_20260321" / "summary.json"
    )
    topology = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_topology_efficiency_20260321" / "summary.json"
    )
    prototype = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_contextual_trainable_prototype_20260320" / "summary.json"
    )

    hl = language["headline_metrics"]
    hb = brain["headline_metrics"]
    ht = topology["headline_metrics"]
    hp = prototype["headline_metrics"]

    local_patch_encoding = _clip01(
        (hl["language_structure_resolution"] + hb["direct_origin_measure_v3"] + hb["direct_feature_measure_v3"]) / 3.0
    )
    transverse_fiber_binding = _clip01(
        (ht["path_superposition_capacity"] + hp["shared_red_consistency"] + hb["direct_feature_measure_v3"]) / 3.0
    )
    route_superposition_binding = _clip01(
        (hb["direct_route_measure_v3"] + hp["route_split_consistency"] + ht["path_superposition_capacity"]) / 3.0
    )
    topology_selective_gate = _clip01(
        (ht["minimal_transport_efficiency"] + ht["topology_grid_efficiency"] + hp["context_split_consistency"]) / 3.0
    )
    contextual_projection = _clip01(
        (hp["context_split_consistency"] + hb["direct_route_measure_v3"] + hl["language_transport_resolution"]) / 3.0
    )
    three_d_encoding_margin = (
        local_patch_encoding
        + transverse_fiber_binding
        + route_superposition_binding
        + topology_selective_gate
        + contextual_projection
    )

    return {
        "headline_metrics": {
            "local_patch_encoding": local_patch_encoding,
            "transverse_fiber_binding": transverse_fiber_binding,
            "route_superposition_binding": route_superposition_binding,
            "topology_selective_gate": topology_selective_gate,
            "contextual_projection": contextual_projection,
            "three_d_encoding_margin": three_d_encoding_margin,
        },
        "mechanism_equation": {
            "patch_term": "E_patch = mean(R_lang_struct, D_origin_v3, D_feature_v3)",
            "fiber_term": "E_fiber = mean(P_super, C_red, D_feature_v3)",
            "route_term": "E_route = mean(D_route_v3, C_route, P_super)",
            "gate_term": "E_gate = mean(T_min, G_3d, C_ctx)",
            "context_term": "E_ctx = mean(C_ctx, D_route_v3, R_lang_transport)",
            "system_term": "M_3d_encode = E_patch + E_fiber + E_route + E_gate + E_ctx",
        },
        "project_readout": {
            "summary": "三维空间网络里的编码单元更像局部片区、横跨纤维、路径叠加和上下文投影的组合，而不是单点标签。",
            "next_question": "下一步要回答的核心问题，是这种三维路径叠加机制在规模化以后如何避免路径爆炸和结构塌缩。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 三维拓扑编码机制报告",
        "",
        f"- local_patch_encoding: {hm['local_patch_encoding']:.6f}",
        f"- transverse_fiber_binding: {hm['transverse_fiber_binding']:.6f}",
        f"- route_superposition_binding: {hm['route_superposition_binding']:.6f}",
        f"- topology_selective_gate: {hm['topology_selective_gate']:.6f}",
        f"- contextual_projection: {hm['contextual_projection']:.6f}",
        f"- three_d_encoding_margin: {hm['three_d_encoding_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_3d_topology_encoding_mechanism_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
