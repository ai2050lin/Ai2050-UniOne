from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v5_summary() -> dict:
    v4 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v4_20260321" / "summary.json"
    )
    topo = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_3d_topology_encoding_mechanism_20260321" / "summary.json"
    )
    curriculum = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_curriculum_20260321" / "summary.json"
    )
    trainable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_trainable_prototype_20260321" / "summary.json"
    )
    topo_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_prototype_20260321" / "summary.json"
    )

    hv = v4["headline_metrics"]
    ht = topo["headline_metrics"]
    hc = curriculum["headline_metrics"]
    hp = trainable["headline_metrics"]
    hl = topo_long["headline_metrics"]

    direct_origin_measure_v5 = _clip01(
        (hv["direct_origin_measure_v4"] + ht["local_patch_encoding"] + hl["topo_long_retention"] + hc["curriculum_structural_guard"]) / 4.0
    )
    direct_feature_measure_v5 = _clip01(
        (
            hv["direct_feature_measure_v4"]
            + ht["transverse_fiber_binding"]
            + hc["shared_route_guard"]
            + hp["path_reuse_score"]
        )
        / 4.0
    )
    direct_structure_measure_v5 = _clip01(
        (
            hv["direct_structure_measure_v4"]
            + hc["curriculum_structural_guard"]
            + hl["topo_long_structural_survival"]
            + hp["structural_persistence"]
        )
        / 4.0
    )
    direct_route_measure_v5 = _clip01(
        (
            hv["direct_route_measure_v4"]
            + ht["route_superposition_binding"]
            + hc["context_generalization_guard"]
            + hl["topo_long_context_survival"]
        )
        / 4.0
    )
    direct_brain_measure_v5 = (
        direct_origin_measure_v5
        + direct_feature_measure_v5
        + direct_structure_measure_v5
        + direct_route_measure_v5
    ) / 4.0
    direct_brain_gap_v5 = 1.0 - direct_brain_measure_v5
    direct_topology_alignment_v5 = (
        direct_structure_measure_v5 + direct_route_measure_v5 + ht["topology_selective_gate"] + ht["contextual_projection"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v5": direct_origin_measure_v5,
            "direct_feature_measure_v5": direct_feature_measure_v5,
            "direct_structure_measure_v5": direct_structure_measure_v5,
            "direct_route_measure_v5": direct_route_measure_v5,
            "direct_brain_measure_v5": direct_brain_measure_v5,
            "direct_brain_gap_v5": direct_brain_gap_v5,
            "direct_topology_alignment_v5": direct_topology_alignment_v5,
        },
        "direct_equation_v5": {
            "origin_term": "D_origin_v5 = mean(D_origin_v4, E_patch, R_topo_long, S_curr)",
            "feature_term": "D_feature_v5 = mean(D_feature_v4, E_fiber, H_curr, R_topo)",
            "structure_term": "D_structure_v5 = mean(D_structure_v4, S_curr, S_topo_long, S_topo)",
            "route_term": "D_route_v5 = mean(D_route_v4, E_route, C_curr, C_topo_long)",
            "system_term": "M_brain_direct_v5 = mean(D_origin_v5, D_feature_v5, D_structure_v5, D_route_v5)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第五版把课程式可塑性强化并进了结构层和路线层，使脑编码直测链开始同时受三维拓扑编码和长期增量学习能力约束。",
            "next_question": "下一步要把这条第五版直测链继续推进到更大原型里，检验更高更新强度下结构和路线是否还稳定。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第五版报告",
        "",
        f"- direct_origin_measure_v5: {hm['direct_origin_measure_v5']:.6f}",
        f"- direct_feature_measure_v5: {hm['direct_feature_measure_v5']:.6f}",
        f"- direct_structure_measure_v5: {hm['direct_structure_measure_v5']:.6f}",
        f"- direct_route_measure_v5: {hm['direct_route_measure_v5']:.6f}",
        f"- direct_brain_measure_v5: {hm['direct_brain_measure_v5']:.6f}",
        f"- direct_brain_gap_v5: {hm['direct_brain_gap_v5']:.6f}",
        f"- direct_topology_alignment_v5: {hm['direct_topology_alignment_v5']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v5_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
