from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v4_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v4_summary() -> dict:
    v3 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v3_20260321" / "summary.json"
    )
    topo = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_3d_topology_encoding_mechanism_20260321" / "summary.json"
    )
    topo_train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_trainable_prototype_20260321" / "summary.json"
    )
    topo_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_prototype_20260321" / "summary.json"
    )
    plasticity = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_boost_20260321" / "summary.json"
    )

    hv3 = v3["headline_metrics"]
    ht = topo["headline_metrics"]
    hp = topo_train["headline_metrics"]
    hl = topo_long["headline_metrics"]
    hb = plasticity["headline_metrics"]

    direct_origin_measure_v4 = _clip01(
        (hv3["direct_origin_measure_v3"] + ht["local_patch_encoding"] + hl["topo_long_retention"]) / 3.0
    )
    direct_feature_measure_v4 = _clip01(
        (
            hv3["direct_feature_measure_v3"]
            + ht["transverse_fiber_binding"]
            + hb["shared_guard_after_boost"]
            + hp["path_reuse_score"]
        )
        / 4.0
    )
    direct_structure_measure_v4 = _clip01(
        (
            hv3["direct_structure_measure_v3"]
            + hp["structural_persistence"]
            + hl["topo_long_structural_survival"]
            + hb["structural_plasticity_balance"]
        )
        / 4.0
    )
    direct_route_measure_v4 = _clip01(
        (
            hv3["direct_route_measure_v3"]
            + ht["topology_selective_gate"]
            + hl["topo_long_context_survival"]
            + hp["route_split_score"]
        )
        / 4.0
    )
    direct_brain_measure_v4 = (
        direct_origin_measure_v4
        + direct_feature_measure_v4
        + direct_structure_measure_v4
        + direct_route_measure_v4
    ) / 4.0
    direct_brain_gap_v4 = 1.0 - direct_brain_measure_v4
    dynamic_structure_balance_v4 = (
        direct_structure_measure_v4 + direct_route_measure_v4 + hl["topo_long_structural_survival"]
    ) / 3.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v4": direct_origin_measure_v4,
            "direct_feature_measure_v4": direct_feature_measure_v4,
            "direct_structure_measure_v4": direct_structure_measure_v4,
            "direct_route_measure_v4": direct_route_measure_v4,
            "direct_brain_measure_v4": direct_brain_measure_v4,
            "direct_brain_gap_v4": direct_brain_gap_v4,
            "dynamic_structure_balance_v4": dynamic_structure_balance_v4,
        },
        "direct_equation_v4": {
            "origin_term": "D_origin_v4 = mean(D_origin_v3, E_patch, R_topo_long)",
            "feature_term": "D_feature_v4 = mean(D_feature_v3, E_fiber, H_boost, R_topo)",
            "structure_term": "D_structure_v4 = mean(D_structure_v3, S_topo, S_topo_long, S_boost)",
            "route_term": "D_route_v4 = mean(D_route_v3, E_gate, C_topo_long, route_split)",
            "system_term": "M_brain_direct_v4 = mean(D_origin_v4, D_feature_v4, D_structure_v4, D_route_v4)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第四版开始把三维拓扑编码、长时间尺度结构生存率和可塑性平衡一起并回直测链，使结构层不再只靠短程一致性支撑。",
            "next_question": "下一步要继续把结构层和路线层推进到更接近原生回路直测，否则脑编码链仍然会卡在中层代理量上。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第四版报告",
        "",
        f"- direct_origin_measure_v4: {hm['direct_origin_measure_v4']:.6f}",
        f"- direct_feature_measure_v4: {hm['direct_feature_measure_v4']:.6f}",
        f"- direct_structure_measure_v4: {hm['direct_structure_measure_v4']:.6f}",
        f"- direct_route_measure_v4: {hm['direct_route_measure_v4']:.6f}",
        f"- direct_brain_measure_v4: {hm['direct_brain_measure_v4']:.6f}",
        f"- direct_brain_gap_v4: {hm['direct_brain_gap_v4']:.6f}",
        f"- dynamic_structure_balance_v4: {hm['dynamic_structure_balance_v4']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v4_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
