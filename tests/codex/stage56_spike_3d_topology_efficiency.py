from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_spike_3d_topology_efficiency_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_spike_3d_topology_efficiency_summary() -> dict:
    sparse = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_sparse_activation_region_analysis_20260320" / "summary.json"
    )
    region = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_region_topology_analysis_20260320" / "summary.json"
    )
    horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321" / "summary.json"
    )
    brain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v3_20260321" / "summary.json"
    )

    hs = sparse["headline_metrics"]
    hr = region["headline_metrics"]
    hh = horizon["headline_metrics"]
    hb = brain["headline_metrics"]

    minimal_transport_efficiency = _clip01((hs["sparse_activation_efficiency"] + hh["shared_fiber_survival"]) / 2.0)
    topology_grid_efficiency = _clip01((hr["region_topology_margin"] + hr["regional_overlap_control"]) / 2.0)
    path_superposition_capacity = _clip01((hb["direct_route_measure_v3"] + hh["contextual_survival"] + hh["shared_fiber_survival"]) / 3.0)
    online_stability_coupling = _clip01((hh["long_horizon_retention"] + hh["structural_survival"]) / 2.0)
    global_steady_coupling = _clip01((hb["direct_structure_measure_v3"] + minimal_transport_efficiency + topology_grid_efficiency) / 3.0)
    topology_encoding_margin = (
        minimal_transport_efficiency
        + topology_grid_efficiency
        + path_superposition_capacity
        + online_stability_coupling
        + global_steady_coupling
    )

    return {
        "headline_metrics": {
            "minimal_transport_efficiency": minimal_transport_efficiency,
            "topology_grid_efficiency": topology_grid_efficiency,
            "path_superposition_capacity": path_superposition_capacity,
            "online_stability_coupling": online_stability_coupling,
            "global_steady_coupling": global_steady_coupling,
            "topology_encoding_margin": topology_encoding_margin,
        },
        "topology_equation": {
            "transport_term": "T_min = mean(A_sparse, H_fiber)",
            "grid_term": "G_3d = mean(M_region, R_overlap)",
            "path_term": "P_super = mean(D_route_v3, H_context, H_fiber)",
            "online_term": "C_online = mean(R_h, H_structure)",
            "steady_term": "C_global = mean(D_structure_v3, T_min, G_3d)",
            "system_term": "M_topology = T_min + G_3d + P_super + C_online + C_global",
        },
        "project_readout": {
            "summary": "如果脉冲网络满足最小传送量原理，且三维拓扑网格具有高路径复用效率，那么路径叠加就会自然成为编码原理，并同时提升在线学习与全局稳态的兼容性。",
            "next_question": "下一步要把这条三维拓扑主线真正并入可训练脉冲原型，否则当前仍然只是中层拓扑解释而不是可施工结构。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 脉冲三维拓扑效率报告",
        "",
        f"- minimal_transport_efficiency: {hm['minimal_transport_efficiency']:.6f}",
        f"- topology_grid_efficiency: {hm['topology_grid_efficiency']:.6f}",
        f"- path_superposition_capacity: {hm['path_superposition_capacity']:.6f}",
        f"- online_stability_coupling: {hm['online_stability_coupling']:.6f}",
        f"- global_steady_coupling: {hm['global_steady_coupling']:.6f}",
        f"- topology_encoding_margin: {hm['topology_encoding_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_spike_3d_topology_efficiency_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
