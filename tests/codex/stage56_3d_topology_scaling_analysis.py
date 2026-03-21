from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_3d_topology_scaling_analysis_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_3d_topology_scaling_analysis_summary() -> dict:
    topology = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_topology_efficiency_20260321" / "summary.json"
    )
    horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321" / "summary.json"
    )
    prototype = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_contextual_trainable_prototype_20260320" / "summary.json"
    )
    mech = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_3d_topology_encoding_mechanism_20260321" / "summary.json"
    )

    ht = topology["headline_metrics"]
    hh = horizon["headline_metrics"]
    hp = prototype["headline_metrics"]
    hm = mech["headline_metrics"]

    scale_transport_retention = _clip01((ht["minimal_transport_efficiency"] + hh["long_horizon_retention"]) / 2.0)
    scale_modular_reuse = _clip01((hh["shared_fiber_survival"] + hp["shared_red_consistency"] + hm["transverse_fiber_binding"]) / 3.0)
    scale_route_density = _clip01((hm["route_superposition_binding"] + hp["route_split_consistency"] + hh["contextual_survival"]) / 3.0)
    scale_collision_penalty = _clip01(1.0 - ((hh["structural_survival"] + ht["topology_grid_efficiency"]) / 2.0))
    scale_structural_risk = _clip01(1.0 - ((hh["structural_survival"] + hp["context_split_consistency"]) / 2.0))
    scale_ready_score = _clip01(
        (scale_transport_retention + scale_modular_reuse + scale_route_density + (1.0 - scale_collision_penalty)) / 4.0
    )

    return {
        "headline_metrics": {
            "scale_transport_retention": scale_transport_retention,
            "scale_modular_reuse": scale_modular_reuse,
            "scale_route_density": scale_route_density,
            "scale_collision_penalty": scale_collision_penalty,
            "scale_structural_risk": scale_structural_risk,
            "scale_ready_score": scale_ready_score,
        },
        "scaling_equation": {
            "transport_term": "S_trans = mean(T_min, R_h_long)",
            "reuse_term": "S_reuse = mean(H_fiber, C_red, E_fiber)",
            "route_term": "S_route = mean(E_route, C_route, H_ctx)",
            "collision_term": "P_collision = 1 - mean(H_structure, G_3d)",
            "risk_term": "P_struct = 1 - mean(H_structure, C_ctx)",
            "system_term": "M_scale = mean(S_trans, S_reuse, S_route, 1 - P_collision)",
        },
        "project_readout": {
            "summary": "三维拓扑规模化真正的难点不是属性纤维复用，而是结构层在持续更新和高路径密度下能否不塌。",
            "next_question": "下一步要把这种规模化判断放进可训练脉冲原型，否则规模化仍然停留在中层推断。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 三维拓扑规模化分析报告",
        "",
        f"- scale_transport_retention: {hm['scale_transport_retention']:.6f}",
        f"- scale_modular_reuse: {hm['scale_modular_reuse']:.6f}",
        f"- scale_route_density: {hm['scale_route_density']:.6f}",
        f"- scale_collision_penalty: {hm['scale_collision_penalty']:.6f}",
        f"- scale_structural_risk: {hm['scale_structural_risk']:.6f}",
        f"- scale_ready_score: {hm['scale_ready_score']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_3d_topology_scaling_analysis_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
