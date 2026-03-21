from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_degradation_probe_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_true_large_scale_route_degradation_probe_summary() -> dict:
    true_scale = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_online_collapse_probe_20260321" / "summary.json"
    )
    brain_v6 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v6_20260321" / "summary.json"
    )
    bridge_v12 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v12_20260321" / "summary.json"
    )
    extreme = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_scale_high_intensity_long_horizon_extreme_20260321" / "summary.json"
    )

    ht = true_scale["headline_metrics"]
    hb = brain_v6["headline_metrics"]
    hr = bridge_v12["headline_metrics"]
    he = extreme["headline_metrics"]

    route_degradation_risk = _clip01(
        (
            (1.0 - ht["true_scale_context_keep"])
            + (1.0 - hb["direct_route_measure_v6"])
            + hr["topology_training_gap_v12"]
            + ht["true_scale_phase_shift_risk"]
        )
        / 4.0
    )
    structure_phase_shift_risk = _clip01(
        (
            ht["true_scale_collapse_risk"]
            + ht["true_scale_phase_shift_risk"]
            + (1.0 - hb["direct_structure_measure_v6"])
            + (1.0 - ht["true_scale_structure_keep"])
        )
        / 4.0
    )
    route_resilience = _clip01(
        (
            ht["true_scale_context_keep"]
            + hb["direct_route_measure_v6"]
            + hr["true_scale_guard_v12"]
            + (1.0 - route_degradation_risk)
        )
        / 4.0
    )
    structure_resilience = _clip01(
        (
            ht["true_scale_structure_keep"]
            + hb["direct_structure_measure_v6"]
            + hr["structure_rule_alignment_v12"]
            + (1.0 - structure_phase_shift_risk)
        )
        / 4.0
    )
    true_scale_reinforced_readiness = _clip01(
        (
            ht["true_scale_language_keep"]
            + structure_resilience
            + route_resilience
            + ht["true_scale_novel_gain"]
            + (1.0 - ht["true_scale_forgetting_penalty"])
            + (1.0 - route_degradation_risk)
            + (1.0 - structure_phase_shift_risk)
            + hr["topology_training_readiness_v12"]
            + he["extreme_readiness"]
        )
        / 9.0
    )
    route_phase_margin = (
        route_resilience
        + structure_resilience
        + true_scale_reinforced_readiness
        + ht["true_scale_language_keep"]
        + ht["true_scale_novel_gain"]
        - ht["true_scale_forgetting_penalty"]
        - route_degradation_risk
        - structure_phase_shift_risk
    )

    return {
        "headline_metrics": {
            "route_degradation_risk": route_degradation_risk,
            "structure_phase_shift_risk": structure_phase_shift_risk,
            "route_resilience": route_resilience,
            "structure_resilience": structure_resilience,
            "true_scale_reinforced_readiness": true_scale_reinforced_readiness,
            "route_phase_margin": route_phase_margin,
        },
        "route_probe_equation": {
            "route_risk_term": "R_route = mean(1 - C_true, 1 - D_route_v6, G_train_v12, Q_true)",
            "phase_risk_term": "R_phase = mean(R_true, Q_true, 1 - D_structure_v6, 1 - S_true)",
            "route_resilience_term": "H_route = mean(C_true, D_route_v6, H_true_v12, 1 - R_route)",
            "structure_resilience_term": "H_struct = mean(S_true, D_structure_v6, B_struct_v12, 1 - R_phase)",
            "system_term": "M_route_phase = H_route + H_struct + A_true - P_true - R_route - R_phase",
        },
        "project_readout": {
            "summary": "真正规模化场景里，路由退化已经开始和结构层塌缩、相变式失稳绑在一起，系统下一阶段的主要工程风险不再只是遗忘，而是路由和结构的共同退化。",
            "next_question": "下一步要把这组路由退化探针结果并回脑编码直测和训练桥，检验主核在更真实路由压力下是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 真正规模化路由退化探针报告",
        "",
        f"- route_degradation_risk: {hm['route_degradation_risk']:.6f}",
        f"- structure_phase_shift_risk: {hm['structure_phase_shift_risk']:.6f}",
        f"- route_resilience: {hm['route_resilience']:.6f}",
        f"- structure_resilience: {hm['structure_resilience']:.6f}",
        f"- true_scale_reinforced_readiness: {hm['true_scale_reinforced_readiness']:.6f}",
        f"- route_phase_margin: {hm['route_phase_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_true_large_scale_route_degradation_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
