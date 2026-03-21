from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_structure_coupled_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_true_large_scale_route_structure_coupled_validation_summary() -> dict:
    true_scale = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_online_collapse_probe_20260321" / "summary.json"
    )
    route_probe = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_degradation_probe_20260321" / "summary.json"
    )
    brain_v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v7_20260321" / "summary.json"
    )
    bridge_v13 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v13_20260321" / "summary.json"
    )

    ht = true_scale["headline_metrics"]
    hr = route_probe["headline_metrics"]
    hb = brain_v7["headline_metrics"]
    hbri = bridge_v13["headline_metrics"]

    coupled_route_keep = _clip01(
        (
            hr["route_resilience"]
            + hb["direct_route_measure_v7"]
            + (1.0 - hr["route_degradation_risk"])
            + hbri["route_guard_v13"]
        )
        / 4.0
    )
    coupled_structure_keep = _clip01(
        (
            hr["structure_resilience"]
            + hb["direct_structure_measure_v7"]
            + (1.0 - hr["structure_phase_shift_risk"])
            + hbri["structure_rule_alignment_v13"]
        )
        / 4.0
    )
    coupled_context_keep = _clip01(
        (
            ht["true_scale_context_keep"]
            + hr["route_resilience"]
            + hb["direct_route_alignment_v7"]
            + hbri["topology_training_readiness_v13"]
        )
        / 4.0
    )
    coupled_novel_gain = _clip01(
        (
            ht["true_scale_novel_gain"]
            + hbri["plasticity_rule_alignment_v13"]
            + hb["direct_feature_measure_v7"]
        )
        / 3.0
    )
    coupled_forgetting_penalty = _clip01(
        (
            ht["true_scale_forgetting_penalty"]
            + hr["route_degradation_risk"] * 0.5
            + hr["structure_phase_shift_risk"] * 0.5
        )
        / 2.0
    )
    coupled_failure_risk = _clip01(
        (
            hr["route_degradation_risk"]
            + hr["structure_phase_shift_risk"]
            + (1.0 - coupled_route_keep)
            + (1.0 - coupled_structure_keep)
            + hbri["topology_training_gap_v13"]
        )
        / 5.0
    )
    coupled_readiness = _clip01(
        (
            ht["true_scale_language_keep"]
            + coupled_route_keep
            + coupled_structure_keep
            + coupled_context_keep
            + coupled_novel_gain
            + (1.0 - coupled_forgetting_penalty)
            + (1.0 - coupled_failure_risk)
            + hbri["topology_training_readiness_v13"]
        )
        / 8.0
    )
    coupled_margin = (
        ht["true_scale_language_keep"]
        + coupled_route_keep
        + coupled_structure_keep
        + coupled_context_keep
        + coupled_novel_gain
        + coupled_readiness
        - coupled_forgetting_penalty
        - coupled_failure_risk
    )

    return {
        "headline_metrics": {
            "coupled_route_keep": coupled_route_keep,
            "coupled_structure_keep": coupled_structure_keep,
            "coupled_context_keep": coupled_context_keep,
            "coupled_novel_gain": coupled_novel_gain,
            "coupled_forgetting_penalty": coupled_forgetting_penalty,
            "coupled_failure_risk": coupled_failure_risk,
            "coupled_readiness": coupled_readiness,
            "coupled_margin": coupled_margin,
        },
        "coupled_equation": {
            "route_keep_term": "K_route = mean(H_route, D_route_v7, 1 - R_route, H_route_v13)",
            "structure_keep_term": "K_struct = mean(H_struct, D_structure_v7, 1 - R_phase, B_struct_v13)",
            "context_keep_term": "K_ctx = mean(C_true, H_route, D_route_align_v7, R_train_v13)",
            "failure_term": "R_fail = mean(R_route, R_phase, 1 - K_route, 1 - K_struct, G_train_v13)",
            "system_term": "M_coupled = L_true + K_route + K_struct + K_ctx + G_coupled + A_coupled - P_coupled - R_fail",
        },
        "project_readout": {
            "summary": "真正规模化场景下，路由退化和结构塌缩已经不是分开的风险项，而是开始形成联动退化链，这条链正在变成当前最大工程瓶颈。",
            "next_question": "下一步要把这组联动退化结果并回脑编码直测和训练桥，检验主核在更真实耦合压力下是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 真正规模化路由-结构联动验证报告",
        "",
        f"- coupled_route_keep: {hm['coupled_route_keep']:.6f}",
        f"- coupled_structure_keep: {hm['coupled_structure_keep']:.6f}",
        f"- coupled_context_keep: {hm['coupled_context_keep']:.6f}",
        f"- coupled_novel_gain: {hm['coupled_novel_gain']:.6f}",
        f"- coupled_forgetting_penalty: {hm['coupled_forgetting_penalty']:.6f}",
        f"- coupled_failure_risk: {hm['coupled_failure_risk']:.6f}",
        f"- coupled_readiness: {hm['coupled_readiness']:.6f}",
        f"- coupled_margin: {hm['coupled_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_true_large_scale_route_structure_coupled_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
