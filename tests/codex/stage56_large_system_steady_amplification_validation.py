from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_steady_amplification_validation_summary() -> dict:
    reinforce = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_strengthening_20260321" / "summary.json"
    )
    amp = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_validation_20260321" / "summary.json"
    )
    brain_v18 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v18_20260321" / "summary.json"
    )
    bridge_v24 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v24_20260321" / "summary.json"
    )

    hr = reinforce["headline_metrics"]
    ha = amp["headline_metrics"]
    hb = brain_v18["headline_metrics"]
    ht = bridge_v24["headline_metrics"]

    steady_amplification_strength = _clip01(
        hr["amplification_strength"] * 0.36
        + hr["amplification_reinforced_score"] * 0.16
        + ha["amplification_score"] * 0.10
        + hb["direct_brain_measure_v18"] * 0.18
        + ht["topology_training_readiness_v24"] * 0.20
    )
    steady_structure_stability = _clip01(
        hr["amplification_structure_stability"] * 0.42
        + ha["amplification_structure"] * 0.12
        + hb["direct_structure_measure_v18"] * 0.22
        + ht["structure_rule_alignment_v24"] * 0.24
    )
    steady_route_stability = _clip01(
        hr["amplification_route_stability"] * 0.42
        + ha["amplification_route"] * 0.12
        + hb["direct_route_measure_v18"] * 0.22
        + ht["reinforcement_guard_v24"] * 0.24
    )
    steady_learning_lift = _clip01(
        hr["amplification_learning_lift"] * 0.38
        + ha["amplification_learning"] * 0.10
        + hb["direct_feature_measure_v18"] * 0.16
        + ht["plasticity_rule_alignment_v24"] * 0.16
        + ht["topology_training_readiness_v24"] * 0.20
    )
    steady_residual_penalty = _clip01(
        hr["amplification_residual_penalty"] * 0.55
        + hb["direct_brain_gap_v18"] * 0.20
        + ht["topology_training_gap_v24"] * 0.25
    )
    steady_readiness = _clip01(
        (
            steady_amplification_strength
            + steady_structure_stability
            + steady_route_stability
            + steady_learning_lift
            + (1.0 - steady_residual_penalty)
        )
        / 5.0
    )
    steady_score = _clip01(
        (
            steady_readiness
            + steady_amplification_strength
            + steady_learning_lift
            + (1.0 - steady_residual_penalty)
        )
        / 4.0
    )
    steady_margin = (
        steady_amplification_strength
        + steady_structure_stability
        + steady_route_stability
        + steady_learning_lift
        + steady_readiness
        + steady_score
        - steady_residual_penalty
    )

    return {
        "headline_metrics": {
            "steady_amplification_strength": steady_amplification_strength,
            "steady_structure_stability": steady_structure_stability,
            "steady_route_stability": steady_route_stability,
            "steady_learning_lift": steady_learning_lift,
            "steady_residual_penalty": steady_residual_penalty,
            "steady_readiness": steady_readiness,
            "steady_score": steady_score,
            "steady_margin": steady_margin,
        },
        "steady_equation": {
            "strength_term": "A_steady = mix(A_reinforce, S_reinforce_score, S_amp_score, M_brain_direct_v18, R_train_v24)",
            "structure_term": "S_steady = mix(S_reinforce, S_amp, D_structure_v18, B_struct_v24)",
            "route_term": "R_steady = mix(R_reinforce, R_amp, D_route_v18, H_reinforce_v24)",
            "learning_term": "L_steady = mix(L_reinforce, L_amp, D_feature_v18, B_plastic_v24, R_train_v24)",
            "system_term": "M_steady = A_steady + S_steady + R_steady + L_steady + R_steady_sys - P_steady",
        },
        "project_readout": {
            "summary": "更大系统稳态放大验证开始直接检测放大趋势是否开始脱离短期强化，走向更稳的系统级放大。",
            "next_question": "下一步要把这组稳态放大结果并回脑编码直测和训练终式，确认放大是否开始真正稳住。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统稳态放大验证报告",
        "",
        f"- steady_amplification_strength: {hm['steady_amplification_strength']:.6f}",
        f"- steady_structure_stability: {hm['steady_structure_stability']:.6f}",
        f"- steady_route_stability: {hm['steady_route_stability']:.6f}",
        f"- steady_learning_lift: {hm['steady_learning_lift']:.6f}",
        f"- steady_residual_penalty: {hm['steady_residual_penalty']:.6f}",
        f"- steady_readiness: {hm['steady_readiness']:.6f}",
        f"- steady_score: {hm['steady_score']:.6f}",
        f"- steady_margin: {hm['steady_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_steady_amplification_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
