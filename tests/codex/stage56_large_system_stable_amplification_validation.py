from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_stable_amplification_validation_summary() -> dict:
    v82 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_reinforcement_20260321" / "summary.json"
    )
    v81 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_validation_20260321" / "summary.json"
    )
    brain_v20 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v20_20260321" / "summary.json"
    )
    bridge_v26 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v26_20260321" / "summary.json"
    )

    hs = v82["headline_metrics"]
    hb = v81["headline_metrics"]
    hd = brain_v20["headline_metrics"]
    ht = bridge_v26["headline_metrics"]

    stable_amplification_strength = _clip01(
        hs["steady_reinforcement_score"] * 0.34
        + hs["steady_reinforcement_readiness"] * 0.18
        + hb["steady_score"] * 0.08
        + hd["direct_brain_measure_v20"] * 0.18
        + ht["topology_training_readiness_v26"] * 0.22
    )
    stable_structure_stability = _clip01(
        hs["steady_reinforcement_structure"] * 0.40
        + hb["steady_structure_stability"] * 0.08
        + hd["direct_structure_measure_v20"] * 0.24
        + ht["structure_rule_alignment_v26"] * 0.28
    )
    stable_route_stability = _clip01(
        hs["steady_reinforcement_route"] * 0.40
        + hb["steady_route_stability"] * 0.08
        + hd["direct_route_measure_v20"] * 0.24
        + ht["steady_guard_v26"] * 0.28
    )
    stable_learning_lift = _clip01(
        hs["steady_reinforcement_learning"] * 0.34
        + hb["steady_learning_lift"] * 0.08
        + hd["direct_feature_measure_v20"] * 0.16
        + ht["plasticity_rule_alignment_v26"] * 0.18
        + ht["topology_training_readiness_v26"] * 0.24
    )
    stable_residual_penalty = _clip01(
        hs["steady_reinforcement_penalty"] * 0.52
        + hd["direct_brain_gap_v20"] * 0.20
        + ht["topology_training_gap_v26"] * 0.28
    )
    stable_readiness = _clip01(
        (
            stable_amplification_strength
            + stable_structure_stability
            + stable_route_stability
            + stable_learning_lift
            + (1.0 - stable_residual_penalty)
        )
        / 5.0
    )
    stable_score = _clip01(
        (
            stable_readiness
            + stable_amplification_strength
            + stable_learning_lift
            + (1.0 - stable_residual_penalty)
        )
        / 4.0
    )
    stable_margin = (
        stable_amplification_strength
        + stable_structure_stability
        + stable_route_stability
        + stable_learning_lift
        + stable_readiness
        + stable_score
        - stable_residual_penalty
    )

    return {
        "headline_metrics": {
            "stable_amplification_strength": stable_amplification_strength,
            "stable_structure_stability": stable_structure_stability,
            "stable_route_stability": stable_route_stability,
            "stable_learning_lift": stable_learning_lift,
            "stable_residual_penalty": stable_residual_penalty,
            "stable_readiness": stable_readiness,
            "stable_score": stable_score,
            "stable_margin": stable_margin,
        },
        "stable_equation": {
            "strength_term": "A_stable = mix(S_steady_plus_score, R_steady_plus, S_steady_score, M_brain_direct_v20, R_train_v26)",
            "structure_term": "S_stable = mix(S_steady_plus, S_steady, D_structure_v20, B_struct_v26)",
            "route_term": "R_stable = mix(R_steady_plus, R_steady, D_route_v20, H_steady_v26)",
            "learning_term": "L_stable = mix(L_steady_plus, L_steady, D_feature_v20, B_plastic_v26, R_train_v26)",
            "system_term": "M_stable = A_stable + S_stable + R_stable + L_stable + R_stable_sys - P_stable",
        },
        "project_readout": {
            "summary": "更大系统稳定放大验证开始直接检验增强后的放大趋势能否进一步转成更稳定的系统放大，而不是停在当前增强区。",
            "next_question": "下一步要把这组稳定放大结果并回脑编码直测和训练终式，确认放大趋势是否开始形成更低风险的稳定承接。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统稳定放大验证报告",
        "",
        f"- stable_amplification_strength: {hm['stable_amplification_strength']:.6f}",
        f"- stable_structure_stability: {hm['stable_structure_stability']:.6f}",
        f"- stable_route_stability: {hm['stable_route_stability']:.6f}",
        f"- stable_learning_lift: {hm['stable_learning_lift']:.6f}",
        f"- stable_residual_penalty: {hm['stable_residual_penalty']:.6f}",
        f"- stable_readiness: {hm['stable_readiness']:.6f}",
        f"- stable_score: {hm['stable_score']:.6f}",
        f"- stable_margin: {hm['stable_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_stable_amplification_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
