from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_stable_amplification_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_stable_amplification_validation_summary() -> dict:
    stable_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_strengthening_20260321" / "summary.json"
    )
    stable = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_stable_amplification_validation_20260321" / "summary.json"
    )
    brain_v22 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v22_20260321" / "summary.json"
    )
    bridge_v28 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v28_20260321" / "summary.json"
    )

    hs = stable_plus["headline_metrics"]
    hb = stable["headline_metrics"]
    hd = brain_v22["headline_metrics"]
    ht = bridge_v28["headline_metrics"]

    systemic_amplification_strength = _clip01(
        hs["stable_reinforced_score"] * 0.34
        + hs["stable_reinforced_readiness"] * 0.18
        + hb["stable_score"] * 0.08
        + hd["direct_brain_measure_v22"] * 0.18
        + ht["topology_training_readiness_v28"] * 0.22
    )
    systemic_structure_stability = _clip01(
        hs["stable_reinforced_structure"] * 0.40
        + hb["stable_structure_stability"] * 0.08
        + hd["direct_structure_measure_v22"] * 0.24
        + ht["structure_rule_alignment_v28"] * 0.28
    )
    systemic_route_stability = _clip01(
        hs["stable_reinforced_route"] * 0.40
        + hb["stable_route_stability"] * 0.08
        + hd["direct_route_measure_v22"] * 0.24
        + ht["stable_guard_v28"] * 0.28
    )
    systemic_learning_lift = _clip01(
        hs["stable_reinforced_learning"] * 0.34
        + hb["stable_learning_lift"] * 0.08
        + hd["direct_feature_measure_v22"] * 0.16
        + ht["plasticity_rule_alignment_v28"] * 0.18
        + ht["topology_training_readiness_v28"] * 0.24
    )
    systemic_residual_penalty = _clip01(
        hs["stable_reinforced_penalty"] * 0.52
        + hd["direct_brain_gap_v22"] * 0.20
        + ht["topology_training_gap_v28"] * 0.28
    )
    systemic_readiness = _clip01(
        (
            systemic_amplification_strength
            + systemic_structure_stability
            + systemic_route_stability
            + systemic_learning_lift
            + (1.0 - systemic_residual_penalty)
        )
        / 5.0
    )
    systemic_score = _clip01(
        (
            systemic_readiness
            + systemic_amplification_strength
            + systemic_learning_lift
            + (1.0 - systemic_residual_penalty)
        )
        / 4.0
    )
    systemic_margin = (
        systemic_amplification_strength
        + systemic_structure_stability
        + systemic_route_stability
        + systemic_learning_lift
        + systemic_readiness
        + systemic_score
        - systemic_residual_penalty
    )

    return {
        "headline_metrics": {
            "systemic_amplification_strength": systemic_amplification_strength,
            "systemic_structure_stability": systemic_structure_stability,
            "systemic_route_stability": systemic_route_stability,
            "systemic_learning_lift": systemic_learning_lift,
            "systemic_residual_penalty": systemic_residual_penalty,
            "systemic_readiness": systemic_readiness,
            "systemic_score": systemic_score,
            "systemic_margin": systemic_margin,
        },
        "systemic_equation": {
            "strength_term": "A_system = mix(S_stable_plus_score, R_stable_plus, S_stable_score, M_brain_direct_v22, R_train_v28)",
            "structure_term": "S_system = mix(S_stable_plus, S_stable, D_structure_v22, B_struct_v28)",
            "route_term": "R_system = mix(R_stable_plus, R_stable, D_route_v22, H_stable_v28)",
            "learning_term": "L_system = mix(L_stable_plus, L_stable, D_feature_v22, B_plastic_v28, R_train_v28)",
            "system_term": "M_system = A_system + S_system + R_system + L_system + R_system_sys - P_system",
        },
        "project_readout": {
            "summary": "更大系统系统级稳定放大验证开始直接检验稳定放大增强能否继续推进成更接近系统级的稳态放大。",
            "next_question": "下一步要把这组系统级结果并回脑编码直测和训练终式，确认稳定放大是否开始进入更低风险区。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统系统级稳定放大验证报告",
        "",
        f"- systemic_amplification_strength: {hm['systemic_amplification_strength']:.6f}",
        f"- systemic_structure_stability: {hm['systemic_structure_stability']:.6f}",
        f"- systemic_route_stability: {hm['systemic_route_stability']:.6f}",
        f"- systemic_learning_lift: {hm['systemic_learning_lift']:.6f}",
        f"- systemic_residual_penalty: {hm['systemic_residual_penalty']:.6f}",
        f"- systemic_readiness: {hm['systemic_readiness']:.6f}",
        f"- systemic_score: {hm['systemic_score']:.6f}",
        f"- systemic_margin: {hm['systemic_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_stable_amplification_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
