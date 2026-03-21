from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_strengthening_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_sustained_amplification_strengthening_summary() -> dict:
    amp = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_amplification_validation_20260321" / "summary.json"
    )
    sustain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_sustained_rebound_validation_20260321" / "summary.json"
    )
    brain_v17 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v17_20260321" / "summary.json"
    )
    bridge_v23 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v23_20260321" / "summary.json"
    )

    ha = amp["headline_metrics"]
    hs = sustain["headline_metrics"]
    hb = brain_v17["headline_metrics"]
    ht = bridge_v23["headline_metrics"]

    amplification_strength = _clip01(
        ha["amplification_score"] * 0.34
        + ha["amplification_readiness"] * 0.18
        + hs["sustained_rebound_score"] * 0.10
        + hb["direct_brain_measure_v17"] * 0.18
        + ht["topology_training_readiness_v23"] * 0.20
    )
    amplification_structure_stability = _clip01(
        ha["amplification_structure"] * 0.40
        + hs["sustained_structure"] * 0.12
        + hb["direct_structure_measure_v17"] * 0.24
        + ht["structure_rule_alignment_v23"] * 0.24
    )
    amplification_route_stability = _clip01(
        ha["amplification_route"] * 0.40
        + hs["sustained_route"] * 0.12
        + hb["direct_route_measure_v17"] * 0.24
        + ht["amplification_guard_v23"] * 0.24
    )
    amplification_learning_lift = _clip01(
        ha["amplification_learning"] * 0.36
        + hs["sustained_learning"] * 0.10
        + hb["direct_feature_measure_v17"] * 0.16
        + ht["plasticity_rule_alignment_v23"] * 0.18
        + ht["topology_training_readiness_v23"] * 0.20
    )
    amplification_residual_penalty = _clip01(
        ha["amplification_penalty"] * 0.55
        + hb["direct_brain_gap_v17"] * 0.20
        + ht["topology_training_gap_v23"] * 0.25
    )
    amplification_reinforced_readiness = _clip01(
        (
            amplification_strength
            + amplification_structure_stability
            + amplification_route_stability
            + amplification_learning_lift
            + (1.0 - amplification_residual_penalty)
        )
        / 5.0
    )
    amplification_reinforced_score = _clip01(
        (
            amplification_reinforced_readiness
            + amplification_strength
            + amplification_learning_lift
            + (1.0 - amplification_residual_penalty)
        )
        / 4.0
    )
    amplification_reinforced_margin = (
        amplification_strength
        + amplification_structure_stability
        + amplification_route_stability
        + amplification_learning_lift
        + amplification_reinforced_readiness
        + amplification_reinforced_score
        - amplification_residual_penalty
    )

    return {
        "headline_metrics": {
            "amplification_strength": amplification_strength,
            "amplification_structure_stability": amplification_structure_stability,
            "amplification_route_stability": amplification_route_stability,
            "amplification_learning_lift": amplification_learning_lift,
            "amplification_residual_penalty": amplification_residual_penalty,
            "amplification_reinforced_readiness": amplification_reinforced_readiness,
            "amplification_reinforced_score": amplification_reinforced_score,
            "amplification_reinforced_margin": amplification_reinforced_margin,
        },
        "amplification_reinforced_equation": {
            "strength_term": "A_reinforce = mix(S_amp_score, R_amp, S_sustain_score, M_brain_direct_v17, R_train_v23)",
            "structure_term": "S_reinforce = mix(S_amp, S_sustain, D_structure_v17, B_struct_v23)",
            "route_term": "R_reinforce = mix(R_amp, R_sustain, D_route_v17, H_amp_v23)",
            "learning_term": "L_reinforce = mix(L_amp, L_sustain, D_feature_v17, B_plastic_v23, R_train_v23)",
            "system_term": "M_reinforce = A_reinforce + S_reinforce + R_reinforce + L_reinforce + R_reinforce_sys - P_reinforce",
        },
        "project_readout": {
            "summary": "更大系统持续放大强化开始直接检测放大趋势能否继续增强，而不是只停留在轻度放大。",
            "next_question": "下一步要把这组强化结果并回脑编码直测和训练终式，确认放大趋势是否开始从弱放大进入稳态放大。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统持续放大强化报告",
        "",
        f"- amplification_strength: {hm['amplification_strength']:.6f}",
        f"- amplification_structure_stability: {hm['amplification_structure_stability']:.6f}",
        f"- amplification_route_stability: {hm['amplification_route_stability']:.6f}",
        f"- amplification_learning_lift: {hm['amplification_learning_lift']:.6f}",
        f"- amplification_residual_penalty: {hm['amplification_residual_penalty']:.6f}",
        f"- amplification_reinforced_readiness: {hm['amplification_reinforced_readiness']:.6f}",
        f"- amplification_reinforced_score: {hm['amplification_reinforced_score']:.6f}",
        f"- amplification_reinforced_margin: {hm['amplification_reinforced_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_sustained_amplification_strengthening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
