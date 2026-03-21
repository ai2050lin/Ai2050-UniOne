from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_enlargement_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_zone_enlargement_validation_summary() -> dict:
    systemic_expand = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_expansion_validation_20260321" / "summary.json"
    )
    brain_v28 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v28_20260321" / "summary.json"
    )
    bridge_v34 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v34_20260321" / "summary.json"
    )
    v90 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v90_20260321" / "summary.json"
    )

    hs = systemic_expand["headline_metrics"]
    hb = brain_v28["headline_metrics"]
    ht = bridge_v34["headline_metrics"]
    hm = v90["headline_metrics"]

    systemic_low_risk_enlargement_strength = _clip01(
        hs["systemic_low_risk_expansion_score"] * 0.31
        + hs["systemic_low_risk_expansion_readiness"] * 0.19
        + (1.0 - hs["systemic_low_risk_expansion_penalty"]) * 0.14
        + hb["direct_brain_measure_v28"] * 0.17
        + ht["topology_training_readiness_v34"] * 0.19
    )
    systemic_low_risk_enlargement_structure = _clip01(
        hs["systemic_low_risk_expansion_structure"] * 0.39
        + hs["systemic_low_risk_expansion_route"] * 0.08
        + hb["direct_structure_measure_v28"] * 0.24
        + ht["structure_rule_alignment_v34"] * 0.29
    )
    systemic_low_risk_enlargement_route = _clip01(
        hs["systemic_low_risk_expansion_route"] * 0.39
        + hs["systemic_low_risk_expansion_structure"] * 0.08
        + hb["direct_route_measure_v28"] * 0.24
        + ht["systemic_low_risk_expansion_guard_v34"] * 0.29
    )
    systemic_low_risk_enlargement_learning = _clip01(
        hs["systemic_low_risk_expansion_learning"] * 0.29
        + hs["systemic_low_risk_expansion_score"] * 0.11
        + hb["direct_feature_measure_v28"] * 0.18
        + ht["plasticity_rule_alignment_v34"] * 0.20
        + ht["topology_training_readiness_v34"] * 0.22
    )
    systemic_low_risk_enlargement_penalty = _clip01(
        hs["systemic_low_risk_expansion_penalty"] * 0.42
        + hb["direct_brain_gap_v28"] * 0.22
        + ht["topology_training_gap_v34"] * 0.23
        + (1.0 - hs["systemic_low_risk_expansion_score"]) * 0.05
        + hm["pressure_term_v90"] * 1e-3 * 0.08
    )
    systemic_low_risk_enlargement_readiness = _clip01(
        (
            systemic_low_risk_enlargement_strength
            + systemic_low_risk_enlargement_structure
            + systemic_low_risk_enlargement_route
            + systemic_low_risk_enlargement_learning
            + (1.0 - systemic_low_risk_enlargement_penalty)
        )
        / 5.0
    )
    systemic_low_risk_enlargement_score = _clip01(
        (
            systemic_low_risk_enlargement_readiness
            + systemic_low_risk_enlargement_strength
            + systemic_low_risk_enlargement_route
            + systemic_low_risk_enlargement_learning
            + (1.0 - systemic_low_risk_enlargement_penalty)
        )
        / 5.0
    )
    systemic_low_risk_enlargement_margin = (
        systemic_low_risk_enlargement_strength
        + systemic_low_risk_enlargement_structure
        + systemic_low_risk_enlargement_route
        + systemic_low_risk_enlargement_learning
        + systemic_low_risk_enlargement_readiness
        + systemic_low_risk_enlargement_score
        - systemic_low_risk_enlargement_penalty
        + hm["encoding_margin_v90"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_enlargement_strength": systemic_low_risk_enlargement_strength,
            "systemic_low_risk_enlargement_structure": systemic_low_risk_enlargement_structure,
            "systemic_low_risk_enlargement_route": systemic_low_risk_enlargement_route,
            "systemic_low_risk_enlargement_learning": systemic_low_risk_enlargement_learning,
            "systemic_low_risk_enlargement_penalty": systemic_low_risk_enlargement_penalty,
            "systemic_low_risk_enlargement_readiness": systemic_low_risk_enlargement_readiness,
            "systemic_low_risk_enlargement_score": systemic_low_risk_enlargement_score,
            "systemic_low_risk_enlargement_margin": systemic_low_risk_enlargement_margin,
        },
        "systemic_low_risk_enlargement_equation": {
            "strength_term": "A_sys_enlarge = mix(S_sys_expand_score, R_sys_expand, 1 - P_sys_expand, M_brain_direct_v28, R_train_v34)",
            "structure_term": "S_sys_enlarge = mix(S_sys_expand, R_sys_expand, D_structure_v28, B_struct_v34)",
            "route_term": "R_sys_enlarge = mix(R_sys_expand, S_sys_expand, D_route_v28, H_sys_expand_v34)",
            "learning_term": "L_sys_enlarge = mix(L_sys_expand, S_sys_expand_score, D_feature_v28, B_plastic_v34, R_train_v34)",
            "system_term": "M_sys_enlarge = A_sys_enlarge + S_sys_enlarge + R_sys_enlarge + L_sys_enlarge - P_sys_enlarge",
        },
        "project_readout": {
            "summary": "systemic low-risk zone enlargement validation checks whether systemic low-risk expansion begins to enlarge into a broader low-risk regime instead of only extending the previous edge.",
            "next_question": "next verify whether this enlargement still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Zone Enlargement Validation Report",
        "",
        f"- systemic_low_risk_enlargement_strength: {hm['systemic_low_risk_enlargement_strength']:.6f}",
        f"- systemic_low_risk_enlargement_structure: {hm['systemic_low_risk_enlargement_structure']:.6f}",
        f"- systemic_low_risk_enlargement_route: {hm['systemic_low_risk_enlargement_route']:.6f}",
        f"- systemic_low_risk_enlargement_learning: {hm['systemic_low_risk_enlargement_learning']:.6f}",
        f"- systemic_low_risk_enlargement_penalty: {hm['systemic_low_risk_enlargement_penalty']:.6f}",
        f"- systemic_low_risk_enlargement_readiness: {hm['systemic_low_risk_enlargement_readiness']:.6f}",
        f"- systemic_low_risk_enlargement_score: {hm['systemic_low_risk_enlargement_score']:.6f}",
        f"- systemic_low_risk_enlargement_margin: {hm['systemic_low_risk_enlargement_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_zone_enlargement_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
