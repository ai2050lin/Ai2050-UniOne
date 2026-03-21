from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_broadening_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_zone_broadening_validation_summary() -> dict:
    enlarge = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_enlargement_validation_20260321" / "summary.json"
    )
    brain_v29 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v29_20260321" / "summary.json"
    )
    bridge_v35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v35_20260321" / "summary.json"
    )
    v91 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v91_20260321" / "summary.json"
    )

    hs = enlarge["headline_metrics"]
    hb = brain_v29["headline_metrics"]
    ht = bridge_v35["headline_metrics"]
    hm = v91["headline_metrics"]

    systemic_low_risk_broadening_strength = _clip01(
        hs["systemic_low_risk_enlargement_score"] * 0.31
        + hs["systemic_low_risk_enlargement_readiness"] * 0.19
        + (1.0 - hs["systemic_low_risk_enlargement_penalty"]) * 0.15
        + hb["direct_brain_measure_v29"] * 0.16
        + ht["topology_training_readiness_v35"] * 0.19
    )
    systemic_low_risk_broadening_structure = _clip01(
        hs["systemic_low_risk_enlargement_structure"] * 0.39
        + hs["systemic_low_risk_enlargement_route"] * 0.08
        + hb["direct_structure_measure_v29"] * 0.24
        + ht["structure_rule_alignment_v35"] * 0.29
    )
    systemic_low_risk_broadening_route = _clip01(
        hs["systemic_low_risk_enlargement_route"] * 0.39
        + hs["systemic_low_risk_enlargement_structure"] * 0.08
        + hb["direct_route_measure_v29"] * 0.24
        + ht["systemic_low_risk_enlargement_guard_v35"] * 0.29
    )
    systemic_low_risk_broadening_learning = _clip01(
        hs["systemic_low_risk_enlargement_learning"] * 0.29
        + hs["systemic_low_risk_enlargement_score"] * 0.11
        + hb["direct_feature_measure_v29"] * 0.18
        + ht["plasticity_rule_alignment_v35"] * 0.20
        + ht["topology_training_readiness_v35"] * 0.22
    )
    systemic_low_risk_broadening_penalty = _clip01(
        hs["systemic_low_risk_enlargement_penalty"] * 0.40
        + hb["direct_brain_gap_v29"] * 0.22
        + ht["topology_training_gap_v35"] * 0.22
        + (1.0 - hs["systemic_low_risk_enlargement_score"]) * 0.05
        + hm["pressure_term_v91"] * 1e-3 * 0.11
    )
    systemic_low_risk_broadening_readiness = _clip01(
        (
            systemic_low_risk_broadening_strength
            + systemic_low_risk_broadening_structure
            + systemic_low_risk_broadening_route
            + systemic_low_risk_broadening_learning
            + (1.0 - systemic_low_risk_broadening_penalty)
        )
        / 5.0
    )
    systemic_low_risk_broadening_score = _clip01(
        (
            systemic_low_risk_broadening_readiness
            + systemic_low_risk_broadening_strength
            + systemic_low_risk_broadening_route
            + systemic_low_risk_broadening_learning
            + (1.0 - systemic_low_risk_broadening_penalty)
        )
        / 5.0
    )
    systemic_low_risk_broadening_margin = (
        systemic_low_risk_broadening_strength
        + systemic_low_risk_broadening_structure
        + systemic_low_risk_broadening_route
        + systemic_low_risk_broadening_learning
        + systemic_low_risk_broadening_readiness
        + systemic_low_risk_broadening_score
        - systemic_low_risk_broadening_penalty
        + hm["encoding_margin_v91"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_broadening_strength": systemic_low_risk_broadening_strength,
            "systemic_low_risk_broadening_structure": systemic_low_risk_broadening_structure,
            "systemic_low_risk_broadening_route": systemic_low_risk_broadening_route,
            "systemic_low_risk_broadening_learning": systemic_low_risk_broadening_learning,
            "systemic_low_risk_broadening_penalty": systemic_low_risk_broadening_penalty,
            "systemic_low_risk_broadening_readiness": systemic_low_risk_broadening_readiness,
            "systemic_low_risk_broadening_score": systemic_low_risk_broadening_score,
            "systemic_low_risk_broadening_margin": systemic_low_risk_broadening_margin,
        },
        "systemic_low_risk_broadening_equation": {
            "strength_term": "A_sys_broad = mix(S_sys_enlarge_score, R_sys_enlarge, 1 - P_sys_enlarge, M_brain_direct_v29, R_train_v35)",
            "structure_term": "S_sys_broad = mix(S_sys_enlarge, R_sys_enlarge, D_structure_v29, B_struct_v35)",
            "route_term": "R_sys_broad = mix(R_sys_enlarge, S_sys_enlarge, D_route_v29, H_sys_enlarge_v35)",
            "learning_term": "L_sys_broad = mix(L_sys_enlarge, S_sys_enlarge_score, D_feature_v29, B_plastic_v35, R_train_v35)",
            "system_term": "M_sys_broad = A_sys_broad + S_sys_broad + R_sys_broad + L_sys_broad - P_sys_broad",
        },
        "project_readout": {
            "summary": "systemic low-risk zone broadening validation checks whether systemic low-risk enlargement begins broadening into a wider low-risk regime instead of only growing the previous band.",
            "next_question": "next verify whether this broadening still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Zone Broadening Validation Report",
        "",
        f"- systemic_low_risk_broadening_strength: {hm['systemic_low_risk_broadening_strength']:.6f}",
        f"- systemic_low_risk_broadening_structure: {hm['systemic_low_risk_broadening_structure']:.6f}",
        f"- systemic_low_risk_broadening_route: {hm['systemic_low_risk_broadening_route']:.6f}",
        f"- systemic_low_risk_broadening_learning: {hm['systemic_low_risk_broadening_learning']:.6f}",
        f"- systemic_low_risk_broadening_penalty: {hm['systemic_low_risk_broadening_penalty']:.6f}",
        f"- systemic_low_risk_broadening_readiness: {hm['systemic_low_risk_broadening_readiness']:.6f}",
        f"- systemic_low_risk_broadening_score: {hm['systemic_low_risk_broadening_score']:.6f}",
        f"- systemic_low_risk_broadening_margin: {hm['systemic_low_risk_broadening_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_zone_broadening_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
