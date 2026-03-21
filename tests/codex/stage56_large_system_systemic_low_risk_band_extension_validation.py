from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_band_extension_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_band_extension_validation_summary() -> dict:
    broad = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_broadening_validation_20260321" / "summary.json"
    )
    brain_v30 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v30_20260321" / "summary.json"
    )
    bridge_v36 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v36_20260321" / "summary.json"
    )
    v92 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v92_20260321" / "summary.json"
    )

    hs = broad["headline_metrics"]
    hb = brain_v30["headline_metrics"]
    ht = bridge_v36["headline_metrics"]
    hm = v92["headline_metrics"]

    systemic_low_risk_band_strength = _clip01(
        hs["systemic_low_risk_broadening_score"] * 0.31
        + hs["systemic_low_risk_broadening_readiness"] * 0.19
        + (1.0 - hs["systemic_low_risk_broadening_penalty"]) * 0.16
        + hb["direct_brain_measure_v30"] * 0.16
        + ht["topology_training_readiness_v36"] * 0.18
    )
    systemic_low_risk_band_structure = _clip01(
        hs["systemic_low_risk_broadening_structure"] * 0.38
        + hs["systemic_low_risk_broadening_route"] * 0.08
        + hb["direct_structure_measure_v30"] * 0.24
        + ht["structure_rule_alignment_v36"] * 0.30
    )
    systemic_low_risk_band_route = _clip01(
        hs["systemic_low_risk_broadening_route"] * 0.38
        + hs["systemic_low_risk_broadening_structure"] * 0.08
        + hb["direct_route_measure_v30"] * 0.24
        + ht["systemic_low_risk_broadening_guard_v36"] * 0.30
    )
    systemic_low_risk_band_learning = _clip01(
        hs["systemic_low_risk_broadening_learning"] * 0.28
        + hs["systemic_low_risk_broadening_score"] * 0.12
        + hb["direct_feature_measure_v30"] * 0.18
        + ht["plasticity_rule_alignment_v36"] * 0.20
        + ht["topology_training_readiness_v36"] * 0.22
    )
    systemic_low_risk_band_penalty = _clip01(
        hs["systemic_low_risk_broadening_penalty"] * 0.38
        + hb["direct_brain_gap_v30"] * 0.22
        + ht["topology_training_gap_v36"] * 0.22
        + (1.0 - hs["systemic_low_risk_broadening_score"]) * 0.05
        + hm["pressure_term_v92"] * 1e-3 * 0.13
    )
    systemic_low_risk_band_readiness = _clip01(
        (
            systemic_low_risk_band_strength
            + systemic_low_risk_band_structure
            + systemic_low_risk_band_route
            + systemic_low_risk_band_learning
            + (1.0 - systemic_low_risk_band_penalty)
        )
        / 5.0
    )
    systemic_low_risk_band_score = _clip01(
        (
            systemic_low_risk_band_readiness
            + systemic_low_risk_band_strength
            + systemic_low_risk_band_route
            + systemic_low_risk_band_learning
            + (1.0 - systemic_low_risk_band_penalty)
        )
        / 5.0
    )
    systemic_low_risk_band_margin = (
        systemic_low_risk_band_strength
        + systemic_low_risk_band_structure
        + systemic_low_risk_band_route
        + systemic_low_risk_band_learning
        + systemic_low_risk_band_readiness
        + systemic_low_risk_band_score
        - systemic_low_risk_band_penalty
        + hm["encoding_margin_v92"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_band_strength": systemic_low_risk_band_strength,
            "systemic_low_risk_band_structure": systemic_low_risk_band_structure,
            "systemic_low_risk_band_route": systemic_low_risk_band_route,
            "systemic_low_risk_band_learning": systemic_low_risk_band_learning,
            "systemic_low_risk_band_penalty": systemic_low_risk_band_penalty,
            "systemic_low_risk_band_readiness": systemic_low_risk_band_readiness,
            "systemic_low_risk_band_score": systemic_low_risk_band_score,
            "systemic_low_risk_band_margin": systemic_low_risk_band_margin,
        },
        "systemic_low_risk_band_equation": {
            "strength_term": "A_sys_band = mix(S_sys_broad_score, R_sys_broad, 1 - P_sys_broad, M_brain_direct_v30, R_train_v36)",
            "structure_term": "S_sys_band = mix(S_sys_broad, R_sys_broad, D_structure_v30, B_struct_v36)",
            "route_term": "R_sys_band = mix(R_sys_broad, S_sys_broad, D_route_v30, H_sys_broad_v36)",
            "learning_term": "L_sys_band = mix(L_sys_broad, S_sys_broad_score, D_feature_v30, B_plastic_v36, R_train_v36)",
            "system_term": "M_sys_band = A_sys_band + S_sys_band + R_sys_band + L_sys_band - P_sys_band",
        },
        "project_readout": {
            "summary": "systemic low-risk band extension validation checks whether the broader low-risk regime starts extending into a more coherent low-risk band instead of only widening the previous band.",
            "next_question": "next verify whether this band extension still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Band Extension Validation Report",
        "",
        f"- systemic_low_risk_band_strength: {hm['systemic_low_risk_band_strength']:.6f}",
        f"- systemic_low_risk_band_structure: {hm['systemic_low_risk_band_structure']:.6f}",
        f"- systemic_low_risk_band_route: {hm['systemic_low_risk_band_route']:.6f}",
        f"- systemic_low_risk_band_learning: {hm['systemic_low_risk_band_learning']:.6f}",
        f"- systemic_low_risk_band_penalty: {hm['systemic_low_risk_band_penalty']:.6f}",
        f"- systemic_low_risk_band_readiness: {hm['systemic_low_risk_band_readiness']:.6f}",
        f"- systemic_low_risk_band_score: {hm['systemic_low_risk_band_score']:.6f}",
        f"- systemic_low_risk_band_margin: {hm['systemic_low_risk_band_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_band_extension_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
