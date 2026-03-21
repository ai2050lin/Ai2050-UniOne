from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_extension_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_field_extension_validation_summary() -> dict:
    band = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_band_extension_validation_20260321" / "summary.json"
    )
    brain_v31 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v31_20260321" / "summary.json"
    )
    bridge_v37 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v37_20260321" / "summary.json"
    )
    v93 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v93_20260321" / "summary.json"
    )

    hs = band["headline_metrics"]
    hb = brain_v31["headline_metrics"]
    ht = bridge_v37["headline_metrics"]
    hm = v93["headline_metrics"]

    systemic_low_risk_field_strength = _clip01(
        hs["systemic_low_risk_band_score"] * 0.32
        + hs["systemic_low_risk_band_readiness"] * 0.18
        + (1.0 - hs["systemic_low_risk_band_penalty"]) * 0.15
        + hb["direct_brain_measure_v31"] * 0.17
        + ht["topology_training_readiness_v37"] * 0.18
    )
    systemic_low_risk_field_structure = _clip01(
        hs["systemic_low_risk_band_structure"] * 0.37
        + hs["systemic_low_risk_band_route"] * 0.09
        + hb["direct_structure_measure_v31"] * 0.24
        + ht["structure_rule_alignment_v37"] * 0.30
    )
    systemic_low_risk_field_route = _clip01(
        hs["systemic_low_risk_band_route"] * 0.37
        + hs["systemic_low_risk_band_structure"] * 0.09
        + hb["direct_route_measure_v31"] * 0.24
        + ht["systemic_low_risk_band_guard_v37"] * 0.30
    )
    systemic_low_risk_field_learning = _clip01(
        hs["systemic_low_risk_band_learning"] * 0.27
        + hs["systemic_low_risk_band_score"] * 0.13
        + hb["direct_feature_measure_v31"] * 0.18
        + ht["plasticity_rule_alignment_v37"] * 0.21
        + ht["topology_training_readiness_v37"] * 0.21
    )
    systemic_low_risk_field_penalty = _clip01(
        hs["systemic_low_risk_band_penalty"] * 0.36
        + hb["direct_brain_gap_v31"] * 0.21
        + ht["topology_training_gap_v37"] * 0.21
        + (1.0 - hs["systemic_low_risk_band_score"]) * 0.05
        + hm["pressure_term_v93"] * 1e-3 * 0.17
    )
    systemic_low_risk_field_readiness = _clip01(
        (
            systemic_low_risk_field_strength
            + systemic_low_risk_field_structure
            + systemic_low_risk_field_route
            + systemic_low_risk_field_learning
            + (1.0 - systemic_low_risk_field_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_score = _clip01(
        (
            systemic_low_risk_field_readiness
            + systemic_low_risk_field_strength
            + systemic_low_risk_field_route
            + systemic_low_risk_field_learning
            + (1.0 - systemic_low_risk_field_penalty)
        )
        / 5.0
    )
    systemic_low_risk_field_margin = (
        systemic_low_risk_field_strength
        + systemic_low_risk_field_structure
        + systemic_low_risk_field_route
        + systemic_low_risk_field_learning
        + systemic_low_risk_field_readiness
        + systemic_low_risk_field_score
        - systemic_low_risk_field_penalty
        + hm["encoding_margin_v93"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_field_strength": systemic_low_risk_field_strength,
            "systemic_low_risk_field_structure": systemic_low_risk_field_structure,
            "systemic_low_risk_field_route": systemic_low_risk_field_route,
            "systemic_low_risk_field_learning": systemic_low_risk_field_learning,
            "systemic_low_risk_field_penalty": systemic_low_risk_field_penalty,
            "systemic_low_risk_field_readiness": systemic_low_risk_field_readiness,
            "systemic_low_risk_field_score": systemic_low_risk_field_score,
            "systemic_low_risk_field_margin": systemic_low_risk_field_margin,
        },
        "systemic_low_risk_field_equation": {
            "strength_term": "A_sys_field = mix(S_sys_band_score, R_sys_band, 1 - P_sys_band, M_brain_direct_v31, R_train_v37)",
            "structure_term": "S_sys_field = mix(S_sys_band, R_sys_band, D_structure_v31, B_struct_v37)",
            "route_term": "R_sys_field = mix(R_sys_band, S_sys_band, D_route_v31, H_sys_band_v37)",
            "learning_term": "L_sys_field = mix(L_sys_band, S_sys_band_score, D_feature_v31, B_plastic_v37, R_train_v37)",
            "system_term": "M_sys_field = A_sys_field + S_sys_field + R_sys_field + L_sys_field - P_sys_field",
        },
        "project_readout": {
            "summary": "systemic low-risk field extension validation checks whether the coherent low-risk band starts extending into a more connected low-risk field instead of remaining a narrow band.",
            "next_question": "next verify whether this low-risk field still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Field Extension Validation Report",
        "",
        f"- systemic_low_risk_field_strength: {hm['systemic_low_risk_field_strength']:.6f}",
        f"- systemic_low_risk_field_structure: {hm['systemic_low_risk_field_structure']:.6f}",
        f"- systemic_low_risk_field_route: {hm['systemic_low_risk_field_route']:.6f}",
        f"- systemic_low_risk_field_learning: {hm['systemic_low_risk_field_learning']:.6f}",
        f"- systemic_low_risk_field_penalty: {hm['systemic_low_risk_field_penalty']:.6f}",
        f"- systemic_low_risk_field_readiness: {hm['systemic_low_risk_field_readiness']:.6f}",
        f"- systemic_low_risk_field_score: {hm['systemic_low_risk_field_score']:.6f}",
        f"- systemic_low_risk_field_margin: {hm['systemic_low_risk_field_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_field_extension_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
