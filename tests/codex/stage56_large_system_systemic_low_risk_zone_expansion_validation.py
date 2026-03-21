from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_expansion_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_systemic_low_risk_zone_expansion_validation_summary() -> dict:
    expansion = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_zone_expansion_validation_20260321" / "summary.json"
    )
    brain_v27 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v27_20260321" / "summary.json"
    )
    bridge_v33 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v33_20260321" / "summary.json"
    )
    v89 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v89_20260321" / "summary.json"
    )

    he = expansion["headline_metrics"]
    hb = brain_v27["headline_metrics"]
    ht = bridge_v33["headline_metrics"]
    hm = v89["headline_metrics"]

    systemic_low_risk_expansion_strength = _clip01(
        he["low_risk_expansion_score"] * 0.32
        + he["low_risk_expansion_readiness"] * 0.18
        + (1.0 - he["low_risk_expansion_penalty"]) * 0.12
        + hb["direct_brain_measure_v27"] * 0.18
        + ht["topology_training_readiness_v33"] * 0.20
    )
    systemic_low_risk_expansion_structure = _clip01(
        he["low_risk_expansion_structure"] * 0.40
        + he["low_risk_expansion_route"] * 0.08
        + hb["direct_structure_measure_v27"] * 0.24
        + ht["structure_rule_alignment_v33"] * 0.28
    )
    systemic_low_risk_expansion_route = _clip01(
        he["low_risk_expansion_route"] * 0.40
        + he["low_risk_expansion_structure"] * 0.08
        + hb["direct_route_measure_v27"] * 0.24
        + ht["low_risk_expansion_guard_v33"] * 0.28
    )
    systemic_low_risk_expansion_learning = _clip01(
        he["low_risk_expansion_learning"] * 0.30
        + he["low_risk_expansion_score"] * 0.10
        + hb["direct_feature_measure_v27"] * 0.18
        + ht["plasticity_rule_alignment_v33"] * 0.20
        + ht["topology_training_readiness_v33"] * 0.22
    )
    systemic_low_risk_expansion_penalty = _clip01(
        he["low_risk_expansion_penalty"] * 0.44
        + hb["direct_brain_gap_v27"] * 0.22
        + ht["topology_training_gap_v33"] * 0.24
        + (1.0 - he["low_risk_expansion_score"]) * 0.06
        + hm["pressure_term_v89"] * 1e-3 * 0.04
    )
    systemic_low_risk_expansion_readiness = _clip01(
        (
            systemic_low_risk_expansion_strength
            + systemic_low_risk_expansion_structure
            + systemic_low_risk_expansion_route
            + systemic_low_risk_expansion_learning
            + (1.0 - systemic_low_risk_expansion_penalty)
        )
        / 5.0
    )
    systemic_low_risk_expansion_score = _clip01(
        (
            systemic_low_risk_expansion_readiness
            + systemic_low_risk_expansion_strength
            + systemic_low_risk_expansion_route
            + systemic_low_risk_expansion_learning
            + (1.0 - systemic_low_risk_expansion_penalty)
        )
        / 5.0
    )
    systemic_low_risk_expansion_margin = (
        systemic_low_risk_expansion_strength
        + systemic_low_risk_expansion_structure
        + systemic_low_risk_expansion_route
        + systemic_low_risk_expansion_learning
        + systemic_low_risk_expansion_readiness
        + systemic_low_risk_expansion_score
        - systemic_low_risk_expansion_penalty
        + hm["encoding_margin_v89"] * 1e-15
    )

    return {
        "headline_metrics": {
            "systemic_low_risk_expansion_strength": systemic_low_risk_expansion_strength,
            "systemic_low_risk_expansion_structure": systemic_low_risk_expansion_structure,
            "systemic_low_risk_expansion_route": systemic_low_risk_expansion_route,
            "systemic_low_risk_expansion_learning": systemic_low_risk_expansion_learning,
            "systemic_low_risk_expansion_penalty": systemic_low_risk_expansion_penalty,
            "systemic_low_risk_expansion_readiness": systemic_low_risk_expansion_readiness,
            "systemic_low_risk_expansion_score": systemic_low_risk_expansion_score,
            "systemic_low_risk_expansion_margin": systemic_low_risk_expansion_margin,
        },
        "systemic_low_risk_expansion_equation": {
            "strength_term": "A_sys_expand = mix(S_expand_score, R_expand, 1 - P_expand, M_brain_direct_v27, R_train_v33)",
            "structure_term": "S_sys_expand = mix(S_expand, R_expand, D_structure_v27, B_struct_v33)",
            "route_term": "R_sys_expand = mix(R_expand, S_expand, D_route_v27, H_expand_v33)",
            "learning_term": "L_sys_expand = mix(L_expand, S_expand_score, D_feature_v27, B_plastic_v33, R_train_v33)",
            "system_term": "M_sys_expand = A_sys_expand + S_sys_expand + R_sys_expand + L_sys_expand - P_sys_expand",
        },
        "project_readout": {
            "summary": "systemic low-risk zone expansion validation checks whether the low-risk steady zone starts expanding at the systemic level instead of only inside the previous local shell.",
            "next_question": "next verify whether this systemic expansion still survives after it is folded into the next brain direct chain and training bridge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Systemic Low-Risk Zone Expansion Validation Report",
        "",
        f"- systemic_low_risk_expansion_strength: {hm['systemic_low_risk_expansion_strength']:.6f}",
        f"- systemic_low_risk_expansion_structure: {hm['systemic_low_risk_expansion_structure']:.6f}",
        f"- systemic_low_risk_expansion_route: {hm['systemic_low_risk_expansion_route']:.6f}",
        f"- systemic_low_risk_expansion_learning: {hm['systemic_low_risk_expansion_learning']:.6f}",
        f"- systemic_low_risk_expansion_penalty: {hm['systemic_low_risk_expansion_penalty']:.6f}",
        f"- systemic_low_risk_expansion_readiness: {hm['systemic_low_risk_expansion_readiness']:.6f}",
        f"- systemic_low_risk_expansion_score: {hm['systemic_low_risk_expansion_score']:.6f}",
        f"- systemic_low_risk_expansion_margin: {hm['systemic_low_risk_expansion_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_systemic_low_risk_zone_expansion_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
