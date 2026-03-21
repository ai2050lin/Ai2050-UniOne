from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_zone_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_low_risk_steady_zone_validation_summary() -> dict:
    low_risk = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_amplification_validation_20260321" / "summary.json"
    )
    systemic_steady = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_steady_amplification_validation_20260321" / "summary.json"
    )
    brain_v25 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v25_20260321" / "summary.json"
    )
    bridge_v31 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v31_20260321" / "summary.json"
    )

    hl = low_risk["headline_metrics"]
    hs = systemic_steady["headline_metrics"]
    hb = brain_v25["headline_metrics"]
    ht = bridge_v31["headline_metrics"]

    low_risk_zone_strength = _clip01(
        hl["low_risk_score"] * 0.36
        + hl["low_risk_readiness"] * 0.20
        + hs["systemic_steady_score"] * 0.06
        + hb["direct_brain_measure_v25"] * 0.18
        + ht["topology_training_readiness_v31"] * 0.20
    )
    low_risk_zone_structure = _clip01(
        hl["low_risk_structure"] * 0.42
        + hs["systemic_steady_structure"] * 0.06
        + hb["direct_structure_measure_v25"] * 0.26
        + ht["structure_rule_alignment_v31"] * 0.26
    )
    low_risk_zone_route = _clip01(
        hl["low_risk_route"] * 0.42
        + hs["systemic_steady_route"] * 0.06
        + hb["direct_route_measure_v25"] * 0.26
        + ht["low_risk_guard_v31"] * 0.26
    )
    low_risk_zone_learning = _clip01(
        hl["low_risk_learning"] * 0.34
        + hs["systemic_steady_learning"] * 0.06
        + hb["direct_feature_measure_v25"] * 0.18
        + ht["plasticity_rule_alignment_v31"] * 0.18
        + ht["topology_training_readiness_v31"] * 0.24
    )
    low_risk_zone_penalty = _clip01(
        hl["low_risk_penalty"] * 0.48
        + hb["direct_brain_gap_v25"] * 0.22
        + ht["topology_training_gap_v31"] * 0.24
        + hs["systemic_steady_penalty"] * 0.06
    )
    low_risk_zone_readiness = _clip01(
        (
            low_risk_zone_strength
            + low_risk_zone_structure
            + low_risk_zone_route
            + low_risk_zone_learning
            + (1.0 - low_risk_zone_penalty)
        )
        / 5.0
    )
    low_risk_zone_score = _clip01(
        (
            low_risk_zone_readiness
            + low_risk_zone_strength
            + low_risk_zone_learning
            + (1.0 - low_risk_zone_penalty)
        )
        / 4.0
    )
    low_risk_zone_margin = (
        low_risk_zone_strength
        + low_risk_zone_structure
        + low_risk_zone_route
        + low_risk_zone_learning
        + low_risk_zone_readiness
        + low_risk_zone_score
        - low_risk_zone_penalty
    )

    return {
        "headline_metrics": {
            "low_risk_zone_strength": low_risk_zone_strength,
            "low_risk_zone_structure": low_risk_zone_structure,
            "low_risk_zone_route": low_risk_zone_route,
            "low_risk_zone_learning": low_risk_zone_learning,
            "low_risk_zone_penalty": low_risk_zone_penalty,
            "low_risk_zone_readiness": low_risk_zone_readiness,
            "low_risk_zone_score": low_risk_zone_score,
            "low_risk_zone_margin": low_risk_zone_margin,
        },
        "low_risk_zone_equation": {
            "strength_term": "A_zone = mix(S_low_score, R_low, S_system_steady_score, M_brain_direct_v25, R_train_v31)",
            "structure_term": "S_zone = mix(S_low, S_system_steady, D_structure_v25, B_struct_v31)",
            "route_term": "R_zone = mix(R_low, R_system_steady, D_route_v25, H_low_v31)",
            "learning_term": "L_zone = mix(L_low, L_system_steady, D_feature_v25, B_plastic_v31, R_train_v31)",
            "system_term": "M_zone = A_zone + S_zone + R_zone + L_zone - P_zone",
        },
        "project_readout": {
            "summary": "low-risk steady zone validation checks whether the low-risk steady amplification trend starts forming a stable low-risk zone.",
            "next_question": "next verify whether this low-risk zone keeps expanding under larger, longer and higher-pressure system conditions.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Low-Risk Steady Zone Validation Report",
        "",
        f"- low_risk_zone_strength: {hm['low_risk_zone_strength']:.6f}",
        f"- low_risk_zone_structure: {hm['low_risk_zone_structure']:.6f}",
        f"- low_risk_zone_route: {hm['low_risk_zone_route']:.6f}",
        f"- low_risk_zone_learning: {hm['low_risk_zone_learning']:.6f}",
        f"- low_risk_zone_penalty: {hm['low_risk_zone_penalty']:.6f}",
        f"- low_risk_zone_readiness: {hm['low_risk_zone_readiness']:.6f}",
        f"- low_risk_zone_score: {hm['low_risk_zone_score']:.6f}",
        f"- low_risk_zone_margin: {hm['low_risk_zone_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_low_risk_steady_zone_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
