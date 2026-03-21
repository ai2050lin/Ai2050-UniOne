from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_amplification_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_low_risk_steady_amplification_validation_summary() -> dict:
    systemic_steady = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_steady_amplification_validation_20260321" / "summary.json"
    )
    systemic = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_stable_amplification_validation_20260321" / "summary.json"
    )
    brain_v24 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v24_20260321" / "summary.json"
    )
    bridge_v30 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v30_20260321" / "summary.json"
    )

    hs = systemic_steady["headline_metrics"]
    hb = systemic["headline_metrics"]
    hd = brain_v24["headline_metrics"]
    ht = bridge_v30["headline_metrics"]

    low_risk_strength = _clip01(
        hs["systemic_steady_score"] * 0.36
        + hs["systemic_steady_readiness"] * 0.20
        + hb["systemic_score"] * 0.06
        + hd["direct_brain_measure_v24"] * 0.18
        + ht["topology_training_readiness_v30"] * 0.20
    )
    low_risk_structure = _clip01(
        hs["systemic_steady_structure"] * 0.42
        + hb["systemic_structure_stability"] * 0.06
        + hd["direct_structure_measure_v24"] * 0.26
        + ht["structure_rule_alignment_v30"] * 0.26
    )
    low_risk_route = _clip01(
        hs["systemic_steady_route"] * 0.42
        + hb["systemic_route_stability"] * 0.06
        + hd["direct_route_measure_v24"] * 0.26
        + ht["systemic_steady_guard_v30"] * 0.26
    )
    low_risk_learning = _clip01(
        hs["systemic_steady_learning"] * 0.34
        + hb["systemic_learning_lift"] * 0.06
        + hd["direct_feature_measure_v24"] * 0.18
        + ht["plasticity_rule_alignment_v30"] * 0.18
        + ht["topology_training_readiness_v30"] * 0.24
    )
    low_risk_penalty = _clip01(
        hs["systemic_steady_penalty"] * 0.48
        + hd["direct_brain_gap_v24"] * 0.22
        + ht["topology_training_gap_v30"] * 0.24
        + hb["systemic_residual_penalty"] * 0.06
    )
    low_risk_readiness = _clip01(
        (
            low_risk_strength
            + low_risk_structure
            + low_risk_route
            + low_risk_learning
            + (1.0 - low_risk_penalty)
        )
        / 5.0
    )
    low_risk_score = _clip01(
        (
            low_risk_readiness
            + low_risk_strength
            + low_risk_learning
            + (1.0 - low_risk_penalty)
        )
        / 4.0
    )
    low_risk_margin = (
        low_risk_strength
        + low_risk_structure
        + low_risk_route
        + low_risk_learning
        + low_risk_readiness
        + low_risk_score
        - low_risk_penalty
    )

    return {
        "headline_metrics": {
            "low_risk_strength": low_risk_strength,
            "low_risk_structure": low_risk_structure,
            "low_risk_route": low_risk_route,
            "low_risk_learning": low_risk_learning,
            "low_risk_penalty": low_risk_penalty,
            "low_risk_readiness": low_risk_readiness,
            "low_risk_score": low_risk_score,
            "low_risk_margin": low_risk_margin,
        },
        "low_risk_equation": {
            "strength_term": "A_low = mix(S_system_steady_score, R_system_steady, S_system_score, M_brain_direct_v24, R_train_v30)",
            "structure_term": "S_low = mix(S_system_steady, S_system, D_structure_v24, B_struct_v30)",
            "route_term": "R_low = mix(R_system_steady, R_system, D_route_v24, H_system_steady_v30)",
            "learning_term": "L_low = mix(L_system_steady, L_system, D_feature_v24, B_plastic_v30, R_train_v30)",
            "system_term": "M_low = A_low + S_low + R_low + L_low - P_low",
        },
        "project_readout": {
            "summary": "low-risk steady amplification validation checks whether systemic steady amplification starts moving into a lower-risk regime.",
            "next_question": "next verify whether this lower-risk trend persists when pushed into larger, longer, higher-pressure systems.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Low-Risk Steady Amplification Validation Report",
        "",
        f"- low_risk_strength: {hm['low_risk_strength']:.6f}",
        f"- low_risk_structure: {hm['low_risk_structure']:.6f}",
        f"- low_risk_route: {hm['low_risk_route']:.6f}",
        f"- low_risk_learning: {hm['low_risk_learning']:.6f}",
        f"- low_risk_penalty: {hm['low_risk_penalty']:.6f}",
        f"- low_risk_readiness: {hm['low_risk_readiness']:.6f}",
        f"- low_risk_score: {hm['low_risk_score']:.6f}",
        f"- low_risk_margin: {hm['low_risk_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_low_risk_steady_amplification_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
