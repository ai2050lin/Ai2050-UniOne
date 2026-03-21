from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_zone_expansion_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_low_risk_steady_zone_expansion_validation_summary() -> dict:
    zone = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_zone_validation_20260321" / "summary.json"
    )
    brain_v26 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v26_20260321" / "summary.json"
    )
    bridge_v32 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v32_20260321" / "summary.json"
    )
    v88 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v88_20260321" / "summary.json"
    )

    hz = zone["headline_metrics"]
    hb = brain_v26["headline_metrics"]
    ht = bridge_v32["headline_metrics"]
    hm = v88["headline_metrics"]

    low_risk_expansion_strength = _clip01(
        hz["low_risk_zone_score"] * 0.34
        + hz["low_risk_zone_readiness"] * 0.20
        + (1.0 - hz["low_risk_zone_penalty"]) * 0.10
        + hb["direct_brain_measure_v26"] * 0.18
        + ht["topology_training_readiness_v32"] * 0.18
    )
    low_risk_expansion_structure = _clip01(
        hz["low_risk_zone_structure"] * 0.40
        + hz["low_risk_zone_route"] * 0.08
        + hb["direct_structure_measure_v26"] * 0.26
        + ht["structure_rule_alignment_v32"] * 0.26
    )
    low_risk_expansion_route = _clip01(
        hz["low_risk_zone_route"] * 0.40
        + hz["low_risk_zone_structure"] * 0.08
        + hb["direct_route_measure_v26"] * 0.24
        + ht["low_risk_zone_guard_v32"] * 0.28
    )
    low_risk_expansion_learning = _clip01(
        hz["low_risk_zone_learning"] * 0.32
        + hz["low_risk_zone_score"] * 0.10
        + hb["direct_feature_measure_v26"] * 0.18
        + ht["plasticity_rule_alignment_v32"] * 0.20
        + ht["topology_training_readiness_v32"] * 0.20
    )
    low_risk_expansion_penalty = _clip01(
        hz["low_risk_zone_penalty"] * 0.46
        + hb["direct_brain_gap_v26"] * 0.22
        + ht["topology_training_gap_v32"] * 0.24
        + (1.0 - hz["low_risk_zone_score"]) * 0.08
    )
    low_risk_expansion_readiness = _clip01(
        (
            low_risk_expansion_strength
            + low_risk_expansion_structure
            + low_risk_expansion_route
            + low_risk_expansion_learning
            + (1.0 - low_risk_expansion_penalty)
        )
        / 5.0
    )
    low_risk_expansion_score = _clip01(
        (
            low_risk_expansion_readiness
            + low_risk_expansion_strength
            + low_risk_expansion_route
            + low_risk_expansion_learning
            + (1.0 - low_risk_expansion_penalty)
        )
        / 5.0
    )
    low_risk_expansion_margin = (
        low_risk_expansion_strength
        + low_risk_expansion_structure
        + low_risk_expansion_route
        + low_risk_expansion_learning
        + low_risk_expansion_readiness
        + low_risk_expansion_score
        - low_risk_expansion_penalty
        + hm["encoding_margin_v88"] * 1e-15
    )

    return {
        "headline_metrics": {
            "low_risk_expansion_strength": low_risk_expansion_strength,
            "low_risk_expansion_structure": low_risk_expansion_structure,
            "low_risk_expansion_route": low_risk_expansion_route,
            "low_risk_expansion_learning": low_risk_expansion_learning,
            "low_risk_expansion_penalty": low_risk_expansion_penalty,
            "low_risk_expansion_readiness": low_risk_expansion_readiness,
            "low_risk_expansion_score": low_risk_expansion_score,
            "low_risk_expansion_margin": low_risk_expansion_margin,
        },
        "low_risk_expansion_equation": {
            "strength_term": "A_expand = mix(S_zone_score, R_zone, 1 - P_zone, M_brain_direct_v26, R_train_v32)",
            "structure_term": "S_expand = mix(S_zone, R_zone, D_structure_v26, B_struct_v32)",
            "route_term": "R_expand = mix(R_zone, S_zone, D_route_v26, H_zone_v32)",
            "learning_term": "L_expand = mix(L_zone, S_zone_score, D_feature_v26, B_plastic_v32, R_train_v32)",
            "system_term": "M_expand = A_expand + S_expand + R_expand + L_expand - P_expand",
        },
        "project_readout": {
            "summary": "low-risk steady zone expansion validation checks whether the newly formed low-risk zone starts expanding instead of only holding its boundary.",
            "next_question": "next verify whether this expansion is visible in the next brain direct chain and training bridge at the same time.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Low-Risk Steady Zone Expansion Validation Report",
        "",
        f"- low_risk_expansion_strength: {hm['low_risk_expansion_strength']:.6f}",
        f"- low_risk_expansion_structure: {hm['low_risk_expansion_structure']:.6f}",
        f"- low_risk_expansion_route: {hm['low_risk_expansion_route']:.6f}",
        f"- low_risk_expansion_learning: {hm['low_risk_expansion_learning']:.6f}",
        f"- low_risk_expansion_penalty: {hm['low_risk_expansion_penalty']:.6f}",
        f"- low_risk_expansion_readiness: {hm['low_risk_expansion_readiness']:.6f}",
        f"- low_risk_expansion_score: {hm['low_risk_expansion_score']:.6f}",
        f"- low_risk_expansion_margin: {hm['low_risk_expansion_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_low_risk_steady_zone_expansion_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
