from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v26_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v26_summary() -> dict:
    v25 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v25_20260321" / "summary.json"
    )
    low_risk_zone = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_zone_validation_20260321" / "summary.json"
    )
    bridge_v31 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v31_20260321" / "summary.json"
    )

    hv = v25["headline_metrics"]
    hs = low_risk_zone["headline_metrics"]
    hb = bridge_v31["headline_metrics"]

    direct_origin_measure_v26 = _clip01(
        hv["direct_origin_measure_v25"] * 0.46
        + hs["low_risk_zone_readiness"] * 0.22
        + (1.0 - hs["low_risk_zone_penalty"]) * 0.15
        + hb["topology_training_readiness_v31"] * 0.17
    )
    direct_feature_measure_v26 = _clip01(
        hv["direct_feature_measure_v25"] * 0.44
        + hs["low_risk_zone_learning"] * 0.26
        + (1.0 - hs["low_risk_zone_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v31"] * 0.20
    )
    direct_structure_measure_v26 = _clip01(
        hv["direct_structure_measure_v25"] * 0.42
        + hs["low_risk_zone_structure"] * 0.28
        + (1.0 - hs["low_risk_zone_penalty"]) * 0.10
        + hb["structure_rule_alignment_v31"] * 0.20
    )
    direct_route_measure_v26 = _clip01(
        hv["direct_route_measure_v25"] * 0.42
        + hs["low_risk_zone_route"] * 0.28
        + hs["low_risk_zone_structure"] * 0.08
        + (1.0 - hs["low_risk_zone_penalty"]) * 0.05
        + hb["low_risk_guard_v31"] * 0.17
    )
    direct_brain_measure_v26 = (
        direct_origin_measure_v26
        + direct_feature_measure_v26
        + direct_structure_measure_v26
        + direct_route_measure_v26
    ) / 4.0
    direct_brain_gap_v26 = 1.0 - direct_brain_measure_v26
    direct_low_risk_zone_alignment_v26 = (
        direct_structure_measure_v26
        + direct_route_measure_v26
        + hs["low_risk_zone_readiness"]
        + hb["topology_training_readiness_v31"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v26": direct_origin_measure_v26,
            "direct_feature_measure_v26": direct_feature_measure_v26,
            "direct_structure_measure_v26": direct_structure_measure_v26,
            "direct_route_measure_v26": direct_route_measure_v26,
            "direct_brain_measure_v26": direct_brain_measure_v26,
            "direct_brain_gap_v26": direct_brain_gap_v26,
            "direct_low_risk_zone_alignment_v26": direct_low_risk_zone_alignment_v26,
        },
        "direct_equation_v26": {
            "origin_term": "D_origin_v26 = 0.46 * D_origin_v25 + 0.22 * R_zone + 0.15 * (1 - P_zone) + 0.17 * R_train_v31",
            "feature_term": "D_feature_v26 = 0.44 * D_feature_v25 + 0.26 * L_zone + 0.10 * (1 - P_zone) + 0.20 * B_plastic_v31",
            "structure_term": "D_structure_v26 = 0.42 * D_structure_v25 + 0.28 * S_zone + 0.10 * (1 - P_zone) + 0.20 * B_struct_v31",
            "route_term": "D_route_v26 = 0.42 * D_route_v25 + 0.28 * R_zone + 0.08 * S_zone + 0.05 * (1 - P_zone) + 0.17 * H_low_v31",
            "system_term": "M_brain_direct_v26 = mean(D_origin_v26, D_feature_v26, D_structure_v26, D_route_v26)",
        },
        "project_readout": {
            "summary": "brain direct refinement v26 checks whether the low-risk steady zone continues to be visible in the brain encoding chain.",
            "next_question": "next verify whether this low-risk zone still survives after being folded into the next training bridge and main core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Brain Encoding Direct Refinement V26 Report",
        "",
        f"- direct_origin_measure_v26: {hm['direct_origin_measure_v26']:.6f}",
        f"- direct_feature_measure_v26: {hm['direct_feature_measure_v26']:.6f}",
        f"- direct_structure_measure_v26: {hm['direct_structure_measure_v26']:.6f}",
        f"- direct_route_measure_v26: {hm['direct_route_measure_v26']:.6f}",
        f"- direct_brain_measure_v26: {hm['direct_brain_measure_v26']:.6f}",
        f"- direct_brain_gap_v26: {hm['direct_brain_gap_v26']:.6f}",
        f"- direct_low_risk_zone_alignment_v26: {hm['direct_low_risk_zone_alignment_v26']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v26_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
