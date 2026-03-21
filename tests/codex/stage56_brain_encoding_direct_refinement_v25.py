from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v25_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v25_summary() -> dict:
    v24 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v24_20260321" / "summary.json"
    )
    low_risk = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_low_risk_steady_amplification_validation_20260321" / "summary.json"
    )
    bridge_v30 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v30_20260321" / "summary.json"
    )

    hv = v24["headline_metrics"]
    hs = low_risk["headline_metrics"]
    hb = bridge_v30["headline_metrics"]

    direct_origin_measure_v25 = _clip01(
        hv["direct_origin_measure_v24"] * 0.46
        + hs["low_risk_readiness"] * 0.22
        + (1.0 - hs["low_risk_penalty"]) * 0.15
        + hb["topology_training_readiness_v30"] * 0.17
    )
    direct_feature_measure_v25 = _clip01(
        hv["direct_feature_measure_v24"] * 0.44
        + hs["low_risk_learning"] * 0.26
        + (1.0 - hs["low_risk_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v30"] * 0.20
    )
    direct_structure_measure_v25 = _clip01(
        hv["direct_structure_measure_v24"] * 0.42
        + hs["low_risk_structure"] * 0.28
        + (1.0 - hs["low_risk_penalty"]) * 0.10
        + hb["structure_rule_alignment_v30"] * 0.20
    )
    direct_route_measure_v25 = _clip01(
        hv["direct_route_measure_v24"] * 0.42
        + hs["low_risk_route"] * 0.28
        + hs["low_risk_structure"] * 0.08
        + (1.0 - hs["low_risk_penalty"]) * 0.05
        + hb["systemic_steady_guard_v30"] * 0.17
    )
    direct_brain_measure_v25 = (
        direct_origin_measure_v25
        + direct_feature_measure_v25
        + direct_structure_measure_v25
        + direct_route_measure_v25
    ) / 4.0
    direct_brain_gap_v25 = 1.0 - direct_brain_measure_v25
    direct_low_risk_alignment_v25 = (
        direct_structure_measure_v25
        + direct_route_measure_v25
        + hs["low_risk_readiness"]
        + hb["topology_training_readiness_v30"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v25": direct_origin_measure_v25,
            "direct_feature_measure_v25": direct_feature_measure_v25,
            "direct_structure_measure_v25": direct_structure_measure_v25,
            "direct_route_measure_v25": direct_route_measure_v25,
            "direct_brain_measure_v25": direct_brain_measure_v25,
            "direct_brain_gap_v25": direct_brain_gap_v25,
            "direct_low_risk_alignment_v25": direct_low_risk_alignment_v25,
        },
        "direct_equation_v25": {
            "origin_term": "D_origin_v25 = 0.46 * D_origin_v24 + 0.22 * R_low + 0.15 * (1 - P_low) + 0.17 * R_train_v30",
            "feature_term": "D_feature_v25 = 0.44 * D_feature_v24 + 0.26 * L_low + 0.10 * (1 - P_low) + 0.20 * B_plastic_v30",
            "structure_term": "D_structure_v25 = 0.42 * D_structure_v24 + 0.28 * S_low + 0.10 * (1 - P_low) + 0.20 * B_struct_v30",
            "route_term": "D_route_v25 = 0.42 * D_route_v24 + 0.28 * R_low + 0.08 * S_low + 0.05 * (1 - P_low) + 0.17 * H_system_steady_v30",
            "system_term": "M_brain_direct_v25 = mean(D_origin_v25, D_feature_v25, D_structure_v25, D_route_v25)",
        },
        "project_readout": {
            "summary": "brain direct refinement v25 folds the low-risk steady amplification layer back into the brain encoding chain.",
            "next_question": "next verify whether low-risk steady amplification remains visible once it is fed back into the rule layer and the main core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Brain Encoding Direct Refinement V25 Report",
        "",
        f"- direct_origin_measure_v25: {hm['direct_origin_measure_v25']:.6f}",
        f"- direct_feature_measure_v25: {hm['direct_feature_measure_v25']:.6f}",
        f"- direct_structure_measure_v25: {hm['direct_structure_measure_v25']:.6f}",
        f"- direct_route_measure_v25: {hm['direct_route_measure_v25']:.6f}",
        f"- direct_brain_measure_v25: {hm['direct_brain_measure_v25']:.6f}",
        f"- direct_brain_gap_v25: {hm['direct_brain_gap_v25']:.6f}",
        f"- direct_low_risk_alignment_v25: {hm['direct_low_risk_alignment_v25']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v25_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
