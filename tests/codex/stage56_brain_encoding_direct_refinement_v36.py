from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v36_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v36_summary() -> dict:
    v35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v35_20260321" / "summary.json"
    )
    field_crystal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_field_crystallization_validation_20260321" / "summary.json"
    )
    bridge_v41 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v41_20260321" / "summary.json"
    )

    hv = v35["headline_metrics"]
    hs = field_crystal["headline_metrics"]
    hb = bridge_v41["headline_metrics"]

    direct_origin_measure_v36 = _clip01(
        hv["direct_origin_measure_v35"] * 0.45
        + hs["systemic_low_risk_field_crystallization_readiness"] * 0.22
        + (1.0 - hs["systemic_low_risk_field_crystallization_penalty"]) * 0.15
        + hb["topology_training_readiness_v41"] * 0.18
    )
    direct_feature_measure_v36 = _clip01(
        hv["direct_feature_measure_v35"] * 0.43
        + hs["systemic_low_risk_field_learning_crystallization"] * 0.27
        + (1.0 - hs["systemic_low_risk_field_crystallization_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v41"] * 0.20
    )
    direct_structure_measure_v36 = _clip01(
        hv["direct_structure_measure_v35"] * 0.41
        + hs["systemic_low_risk_field_structure_crystallization"] * 0.29
        + (1.0 - hs["systemic_low_risk_field_crystallization_penalty"]) * 0.10
        + hb["structure_rule_alignment_v41"] * 0.20
    )
    direct_route_measure_v36 = _clip01(
        hv["direct_route_measure_v35"] * 0.41
        + hs["systemic_low_risk_field_route_crystallization"] * 0.29
        + hs["systemic_low_risk_field_structure_crystallization"] * 0.08
        + (1.0 - hs["systemic_low_risk_field_crystallization_penalty"]) * 0.05
        + hb["systemic_low_risk_field_solidification_guard_v41"] * 0.17
    )
    direct_brain_measure_v36 = (
        direct_origin_measure_v36
        + direct_feature_measure_v36
        + direct_structure_measure_v36
        + direct_route_measure_v36
    ) / 4.0
    direct_brain_gap_v36 = 1.0 - direct_brain_measure_v36
    direct_systemic_field_crystallization_alignment_v36 = (
        direct_structure_measure_v36
        + direct_route_measure_v36
        + hs["systemic_low_risk_field_crystallization_readiness"]
        + hb["topology_training_readiness_v41"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v36": direct_origin_measure_v36,
            "direct_feature_measure_v36": direct_feature_measure_v36,
            "direct_structure_measure_v36": direct_structure_measure_v36,
            "direct_route_measure_v36": direct_route_measure_v36,
            "direct_brain_measure_v36": direct_brain_measure_v36,
            "direct_brain_gap_v36": direct_brain_gap_v36,
            "direct_systemic_field_crystallization_alignment_v36": direct_systemic_field_crystallization_alignment_v36,
        },
        "direct_equation_v36": {
            "origin_term": "D_origin_v36 = 0.45 * D_origin_v35 + 0.22 * R_sys_field_crystal + 0.15 * (1 - P_sys_field_crystal) + 0.18 * R_train_v41",
            "feature_term": "D_feature_v36 = 0.43 * D_feature_v35 + 0.27 * L_sys_field_crystal + 0.10 * (1 - P_sys_field_crystal) + 0.20 * B_plastic_v41",
            "structure_term": "D_structure_v36 = 0.41 * D_structure_v35 + 0.29 * S_sys_field_crystal + 0.10 * (1 - P_sys_field_crystal) + 0.20 * B_struct_v41",
            "route_term": "D_route_v36 = 0.41 * D_route_v35 + 0.29 * R_sys_field_crystal + 0.08 * S_sys_field_crystal + 0.05 * (1 - P_sys_field_crystal) + 0.17 * H_sys_field_solid_v41",
            "system_term": "M_brain_direct_v36 = mean(D_origin_v36, D_feature_v36, D_structure_v36, D_route_v36)",
        },
        "project_readout": {
            "summary": "brain direct refinement v36 checks whether systemic low-risk field crystallization continues to stay visible in the brain encoding chain.",
            "next_question": "next verify whether this crystallized field survives after it is folded into the next training bridge and main core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Brain Encoding Direct Refinement V36 Report",
        "",
        f"- direct_origin_measure_v36: {hm['direct_origin_measure_v36']:.6f}",
        f"- direct_feature_measure_v36: {hm['direct_feature_measure_v36']:.6f}",
        f"- direct_structure_measure_v36: {hm['direct_structure_measure_v36']:.6f}",
        f"- direct_route_measure_v36: {hm['direct_route_measure_v36']:.6f}",
        f"- direct_brain_measure_v36: {hm['direct_brain_measure_v36']:.6f}",
        f"- direct_brain_gap_v36: {hm['direct_brain_gap_v36']:.6f}",
        f"- direct_systemic_field_crystallization_alignment_v36: {hm['direct_systemic_field_crystallization_alignment_v36']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v36_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
