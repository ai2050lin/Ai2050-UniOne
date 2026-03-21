from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v29_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v29_summary() -> dict:
    v28 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v28_20260321" / "summary.json"
    )
    systemic_enlarge = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_low_risk_zone_enlargement_validation_20260321" / "summary.json"
    )
    bridge_v34 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v34_20260321" / "summary.json"
    )

    hv = v28["headline_metrics"]
    hs = systemic_enlarge["headline_metrics"]
    hb = bridge_v34["headline_metrics"]

    direct_origin_measure_v29 = _clip01(
        hv["direct_origin_measure_v28"] * 0.45
        + hs["systemic_low_risk_enlargement_readiness"] * 0.22
        + (1.0 - hs["systemic_low_risk_enlargement_penalty"]) * 0.15
        + hb["topology_training_readiness_v34"] * 0.18
    )
    direct_feature_measure_v29 = _clip01(
        hv["direct_feature_measure_v28"] * 0.43
        + hs["systemic_low_risk_enlargement_learning"] * 0.27
        + (1.0 - hs["systemic_low_risk_enlargement_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v34"] * 0.20
    )
    direct_structure_measure_v29 = _clip01(
        hv["direct_structure_measure_v28"] * 0.41
        + hs["systemic_low_risk_enlargement_structure"] * 0.29
        + (1.0 - hs["systemic_low_risk_enlargement_penalty"]) * 0.10
        + hb["structure_rule_alignment_v34"] * 0.20
    )
    direct_route_measure_v29 = _clip01(
        hv["direct_route_measure_v28"] * 0.41
        + hs["systemic_low_risk_enlargement_route"] * 0.29
        + hs["systemic_low_risk_enlargement_structure"] * 0.08
        + (1.0 - hs["systemic_low_risk_enlargement_penalty"]) * 0.05
        + hb["systemic_low_risk_expansion_guard_v34"] * 0.17
    )
    direct_brain_measure_v29 = (
        direct_origin_measure_v29
        + direct_feature_measure_v29
        + direct_structure_measure_v29
        + direct_route_measure_v29
    ) / 4.0
    direct_brain_gap_v29 = 1.0 - direct_brain_measure_v29
    direct_systemic_enlargement_alignment_v29 = (
        direct_structure_measure_v29
        + direct_route_measure_v29
        + hs["systemic_low_risk_enlargement_readiness"]
        + hb["topology_training_readiness_v34"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v29": direct_origin_measure_v29,
            "direct_feature_measure_v29": direct_feature_measure_v29,
            "direct_structure_measure_v29": direct_structure_measure_v29,
            "direct_route_measure_v29": direct_route_measure_v29,
            "direct_brain_measure_v29": direct_brain_measure_v29,
            "direct_brain_gap_v29": direct_brain_gap_v29,
            "direct_systemic_enlargement_alignment_v29": direct_systemic_enlargement_alignment_v29,
        },
        "direct_equation_v29": {
            "origin_term": "D_origin_v29 = 0.45 * D_origin_v28 + 0.22 * R_sys_enlarge + 0.15 * (1 - P_sys_enlarge) + 0.18 * R_train_v34",
            "feature_term": "D_feature_v29 = 0.43 * D_feature_v28 + 0.27 * L_sys_enlarge + 0.10 * (1 - P_sys_enlarge) + 0.20 * B_plastic_v34",
            "structure_term": "D_structure_v29 = 0.41 * D_structure_v28 + 0.29 * S_sys_enlarge + 0.10 * (1 - P_sys_enlarge) + 0.20 * B_struct_v34",
            "route_term": "D_route_v29 = 0.41 * D_route_v28 + 0.29 * R_sys_enlarge + 0.08 * S_sys_enlarge + 0.05 * (1 - P_sys_enlarge) + 0.17 * H_sys_expand_v34",
            "system_term": "M_brain_direct_v29 = mean(D_origin_v29, D_feature_v29, D_structure_v29, D_route_v29)",
        },
        "project_readout": {
            "summary": "brain direct refinement v29 checks whether systemic low-risk enlargement continues to stay visible in the brain encoding chain.",
            "next_question": "next verify whether this enlargement survives after it is folded into the next training bridge and main core.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Brain Encoding Direct Refinement V29 Report",
        "",
        f"- direct_origin_measure_v29: {hm['direct_origin_measure_v29']:.6f}",
        f"- direct_feature_measure_v29: {hm['direct_feature_measure_v29']:.6f}",
        f"- direct_structure_measure_v29: {hm['direct_structure_measure_v29']:.6f}",
        f"- direct_route_measure_v29: {hm['direct_route_measure_v29']:.6f}",
        f"- direct_brain_measure_v29: {hm['direct_brain_measure_v29']:.6f}",
        f"- direct_brain_gap_v29: {hm['direct_brain_gap_v29']:.6f}",
        f"- direct_systemic_enlargement_alignment_v29: {hm['direct_systemic_enlargement_alignment_v29']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v29_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
