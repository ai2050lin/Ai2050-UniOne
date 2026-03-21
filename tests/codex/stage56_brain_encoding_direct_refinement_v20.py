from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v20_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v20_summary() -> dict:
    v19 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v19_20260321" / "summary.json"
    )
    steady_plus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_steady_amplification_reinforcement_20260321" / "summary.json"
    )
    bridge_v25 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v25_20260321" / "summary.json"
    )

    hv = v19["headline_metrics"]
    hs = steady_plus["headline_metrics"]
    hb = bridge_v25["headline_metrics"]

    direct_origin_measure_v20 = _clip01(
        hv["direct_origin_measure_v19"] * 0.46
        + hs["steady_reinforcement_readiness"] * 0.22
        + (1.0 - hs["steady_reinforcement_penalty"]) * 0.15
        + hb["topology_training_readiness_v25"] * 0.17
    )
    direct_feature_measure_v20 = _clip01(
        hv["direct_feature_measure_v19"] * 0.44
        + hs["steady_reinforcement_learning"] * 0.26
        + (1.0 - hs["steady_reinforcement_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v25"] * 0.20
    )
    direct_structure_measure_v20 = _clip01(
        hv["direct_structure_measure_v19"] * 0.42
        + hs["steady_reinforcement_structure"] * 0.28
        + (1.0 - hs["steady_reinforcement_penalty"]) * 0.10
        + hb["structure_rule_alignment_v25"] * 0.20
    )
    direct_route_measure_v20 = _clip01(
        hv["direct_route_measure_v19"] * 0.42
        + hs["steady_reinforcement_route"] * 0.28
        + hs["steady_reinforcement_structure"] * 0.08
        + (1.0 - hs["steady_reinforcement_penalty"]) * 0.05
        + hb["steady_guard_v25"] * 0.17
    )
    direct_brain_measure_v20 = (
        direct_origin_measure_v20
        + direct_feature_measure_v20
        + direct_structure_measure_v20
        + direct_route_measure_v20
    ) / 4.0
    direct_brain_gap_v20 = 1.0 - direct_brain_measure_v20
    direct_steady_alignment_v20 = (
        direct_structure_measure_v20
        + direct_route_measure_v20
        + hs["steady_reinforcement_readiness"]
        + hb["topology_training_readiness_v25"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v20": direct_origin_measure_v20,
            "direct_feature_measure_v20": direct_feature_measure_v20,
            "direct_structure_measure_v20": direct_structure_measure_v20,
            "direct_route_measure_v20": direct_route_measure_v20,
            "direct_brain_measure_v20": direct_brain_measure_v20,
            "direct_brain_gap_v20": direct_brain_gap_v20,
            "direct_steady_alignment_v20": direct_steady_alignment_v20,
        },
        "direct_equation_v20": {
            "origin_term": "D_origin_v20 = 0.46 * D_origin_v19 + 0.22 * R_steady_plus + 0.15 * (1 - P_steady_plus) + 0.17 * R_train_v25",
            "feature_term": "D_feature_v20 = 0.44 * D_feature_v19 + 0.26 * L_steady_plus + 0.10 * (1 - P_steady_plus) + 0.20 * B_plastic_v25",
            "structure_term": "D_structure_v20 = 0.42 * D_structure_v19 + 0.28 * S_steady_plus + 0.10 * (1 - P_steady_plus) + 0.20 * B_struct_v25",
            "route_term": "D_route_v20 = 0.42 * D_route_v19 + 0.28 * R_steady_plus + 0.08 * S_steady_plus + 0.05 * (1 - P_steady_plus) + 0.17 * H_steady_v25",
            "system_term": "M_brain_direct_v20 = mean(D_origin_v20, D_feature_v20, D_structure_v20, D_route_v20)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第二十版开始把更稳的放大强化并回脑编码链，检查放大趋势是否继续稳态化落在脑编码层。",
            "next_question": "下一步要把第二十版直测链并回训练终式和主核，确认更稳的放大是否继续在脑编码层承接。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第二十版报告",
        "",
        f"- direct_origin_measure_v20: {hm['direct_origin_measure_v20']:.6f}",
        f"- direct_feature_measure_v20: {hm['direct_feature_measure_v20']:.6f}",
        f"- direct_structure_measure_v20: {hm['direct_structure_measure_v20']:.6f}",
        f"- direct_route_measure_v20: {hm['direct_route_measure_v20']:.6f}",
        f"- direct_brain_measure_v20: {hm['direct_brain_measure_v20']:.6f}",
        f"- direct_brain_gap_v20: {hm['direct_brain_gap_v20']:.6f}",
        f"- direct_steady_alignment_v20: {hm['direct_steady_alignment_v20']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v20_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
