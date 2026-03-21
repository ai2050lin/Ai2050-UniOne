from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v12_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v12_summary() -> dict:
    v11 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v11_20260321" / "summary.json"
    )
    propagation = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_plateau_break_propagation_probe_20260321" / "summary.json"
    )
    bridge_v17 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v17_20260321" / "summary.json"
    )

    hv = v11["headline_metrics"]
    hp = propagation["headline_metrics"]
    hb = bridge_v17["headline_metrics"]

    direct_origin_measure_v12 = _clip01(
        hv["direct_origin_measure_v11"] * 0.60
        + hp["propagation_readiness"] * 0.15
        + (1.0 - hp["propagation_penalty"]) * 0.15
        + hb["topology_training_readiness_v17"] * 0.10
    )
    direct_feature_measure_v12 = _clip01(
        hv["direct_feature_measure_v11"] * 0.58
        + hp["propagation_learning"] * 0.17
        + (1.0 - hp["propagation_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v17"] * 0.15
    )
    direct_structure_measure_v12 = _clip01(
        hv["direct_structure_measure_v11"] * 0.54
        + hp["propagation_structure"] * 0.21
        + (1.0 - hp["propagation_penalty"]) * 0.10
        + hb["structure_rule_alignment_v17"] * 0.15
    )
    direct_route_measure_v12 = _clip01(
        hv["direct_route_measure_v11"] * 0.54
        + hp["propagation_route"] * 0.21
        + hp["propagation_context"] * 0.10
        + (1.0 - hp["propagation_penalty"]) * 0.05
        + hb["plateau_guard_v17"] * 0.10
    )
    direct_brain_measure_v12 = (
        direct_origin_measure_v12
        + direct_feature_measure_v12
        + direct_structure_measure_v12
        + direct_route_measure_v12
    ) / 4.0
    direct_brain_gap_v12 = 1.0 - direct_brain_measure_v12
    direct_propagation_alignment_v12 = (
        direct_structure_measure_v12
        + direct_route_measure_v12
        + hp["propagation_readiness"]
        + hb["topology_training_readiness_v17"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v12": direct_origin_measure_v12,
            "direct_feature_measure_v12": direct_feature_measure_v12,
            "direct_structure_measure_v12": direct_structure_measure_v12,
            "direct_route_measure_v12": direct_route_measure_v12,
            "direct_brain_measure_v12": direct_brain_measure_v12,
            "direct_brain_gap_v12": direct_brain_gap_v12,
            "direct_propagation_alignment_v12": direct_propagation_alignment_v12,
        },
        "direct_equation_v12": {
            "origin_term": "D_origin_v12 = 0.60 * D_origin_v11 + 0.15 * R_prop + 0.15 * (1 - P_prop) + 0.10 * R_train_v17",
            "feature_term": "D_feature_v12 = 0.58 * D_feature_v11 + 0.17 * T_learn + 0.10 * (1 - P_prop) + 0.15 * B_plastic_v17",
            "structure_term": "D_structure_v12 = 0.54 * D_structure_v11 + 0.21 * T_struct + 0.10 * (1 - P_prop) + 0.15 * B_struct_v17",
            "route_term": "D_route_v12 = 0.54 * D_route_v11 + 0.21 * T_route + 0.10 * T_ctx + 0.05 * (1 - P_prop) + 0.10 * H_break_v17",
            "system_term": "M_brain_direct_v12 = mean(D_origin_v12, D_feature_v12, D_structure_v12, D_route_v12)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第十二版开始直接吸收破平台传播结果，开始检验平台期松动是否能真实传导到脑编码链。",
            "next_question": "下一步要把第十二版直测链并回训练终式和主核，检验传播级突破是否真的出现。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第十二版报告",
        "",
        f"- direct_origin_measure_v12: {hm['direct_origin_measure_v12']:.6f}",
        f"- direct_feature_measure_v12: {hm['direct_feature_measure_v12']:.6f}",
        f"- direct_structure_measure_v12: {hm['direct_structure_measure_v12']:.6f}",
        f"- direct_route_measure_v12: {hm['direct_route_measure_v12']:.6f}",
        f"- direct_brain_measure_v12: {hm['direct_brain_measure_v12']:.6f}",
        f"- direct_brain_gap_v12: {hm['direct_brain_gap_v12']:.6f}",
        f"- direct_propagation_alignment_v12: {hm['direct_propagation_alignment_v12']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v12_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
