from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v15_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v15_summary() -> dict:
    v14 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v14_20260321" / "summary.json"
    )
    persistence = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_anti_attenuation_persistence_20260321" / "summary.json"
    )
    bridge_v20 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v20_20260321" / "summary.json"
    )

    hv = v14["headline_metrics"]
    hp = persistence["headline_metrics"]
    hb = bridge_v20["headline_metrics"]

    direct_origin_measure_v15 = _clip01(
        hv["direct_origin_measure_v14"] * 0.60
        + hp["persistence_readiness"] * 0.15
        + (1.0 - hp["persistence_penalty"]) * 0.15
        + hb["topology_training_readiness_v20"] * 0.10
    )
    direct_feature_measure_v15 = _clip01(
        hv["direct_feature_measure_v14"] * 0.58
        + hp["persistence_learning"] * 0.17
        + (1.0 - hp["persistence_penalty"]) * 0.10
        + hb["plasticity_rule_alignment_v20"] * 0.15
    )
    direct_structure_measure_v15 = _clip01(
        hv["direct_structure_measure_v14"] * 0.54
        + hp["persistence_structure"] * 0.21
        + (1.0 - hp["persistence_penalty"]) * 0.10
        + hb["structure_rule_alignment_v20"] * 0.15
    )
    direct_route_measure_v15 = _clip01(
        hv["direct_route_measure_v14"] * 0.54
        + hp["persistence_route"] * 0.21
        + hp["persistence_context"] * 0.10
        + (1.0 - hp["persistence_penalty"]) * 0.05
        + hb["anti_attenuation_guard_v20"] * 0.10
    )
    direct_brain_measure_v15 = (
        direct_origin_measure_v15
        + direct_feature_measure_v15
        + direct_structure_measure_v15
        + direct_route_measure_v15
    ) / 4.0
    direct_brain_gap_v15 = 1.0 - direct_brain_measure_v15
    direct_persistence_alignment_v15 = (
        direct_structure_measure_v15
        + direct_route_measure_v15
        + hp["persistence_readiness"]
        + hb["topology_training_readiness_v20"]
    ) / 4.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v15": direct_origin_measure_v15,
            "direct_feature_measure_v15": direct_feature_measure_v15,
            "direct_structure_measure_v15": direct_structure_measure_v15,
            "direct_route_measure_v15": direct_route_measure_v15,
            "direct_brain_measure_v15": direct_brain_measure_v15,
            "direct_brain_gap_v15": direct_brain_gap_v15,
            "direct_persistence_alignment_v15": direct_persistence_alignment_v15,
        },
        "direct_equation_v15": {
            "origin_term": "D_origin_v15 = 0.60 * D_origin_v14 + 0.15 * R_persist + 0.15 * (1 - P_persist) + 0.10 * R_train_v20",
            "feature_term": "D_feature_v15 = 0.58 * D_feature_v14 + 0.17 * L_persist + 0.10 * (1 - P_persist) + 0.15 * B_plastic_v20",
            "structure_term": "D_structure_v15 = 0.54 * D_structure_v14 + 0.21 * S_persist + 0.10 * (1 - P_persist) + 0.15 * B_struct_v20",
            "route_term": "D_route_v15 = 0.54 * D_route_v14 + 0.21 * R_persist + 0.10 * C_persist + 0.05 * (1 - P_persist) + 0.10 * H_anti_att_v20",
            "system_term": "M_brain_direct_v15 = mean(D_origin_v15, D_feature_v15, D_structure_v15, D_route_v15)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测第十五版开始直接吸收反衰减持续性结果，检验脑编码链的回升能否继续站住。",
            "next_question": "下一步要把第十五版直测链并回训练终式和主核，检验持续回升是否开始稳定化。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第十五版报告",
        "",
        f"- direct_origin_measure_v15: {hm['direct_origin_measure_v15']:.6f}",
        f"- direct_feature_measure_v15: {hm['direct_feature_measure_v15']:.6f}",
        f"- direct_structure_measure_v15: {hm['direct_structure_measure_v15']:.6f}",
        f"- direct_route_measure_v15: {hm['direct_route_measure_v15']:.6f}",
        f"- direct_brain_measure_v15: {hm['direct_brain_measure_v15']:.6f}",
        f"- direct_brain_gap_v15: {hm['direct_brain_gap_v15']:.6f}",
        f"- direct_persistence_alignment_v15: {hm['direct_persistence_alignment_v15']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v15_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
