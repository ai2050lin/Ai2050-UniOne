from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v20_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v20_summary() -> dict:
    v19 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v19_20260321" / "summary.json"
    )
    attenuation = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_propagation_attenuation_probe_20260321" / "summary.json"
    )
    brain_v14 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v14_20260321" / "summary.json"
    )

    hv = v19["headline_metrics"]
    ha = attenuation["headline_metrics"]
    hb = brain_v14["headline_metrics"]

    plasticity_rule_alignment_v20 = _clip01(
        (
            hv["plasticity_rule_alignment_v19"]
            + (1.0 - ha["attenuation_learning"])
            + (1.0 - ha["attenuation_penalty"])
            + hb["direct_feature_measure_v14"]
            + (1.0 - hb["direct_brain_gap_v14"])
        )
        / 5.0
    )
    structure_rule_alignment_v20 = _clip01(
        (
            hv["structure_rule_alignment_v19"]
            + (1.0 - ha["attenuation_structure"])
            + (1.0 - ha["attenuation_route"])
            + (1.0 - ha["attenuation_penalty"])
            + hb["direct_structure_measure_v14"]
        )
        / 5.0
    )
    topology_training_readiness_v20 = _clip01(
        (
            hv["topology_training_readiness_v19"]
            + plasticity_rule_alignment_v20
            + structure_rule_alignment_v20
            + ha["anti_attenuation_readiness"]
            + hb["direct_anti_attenuation_alignment_v14"]
            + (1.0 - ha["attenuation_penalty"])
        )
        / 6.0
    )
    topology_training_gap_v20 = max(0.0, 1.0 - topology_training_readiness_v20)
    anti_attenuation_guard_v20 = _clip01(
        (
            (1.0 - ha["attenuation_structure"])
            + (1.0 - ha["attenuation_context"])
            + (1.0 - ha["attenuation_route"])
            + topology_training_readiness_v20
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v20": plasticity_rule_alignment_v20,
            "structure_rule_alignment_v20": structure_rule_alignment_v20,
            "topology_training_readiness_v20": topology_training_readiness_v20,
            "topology_training_gap_v20": topology_training_gap_v20,
            "anti_attenuation_guard_v20": anti_attenuation_guard_v20,
        },
        "bridge_equation_v20": {
            "plasticity_term": "B_plastic_v20 = mean(B_plastic_v19, 1 - A_learn, 1 - P_att, D_feature_v14, 1 - G_brain_v14)",
            "structure_term": "B_struct_v20 = mean(B_struct_v19, 1 - A_struct, 1 - A_route, 1 - P_att, D_structure_v14)",
            "readiness_term": "R_train_v20 = mean(R_train_v19, B_plastic_v20, B_struct_v20, R_anti_att, D_align_v14, 1 - P_att)",
            "gap_term": "G_train_v20 = 1 - R_train_v20",
            "guard_term": "H_anti_att_v20 = mean(1 - A_struct, 1 - A_ctx, 1 - A_route, R_train_v20)",
        },
        "project_readout": {
            "summary": "训练终式第二十桥开始直接吸收传播衰减探针，检验系统能否从衰减走向规则层补偿。",
            "next_question": "下一步要把第二十桥并回主核，检验是否终于出现补偿级传播。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第二十桥报告",
        "",
        f"- plasticity_rule_alignment_v20: {hm['plasticity_rule_alignment_v20']:.6f}",
        f"- structure_rule_alignment_v20: {hm['structure_rule_alignment_v20']:.6f}",
        f"- topology_training_readiness_v20: {hm['topology_training_readiness_v20']:.6f}",
        f"- topology_training_gap_v20: {hm['topology_training_gap_v20']:.6f}",
        f"- anti_attenuation_guard_v20: {hm['anti_attenuation_guard_v20']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v20_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
