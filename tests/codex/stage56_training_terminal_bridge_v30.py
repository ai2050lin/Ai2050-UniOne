from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v30_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v30_summary() -> dict:
    v29 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v29_20260321" / "summary.json"
    )
    systemic_steady = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_system_systemic_steady_amplification_validation_20260321" / "summary.json"
    )
    brain_v24 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v24_20260321" / "summary.json"
    )

    hv = v29["headline_metrics"]
    hs = systemic_steady["headline_metrics"]
    hb = brain_v24["headline_metrics"]

    plasticity_rule_alignment_v30 = _clip01(
        hv["plasticity_rule_alignment_v29"] * 0.28
        + hs["systemic_steady_learning"] * 0.24
        + (1.0 - hs["systemic_steady_penalty"]) * 0.14
        + hb["direct_feature_measure_v24"] * 0.14
        + (1.0 - hb["direct_brain_gap_v24"]) * 0.20
    )
    structure_rule_alignment_v30 = _clip01(
        hv["structure_rule_alignment_v29"] * 0.28
        + hs["systemic_steady_structure"] * 0.24
        + hs["systemic_steady_route"] * 0.14
        + (1.0 - hs["systemic_steady_penalty"]) * 0.10
        + hb["direct_structure_measure_v24"] * 0.24
    )
    topology_training_readiness_v30 = _clip01(
        hv["topology_training_readiness_v29"] * 0.30
        + plasticity_rule_alignment_v30 * 0.15
        + structure_rule_alignment_v30 * 0.15
        + hs["systemic_steady_readiness"] * 0.15
        + hb["direct_systemic_steady_alignment_v24"] * 0.15
        + (1.0 - hs["systemic_steady_penalty"]) * 0.10
    )
    topology_training_gap_v30 = max(0.0, 1.0 - topology_training_readiness_v30)
    systemic_steady_guard_v30 = _clip01(
        (
            hs["systemic_steady_structure"]
            + hs["systemic_steady_route"]
            + hs["systemic_steady_strength"]
            + topology_training_readiness_v30
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v30": plasticity_rule_alignment_v30,
            "structure_rule_alignment_v30": structure_rule_alignment_v30,
            "topology_training_readiness_v30": topology_training_readiness_v30,
            "topology_training_gap_v30": topology_training_gap_v30,
            "systemic_steady_guard_v30": systemic_steady_guard_v30,
        },
        "bridge_equation_v30": {
            "plasticity_term": "B_plastic_v30 = mix(B_plastic_v29, L_system_steady, 1 - P_system_steady, D_feature_v24, 1 - G_brain_v24)",
            "structure_term": "B_struct_v30 = mix(B_struct_v29, S_system_steady, R_system_steady, 1 - P_system_steady, D_structure_v24)",
            "readiness_term": "R_train_v30 = mix(R_train_v29, B_plastic_v30, B_struct_v30, R_system_steady, D_align_v24, 1 - P_system_steady)",
            "gap_term": "G_train_v30 = 1 - R_train_v30",
            "guard_term": "H_system_steady_v30 = mean(S_system_steady, R_system_steady, A_system_steady, R_train_v30)",
        },
        "project_readout": {
            "summary": "训练终式第三十桥开始吸收系统级稳态放大验证和脑编码第二十四版，检查系统级稳态承接是否继续压低规则层风险。",
            "next_question": "下一步要把第三十桥并回主核，验证系统级稳态放大是否继续走向更低风险施工区。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第三十桥报告",
        "",
        f"- plasticity_rule_alignment_v30: {hm['plasticity_rule_alignment_v30']:.6f}",
        f"- structure_rule_alignment_v30: {hm['structure_rule_alignment_v30']:.6f}",
        f"- topology_training_readiness_v30: {hm['topology_training_readiness_v30']:.6f}",
        f"- topology_training_gap_v30: {hm['topology_training_gap_v30']:.6f}",
        f"- systemic_steady_guard_v30: {hm['systemic_steady_guard_v30']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v30_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
