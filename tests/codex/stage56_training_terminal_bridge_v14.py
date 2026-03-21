from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v14_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v14_summary() -> dict:
    v13 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v13_20260321" / "summary.json"
    )
    coupled = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_structure_coupled_validation_20260321" / "summary.json"
    )
    brain_v8 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v8_20260321" / "summary.json"
    )

    hv = v13["headline_metrics"]
    hc = coupled["headline_metrics"]
    hb = brain_v8["headline_metrics"]

    plasticity_rule_alignment_v14 = _clip01(
        (
            hv["plasticity_rule_alignment_v13"]
            + hc["coupled_novel_gain"]
            + (1.0 - hc["coupled_forgetting_penalty"])
            + hb["direct_feature_measure_v8"]
            + (1.0 - hb["direct_brain_gap_v8"])
        )
        / 5.0
    )
    structure_rule_alignment_v14 = _clip01(
        (
            hv["structure_rule_alignment_v13"]
            + hc["coupled_structure_keep"]
            + hc["coupled_route_keep"]
            + (1.0 - hc["coupled_failure_risk"])
            + hb["direct_structure_measure_v8"]
        )
        / 5.0
    )
    topology_training_readiness_v14 = _clip01(
        (
            hv["topology_training_readiness_v13"]
            + plasticity_rule_alignment_v14
            + structure_rule_alignment_v14
            + hc["coupled_readiness"]
            + hb["direct_coupled_alignment_v8"]
            + (1.0 - hc["coupled_failure_risk"])
        )
        / 6.0
    )
    topology_training_gap_v14 = max(0.0, 1.0 - topology_training_readiness_v14)
    coupled_guard_v14 = _clip01(
        (
            hc["coupled_route_keep"]
            + hc["coupled_structure_keep"]
            + topology_training_readiness_v14
            + hb["direct_route_measure_v8"]
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v14": plasticity_rule_alignment_v14,
            "structure_rule_alignment_v14": structure_rule_alignment_v14,
            "topology_training_readiness_v14": topology_training_readiness_v14,
            "topology_training_gap_v14": topology_training_gap_v14,
            "coupled_guard_v14": coupled_guard_v14,
        },
        "bridge_equation_v14": {
            "plasticity_term": "B_plastic_v14 = mean(B_plastic_v13, G_coupled, 1 - P_coupled, D_feature_v8, 1 - G_brain_v8)",
            "structure_term": "B_struct_v14 = mean(B_struct_v13, K_struct, K_route, 1 - R_fail, D_structure_v8)",
            "readiness_term": "R_train_v14 = mean(R_train_v13, B_plastic_v14, B_struct_v14, A_coupled, D_align_v8, 1 - R_fail)",
            "gap_term": "G_train_v14 = 1 - R_train_v14",
            "guard_term": "H_coupled_v14 = mean(K_route, K_struct, R_train_v14, D_route_v8)",
        },
        "project_readout": {
            "summary": "训练终式第十四桥开始显式吸收路由-结构联动退化链，使训练规则开始面向更真实的联动失稳，而不是分别处理单项风险。",
            "next_question": "下一步要把第十四桥并回主核，检验主核在联动退化压力下是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第十四桥报告",
        "",
        f"- plasticity_rule_alignment_v14: {hm['plasticity_rule_alignment_v14']:.6f}",
        f"- structure_rule_alignment_v14: {hm['structure_rule_alignment_v14']:.6f}",
        f"- topology_training_readiness_v14: {hm['topology_training_readiness_v14']:.6f}",
        f"- topology_training_gap_v14: {hm['topology_training_gap_v14']:.6f}",
        f"- coupled_guard_v14: {hm['coupled_guard_v14']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v14_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
