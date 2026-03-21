from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v11_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v11_summary() -> dict:
    v10 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v10_20260321" / "summary.json"
    )
    extreme = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_scale_high_intensity_long_horizon_extreme_20260321" / "summary.json"
    )
    brain_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321" / "summary.json"
    )

    hv = v10["headline_metrics"]
    he = extreme["headline_metrics"]
    hb = brain_v5["headline_metrics"]

    plasticity_rule_alignment_v11 = _clip01(
        (
            hv["plasticity_rule_alignment_v10"]
            + he["extreme_novel_gain"]
            + he["extreme_language_keep"]
            + (1.0 - he["extreme_forgetting_penalty"])
        )
        / 4.0
    )
    structure_rule_alignment_v11 = _clip01(
        (
            hv["structure_rule_alignment_v10"]
            + he["extreme_structure_keep"]
            + he["extreme_context_keep"]
            + hb["direct_structure_measure_v5"]
        )
        / 4.0
    )
    topology_training_readiness_v11 = _clip01(
        (
            hv["topology_training_readiness_v10"]
            + plasticity_rule_alignment_v11
            + structure_rule_alignment_v11
            + he["extreme_readiness"]
            + (1.0 - he["extreme_collapse_risk"])
        )
        / 5.0
    )
    topology_training_gap_v11 = max(0.0, 1.0 - topology_training_readiness_v11)
    extreme_guard_v11 = _clip01(
        (
            he["extreme_language_keep"]
            + he["extreme_structure_keep"]
            + he["extreme_context_keep"]
            + topology_training_readiness_v11
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v11": plasticity_rule_alignment_v11,
            "structure_rule_alignment_v11": structure_rule_alignment_v11,
            "topology_training_readiness_v11": topology_training_readiness_v11,
            "topology_training_gap_v11": topology_training_gap_v11,
            "extreme_guard_v11": extreme_guard_v11,
        },
        "bridge_equation_v11": {
            "plasticity_term": "B_plastic_v11 = mean(B_plastic_v10, G_ext, L_ext, 1 - P_ext)",
            "structure_term": "B_struct_v11 = mean(B_struct_v10, S_ext, C_ext, D_structure_v5)",
            "readiness_term": "R_train_v11 = mean(R_train_v10, B_plastic_v11, B_struct_v11, A_ext, 1 - R_ext)",
            "gap_term": "G_train_v11 = 1 - R_train_v11",
            "guard_term": "H_ext_v11 = mean(L_ext, S_ext, C_ext, R_train_v11)",
        },
        "project_readout": {
            "summary": "训练终式第十一桥开始显式吸收更严苛的规模化高压长时场景，重点看训练规则是否还能压住累积遗忘、长程泛化退化和结构塌缩。",
            "next_question": "下一步要把第十一桥并回主核，检验主核在最严苛在线场景下是否还继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第十一桥报告",
        "",
        f"- plasticity_rule_alignment_v11: {hm['plasticity_rule_alignment_v11']:.6f}",
        f"- structure_rule_alignment_v11: {hm['structure_rule_alignment_v11']:.6f}",
        f"- topology_training_readiness_v11: {hm['topology_training_readiness_v11']:.6f}",
        f"- topology_training_gap_v11: {hm['topology_training_gap_v11']:.6f}",
        f"- extreme_guard_v11: {hm['extreme_guard_v11']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v11_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
