from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v10_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v10_summary() -> dict:
    v9 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v9_20260321" / "summary.json"
    )
    scale_ctx = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_scale_long_context_online_validation_20260321" / "summary.json"
    )
    brain_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321" / "summary.json"
    )

    hv = v9["headline_metrics"]
    hs = scale_ctx["headline_metrics"]
    hb = brain_v5["headline_metrics"]

    plasticity_rule_alignment_v10 = _clip01(
        (
            hv["plasticity_rule_alignment_v9"]
            + hs["scale_novel_gain"]
            + hs["scale_language_keep"]
            + (1.0 - hs["scale_forgetting_penalty"])
        )
        / 4.0
    )
    structure_rule_alignment_v10 = _clip01(
        (
            hv["structure_rule_alignment_v9"]
            + hs["scale_structure_keep"]
            + hs["long_context_generalization"]
            + hb["direct_structure_measure_v5"]
        )
        / 4.0
    )
    topology_training_readiness_v10 = _clip01(
        (
            hv["topology_training_readiness_v9"]
            + plasticity_rule_alignment_v10
            + structure_rule_alignment_v10
            + hs["scale_readiness"]
            + (1.0 - hs["scale_collapse_risk"])
        )
        / 5.0
    )
    topology_training_gap_v10 = max(0.0, 1.0 - topology_training_readiness_v10)
    scale_guard_v10 = _clip01(
        (
            hs["scale_language_keep"]
            + hs["scale_structure_keep"]
            + hs["long_context_generalization"]
            + topology_training_readiness_v10
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v10": plasticity_rule_alignment_v10,
            "structure_rule_alignment_v10": structure_rule_alignment_v10,
            "topology_training_readiness_v10": topology_training_readiness_v10,
            "topology_training_gap_v10": topology_training_gap_v10,
            "scale_guard_v10": scale_guard_v10,
        },
        "bridge_equation_v10": {
            "plasticity_term": "B_plastic_v10 = mean(B_plastic_v9, G_scale, L_scale, 1 - P_scale)",
            "structure_term": "B_struct_v10 = mean(B_struct_v9, S_scale, C_scale, D_structure_v5)",
            "readiness_term": "R_train_v10 = mean(R_train_v9, B_plastic_v10, B_struct_v10, A_scale, 1 - R_scale)",
            "gap_term": "G_train_v10 = 1 - R_train_v10",
            "guard_term": "H_scale_v10 = mean(L_scale, S_scale, C_scale, R_train_v10)",
        },
        "project_readout": {
            "summary": "训练终式第十桥开始显式吸收更大对象集和更长上下文场景，重点检验训练规则在规模化后是否还能压住结构塌缩和长程遗忘。",
            "next_question": "下一步要把第十桥并回主核，看主核在规模化长上下文场景下是否继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第十桥报告",
        "",
        f"- plasticity_rule_alignment_v10: {hm['plasticity_rule_alignment_v10']:.6f}",
        f"- structure_rule_alignment_v10: {hm['structure_rule_alignment_v10']:.6f}",
        f"- topology_training_readiness_v10: {hm['topology_training_readiness_v10']:.6f}",
        f"- topology_training_gap_v10: {hm['topology_training_gap_v10']:.6f}",
        f"- scale_guard_v10: {hm['scale_guard_v10']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v10_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
