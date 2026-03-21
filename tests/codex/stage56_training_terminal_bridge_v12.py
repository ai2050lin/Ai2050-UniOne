from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v12_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v12_summary() -> dict:
    v11 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v11_20260321" / "summary.json"
    )
    true_scale = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_online_collapse_probe_20260321" / "summary.json"
    )
    brain_v6 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v6_20260321" / "summary.json"
    )

    hv = v11["headline_metrics"]
    ht = true_scale["headline_metrics"]
    hb = brain_v6["headline_metrics"]

    plasticity_rule_alignment_v12 = _clip01(
        (
            hv["plasticity_rule_alignment_v11"]
            + ht["true_scale_novel_gain"]
            + ht["true_scale_language_keep"]
            + (1.0 - ht["true_scale_forgetting_penalty"])
            + hb["direct_feature_measure_v6"]
        )
        / 5.0
    )
    structure_rule_alignment_v12 = _clip01(
        (
            hv["structure_rule_alignment_v11"]
            + ht["true_scale_structure_keep"]
            + ht["true_scale_context_keep"]
            + (1.0 - ht["true_scale_collapse_risk"])
            + hb["direct_structure_measure_v6"]
        )
        / 5.0
    )
    topology_training_readiness_v12 = _clip01(
        (
            hv["topology_training_readiness_v11"]
            + plasticity_rule_alignment_v12
            + structure_rule_alignment_v12
            + ht["true_scale_readiness"]
            + (1.0 - ht["true_scale_phase_shift_risk"])
            + hb["direct_scale_alignment_v6"]
        )
        / 6.0
    )
    topology_training_gap_v12 = max(0.0, 1.0 - topology_training_readiness_v12)
    true_scale_guard_v12 = _clip01(
        (
            ht["true_scale_language_keep"]
            + ht["true_scale_structure_keep"]
            + ht["true_scale_context_keep"]
            + topology_training_readiness_v12
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v12": plasticity_rule_alignment_v12,
            "structure_rule_alignment_v12": structure_rule_alignment_v12,
            "topology_training_readiness_v12": topology_training_readiness_v12,
            "topology_training_gap_v12": topology_training_gap_v12,
            "true_scale_guard_v12": true_scale_guard_v12,
        },
        "bridge_equation_v12": {
            "plasticity_term": "B_plastic_v12 = mean(B_plastic_v11, G_true, L_true, 1 - P_true, D_feature_v6)",
            "structure_term": "B_struct_v12 = mean(B_struct_v11, S_true, C_true, 1 - R_true, D_structure_v6)",
            "readiness_term": "R_train_v12 = mean(R_train_v11, B_plastic_v12, B_struct_v12, A_true, 1 - Q_true, D_scale_v6)",
            "gap_term": "G_train_v12 = 1 - R_train_v12",
            "guard_term": "H_true_v12 = mean(L_true, S_true, C_true, R_train_v12)",
        },
        "project_readout": {
            "summary": "训练终式第十二桥开始显式吸收真正规模化塌缩探针和脑编码第六版直测链，使训练规则不只面对极端样本压力，而是开始面对更真实的大对象集、长上下文和相变式失稳风险。",
            "next_question": "下一步要把第十二桥并回主核，检验主核在更真实规模化条件下是否还能继续保持收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第十二桥报告",
        "",
        f"- plasticity_rule_alignment_v12: {hm['plasticity_rule_alignment_v12']:.6f}",
        f"- structure_rule_alignment_v12: {hm['structure_rule_alignment_v12']:.6f}",
        f"- topology_training_readiness_v12: {hm['topology_training_readiness_v12']:.6f}",
        f"- topology_training_gap_v12: {hm['topology_training_gap_v12']:.6f}",
        f"- true_scale_guard_v12: {hm['true_scale_guard_v12']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v12_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
