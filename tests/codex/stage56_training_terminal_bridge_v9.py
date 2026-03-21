from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v9_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_training_terminal_bridge_v9_summary() -> dict:
    v8 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v8_20260321" / "summary.json"
    )
    hi_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_long_horizon_20260321" / "summary.json"
    )
    brain_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321" / "summary.json"
    )

    hv = v8["headline_metrics"]
    hh = hi_long["headline_metrics"]
    hb = brain_v5["headline_metrics"]

    plasticity_rule_alignment_v9 = _clip01(
        (
            hv["plasticity_rule_alignment_v8"]
            + hh["cumulative_novel_gain"]
            + hh["cumulative_language_keep"]
            + (1.0 - hh["cumulative_forgetting_penalty"])
        )
        / 4.0
    )
    structure_rule_alignment_v9 = _clip01(
        (
            hv["structure_rule_alignment_v8"]
            + hh["cumulative_structure_keep"]
            + hb["direct_structure_measure_v5"]
            + (1.0 - hh["cumulative_instability_risk"])
        )
        / 4.0
    )
    topology_training_readiness_v9 = _clip01(
        (
            hv["topology_training_readiness_v8"]
            + plasticity_rule_alignment_v9
            + structure_rule_alignment_v9
            + hh["cumulative_readiness"]
            + hb["direct_brain_measure_v5"]
        )
        / 5.0
    )
    topology_training_gap_v9 = max(0.0, 1.0 - topology_training_readiness_v9)
    cumulative_guard_v9 = _clip01(
        (
            hh["cumulative_language_keep"]
            + hh["cumulative_structure_keep"]
            + (1.0 - hh["cumulative_instability_risk"])
            + topology_training_readiness_v9
        )
        / 4.0
    )

    return {
        "headline_metrics": {
            "plasticity_rule_alignment_v9": plasticity_rule_alignment_v9,
            "structure_rule_alignment_v9": structure_rule_alignment_v9,
            "topology_training_readiness_v9": topology_training_readiness_v9,
            "topology_training_gap_v9": topology_training_gap_v9,
            "cumulative_guard_v9": cumulative_guard_v9,
        },
        "bridge_equation_v9": {
            "plasticity_term": "B_plastic_v9 = mean(B_plastic_v8, G_hi_long, L_hi_long, 1 - P_hi_long)",
            "structure_term": "B_struct_v9 = mean(B_struct_v8, S_hi_long, D_structure_v5, 1 - I_hi_long)",
            "readiness_term": "R_train_v9 = mean(R_train_v8, B_plastic_v9, B_struct_v9, R_hi_long, M_brain_direct_v5)",
            "gap_term": "G_train_v9 = 1 - R_train_v9",
            "guard_term": "H_guard_v9 = mean(L_hi_long, S_hi_long, 1 - I_hi_long, R_train_v9)",
        },
        "project_readout": {
            "summary": "训练终式第九桥开始显式吸收更长时间尺度高强度更新结果，重点看训练规则能否同时约束累积遗忘和结构失稳。",
            "next_question": "下一步要把第九桥并回主核，验证高压长时在线场景下主核是否继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式第九桥报告",
        "",
        f"- plasticity_rule_alignment_v9: {hm['plasticity_rule_alignment_v9']:.6f}",
        f"- structure_rule_alignment_v9: {hm['structure_rule_alignment_v9']:.6f}",
        f"- topology_training_readiness_v9: {hm['topology_training_readiness_v9']:.6f}",
        f"- topology_training_gap_v9: {hm['topology_training_gap_v9']:.6f}",
        f"- cumulative_guard_v9: {hm['cumulative_guard_v9']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_bridge_v9_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
