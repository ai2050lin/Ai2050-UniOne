from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_update_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_online_high_intensity_update_summary() -> dict:
    large_online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_prototype_validation_20260321" / "summary.json"
    )
    curriculum = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_curriculum_20260321" / "summary.json"
    )
    brain_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v5_20260321" / "summary.json"
    )
    bridge_v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v7_20260321" / "summary.json"
    )

    hl = large_online["headline_metrics"]
    hc = curriculum["headline_metrics"]
    hb = brain_v5["headline_metrics"]
    ht = bridge_v7["headline_metrics"]

    high_intensity_language_keep = _clip01(
        (
            hl["large_online_language_keep"]
            + ht["scaling_guard_v7"]
            + hb["direct_feature_measure_v5"]
        )
        / 3.0
    )
    high_intensity_novel_gain = _clip01(
        (
            hl["large_online_novel_gain"]
            + hc["curriculum_plasticity_gain"]
            + hc["long_horizon_growth_v2"]
        )
        / 3.0
    )
    high_intensity_structure_keep = _clip01(
        (
            hl["large_online_structure_keep"]
            + hb["direct_structure_measure_v5"]
            + ht["structure_rule_alignment_v7"]
        )
        / 3.0
    )
    high_intensity_forgetting_penalty = _clip01(
        (
            hl["large_online_forgetting_penalty"]
            + (1.0 - ht["topology_training_readiness_v7"])
            + max(0.0, 1.0 - hb["direct_brain_measure_v5"]) * 0.5
        )
        / 2.5
    )
    high_intensity_stability = _clip01(
        (
            high_intensity_language_keep
            + high_intensity_structure_keep
            + (1.0 - high_intensity_forgetting_penalty)
            + ht["scaling_guard_v7"]
        )
        / 4.0
    )
    high_intensity_margin = (
        high_intensity_language_keep
        + high_intensity_novel_gain
        + high_intensity_structure_keep
        + high_intensity_stability
        - high_intensity_forgetting_penalty
    )

    return {
        "headline_metrics": {
            "high_intensity_language_keep": high_intensity_language_keep,
            "high_intensity_novel_gain": high_intensity_novel_gain,
            "high_intensity_structure_keep": high_intensity_structure_keep,
            "high_intensity_forgetting_penalty": high_intensity_forgetting_penalty,
            "high_intensity_stability": high_intensity_stability,
            "high_intensity_margin": high_intensity_margin,
        },
        "high_intensity_equation": {
            "language_term": "L_hi = mean(L_large, H_scale_v7, D_feature_v5)",
            "novel_term": "G_hi = mean(G_large, P_curr, G_curr)",
            "structure_term": "S_hi = mean(S_large, D_structure_v5, B_struct_v7)",
            "forgetting_term": "P_hi = mean(P_large, 1 - R_train_v7, 0.5 * (1 - M_brain_direct_v5))",
            "stability_term": "R_hi = mean(L_hi, S_hi, 1 - P_hi, H_scale_v7)",
            "system_term": "M_hi = L_hi + G_hi + S_hi + R_hi - P_hi",
        },
        "project_readout": {
            "summary": "更高更新强度口径开始把大原型的语言保持、增量学习、结构保持和遗忘惩罚放到更严苛的条件下评估，目的是提前暴露系统性失稳风险。",
            "next_question": "下一步要把这组高强度量并回训练桥，看训练规则在更严苛场景下是否还能保持收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更高更新强度在线原型报告",
        "",
        f"- high_intensity_language_keep: {hm['high_intensity_language_keep']:.6f}",
        f"- high_intensity_novel_gain: {hm['high_intensity_novel_gain']:.6f}",
        f"- high_intensity_structure_keep: {hm['high_intensity_structure_keep']:.6f}",
        f"- high_intensity_forgetting_penalty: {hm['high_intensity_forgetting_penalty']:.6f}",
        f"- high_intensity_stability: {hm['high_intensity_stability']:.6f}",
        f"- high_intensity_margin: {hm['high_intensity_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_online_high_intensity_update_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
