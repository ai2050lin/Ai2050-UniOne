from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_long_horizon_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_online_high_intensity_long_horizon_summary() -> dict:
    high_intensity = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_update_20260321" / "summary.json"
    )
    long_horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_prototype_20260321" / "summary.json"
    )
    online_horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321" / "summary.json"
    )
    large_online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_prototype_validation_20260321" / "summary.json"
    )
    curriculum = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_3d_long_horizon_plasticity_curriculum_20260321" / "summary.json"
    )

    hh = high_intensity["headline_metrics"]
    hl = long_horizon["headline_metrics"]
    ho = online_horizon["headline_metrics"]
    hg = large_online["headline_metrics"]
    hc = curriculum["headline_metrics"]

    cumulative_language_keep = _clip01(
        (
            hh["high_intensity_language_keep"]
            + hg["large_online_language_keep"]
            + hl["topo_long_retention"]
        )
        / 3.0
    )
    cumulative_structure_keep = _clip01(
        (
            hh["high_intensity_structure_keep"]
            + hg["large_online_structure_keep"]
            + hl["topo_long_structural_survival"]
            + ho["structural_survival"]
        )
        / 4.0
    )
    cumulative_novel_gain = _clip01(
        (
            hh["high_intensity_novel_gain"]
            + hg["large_online_novel_gain"]
            + hl["topo_long_plasticity"]
            + hc["long_horizon_growth_v2"]
        )
        / 4.0
    )
    cumulative_forgetting_penalty = _clip01(
        (
            hh["high_intensity_forgetting_penalty"]
            + hg["large_online_forgetting_penalty"]
            + ho["cumulative_rollback"]
        )
        / 3.0
    )
    cumulative_instability_risk = _clip01(
        (
            cumulative_forgetting_penalty
            + (1.0 - cumulative_structure_keep)
            + (1.0 - hl["topo_long_context_survival"])
        )
        / 3.0
    )
    cumulative_readiness = _clip01(
        (
            cumulative_language_keep
            + cumulative_structure_keep
            + cumulative_novel_gain
            + (1.0 - cumulative_forgetting_penalty)
            + (1.0 - cumulative_instability_risk)
        )
        / 5.0
    )
    cumulative_margin = (
        cumulative_language_keep
        + cumulative_structure_keep
        + cumulative_novel_gain
        + cumulative_readiness
        - cumulative_forgetting_penalty
        - cumulative_instability_risk
    )

    return {
        "headline_metrics": {
            "cumulative_language_keep": cumulative_language_keep,
            "cumulative_structure_keep": cumulative_structure_keep,
            "cumulative_novel_gain": cumulative_novel_gain,
            "cumulative_forgetting_penalty": cumulative_forgetting_penalty,
            "cumulative_instability_risk": cumulative_instability_risk,
            "cumulative_readiness": cumulative_readiness,
            "cumulative_margin": cumulative_margin,
        },
        "high_intensity_long_horizon_equation": {
            "language_term": "L_hi_long = mean(L_hi, L_large, R_topo_long)",
            "structure_term": "S_hi_long = mean(S_hi, S_large, S_topo_long, S_long)",
            "novel_term": "G_hi_long = mean(G_hi, G_large, P_topo_long, G_curr_v2)",
            "forgetting_term": "P_hi_long = mean(P_hi, P_large, rollback_long)",
            "instability_term": "I_hi_long = mean(P_hi_long, 1 - S_hi_long, 1 - C_topo_long)",
            "system_term": "M_hi_long = L_hi_long + S_hi_long + G_hi_long + R_hi_long - P_hi_long - I_hi_long",
        },
        "project_readout": {
            "summary": "更长时间尺度高强度在线原型开始直接量化累积性遗忘、结构保持和系统失稳风险，目的是确认高压持续更新下系统是否会出现延迟性塌缩。",
            "next_question": "下一步要把这组高强度长时间尺度量并回训练桥，检验训练规则是否仍然能压住累积失稳。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更长时间尺度高强度在线原型报告",
        "",
        f"- cumulative_language_keep: {hm['cumulative_language_keep']:.6f}",
        f"- cumulative_structure_keep: {hm['cumulative_structure_keep']:.6f}",
        f"- cumulative_novel_gain: {hm['cumulative_novel_gain']:.6f}",
        f"- cumulative_forgetting_penalty: {hm['cumulative_forgetting_penalty']:.6f}",
        f"- cumulative_instability_risk: {hm['cumulative_instability_risk']:.6f}",
        f"- cumulative_readiness: {hm['cumulative_readiness']:.6f}",
        f"- cumulative_margin: {hm['cumulative_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_online_high_intensity_long_horizon_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
