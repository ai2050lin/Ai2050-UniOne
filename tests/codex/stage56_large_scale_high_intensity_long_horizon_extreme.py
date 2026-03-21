from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_scale_high_intensity_long_horizon_extreme_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_scale_high_intensity_long_horizon_extreme_summary() -> dict:
    scale_ctx = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_scale_long_context_online_validation_20260321" / "summary.json"
    )
    hi_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_long_horizon_20260321" / "summary.json"
    )
    high_intensity = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_update_20260321" / "summary.json"
    )
    bridge_v10 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v10_20260321" / "summary.json"
    )

    hs = scale_ctx["headline_metrics"]
    hh = hi_long["headline_metrics"]
    hi = high_intensity["headline_metrics"]
    hb = bridge_v10["headline_metrics"]

    extreme_language_keep = _clip01(
        (
            hs["scale_language_keep"]
            + hh["cumulative_language_keep"]
            + hi["high_intensity_language_keep"]
        )
        / 3.0
    )
    extreme_structure_keep = _clip01(
        (
            hs["scale_structure_keep"]
            + hh["cumulative_structure_keep"]
            + hi["high_intensity_structure_keep"]
        )
        / 3.0
    )
    extreme_context_keep = _clip01(
        (
            hs["long_context_generalization"]
            + (1.0 - hs["scale_collapse_risk"])
            + hb["scale_guard_v10"]
        )
        / 3.0
    )
    extreme_novel_gain = _clip01(
        (
            hs["scale_novel_gain"]
            + hh["cumulative_novel_gain"]
            + hi["high_intensity_novel_gain"]
        )
        / 3.0
    )
    extreme_forgetting_penalty = _clip01(
        (
            hs["scale_forgetting_penalty"]
            + hh["cumulative_forgetting_penalty"]
            + hi["high_intensity_forgetting_penalty"]
        )
        / 3.0
    )
    extreme_collapse_risk = _clip01(
        (
            hs["scale_collapse_risk"]
            + hh["cumulative_instability_risk"]
            + (1.0 - extreme_structure_keep)
            + (1.0 - extreme_context_keep)
        )
        / 4.0
    )
    extreme_readiness = _clip01(
        (
            extreme_language_keep
            + extreme_structure_keep
            + extreme_context_keep
            + extreme_novel_gain
            + (1.0 - extreme_forgetting_penalty)
            + (1.0 - extreme_collapse_risk)
            + hb["topology_training_readiness_v10"]
        )
        / 7.0
    )
    extreme_margin = (
        extreme_language_keep
        + extreme_structure_keep
        + extreme_context_keep
        + extreme_novel_gain
        + extreme_readiness
        - extreme_forgetting_penalty
        - extreme_collapse_risk
    )

    return {
        "headline_metrics": {
            "extreme_language_keep": extreme_language_keep,
            "extreme_structure_keep": extreme_structure_keep,
            "extreme_context_keep": extreme_context_keep,
            "extreme_novel_gain": extreme_novel_gain,
            "extreme_forgetting_penalty": extreme_forgetting_penalty,
            "extreme_collapse_risk": extreme_collapse_risk,
            "extreme_readiness": extreme_readiness,
            "extreme_margin": extreme_margin,
        },
        "extreme_online_equation": {
            "language_term": "L_ext = mean(L_scale, L_hi_long, L_hi)",
            "structure_term": "S_ext = mean(S_scale, S_hi_long, S_hi)",
            "context_term": "C_ext = mean(C_scale, 1 - R_scale, H_scale_v10)",
            "novel_term": "G_ext = mean(G_scale, G_hi_long, G_hi)",
            "forgetting_term": "P_ext = mean(P_scale, P_hi_long, P_hi)",
            "collapse_term": "R_ext = mean(R_scale, I_hi_long, 1 - S_ext, 1 - C_ext)",
            "system_term": "M_ext = L_ext + S_ext + C_ext + G_ext + A_ext - P_ext - R_ext",
        },
        "project_readout": {
            "summary": "更大对象集、长上下文、高更新强度和长时间尺度同时并场后，系统开始更真实地暴露语言保持、结构保持、长程上下文保持与累积塌缩之间的张力。",
            "next_question": "下一步要把这组极端场景结果并回训练桥和主核，检验主核在最严苛在线学习口径下是否还继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大对象集长上下文高压长时在线原型报告",
        "",
        f"- extreme_language_keep: {hm['extreme_language_keep']:.6f}",
        f"- extreme_structure_keep: {hm['extreme_structure_keep']:.6f}",
        f"- extreme_context_keep: {hm['extreme_context_keep']:.6f}",
        f"- extreme_novel_gain: {hm['extreme_novel_gain']:.6f}",
        f"- extreme_forgetting_penalty: {hm['extreme_forgetting_penalty']:.6f}",
        f"- extreme_collapse_risk: {hm['extreme_collapse_risk']:.6f}",
        f"- extreme_readiness: {hm['extreme_readiness']:.6f}",
        f"- extreme_margin: {hm['extreme_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_scale_high_intensity_long_horizon_extreme_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
