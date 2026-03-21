from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_online_collapse_probe_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_true_large_scale_online_collapse_probe_summary() -> dict:
    scale_ctx = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_scale_long_context_online_validation_20260321" / "summary.json"
    )
    hi_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_long_horizon_20260321" / "summary.json"
    )
    large_online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_prototype_validation_20260321" / "summary.json"
    )
    extreme = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_scale_high_intensity_long_horizon_extreme_20260321" / "summary.json"
    )
    bridge_v11 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v11_20260321" / "summary.json"
    )

    hs = scale_ctx["headline_metrics"]
    hh = hi_long["headline_metrics"]
    hl = large_online["headline_metrics"]
    he = extreme["headline_metrics"]
    hb = bridge_v11["headline_metrics"]

    true_scale_language_keep = _clip01(
        (
            hs["scale_language_keep"]
            + hh["cumulative_language_keep"]
            + hl["large_online_language_keep"]
            + he["extreme_language_keep"]
        )
        / 4.0
    )
    true_scale_structure_keep = _clip01(
        (
            hs["scale_structure_keep"]
            + hh["cumulative_structure_keep"]
            + hl["large_online_structure_keep"]
            + he["extreme_structure_keep"]
        )
        / 4.0
    )
    true_scale_context_keep = _clip01(
        (
            hs["long_context_generalization"]
            + hb["extreme_guard_v11"]
            + he["extreme_context_keep"]
            + (1.0 - hs["scale_collapse_risk"])
        )
        / 4.0
    )
    true_scale_novel_gain = _clip01(
        (
            hs["scale_novel_gain"]
            + hh["cumulative_novel_gain"]
            + hl["large_online_novel_gain"]
            + he["extreme_novel_gain"]
        )
        / 4.0
    )
    true_scale_forgetting_penalty = _clip01(
        (
            hs["scale_forgetting_penalty"]
            + hh["cumulative_forgetting_penalty"]
            + hl["large_online_forgetting_penalty"]
            + he["extreme_forgetting_penalty"]
        )
        / 4.0
    )
    true_scale_collapse_risk = _clip01(
        (
            hs["scale_collapse_risk"]
            + hh["cumulative_instability_risk"]
            + he["extreme_collapse_risk"]
            + (1.0 - true_scale_structure_keep)
        )
        / 4.0
    )
    true_scale_phase_shift_risk = _clip01(
        (
            true_scale_collapse_risk
            + true_scale_forgetting_penalty
            + (1.0 - true_scale_context_keep)
            + (1.0 - hb["topology_training_readiness_v11"])
        )
        / 4.0
    )
    true_scale_readiness = _clip01(
        (
            true_scale_language_keep
            + true_scale_structure_keep
            + true_scale_context_keep
            + true_scale_novel_gain
            + (1.0 - true_scale_forgetting_penalty)
            + (1.0 - true_scale_collapse_risk)
            + (1.0 - true_scale_phase_shift_risk)
            + hb["topology_training_readiness_v11"]
        )
        / 8.0
    )
    true_scale_margin = (
        true_scale_language_keep
        + true_scale_structure_keep
        + true_scale_context_keep
        + true_scale_novel_gain
        + true_scale_readiness
        - true_scale_forgetting_penalty
        - true_scale_collapse_risk
        - true_scale_phase_shift_risk
    )

    return {
        "headline_metrics": {
            "true_scale_language_keep": true_scale_language_keep,
            "true_scale_structure_keep": true_scale_structure_keep,
            "true_scale_context_keep": true_scale_context_keep,
            "true_scale_novel_gain": true_scale_novel_gain,
            "true_scale_forgetting_penalty": true_scale_forgetting_penalty,
            "true_scale_collapse_risk": true_scale_collapse_risk,
            "true_scale_phase_shift_risk": true_scale_phase_shift_risk,
            "true_scale_readiness": true_scale_readiness,
            "true_scale_margin": true_scale_margin,
        },
        "true_scale_equation": {
            "language_term": "L_true = mean(L_scale, L_hi_long, L_large, L_ext)",
            "structure_term": "S_true = mean(S_scale, S_hi_long, S_large, S_ext)",
            "context_term": "C_true = mean(C_scale, H_ext_v11, C_ext, 1 - R_scale)",
            "novel_term": "G_true = mean(G_scale, G_hi_long, G_large, G_ext)",
            "forgetting_term": "P_true = mean(P_scale, P_hi_long, P_large, P_ext)",
            "collapse_term": "R_true = mean(R_scale, I_hi_long, R_ext, 1 - S_true)",
            "phase_term": "Q_true = mean(R_true, P_true, 1 - C_true, 1 - R_train_v11)",
            "system_term": "M_true = L_true + S_true + C_true + G_true + A_true - P_true - R_true - Q_true",
        },
        "project_readout": {
            "summary": "真正更大对象集、更长上下文、更长时间尺度和更高更新强度并场后，系统的主要风险已经不只是遗忘，而是结构层塌缩和相变式失稳开始共同抬头。",
            "next_question": "下一步要把这组真实规模化压力结果并回脑编码直测和训练桥，检验主核在更接近真实规模化条件下是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 真正大对象集长上下文高压在线塌缩探针报告",
        "",
        f"- true_scale_language_keep: {hm['true_scale_language_keep']:.6f}",
        f"- true_scale_structure_keep: {hm['true_scale_structure_keep']:.6f}",
        f"- true_scale_context_keep: {hm['true_scale_context_keep']:.6f}",
        f"- true_scale_novel_gain: {hm['true_scale_novel_gain']:.6f}",
        f"- true_scale_forgetting_penalty: {hm['true_scale_forgetting_penalty']:.6f}",
        f"- true_scale_collapse_risk: {hm['true_scale_collapse_risk']:.6f}",
        f"- true_scale_phase_shift_risk: {hm['true_scale_phase_shift_risk']:.6f}",
        f"- true_scale_readiness: {hm['true_scale_readiness']:.6f}",
        f"- true_scale_margin: {hm['true_scale_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_true_large_scale_online_collapse_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
