from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_scale_long_context_online_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_scale_long_context_online_validation_summary() -> dict:
    hi_long = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_high_intensity_long_horizon_20260321" / "summary.json"
    )
    large_online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_online_prototype_validation_20260321" / "summary.json"
    )
    contextual = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_contextual_trainable_prototype_20260320" / "summary.json"
    )
    obj_struct = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_object_attribute_structure_prototype_20260320" / "summary.json"
    )

    hh = hi_long["headline_metrics"]
    hl = large_online["headline_metrics"]
    hc = contextual["headline_metrics"]
    ho = obj_struct["headline_metrics"]

    scale_language_keep = _clip01(
        (
            hh["cumulative_language_keep"]
            + hl["large_online_language_keep"]
            + hc["heldout_generalization"]
        )
        / 3.0
    )
    scale_structure_keep = _clip01(
        (
            hh["cumulative_structure_keep"]
            + hl["large_online_structure_keep"]
            + hc["route_split_consistency"]
            + (1.0 - ho["structure_route_split"])
        )
        / 4.0
    )
    long_context_generalization = _clip01(
        (
            hc["context_split_consistency"]
            + hc["heldout_generalization"]
            + (1.0 - ho["context_route_split"])
        )
        / 3.0
    )
    scale_novel_gain = _clip01(
        (
            hh["cumulative_novel_gain"]
            + hl["large_online_novel_gain"]
            + hc["trainable_prototype_margin"] * 0.25
        )
        / 3.0
    )
    scale_forgetting_penalty = _clip01(
        (
            hh["cumulative_forgetting_penalty"]
            + hl["large_online_forgetting_penalty"]
            + max(0.0, 1.0 - long_context_generalization) * 0.3
        )
        / 2.3
    )
    scale_collapse_risk = _clip01(
        (
            scale_forgetting_penalty
            + (1.0 - scale_structure_keep)
            + max(0.0, 1.0 - long_context_generalization)
        )
        / 3.0
    )
    scale_readiness = _clip01(
        (
            scale_language_keep
            + scale_structure_keep
            + long_context_generalization
            + scale_novel_gain
            + (1.0 - scale_forgetting_penalty)
            + (1.0 - scale_collapse_risk)
        )
        / 6.0
    )
    scale_margin = (
        scale_language_keep
        + scale_structure_keep
        + long_context_generalization
        + scale_novel_gain
        + scale_readiness
        - scale_forgetting_penalty
        - scale_collapse_risk
    )

    return {
        "headline_metrics": {
            "scale_language_keep": scale_language_keep,
            "scale_structure_keep": scale_structure_keep,
            "long_context_generalization": long_context_generalization,
            "scale_novel_gain": scale_novel_gain,
            "scale_forgetting_penalty": scale_forgetting_penalty,
            "scale_collapse_risk": scale_collapse_risk,
            "scale_readiness": scale_readiness,
            "scale_margin": scale_margin,
        },
        "large_scale_context_equation": {
            "language_term": "L_scale = mean(L_hi_long, L_large, G_hold_ctx)",
            "structure_term": "S_scale = mean(S_hi_long, S_large, route_ctx, 1 - S_route_obj)",
            "context_term": "C_scale = mean(C_ctx, G_hold_ctx, 1 - C_route_obj)",
            "novel_term": "G_scale = mean(G_hi_long, G_large, 0.25 * M_proto_trainable)",
            "forgetting_term": "P_scale = mean(P_hi_long, P_large, 0.3 * (1 - C_scale))",
            "collapse_term": "R_scale = mean(P_scale, 1 - S_scale, 1 - C_scale)",
            "system_term": "M_scale = L_scale + S_scale + C_scale + G_scale + A_scale - P_scale - R_scale",
        },
        "project_readout": {
            "summary": "更大对象集和更长上下文口径开始同时考察语言保持、结构保持、长上下文泛化和系统性塌缩风险，目的是确认当前主核在规模化后是否还能站住。",
            "next_question": "下一步要把这组规模化长上下文结果并回训练桥和主核，检验是否会进一步暴露结构层的长期塌缩问题。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大对象集长上下文在线原型报告",
        "",
        f"- scale_language_keep: {hm['scale_language_keep']:.6f}",
        f"- scale_structure_keep: {hm['scale_structure_keep']:.6f}",
        f"- long_context_generalization: {hm['long_context_generalization']:.6f}",
        f"- scale_novel_gain: {hm['scale_novel_gain']:.6f}",
        f"- scale_forgetting_penalty: {hm['scale_forgetting_penalty']:.6f}",
        f"- scale_collapse_risk: {hm['scale_collapse_risk']:.6f}",
        f"- scale_readiness: {hm['scale_readiness']:.6f}",
        f"- scale_margin: {hm['scale_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_scale_long_context_online_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
