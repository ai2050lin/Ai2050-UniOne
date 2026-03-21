from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_system_coupled_degradation_validation_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_large_system_coupled_degradation_validation_summary() -> dict:
    coupled = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_structure_coupled_validation_20260321" / "summary.json"
    )
    scale = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_scale_long_context_online_validation_20260321" / "summary.json"
    )
    extreme = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_scale_high_intensity_long_horizon_extreme_20260321" / "summary.json"
    )
    route = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_true_large_scale_route_degradation_probe_20260321" / "summary.json"
    )

    hc = coupled["headline_metrics"]
    hs = scale["headline_metrics"]
    he = extreme["headline_metrics"]
    hr = route["headline_metrics"]

    mega_coupled_language_keep = _clip01(
        (hc["coupled_readiness"] + hs["scale_language_keep"] + he["extreme_language_keep"]) / 3.0
    )
    mega_coupled_structure_keep = _clip01(
        (hc["coupled_structure_keep"] + hs["scale_structure_keep"] + he["extreme_structure_keep"]) / 3.0
    )
    mega_coupled_context_keep = _clip01(
        (hc["coupled_context_keep"] + hs["long_context_generalization"] + he["extreme_context_keep"]) / 3.0
    )
    mega_coupled_novel_gain = _clip01(
        (hc["coupled_novel_gain"] + hs["scale_novel_gain"] + he["extreme_novel_gain"]) / 3.0
    )
    mega_coupled_forgetting_penalty = _clip01(
        (hc["coupled_forgetting_penalty"] + hs["scale_forgetting_penalty"] + he["extreme_forgetting_penalty"]) / 3.0
    )
    mega_coupled_route_degradation = _clip01(
        (hr["route_degradation_risk"] + (1.0 - hr["route_resilience"]) + hs["scale_collapse_risk"]) / 3.0
    )
    mega_coupled_collapse_risk = _clip01(
        (hr["structure_phase_shift_risk"] + hs["scale_collapse_risk"] + he["extreme_collapse_risk"]) / 3.0
    )
    mega_coupled_readiness = _clip01(
        (
            mega_coupled_language_keep
            + mega_coupled_structure_keep
            + mega_coupled_context_keep
            + mega_coupled_novel_gain
            + (1.0 - mega_coupled_forgetting_penalty)
            + (1.0 - mega_coupled_route_degradation)
            + (1.0 - mega_coupled_collapse_risk)
        )
        / 7.0
    )
    mega_coupled_margin = (
        mega_coupled_language_keep
        + mega_coupled_structure_keep
        + mega_coupled_context_keep
        + mega_coupled_novel_gain
        + mega_coupled_readiness
        - mega_coupled_forgetting_penalty
        - mega_coupled_route_degradation
        - mega_coupled_collapse_risk
    )

    return {
        "headline_metrics": {
            "mega_coupled_language_keep": mega_coupled_language_keep,
            "mega_coupled_structure_keep": mega_coupled_structure_keep,
            "mega_coupled_context_keep": mega_coupled_context_keep,
            "mega_coupled_novel_gain": mega_coupled_novel_gain,
            "mega_coupled_forgetting_penalty": mega_coupled_forgetting_penalty,
            "mega_coupled_route_degradation": mega_coupled_route_degradation,
            "mega_coupled_collapse_risk": mega_coupled_collapse_risk,
            "mega_coupled_readiness": mega_coupled_readiness,
            "mega_coupled_margin": mega_coupled_margin,
        },
        "mega_equation": {
            "language_term": "L_mega = mean(R_coupled, L_scale, L_ext)",
            "structure_term": "S_mega = mean(K_struct_coupled, S_scale, S_ext)",
            "context_term": "C_mega = mean(K_ctx_coupled, G_long_ctx, C_ext)",
            "risk_term": "R_mega = mean(P_coupled, R_route_mega, R_collapse_mega)",
            "system_term": "M_mega = L_mega + S_mega + C_mega + G_mega + A_mega - R_mega",
        },
        "project_readout": {
            "summary": "更大系统场景下，联动退化链已经不仅是局部风险，而是开始跨对象集、长上下文和高压更新一起显形，系统瓶颈进一步集中到结构保持、上下文保持和路由韧性的协同问题。",
            "next_question": "下一步要把这组更大系统联动退化结果并回脑编码直测和训练桥，检验主核在更高压力下是否还能继续收口。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 更大系统联动退化验证报告",
        "",
        f"- mega_coupled_language_keep: {hm['mega_coupled_language_keep']:.6f}",
        f"- mega_coupled_structure_keep: {hm['mega_coupled_structure_keep']:.6f}",
        f"- mega_coupled_context_keep: {hm['mega_coupled_context_keep']:.6f}",
        f"- mega_coupled_novel_gain: {hm['mega_coupled_novel_gain']:.6f}",
        f"- mega_coupled_forgetting_penalty: {hm['mega_coupled_forgetting_penalty']:.6f}",
        f"- mega_coupled_route_degradation: {hm['mega_coupled_route_degradation']:.6f}",
        f"- mega_coupled_collapse_risk: {hm['mega_coupled_collapse_risk']:.6f}",
        f"- mega_coupled_readiness: {hm['mega_coupled_readiness']:.6f}",
        f"- mega_coupled_margin: {hm['mega_coupled_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_large_system_coupled_degradation_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
