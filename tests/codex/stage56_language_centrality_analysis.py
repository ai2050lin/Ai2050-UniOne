from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_language_centrality_analysis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_language_centrality_summary() -> dict:
    synth = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_dnn_brain_math_theory_synthesis_20260320" / "summary.json"
    )
    gap = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_gap_bottleneck_analysis_20260320" / "summary.json"
    )
    spike = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spiking_network_path_analysis_20260320" / "summary.json"
    )

    hs = synth["headline_metrics"]
    hg = gap["headline_metrics"]
    hp = spike["headline_metrics"]

    dnn_language_norm = _clip01(hs["dnn_language_core"] / 3.0)
    language_centrality = (
        dnn_language_norm
        + hg["language_theory_completion"]
        + hp["feature_extraction_unlock"]
    ) / 3.0
    language_bridge_power = (
        hg["brain_structure_path_readiness"] + hp["structure_generation_unlock"]
    ) / 2.0
    language_specialness = (
        language_centrality + language_bridge_power
    ) / 2.0
    language_residual = 1.0 - language_specialness

    return {
        "headline_metrics": {
            "dnn_language_norm": dnn_language_norm,
            "language_centrality": language_centrality,
            "language_bridge_power": language_bridge_power,
            "language_specialness": language_specialness,
            "language_residual": language_residual,
        },
        "centrality_equation": {
            "centrality_term": "C_lang = mean(T_dnn_norm, R_lang_theory, U_feature)",
            "bridge_term": "B_lang = mean(R_brain_path, U_structure)",
            "specialness_term": "S_lang = mean(C_lang, B_lang)",
            "residual_term": "R_lang = 1 - S_lang",
        },
        "project_readout": {
            "summary": "语言在当前理论里依然是最强入口和最强桥接层，但它更像智能理论的主入口，而不是单独足够的全部理论。",
            "next_question": "下一步要判断语言的中心性是否已经高到足以单独推出统一智能理论，还是仍然需要数学闭合和训练终式共同参与。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 语言中心性分析报告",
        "",
        f"- dnn_language_norm: {hm['dnn_language_norm']:.6f}",
        f"- language_centrality: {hm['language_centrality']:.6f}",
        f"- language_bridge_power: {hm['language_bridge_power']:.6f}",
        f"- language_specialness: {hm['language_specialness']:.6f}",
        f"- language_residual: {hm['language_residual']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_language_centrality_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
