from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_language_sufficiency_analysis_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_language_sufficiency_summary() -> dict:
    centrality = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_centrality_analysis_20260320" / "summary.json"
    )
    gap = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_language_gap_bottleneck_analysis_20260320" / "summary.json"
    )
    spike = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spiking_network_path_analysis_20260320" / "summary.json"
    )
    unified = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_unified_intelligence_theory_possibility_20260320" / "summary.json"
    )

    hc = centrality["headline_metrics"]
    hg = gap["headline_metrics"]
    hs = spike["headline_metrics"]
    hu = unified["headline_metrics"]

    language_only_sufficiency = (
        hc["language_specialness"]
        + hg["language_theory_completion"]
        + hs["feature_extraction_unlock"]
    ) / 3.0
    intelligence_theory_completion = (
        hg["math_unification_readiness"]
        + hs["direct_agi_unlock"]
        + hu["higher_unified_intelligence_possibility"]
    ) / 3.0
    language_to_all_gap = max(0.0, intelligence_theory_completion - language_only_sufficiency)
    missing_nonlanguage_mass = (
        hg["math_unification_gap"] + hg["agi_realization_gap"] + hs["overlinearity_penalty"]
    ) / 3.0
    language_solves_all_score = max(0.0, language_only_sufficiency - missing_nonlanguage_mass / 2.0)

    return {
        "headline_metrics": {
            "language_only_sufficiency": language_only_sufficiency,
            "intelligence_theory_completion": intelligence_theory_completion,
            "language_to_all_gap": language_to_all_gap,
            "missing_nonlanguage_mass": missing_nonlanguage_mass,
            "language_solves_all_score": language_solves_all_score,
        },
        "sufficiency_equation": {
            "language_term": "S_lang_only = mean(S_lang, R_lang_theory, U_feature)",
            "intelligence_term": "S_intel = mean(R_math_unified, U_agi, P_unified)",
            "gap_term": "G_lang_to_all = max(0, S_intel - S_lang_only)",
            "missing_term": "M_nonlang = mean(G_math, G_agi, P_linear)",
            "score_term": "P_lang_all = S_lang_only - 0.5 * M_nonlang",
        },
        "project_readout": {
            "summary": "语言本身极其特殊，也确实是当前理论最强入口，但它单独还不足以完成全部智能理论。",
            "next_question": "下一步要把语言入口与数学闭合、训练终式、工程落地一起并核，而不是把语言理论当成单独终点。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 语言充分性分析报告",
        "",
        f"- language_only_sufficiency: {hm['language_only_sufficiency']:.6f}",
        f"- intelligence_theory_completion: {hm['intelligence_theory_completion']:.6f}",
        f"- language_to_all_gap: {hm['language_to_all_gap']:.6f}",
        f"- missing_nonlanguage_mass: {hm['missing_nonlanguage_mass']:.6f}",
        f"- language_solves_all_score: {hm['language_solves_all_score']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_language_sufficiency_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
