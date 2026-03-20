from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_kernel_cross_asset_validation_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def build_encoding_kernel_cross_asset_validation_summary() -> dict:
    refined = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_native_refinement_20260320" / "summary.json"
    )
    predictor = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_kernel_regime_predictor_v2_20260320" / "summary.json"
    )
    corpus = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_real_corpus_shortform_validation_20260320" / "summary.json"
    )
    large_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_model_native_variable_refinement_20260320" / "summary.json"
    )
    large_formula = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_large_model_formula_validation_20260320" / "summary.json"
    )

    rhm = refined["headline_metrics"]
    lhm = large_native["headline_metrics"]
    fhm = large_formula["headline_metrics"]
    cor = corpus["correlations"]

    small_support = rhm["encode_balance_refined"]
    predictor_support = predictor["match_ratio"]
    corpus_g_support = _mean(list(cor["G_corpus_proxy"].values()))
    corpus_load_support = _mean([-value for value in cor["L_base_corpus_proxy"].values()])
    corpus_select_support = _mean(list(cor["L_select_corpus_proxy"].values()))
    corpus_support = _mean([corpus_g_support, corpus_load_support, corpus_select_support])
    large_native_support = lhm["native_balance"] / (1.0 + lhm["native_balance"])
    formula_support = fhm["formula_support_score"]

    support_values = [small_support, predictor_support, corpus_support, large_native_support, formula_support]
    cross_asset_support = _mean(support_values)
    support_gap = max(support_values) - min(support_values)

    return {
        "headline_metrics": {
            "small_support": small_support,
            "predictor_support": predictor_support,
            "corpus_support": corpus_support,
            "large_native_support": large_native_support,
            "formula_support": formula_support,
            "cross_asset_support": cross_asset_support,
            "support_gap": support_gap,
        },
        "cross_asset_equation": {
            "small_term": "S_small ~ encode_balance_refined",
            "predictor_term": "S_predict ~ regime_match_ratio",
            "corpus_term": "S_corpus ~ mean(G_positive, -L_base_negative, L_select_positive)",
            "large_term": "S_large ~ native_balance / (1 + native_balance)",
            "total_term": "S_cross = mean(S_small, S_predict, S_corpus, S_large, S_formula)",
        },
        "project_readout": {
            "summary": "这一版把编码核从单资产验证推进到跨资产支持度比较，检查它是否能同时在小样本、真实语料和大模型代理上保持一致方向。",
            "next_question": "下一步要把跨资产支持度并回第三版闭式核，检验编码核能否兼顾解释力和跨资产稳定性。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码核跨资产验证报告",
        "",
        f"- small_support: {hm['small_support']:.6f}",
        f"- predictor_support: {hm['predictor_support']:.6f}",
        f"- corpus_support: {hm['corpus_support']:.6f}",
        f"- large_native_support: {hm['large_native_support']:.6f}",
        f"- formula_support: {hm['formula_support']:.6f}",
        f"- cross_asset_support: {hm['cross_asset_support']:.6f}",
        f"- support_gap: {hm['support_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_kernel_cross_asset_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
