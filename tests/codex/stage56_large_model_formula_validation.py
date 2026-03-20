from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_model_formula_validation_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_formula_validation_summary() -> dict:
    checkpoint_summary = _load_json(ROOT / "tests" / "codex_temp" / "stage56_large_model_checkpoint_alignment_20260320" / "summary.json")
    stability_summary = _load_json(ROOT / "tests" / "codex_temp" / "stage56_large_model_online_stability_proxy_20260320" / "summary.json")
    real_shortform = _load_json(ROOT / "tests" / "codex_temp" / "stage56_real_corpus_shortform_validation_20260320" / "summary.json")

    ordered_ratio = checkpoint_summary["headline_metrics"]["ordered_case_ratio"]
    plasticity_mean = stability_summary["headline_metrics"]["plasticity_mean"]
    stability_mean = stability_summary["headline_metrics"]["stability_mean"]
    risk_mean = stability_summary["headline_metrics"]["risk_load_mean"]

    sign_matrix = real_shortform["sign_matrix"]

    def _sign_ratio(feature: str, sign: str) -> float:
        values = list(sign_matrix.get(feature, {}).values())
        if not values:
            return 0.0
        return sum(1 for value in values if value == sign) / len(values)

    g_proxy = _sign_ratio("G_corpus_proxy", "positive")
    lbase_proxy = _sign_ratio("L_base_corpus_proxy", "negative")
    lselect_proxy = _sign_ratio("L_select_corpus_proxy", "positive")

    summary = {
        "headline_metrics": {
            "ordering_support": ordered_ratio,
            "plasticity_mean": plasticity_mean,
            "stability_mean": stability_mean,
            "risk_load_mean": risk_mean,
            "g_proxy": g_proxy,
            "l_base_proxy": lbase_proxy,
            "l_select_proxy": lselect_proxy,
            "formula_support_score": mean([
                ordered_ratio,
                max(0.0, min(1.0, plasticity_mean)),
                max(0.0, min(1.0, stability_mean)),
                g_proxy,
                lbase_proxy,
                lselect_proxy,
            ]),
        },
        "project_readout": {
            "summary": (
                "这一步把大模型训练阶段顺序、大模型在线稳定性代理、"
                "以及当前短式里的 G/L_base/L_select 口径并回同一条验证链。"
                "如果排序支持、正负号结构和在线代理同时站住，就说明系统短式开始跨规模成立。"
            ),
            "next_question": "下一步要把这些验证量继续压到更少的闭式变量，并减少代理依赖。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 大模型系统公式验证报告",
        "",
        f"- 排序支持: {hm['ordering_support']:.6f}",
        f"- 平均可塑性增益: {hm['plasticity_mean']:.6f}",
        f"- 平均稳定性分数: {hm['stability_mean']:.6f}",
        f"- 平均风险负载: {hm['risk_load_mean']:.6f}",
        f"- G 代理支持: {hm['g_proxy']:.6f}",
        f"- L_base 代理支持: {hm['l_base_proxy']:.6f}",
        f"- L_select 代理支持: {hm['l_select_proxy']:.6f}",
        f"- 系统公式支持分数: {hm['formula_support_score']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_formula_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
