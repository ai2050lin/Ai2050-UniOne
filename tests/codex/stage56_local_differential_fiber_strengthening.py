from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_local_differential_fiber_strengthening_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_local_differential_fiber_strengthening_summary() -> dict:
    fibers = _load_json(ROOT / "tests" / "codex_temp" / "stage56_attribute_fiber_nativeization_20260320" / "summary.json")
    charts = _load_json(ROOT / "tests" / "codex_temp" / "stage56_concept_local_chart_expansion_20260320" / "summary.json")

    family_rows = {}
    strengths = []

    for family, row in fibers["family_attribute_systems"].items():
        local_fibers = row["local_fibers"]
        family_strength = float(
            np.mean(
                [
                    np.mean([abs(v) for v in coeffs.values()])
                    for coeffs in local_fibers.values()
                ]
            )
        ) if local_fibers else 0.0
        chart_support = charts["family_charts"][family]["chart_support"]
        strengthened = family_strength * (1.0 + chart_support)
        family_rows[family] = {
            "raw_local_strength": family_strength,
            "chart_support": chart_support,
            "strengthened_local_strength": strengthened,
            "local_attributes": row["local_attributes"],
        }
        strengths.append(strengthened)

    apple_local = fibers["apple_nativeized_fibers"]["local_fiber_coefficients"]
    apple_strengthened_margin = abs(apple_local["round"]) + abs(apple_local["elongated"])

    return {
        "headline_metrics": {
            "mean_strengthened_local_fiber": float(np.mean(strengths)),
            "max_strengthened_local_fiber": float(np.max(strengths)),
            "apple_strengthened_local_margin": apple_strengthened_margin,
            "family_count": len(family_rows),
        },
        "family_strengths": family_rows,
        "project_readout": {
            "summary": (
                "这一轮把局部差分属性纤维和家族局部图册并场，"
                "让局部纤维不再只是弱小残差，而是能借助图册支持度被增强。"
            ),
            "next_question": (
                "下一步要把增强后的局部纤维项并回概念形成核，"
                "检查它能否显著缩小概念形成里家族项与局部项的失衡。"
            ),
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 局部差分纤维强化报告",
        "",
        f"- mean_strengthened_local_fiber: {hm['mean_strengthened_local_fiber']:.6f}",
        f"- max_strengthened_local_fiber: {hm['max_strengthened_local_fiber']:.6f}",
        f"- apple_strengthened_local_margin: {hm['apple_strengthened_local_margin']:.6f}",
        f"- family_count: {hm['family_count']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_local_differential_fiber_strengthening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
