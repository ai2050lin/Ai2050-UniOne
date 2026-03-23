#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

from wordclass_neuron_basic_probe_lib import clamp01


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE124_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323"
STAGE130_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage130_multisyntax_noun_context_probe_20260323"
STAGE131_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage131_l1_l3_l11_propagation_bridge_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage132_unified_variable_fit_20260323"

WEIGHT_STEP = 0.05


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return load_json(summary_path)
    return None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 1e-12 or y_var <= 1e-12:
        return 0.0
    return cov / math.sqrt(x_var * y_var)


def mean_abs_error(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def align_family_rows() -> List[Dict[str, object]]:
    stage124 = load_json(STAGE124_DIR / "summary.json")
    stage130 = load_json(STAGE130_DIR / "summary.json")
    stage131 = load_json(STAGE131_DIR / "summary.json")

    static_score = float(stage124["dominant_general_layer_score"])
    family130 = {row["family_name"]: row for row in stage130["family_rows"]}
    family131 = {row["family_name"]: row for row in stage131["family_rows"]}

    rows = []
    for family_name in family130:
        row130 = family130[family_name]
        row131 = family131[family_name]
        context_score = clamp01(float(row130["dominant_general_layer_score"]))
        bridge_score = clamp01(float(row131["family_path_score"]))
        empirical_target = 0.40 * static_score + 0.40 * context_score + 0.20 * bridge_score
        rows.append(
            {
                "family_name": family_name,
                "static_score": static_score,
                "context_score": context_score,
                "bridge_score": bridge_score,
                "empirical_target": empirical_target,
                "a_proxy": context_score,
                "q_proxy": clamp01((float(row131["l1_l11_corr"]) + 1.0) / 2.0),
                "g_proxy": clamp01((float(row131["l1_l3_corr"]) + 1.0) / 2.0),
                "f_proxy": clamp01((float(row131["l3_l11_corr"]) + 1.0) / 2.0),
            }
        )
    return rows


def iter_weight_sets() -> List[Dict[str, float]]:
    values = [round(i * WEIGHT_STEP, 2) for i in range(int(1 / WEIGHT_STEP) + 1)]
    rows: List[Dict[str, float]] = []
    for wa, wq, wg in itertools.product(values, repeat=3):
        wf = round(1.0 - wa - wq - wg, 2)
        if wf < -1e-9:
            continue
        if wf not in values:
            continue
        rows.append({"a": wa, "q": wq, "g": wg, "f": wf})
    return rows


def score_weight_set(family_rows: Sequence[Dict[str, object]], weights: Dict[str, float]) -> Dict[str, object]:
    predicted = []
    target = []
    enriched_rows = []
    for row in family_rows:
        pred = (
            weights["a"] * float(row["a_proxy"])
            + weights["q"] * float(row["q_proxy"])
            + weights["g"] * float(row["g_proxy"])
            + weights["f"] * float(row["f_proxy"])
        )
        predicted.append(pred)
        target.append(float(row["empirical_target"]))
        enriched = dict(row)
        enriched["predicted_score"] = pred
        enriched_rows.append(enriched)

    corr = pearson(predicted, target)
    mae = mean_abs_error(predicted, target)
    score = 0.65 * clamp01((corr + 1.0) / 2.0) + 0.35 * clamp01(1.0 - mae)
    return {
        "weights": weights,
        "predicted": predicted,
        "target": target,
        "correlation": corr,
        "mae": mae,
        "fit_score": score,
        "family_rows": enriched_rows,
    }


def choose_best_fit(family_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    candidates = [score_weight_set(family_rows, weights) for weights in iter_weight_sets()]
    candidates.sort(
        key=lambda row: (
            float(row["fit_score"]),
            float(row["correlation"]),
            -float(row["mae"]),
            float(row["weights"]["a"]),
            float(row["weights"]["q"]),
        ),
        reverse=True,
    )
    return candidates[0]


def build_summary(best_fit: Dict[str, object], family_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    mean_a = sum(float(row["a_proxy"]) for row in family_rows) / len(family_rows)
    mean_q = sum(float(row["q_proxy"]) for row in family_rows) / len(family_rows)
    mean_g = sum(float(row["g_proxy"]) for row in family_rows) / len(family_rows)
    mean_f = sum(float(row["f_proxy"]) for row in family_rows) / len(family_rows)
    proxy_means = {
        "a_proxy_mean": mean_a,
        "q_proxy_mean": mean_q,
        "g_proxy_mean": mean_g,
        "f_proxy_mean": mean_f,
    }
    weakest_proxy_name = min(proxy_means.items(), key=lambda item: item[1])[0]
    weights = best_fit["weights"]
    formula = (
        f"noun_proxy = {weights['a']:.2f}*a + {weights['q']:.2f}*q + "
        f"{weights['g']:.2f}*g + {weights['f']:.2f}*f"
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage132_unified_variable_fit",
        "title": "名词统一变量拟合",
        "status_short": "gpt2_noun_unified_variable_fit_ready",
        "family_count": len(family_rows),
        "target_reference_law": "noun_core = 0.40*s + 0.40*c + 0.20*b",
        "best_law_name": "aqgf_linear_grid",
        "best_formula": formula,
        "best_weights": weights,
        "best_correlation": best_fit["correlation"],
        "best_mae": best_fit["mae"],
        "noun_unified_variable_fit_score": best_fit["fit_score"],
        "a_proxy_mean": mean_a,
        "q_proxy_mean": mean_q,
        "g_proxy_mean": mean_g,
        "f_proxy_mean": mean_f,
        "weakest_proxy_name": weakest_proxy_name,
        "family_rows": best_fit["family_rows"],
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage132: 名词统一变量拟合",
        "",
        "## 核心结果",
        f"- 家族数量: {summary['family_count']}",
        f"- 参考律: `{summary['target_reference_law']}`",
        f"- 最优律: `{summary['best_formula']}`",
        f"- 相关系数: {summary['best_correlation']:.4f}",
        f"- 平均绝对误差: {summary['best_mae']:.4f}",
        f"- 拟合分数: {summary['noun_unified_variable_fit_score']:.4f}",
        f"- 最弱代理量: {summary['weakest_proxy_name']}",
        "",
        "## 代理量均值",
        f"- a_proxy_mean = {summary['a_proxy_mean']:.4f}",
        f"- q_proxy_mean = {summary['q_proxy_mean']:.4f}",
        f"- g_proxy_mean = {summary['g_proxy_mean']:.4f}",
        f"- f_proxy_mean = {summary['f_proxy_mean']:.4f}",
        "",
        "## 各句法簇",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "- "
            f"{row['family_name']}: "
            f"target={row['empirical_target']:.4f}, "
            f"pred={row['predicted_score']:.4f}, "
            f"a={row['a_proxy']:.4f}, "
            f"q={row['q_proxy']:.4f}, "
            f"g={row['g_proxy']:.4f}, "
            f"f={row['f_proxy']:.4f}"
        )
    lines.extend(
        [
            "",
            "## 理论提示",
            "- a 代理量对应句中名词位的早层定锚强度。",
            "- q 代理量对应早层到后层的保持性，也就是上下文保形能力。",
            "- g 代理量对应 L1 到 L3 的前向选路耦合。",
            "- f 代理量对应 L3 到 L11 的跨层续接强度。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    (output_dir / "STAGE132_UNIFIED_VARIABLE_FIT_REPORT.md").write_text(
        build_report(summary),
        encoding="utf-8-sig",
    )
    (output_dir / "family_rows.json").write_text(
        json.dumps(summary["family_rows"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force:
        cached = load_cached_summary(output_dir)
        if cached is not None:
            return cached

    family_rows = align_family_rows()
    best_fit = choose_best_fit(family_rows)
    summary = build_summary(best_fit, family_rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="名词统一变量拟合")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage132 输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    summary = run_analysis(output_dir=Path(args.output_dir), force=args.force)
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "best_formula": summary["best_formula"],
                "noun_unified_variable_fit_score": summary["noun_unified_variable_fit_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
