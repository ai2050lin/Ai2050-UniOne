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
STAGE121_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage121_adverb_gate_bridge_probe_20260323"
STAGE123_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage123_route_shift_layer_localization_20260323"
STAGE134_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage134_noun_verb_joint_propagation_20260323"
STAGE136_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage136_anaphora_ellipsis_propagation_20260323"
STAGE137_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage137_noun_verb_result_chain_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage138_conditional_gating_field_reconstruction_20260323"

WEIGHT_STEP = 0.1


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


def align_family_rows() -> Dict[str, object]:
    stage121 = load_json(STAGE121_DIR / "summary.json")
    stage123 = load_json(STAGE123_DIR / "summary.json")
    stage134 = load_json(STAGE134_DIR / "summary.json")
    stage136 = load_json(STAGE136_DIR / "summary.json")
    stage137 = load_json(STAGE137_DIR / "summary.json")

    family134 = {row["family_name"]: row for row in stage134["family_rows"]}
    family136 = {row["family_name"]: row for row in stage136["family_rows"]}
    family137 = {row["family_name"]: row for row in stage137["family_rows"]}

    rows = []
    for family_name in sorted(set(family134) & set(family136) & set(family137)):
        row134 = family134[family_name]
        row136 = family136[family_name]
        row137 = family137[family_name]
        q_proxy = clamp01(
            0.35 * ((float(row136["noun_pronoun_late_corr"]) + 1.0) / 2.0)
            + 0.35 * ((float(row136["noun_ellipsis_late_corr"]) + 1.0) / 2.0)
            + 0.30 * float(row136["pronoun_sign_consistency_rate"])
        )
        b_proxy = clamp01(float(row134["route_band_gap"]) / 0.08)
        g_proxy = clamp01(
            0.50 * ((float(row134["noun_route_corr"]) + 1.0) / 2.0)
            + 0.50 * ((float(row137["verb_result_corr"]) + 1.0) / 2.0)
        )
        empirical_target = (
            0.45 * float(row137["chain_family_score"])
            + 0.35 * float(row136["family_score"])
            + 0.20 * float(row134["joint_family_score"])
        )
        rows.append(
            {
                "family_name": family_name,
                "q_proxy": q_proxy,
                "b_proxy": b_proxy,
                "g_proxy": g_proxy,
                "empirical_target": empirical_target,
            }
        )

    return {
        "stage121_adverb_gate_bridge_score": float(stage121["adverb_gate_bridge_score"]),
        "stage121_adverb_balance_mean": float(stage121["adverb_action_function_balance_mean"]),
        "stage123_route_shift_layer_localization_score": float(stage123["route_shift_layer_localization_score"]),
        "rows": rows,
    }


def iter_weight_sets() -> List[Dict[str, float]]:
    values = [round(i * WEIGHT_STEP, 2) for i in range(int(1 / WEIGHT_STEP) + 1)]
    rows: List[Dict[str, float]] = []
    for wq, wb in itertools.product(values, repeat=2):
        wg = round(1.0 - wq - wb, 2)
        if wg < -1e-9:
            continue
        if wg not in values:
            continue
        rows.append({"q": wq, "b": wb, "g": wg})
    return rows


def score_weight_set(family_rows: Sequence[Dict[str, object]], weights: Dict[str, float]) -> Dict[str, object]:
    predicted = []
    target = []
    enriched_rows = []
    for row in family_rows:
        pred = (
            weights["q"] * float(row["q_proxy"])
            + weights["b"] * float(row["b_proxy"])
            + weights["g"] * float(row["g_proxy"])
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
            float(row["weights"]["q"]),
        ),
        reverse=True,
    )
    return candidates[0]


def build_summary(aligned: Dict[str, object], best_fit: Dict[str, object]) -> Dict[str, object]:
    family_rows = best_fit["family_rows"]
    mean_q = sum(float(row["q_proxy"]) for row in family_rows) / len(family_rows)
    mean_b = sum(float(row["b_proxy"]) for row in family_rows) / len(family_rows)
    mean_g = sum(float(row["g_proxy"]) for row in family_rows) / len(family_rows)
    proxy_means = {
        "q_proxy_mean": mean_q,
        "b_proxy_mean": mean_b,
        "g_proxy_mean": mean_g,
    }
    weakest_proxy_name = min(proxy_means.items(), key=lambda item: item[1])[0]
    strongest_proxy_name = max(proxy_means.items(), key=lambda item: item[1])[0]
    weights = best_fit["weights"]
    formula = f"field = {weights['q']:.2f}*q + {weights['b']:.2f}*b + {weights['g']:.2f}*g"
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage138_conditional_gating_field_reconstruction",
        "title": "条件门控场重建块",
        "status_short": "gpt2_conditional_gating_field_ready",
        "family_count": len(family_rows),
        "best_law_name": "qbg_linear_grid",
        "best_formula": formula,
        "best_weights": weights,
        "best_correlation": best_fit["correlation"],
        "best_mae": best_fit["mae"],
        "conditional_gating_field_score": best_fit["fit_score"],
        "q_proxy_mean": mean_q,
        "b_proxy_mean": mean_b,
        "g_proxy_mean": mean_g,
        "stage121_adverb_gate_bridge_score": float(aligned["stage121_adverb_gate_bridge_score"]),
        "stage121_adverb_balance_mean": float(aligned["stage121_adverb_balance_mean"]),
        "stage123_route_shift_layer_localization_score": float(aligned["stage123_route_shift_layer_localization_score"]),
        "weakest_proxy_name": weakest_proxy_name,
        "strongest_proxy_name": strongest_proxy_name,
        "family_rows": family_rows,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage138: 条件门控场重建块",
        "",
        "## 核心结果",
        f"- 家族数量: {summary['family_count']}",
        f"- 最优律: `{summary['best_formula']}`",
        f"- 相关系数: {summary['best_correlation']:.4f}",
        f"- 平均绝对误差: {summary['best_mae']:.4f}",
        f"- 条件门控场分数: {summary['conditional_gating_field_score']:.4f}",
        f"- 最强代理量: {summary['strongest_proxy_name']}",
        f"- 最弱代理量: {summary['weakest_proxy_name']}",
        "",
        "## 代理量均值",
        f"- q_proxy_mean = {summary['q_proxy_mean']:.4f}",
        f"- b_proxy_mean = {summary['b_proxy_mean']:.4f}",
        f"- g_proxy_mean = {summary['g_proxy_mean']:.4f}",
        "",
        "## 各语篇家族",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "- "
            f"{row['family_name']}: "
            f"target={row['empirical_target']:.4f}, "
            f"pred={row['predicted_score']:.4f}, "
            f"q={row['q_proxy']:.4f}, "
            f"b={row['b_proxy']:.4f}, "
            f"g={row['g_proxy']:.4f}"
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "STAGE138_CONDITIONAL_GATING_FIELD_RECONSTRUCTION_REPORT.md").write_text(
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

    aligned = align_family_rows()
    best_fit = choose_best_fit(aligned["rows"])
    summary = build_summary(aligned, best_fit)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="条件门控场重建块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage138 输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    summary = run_analysis(output_dir=Path(args.output_dir), force=args.force)
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "best_formula": summary["best_formula"],
                "conditional_gating_field_score": summary["conditional_gating_field_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
