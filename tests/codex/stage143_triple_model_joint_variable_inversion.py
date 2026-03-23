#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

from cross_model_language_shared import PROJECT_ROOT, build_all_model_bundles, clamp01, corr_to_unit


OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage143_triple_model_joint_variable_inversion_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE143_TRIPLE_MODEL_JOINT_VARIABLE_INVERSION_REPORT.md"
WEIGHT_STEP = 0.10


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
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


def bundle_to_family_maps(bundle: Dict[str, object]) -> Dict[str, Dict[str, Dict[str, object]]]:
    if bundle["model_key"] == "gpt2":
        return {
            "discourse": {row["family_name"]: row for row in bundle["stage133"]["family_rows"]},
            "joint": {row["family_name"]: row for row in bundle["stage134"]["family_rows"]},
            "anaphora": {row["family_name"]: row for row in bundle["stage136"]["family_rows"]},
            "result": {row["family_name"]: row for row in bundle["stage137"]["family_rows"]},
            "field": {row["family_name"]: row for row in bundle["stage138"]["family_rows"]},
            "syntax_stability_rate": {"global": {"value": float(bundle["stage130"].get("syntax_stability_rate", 1.0))}},
        }
    return {
        "discourse": {row["family_name"]: row for row in bundle["discourse"]["family_rows"]},
        "joint": {row["family_name"]: row for row in bundle["joint"]["family_rows"]},
        "anaphora": {row["family_name"]: row for row in bundle["anaphora"]["family_rows"]},
        "result": {row["family_name"]: row for row in bundle["result"]["family_rows"]},
        "field": {row["family_name"]: row for row in bundle["field"]["family_rows"]},
        "syntax_stability_rate": {"global": {"value": float(bundle["transfer"]["qwen_core_metrics"]["syntax_stability_rate"])}},
    }


def build_aligned_rows() -> List[Dict[str, object]]:
    bundles = build_all_model_bundles()
    out: List[Dict[str, object]] = []
    for model_key, bundle in bundles.items():
        maps = bundle_to_family_maps(bundle)
        family_names = sorted(set(maps["discourse"]) & set(maps["joint"]) & set(maps["anaphora"]) & set(maps["result"]) & set(maps["field"]))
        syntax_stability = float(maps["syntax_stability_rate"]["global"]["value"])
        for family_name in family_names:
            discourse = maps["discourse"][family_name]
            joint = maps["joint"][family_name]
            anaphora = maps["anaphora"][family_name]
            result = maps["result"][family_name]
            field = maps["field"][family_name]

            early_anchor = 0.5 * float(discourse.get("early_sign_consistency_rate", 0.0)) + 0.5 * corr_to_unit(float(discourse.get("early_remention_corr", 0.0)))
            a_proxy = 0.5 * syntax_stability + 0.5 * early_anchor

            pronoun_late = corr_to_unit(float(anaphora.get("noun_pronoun_late_corr", 0.0)))
            ellipsis_late = corr_to_unit(float(anaphora.get("noun_ellipsis_late_corr", 0.0)))
            pronoun_consistency = float(anaphora.get("pronoun_sign_consistency_rate", 0.0))
            ellipsis_consistency = float(anaphora.get("ellipsis_sign_consistency_rate", 0.0))
            r_proxy = 0.25 * pronoun_late + 0.25 * ellipsis_late + 0.25 * pronoun_consistency + 0.25 * ellipsis_consistency

            q_proxy = float(field.get("q_proxy", 0.0))
            b_proxy = float(field.get("b_proxy", 0.0))
            field_g_proxy = float(field.get("g_proxy", 0.0))
            route_corr = corr_to_unit(float(joint.get("noun_route_corr", 0.0)))
            result_corr = corr_to_unit(float(result.get("verb_result_corr", 0.0)))
            g_proxy = 0.5 * field_g_proxy + 0.25 * route_corr + 0.25 * result_corr

            late_remention = corr_to_unit(float(discourse.get("late_remention_corr", 0.0)))
            late_consistency = float(discourse.get("late_sign_consistency_rate", 0.0))
            f_proxy = 0.5 * late_remention + 0.5 * late_consistency

            empirical_target = (
                0.30 * float(discourse.get("discourse_family_score", 0.0))
                + 0.20 * float(joint.get("joint_family_score", 0.0))
                + 0.20 * float(result.get("chain_family_score", 0.0))
                + 0.20 * float(anaphora.get("family_score", 0.0))
                + 0.10 * float(field.get("empirical_target", 0.0))
            )

            out.append(
                {
                    "model_key": model_key,
                    "display_name": bundle["display_name"],
                    "family_name": family_name,
                    "a_proxy": a_proxy,
                    "r_proxy": r_proxy,
                    "q_proxy": q_proxy,
                    "g_proxy": g_proxy,
                    "f_proxy": f_proxy,
                    "b_proxy": b_proxy,
                    "empirical_target": empirical_target,
                }
            )
    return out


def iter_weight_sets() -> List[Dict[str, float]]:
    values = [round(i * WEIGHT_STEP, 2) for i in range(int(1 / WEIGHT_STEP) + 1)]
    rows: List[Dict[str, float]] = []
    for wa, wr, wq, wg, wf in itertools.product(values, repeat=5):
        wb = round(1.0 - wa - wr - wq - wg - wf, 2)
        if wb < -1e-9:
            continue
        if wb not in values:
            continue
        rows.append({"a": wa, "r": wr, "q": wq, "g": wg, "f": wf, "b": wb})
    return rows


def score_weight_set(rows: Sequence[Dict[str, object]], weights: Dict[str, float]) -> Dict[str, object]:
    predicted = []
    target = []
    for row in rows:
        pred = (
            weights["a"] * float(row["a_proxy"])
            + weights["r"] * float(row["r_proxy"])
            + weights["q"] * float(row["q_proxy"])
            + weights["g"] * float(row["g_proxy"])
            + weights["f"] * float(row["f_proxy"])
            + weights["b"] * float(row["b_proxy"])
        )
        predicted.append(pred)
        target.append(float(row["empirical_target"]))
    corr = pearson(predicted, target)
    mae = mean_abs_error(predicted, target)
    fit_score = 0.65 * clamp01((corr + 1.0) / 2.0) + 0.35 * clamp01(1.0 - mae)
    return {"weights": weights, "correlation": corr, "mae": mae, "fit_score": fit_score}


def choose_best_fit(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    candidates = [score_weight_set(rows, weights) for weights in iter_weight_sets()]
    candidates.sort(
        key=lambda item: (
            float(item["fit_score"]),
            float(item["correlation"]),
            -float(item["mae"]),
            float(item["weights"]["a"]),
            float(item["weights"]["g"]),
        ),
        reverse=True,
    )
    return candidates[0]


def build_summary(rows: Sequence[Dict[str, object]], best_fit: Dict[str, object]) -> Dict[str, object]:
    proxy_means = {
        "a_proxy_mean": sum(float(row["a_proxy"]) for row in rows) / len(rows),
        "r_proxy_mean": sum(float(row["r_proxy"]) for row in rows) / len(rows),
        "q_proxy_mean": sum(float(row["q_proxy"]) for row in rows) / len(rows),
        "g_proxy_mean": sum(float(row["g_proxy"]) for row in rows) / len(rows),
        "f_proxy_mean": sum(float(row["f_proxy"]) for row in rows) / len(rows),
        "b_proxy_mean": sum(float(row["b_proxy"]) for row in rows) / len(rows),
    }
    weakest_proxy_name = min(proxy_means.items(), key=lambda item: item[1])[0]
    strongest_proxy_name = max(proxy_means.items(), key=lambda item: item[1])[0]
    weights = best_fit["weights"]
    formula = (
        f"triple_inversion = {weights['a']:.2f}*a + {weights['r']:.2f}*r + {weights['q']:.2f}*q + "
        f"{weights['g']:.2f}*g + {weights['f']:.2f}*f + {weights['b']:.2f}*b"
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage143_triple_model_joint_variable_inversion",
        "title": "三模型联合变量反演块",
        "status_short": "triple_model_joint_inversion_ready",
        "row_count": len(rows),
        "model_count": len({row['model_key'] for row in rows}),
        "family_count": len({row['family_name'] for row in rows}),
        "best_formula": formula,
        "best_weights": weights,
        "best_correlation": best_fit["correlation"],
        "best_mae": best_fit["mae"],
        "joint_inversion_score": best_fit["fit_score"],
        **proxy_means,
        "weakest_proxy_name": weakest_proxy_name,
        "strongest_proxy_name": strongest_proxy_name,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage143: 三模型联合变量反演块",
        "",
        "## 核心结果",
        f"- 样本行数: {summary['row_count']}",
        f"- 模型数量: {summary['model_count']}",
        f"- 家族数量: {summary['family_count']}",
        f"- 最优式: `{summary['best_formula']}`",
        f"- 相关系数: {summary['best_correlation']:.4f}",
        f"- 平均绝对误差: {summary['best_mae']:.4f}",
        f"- 联合反演分数: {summary['joint_inversion_score']:.4f}",
        f"- 最强代理量: {summary['strongest_proxy_name']}",
        f"- 最弱代理量: {summary['weakest_proxy_name']}",
    ]
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force:
        cached = load_cached_summary(output_dir)
        if cached is not None:
            return cached
    rows = build_aligned_rows()
    best_fit = choose_best_fit(rows)
    summary = build_summary(rows, best_fit)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="三模型联合变量反演块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重算")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
