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
STAGE132_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage132_unified_variable_fit_20260323"
STAGE133_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage133_complex_discourse_noun_propagation_20260323"
STAGE134_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage134_noun_verb_joint_propagation_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage135_unified_variable_inversion_20260323"

WEIGHT_STEP = 0.10


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
    stage132 = load_json(STAGE132_DIR / "summary.json")
    stage133 = load_json(STAGE133_DIR / "summary.json")
    stage134 = load_json(STAGE134_DIR / "summary.json")

    family133 = {row["family_name"]: row for row in stage133["family_rows"]}
    family134 = {row["family_name"]: row for row in stage134["family_rows"]}

    aligned_rows = []
    for family_name in sorted(set(family133) & set(family134)):
        row133 = family133[family_name]
        row134 = family134[family_name]
        a_proxy = float(row133["early_sign_consistency_rate"])
        r_proxy = clamp01((float(row133["early_remention_corr"]) + 1.0) / 2.0)
        f_proxy = clamp01((float(row133["late_remention_corr"]) + 1.0) / 2.0)
        g_proxy = clamp01((float(row134["noun_route_corr"]) + 1.0) / 2.0)
        b_proxy = clamp01(float(row134["route_band_gap"]) / 0.08)
        q_proxy = clamp01(0.5 * a_proxy + 0.5 * r_proxy)
        empirical_target = 0.55 * float(row133["discourse_family_score"]) + 0.45 * float(row134["joint_family_score"])
        aligned_rows.append(
            {
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

    return {
        "stage132_q_proxy_mean": float(stage132["q_proxy_mean"]),
        "family_rows": aligned_rows,
    }


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


def score_weight_set(family_rows: Sequence[Dict[str, object]], weights: Dict[str, float]) -> Dict[str, object]:
    predicted = []
    target = []
    enriched_rows = []
    for row in family_rows:
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
            float(row["weights"]["a"]),
            float(row["weights"]["r"]),
        ),
        reverse=True,
    )
    return candidates[0]


def build_summary(aligned: Dict[str, object], best_fit: Dict[str, object]) -> Dict[str, object]:
    family_rows = best_fit["family_rows"]
    mean_a = sum(float(row["a_proxy"]) for row in family_rows) / len(family_rows)
    mean_r = sum(float(row["r_proxy"]) for row in family_rows) / len(family_rows)
    mean_q = sum(float(row["q_proxy"]) for row in family_rows) / len(family_rows)
    mean_g = sum(float(row["g_proxy"]) for row in family_rows) / len(family_rows)
    mean_f = sum(float(row["f_proxy"]) for row in family_rows) / len(family_rows)
    mean_b = sum(float(row["b_proxy"]) for row in family_rows) / len(family_rows)

    proxy_means = {
        "a_proxy_mean": mean_a,
        "r_proxy_mean": mean_r,
        "q_proxy_mean": mean_q,
        "g_proxy_mean": mean_g,
        "f_proxy_mean": mean_f,
        "b_proxy_mean": mean_b,
    }
    weakest_proxy_name = min(proxy_means.items(), key=lambda item: item[1])[0]
    strongest_proxy_name = max(proxy_means.items(), key=lambda item: item[1])[0]
    weights = best_fit["weights"]
    formula = (
        f"inversion = {weights['a']:.2f}*a + {weights['r']:.2f}*r + {weights['q']:.2f}*q + "
        f"{weights['g']:.2f}*g + {weights['f']:.2f}*f + {weights['b']:.2f}*b"
    )
    q_reference_gap = abs(mean_q - float(aligned["stage132_q_proxy_mean"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage135_unified_variable_inversion",
        "title": "统一变量反演块",
        "status_short": "gpt2_unified_variable_inversion_ready",
        "family_count": len(family_rows),
        "best_law_name": "arqgfb_linear_grid",
        "best_formula": formula,
        "best_weights": weights,
        "best_correlation": best_fit["correlation"],
        "best_mae": best_fit["mae"],
        "unified_variable_inversion_score": best_fit["fit_score"],
        "a_proxy_mean": mean_a,
        "r_proxy_mean": mean_r,
        "q_proxy_mean": mean_q,
        "g_proxy_mean": mean_g,
        "f_proxy_mean": mean_f,
        "b_proxy_mean": mean_b,
        "stage132_q_proxy_mean": float(aligned["stage132_q_proxy_mean"]),
        "q_reference_gap": q_reference_gap,
        "weakest_proxy_name": weakest_proxy_name,
        "strongest_proxy_name": strongest_proxy_name,
        "family_rows": family_rows,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage135: 统一变量反演块",
        "",
        "## 核心结果",
        f"- 家族数量: {summary['family_count']}",
        f"- 最优律: `{summary['best_formula']}`",
        f"- 相关系数: {summary['best_correlation']:.4f}",
        f"- 平均绝对误差: {summary['best_mae']:.4f}",
        f"- 反演分数: {summary['unified_variable_inversion_score']:.4f}",
        f"- 最强代理量: {summary['strongest_proxy_name']}",
        f"- 最弱代理量: {summary['weakest_proxy_name']}",
        f"- q 参考差: {summary['q_reference_gap']:.4f}",
        "",
        "## 代理量均值",
        f"- a_proxy_mean = {summary['a_proxy_mean']:.4f}",
        f"- r_proxy_mean = {summary['r_proxy_mean']:.4f}",
        f"- q_proxy_mean = {summary['q_proxy_mean']:.4f}",
        f"- g_proxy_mean = {summary['g_proxy_mean']:.4f}",
        f"- f_proxy_mean = {summary['f_proxy_mean']:.4f}",
        f"- b_proxy_mean = {summary['b_proxy_mean']:.4f}",
        "",
        "## 各语篇家族",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "- "
            f"{row['family_name']}: "
            f"target={row['empirical_target']:.4f}, "
            f"pred={row['predicted_score']:.4f}, "
            f"a={row['a_proxy']:.4f}, "
            f"r={row['r_proxy']:.4f}, "
            f"q={row['q_proxy']:.4f}, "
            f"g={row['g_proxy']:.4f}, "
            f"f={row['f_proxy']:.4f}, "
            f"b={row['b_proxy']:.4f}"
        )
    lines.extend(
        [
            "",
            "## 理论提示",
            "- a 对应早层定锚稳定度。",
            "- r 对应语篇重提回返一致性。",
            "- q 对应由定锚与回返共同构成的上下文保持量。",
            "- g 对应名词到动词的选路耦合。",
            "- f 对应后层续接与聚合。",
            "- b 对应名词尺度带给动词路由带来的上下文偏置。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "STAGE135_UNIFIED_VARIABLE_INVERSION_REPORT.md").write_text(
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
    best_fit = choose_best_fit(aligned["family_rows"])
    summary = build_summary(aligned, best_fit)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="统一变量反演块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage135 输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    summary = run_analysis(output_dir=Path(args.output_dir), force=args.force)
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "best_formula": summary["best_formula"],
                "unified_variable_inversion_score": summary["unified_variable_inversion_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
