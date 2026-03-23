#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE124_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323"
STAGE127_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage127_noun_context_neuron_probe_20260323"
STAGE128_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage128_noun_static_route_bridge_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage129_noun_neuron_rule_compression_20260323"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def percentile_threshold(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def collect_bridge_scores(bridge_summary: Dict[str, object]) -> Dict[Tuple[int, int], float]:
    best_scores: Dict[Tuple[int, int], float] = {}
    for row in bridge_summary["strongest_bridge_pairs"]:
        key = (int(row["noun_layer_index"]), int(row["noun_neuron_index"]))
        best_scores[key] = max(best_scores.get(key, -1.0), float(row["cosine_alignment"]))
    return best_scores


def build_feature_rows(
    static_summary: Dict[str, object],
    context_summary: Dict[str, object],
    bridge_summary: Dict[str, object],
) -> List[Dict[str, object]]:
    static_rows = {
        (int(row["layer_index"]), int(row["neuron_index"])): row
        for row in static_summary["top_general_neurons"]
        if int(row["layer_index"]) == 11
    }
    context_rows = {
        (int(row["layer_index"]), int(row["neuron_index"])): row
        for row in context_summary["top_general_neurons"]
        if int(row["layer_index"]) == 11
    }
    bridge_scores = collect_bridge_scores(bridge_summary)
    keys = sorted(set(static_rows) | set(context_rows) | set(bridge_scores))

    feature_rows = []
    for key in keys:
        static_row = static_rows.get(key)
        context_row = context_rows.get(key)
        static_score = float(static_row["general_rule_score"]) if static_row else 0.0
        context_score = float(context_row["general_rule_score"]) if context_row else 0.0
        static_support = float(static_row["group_support_ratio"]) if static_row else 0.0
        context_support = float(context_row["group_support_ratio"]) if context_row else 0.0
        bridge_score = float(bridge_scores.get(key, 0.0))
        support_mean = 0.5 * (static_support + context_support)
        feature_rows.append(
            {
                "layer_index": int(key[0]),
                "neuron_index": int(key[1]),
                "static_generality": static_score,
                "context_persistence": context_score,
                "route_bridge": bridge_score,
                "support_mean": support_mean,
            }
        )
    return feature_rows


def add_empirical_targets(feature_rows: List[Dict[str, object]]) -> None:
    static_values = [row["static_generality"] for row in feature_rows]
    context_values = [row["context_persistence"] for row in feature_rows]
    bridge_values = [row["route_bridge"] for row in feature_rows]
    support_values = [row["support_mean"] for row in feature_rows]

    static_thr = percentile_threshold(static_values, 0.75)
    context_thr = percentile_threshold(context_values, 0.75)
    bridge_thr = percentile_threshold(bridge_values, 0.60)
    support_thr = percentile_threshold(support_values, 0.75)

    for row in feature_rows:
        hit_count = 0
        hit_count += int(row["static_generality"] >= static_thr)
        hit_count += int(row["context_persistence"] >= context_thr)
        hit_count += int(row["route_bridge"] >= bridge_thr)
        hit_count += int(row["support_mean"] >= support_thr)
        row["empirical_target"] = hit_count / 4.0


def law_scores(row: Dict[str, object]) -> Dict[str, float]:
    s = max(0.0, row["static_generality"])
    c = max(0.0, row["context_persistence"])
    b = max(0.0, row["route_bridge"])
    p = max(0.0, row["support_mean"])
    return {
        "linear_mean": 0.40 * s + 0.40 * c + 0.20 * b,
        "geometric_bridge": math.sqrt(max(1e-8, s * c)) * (0.5 + 0.5 * b),
        "support_gated_min": min(s, c) * (0.5 + 0.5 * p),
        "bridge_dominant": b * (0.5 + 0.5 * s) * (0.5 + 0.5 * c),
    }


def correlation(xs: List[float], ys: List[float]) -> float:
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


def evaluate_laws(feature_rows: List[Dict[str, object]]) -> Dict[str, object]:
    law_names = list(law_scores(feature_rows[0]).keys())
    empirical_targets = [row["empirical_target"] for row in feature_rows]
    law_rows = []
    scored_rows = {name: [] for name in law_names}

    for row in feature_rows:
        scores = law_scores(row)
        row["candidate_laws"] = scores
        for name, value in scores.items():
            scored_rows[name].append((value, row))

    for name in law_names:
        values = [value for value, _row in scored_rows[name]]
        corr = correlation(values, empirical_targets)
        top_k = max(1, len(values) // 3)
        top_rows = sorted(scored_rows[name], key=lambda item: item[0], reverse=True)[:top_k]
        top_precision = sum(row["empirical_target"] >= 0.75 for _value, row in top_rows) / top_k
        total_score = 0.70 * clamp01((corr + 1.0) / 2.0) + 0.30 * top_precision
        law_rows.append(
            {
                "law_name": name,
                "correlation_to_empirical_target": corr,
                "top_precision": top_precision,
                "law_score": total_score,
            }
        )

    law_rows.sort(key=lambda row: row["law_score"], reverse=True)
    best_law = law_rows[0]
    return {
        "law_rows": law_rows,
        "best_law": best_law,
        "feature_rows": feature_rows,
    }


def best_law_formula(name: str) -> str:
    formulas = {
        "linear_mean": "noun_core = 0.40*s + 0.40*c + 0.20*b",
        "geometric_bridge": "noun_core = sqrt(s*c) * (0.5 + 0.5*b)",
        "support_gated_min": "noun_core = min(s,c) * (0.5 + 0.5*p)",
        "bridge_dominant": "noun_core = b * (0.5 + 0.5*s) * (0.5 + 0.5*c)",
    }
    return formulas[name]


def build_summary(evaluation: Dict[str, object]) -> Dict[str, object]:
    best_law = evaluation["best_law"]
    feature_rows = evaluation["feature_rows"]
    top_neurons = sorted(
        feature_rows,
        key=lambda row: row["candidate_laws"][best_law["law_name"]],
        reverse=True,
    )[:12]
    compression_score = (
        0.45 * best_law["law_score"]
        + 0.35 * clamp01(best_law["correlation_to_empirical_target"])
        + 0.20 * best_law["top_precision"]
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage129_noun_neuron_rule_compression",
        "title": "Noun 神经元规则压缩",
        "status_short": "gpt2_noun_neuron_rule_compression_ready",
        "feature_row_count": len(feature_rows),
        "best_law_name": best_law["law_name"],
        "best_law_formula": best_law_formula(best_law["law_name"]),
        "best_law_score": best_law["law_score"],
        "best_law_correlation": best_law["correlation_to_empirical_target"],
        "best_law_top_precision": best_law["top_precision"],
        "noun_neuron_rule_compression_score": float(compression_score),
        "law_rows": evaluation["law_rows"],
        "top_compressed_neurons": top_neurons,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage129: Noun 神经元规则压缩",
        "",
        "## 核心结果",
        f"- 参与压缩的 L11 神经元数: {summary['feature_row_count']}",
        f"- 最优候选律: {summary['best_law_name']}",
        f"- 候选律公式: {summary['best_law_formula']}",
        f"- 候选律得分: {summary['best_law_score']:.4f}",
        f"- 候选律相关性: {summary['best_law_correlation']:.4f}",
        f"- 候选律头部精度: {summary['best_law_top_precision']:.4f}",
        f"- 神经元规则压缩分数: {summary['noun_neuron_rule_compression_score']:.4f}",
        "",
        "## 候选律排行",
    ]
    for row in summary["law_rows"]:
        lines.append(
            "- "
            f"{row['law_name']}: score={row['law_score']:.4f}, "
            f"corr={row['correlation_to_empirical_target']:.4f}, "
            f"top_precision={row['top_precision']:.4f}"
        )
    lines.extend(["", "## 压缩后的代表神经元"])
    for row in summary["top_compressed_neurons"][:10]:
        lines.append(
            "- "
            f"L{row['layer_index']} N{row['neuron_index']}: "
            f"s={row['static_generality']:.4f}, "
            f"c={row['context_persistence']:.4f}, "
            f"b={row['route_bridge']:.4f}, "
            f"p={row['support_mean']:.4f}"
        )
    lines.extend(
        [
            "",
            "## 理论提示",
            "- 这里的 s 表示静态通用性，c 表示上下文保持性，b 表示选路桥接性，p 表示覆盖稳定性。",
            "- 如果少数候选律就能较好压缩高覆盖神经元，说明名词编码有机会进一步逼近统一变量形式。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "STAGE129_NOUN_NEURON_RULE_COMPRESSION_REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")
    (output_dir / "law_rows.json").write_text(json.dumps(summary["law_rows"], ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "top_compressed_neurons.json").write_text(
        json.dumps(summary["top_compressed_neurons"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR) -> Dict[str, object]:
    static_summary = load_json(STAGE124_DIR / "summary.json")
    context_summary = load_json(STAGE127_DIR / "summary.json")
    bridge_summary = load_json(STAGE128_DIR / "summary.json")
    feature_rows = build_feature_rows(static_summary, context_summary, bridge_summary)
    add_empirical_targets(feature_rows)
    evaluation = evaluate_laws(feature_rows)
    summary = build_summary(evaluation)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Noun 神经元规则压缩")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage129 输出目录")
    args = parser.parse_args()

    summary = run_analysis(output_dir=Path(args.output_dir))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "best_law_name": summary["best_law_name"],
                "noun_neuron_rule_compression_score": summary["noun_neuron_rule_compression_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
