#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage540_invariant_recheck_20260405"
STAGE532_PATH = (
    PROJECT_ROOT / "tests" / "codex_temp"
    / "stage532_multinoun_causal_qwen3_20260404" / "summary.json"
)
STAGE533_PATH = (
    PROJECT_ROOT / "tests" / "codex_temp"
    / "stage533_multinoun_causal_deepseek7b_20260404" / "summary.json"
)
STAGE534_PATH = (
    PROJECT_ROOT / "tests" / "codex_temp"
    / "stage534_causal_law_synthesis_20260404" / "summary.json"
)

FAMILY_LABELS = {
    "fruit": "水果",
    "animal": "动物",
    "tool": "工具",
    "organization": "组织",
    "celestial": "天体",
    "abstract": "抽象",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pearson_r(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def spearman_rho(order_a: Dict[str, int], order_b: Dict[str, int]) -> float:
    keys = [k for k in order_a if k in order_b]
    if len(keys) < 2:
        return 0.0
    xs = [float(order_a[k]) for k in keys]
    ys = [float(order_b[k]) for k in keys]
    return pearson_r(xs, ys)


def summarize_distance_block(summary: dict) -> dict:
    calibration = summary["calibration_data"]
    dist = summary["distance_matrix"]
    same_pairs = [row["word_pair"] for row in calibration if row["same_family"] == 1]
    diff_pairs = [row["word_pair"] for row in calibration if row["same_family"] == 0]

    intra = [float(dist[p]) for p in same_pairs if p in dist]
    inter = [float(dist[p]) for p in diff_pairs if p in dist]
    avg_intra = sum(intra) / max(len(intra), 1)
    avg_inter = sum(inter) / max(len(inter), 1)
    ratio = avg_intra / max(avg_inter, 1e-8)
    return {
        "pair_count": len(dist),
        "same_family_pair_count": len(intra),
        "cross_family_pair_count": len(inter),
        "avg_intra": avg_intra,
        "avg_inter": avg_inter,
        "intra_inter_ratio": ratio,
    }


def family_best_rows(summary: dict) -> List[dict]:
    rows = []
    for family_key in FAMILY_LABELS:
        row = summary["family_sensitivity"][family_key]
        rows.append(
            {
                "family": family_key,
                "label_zh": FAMILY_LABELS[family_key],
                "best_layer": row["best_layer"],
                "best_component": row["best_component"],
                "best_total_effect": row["best_total_effect"],
            }
        )
    return rows


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()

    s532 = load_json(STAGE532_PATH)
    s533 = load_json(STAGE533_PATH)
    s534 = load_json(STAGE534_PATH)

    dist_q = summarize_distance_block(s532)
    dist_d = summarize_distance_block(s533)
    common_pairs = sorted(set(s532["distance_matrix"]) & set(s533["distance_matrix"]))
    topo_r = pearson_r(
        [float(s532["distance_matrix"][p]) for p in common_pairs],
        [float(s533["distance_matrix"][p]) for p in common_pairs],
    )

    q_rows = family_best_rows(s532)
    d_rows = family_best_rows(s533)
    q_map = {r["family"]: r for r in q_rows}
    d_map = {r["family"]: r for r in d_rows}
    exact_match_count = sum(
        1
        for family in FAMILY_LABELS
        if q_map[family]["best_layer"] == d_map[family]["best_layer"]
        and q_map[family]["best_component"] == d_map[family]["best_component"]
    )

    effect_rank_q = {
        row["family"]: rank
        for rank, row in enumerate(sorted(q_rows, key=lambda x: x["best_total_effect"], reverse=True))
    }
    effect_rank_d = {
        row["family"]: rank
        for rank, row in enumerate(sorted(d_rows, key=lambda x: x["best_total_effect"], reverse=True))
    }
    family_effect_rank_rho = spearman_rho(effect_rank_q, effect_rank_d)

    corrected = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage540_invariant_recheck",
        "title": "不变量复核：修正版拓扑与层位分析",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage532": str(STAGE532_PATH),
            "stage533": str(STAGE533_PATH),
            "stage534_original": str(STAGE534_PATH),
        },
        "corrected_distance_metrics": {
            "qwen3": dist_q,
            "deepseek7b": dist_d,
            "common_pairs": len(common_pairs),
            "topology_pearson_r": topo_r,
        },
        "layer_non_invariance": {
            "exact_match_count": exact_match_count,
            "family_count": len(FAMILY_LABELS),
            "match_rate": exact_match_count / len(FAMILY_LABELS),
            "qwen3_family_best": q_rows,
            "deepseek7b_family_best": d_rows,
        },
        "family_effect_rank_consistency": {
            "qwen3_rank": effect_rank_q,
            "deepseek7b_rank": effect_rank_d,
            "spearman_rho": family_effect_rank_rho,
        },
        "comparison_to_stage534": {
            "original_distance_correlation": s534["distance_correlation"],
            "note": (
                "stage534 原脚本把 DeepSeek7B 的家族内平均距离误用成了 Qwen3 的矩阵。"
                "修正版保留拓扑相关结论，但纠正家族内聚精确数值。"
            ),
        },
        "core_answer": (
            "修正版复核后，最稳的不变量仍然是编码拓扑排序而不是具体层号。"
            f"两模型距离矩阵 Pearson r={topo_r:.6f}，说明家族级几何关系高度一致；"
            f"但最敏感层位完全匹配只有 {exact_match_count}/{len(FAMILY_LABELS)}，"
            "说明抽象分工可对齐，具体层号不可直接对齐。"
        ),
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# stage540 不变量复核",
        "",
        corrected["core_answer"],
        "",
        "## 修正后的家族内聚性",
        f"- Qwen3: 家族内 {dist_q['avg_intra']:.6f}, 家族间 {dist_q['avg_inter']:.6f}, 比值 {dist_q['intra_inter_ratio']:.6f}",
        f"- DeepSeek7B: 家族内 {dist_d['avg_intra']:.6f}, 家族间 {dist_d['avg_inter']:.6f}, 比值 {dist_d['intra_inter_ratio']:.6f}",
        "",
        "## 拓扑与层位",
        f"- 编码拓扑 Pearson r: {topo_r:.6f}",
        f"- 精确层位匹配: {exact_match_count}/{len(FAMILY_LABELS)}",
        f"- 家族效果排序 Spearman rho: {family_effect_rank_rho:.6f}",
        "",
        "## 严格说明",
        "- 这个修正版只纠正统计口径，不改动原始实验数据。",
        "- 结果支持“拓扑不变量”，但不支持把原来的家族内聚精确区间直接当最终定律。",
    ]
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(corrected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
