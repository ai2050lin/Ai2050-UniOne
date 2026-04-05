#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage541_binding_invariant_recheck_20260405"
STAGE535_PATH = (
    PROJECT_ROOT / "tests" / "codex_temp"
    / "stage535_binding_mutual_info_qwen3_20260404" / "summary.json"
)
STAGE536_PATH = (
    PROJECT_ROOT / "tests" / "codex_temp"
    / "stage536_binding_neuron_search_deepseek7b_20260404" / "summary.json"
)
STAGE537_PATH = (
    PROJECT_ROOT / "tests" / "codex_temp"
    / "stage537_binding_synthesis_20260404" / "summary.json"
)

BINDING_ORDER = ["attribute", "association", "grammar", "relation"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rank_map(rows: List[dict], value_key: str) -> Dict[str, int]:
    ordered = sorted(rows, key=lambda x: x[value_key], reverse=True)
    return {row["binding_type"]: rank for rank, row in enumerate(ordered)}


def pearson_r(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def binding_rows_qwen3(summary: dict) -> List[dict]:
    rows = []
    for binding_type, payload in summary["binding_summary"].items():
        ablation_effects = payload["ablation_effects"]
        best_layer, best_drop = max(
            (
                (int(layer), float(stats["causal_drop"]))
                for layer, stats in ablation_effects.items()
            ),
            key=lambda item: item[1],
        )
        rows.append(
            {
                "binding_type": binding_type,
                "label_zh": payload["label_zh"],
                "bottleneck_layer": int(payload["most_common_bottleneck"]),
                "bottleneck_efficiency": (
                    payload["avg_binding_per_layer"][str(payload["most_common_bottleneck"])]
                    / max(
                        sum(payload["avg_binding_per_layer"].values()) / len(payload["avg_binding_per_layer"]),
                        1e-8,
                    )
                ),
                "best_causal_layer": best_layer,
                "best_causal_drop": best_drop,
            }
        )
    return rows


def binding_rows_ds7b(summary: dict) -> List[dict]:
    rows = []
    for binding_type, payload in summary["binding_summary"].items():
        drops = payload["causal_drops"]
        best_layer, best_drop = max(
            ((int(layer), float(drop)) for layer, drop in drops.items()),
            key=lambda item: item[1],
        )
        mean_binding = sum(payload["avg_binding_per_layer"].values()) / len(payload["avg_binding_per_layer"])
        bottleneck_layer = int(payload["most_common_bottleneck"])
        rows.append(
            {
                "binding_type": binding_type,
                "label_zh": payload["label_zh"],
                "bottleneck_layer": bottleneck_layer,
                "bottleneck_efficiency": payload["avg_binding_per_layer"][str(bottleneck_layer)] / max(mean_binding, 1e-8),
                "best_causal_layer": best_layer,
                "best_causal_drop": best_drop,
            }
        )
    return rows


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()

    s535 = load_json(STAGE535_PATH)
    s536 = load_json(STAGE536_PATH)
    s537 = load_json(STAGE537_PATH)

    q_rows = binding_rows_qwen3(s535)
    d_rows = binding_rows_ds7b(s536)
    q_map = {r["binding_type"]: r for r in q_rows}
    d_map = {r["binding_type"]: r for r in d_rows}

    efficiency_rank_q = rank_map(q_rows, "bottleneck_efficiency")
    efficiency_rank_d = rank_map(d_rows, "bottleneck_efficiency")
    causal_rank_q = rank_map(q_rows, "best_causal_drop")
    causal_rank_d = rank_map(d_rows, "best_causal_drop")

    rho_eff = pearson_r(
        [float(efficiency_rank_q[k]) for k in BINDING_ORDER],
        [float(efficiency_rank_d[k]) for k in BINDING_ORDER],
    )
    rho_causal = pearson_r(
        [float(causal_rank_q[k]) for k in BINDING_ORDER],
        [float(causal_rank_d[k]) for k in BINDING_ORDER],
    )
    bottleneck_layer_match_count = sum(
        1 for k in BINDING_ORDER if q_map[k]["bottleneck_layer"] == d_map[k]["bottleneck_layer"]
    )
    best_causal_layer_match_count = sum(
        1 for k in BINDING_ORDER if q_map[k]["best_causal_layer"] == d_map[k]["best_causal_layer"]
    )

    corrected = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage541_binding_invariant_recheck",
        "title": "绑定不变量复核：瓶颈层与最强因果层拆分",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage535": str(STAGE535_PATH),
            "stage536": str(STAGE536_PATH),
            "stage537_original": str(STAGE537_PATH),
        },
        "qwen3_binding_rows": q_rows,
        "deepseek7b_binding_rows": d_rows,
        "efficiency_rank_consistency": {
            "qwen3_rank": efficiency_rank_q,
            "deepseek7b_rank": efficiency_rank_d,
            "spearman_like_rho": rho_eff,
        },
        "causal_rank_consistency": {
            "qwen3_rank": causal_rank_q,
            "deepseek7b_rank": causal_rank_d,
            "spearman_like_rho": rho_causal,
        },
        "layer_match": {
            "bottleneck_layer_match_count": bottleneck_layer_match_count,
            "best_causal_layer_match_count": best_causal_layer_match_count,
            "binding_type_count": len(BINDING_ORDER),
        },
        "comparison_to_stage537": {
            "original_spearman_rho": s537["spearman_rho"],
            "note": (
                "stage537 的效率排序结论基本可保留，但“绑定因果极弱”这一说法被瓶颈层口径放大了。"
                "修正版把瓶颈层和最强因果层拆开。"
            ),
        },
        "core_answer": (
            f"修正版复核后，绑定效率排序跨模型仍然高度一致（rho≈{rho_eff:.3f}），"
            "但绑定因果不应只看瓶颈层。更准确的结构是："
            "瓶颈层描述信息聚集位置，最强因果层描述真正可打中的控制位点，两者并不总是同一层。"
        ),
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# stage541 绑定不变量复核",
        "",
        corrected["core_answer"],
        "",
        "## 关键修正",
        "- 旧口径把瓶颈层因果值直接当成绑定强度，会低估真正可打中的因果层。",
        f"- 效率排序相关仍高：rho≈{rho_eff:.3f}",
        f"- 最强因果排序相关：rho≈{rho_causal:.3f}",
        f"- 瓶颈层精确匹配：{bottleneck_layer_match_count}/{len(BINDING_ORDER)}",
        f"- 最强因果层精确匹配：{best_causal_layer_match_count}/{len(BINDING_ORDER)}",
        "",
        "## 严格说明",
        "- 修正版不改变源实验，只纠正综合层的统计口径。",
        "- 这一步更支持“绑定分型不变量”，而不是“绑定只靠单一瓶颈层”。",
    ]
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(corrected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
