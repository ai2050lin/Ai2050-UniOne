#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage521_language_layer_band_dynamics_synthesis_20260404"
)
STAGE519_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage519_noun_attribute_bridge_layer_atlas_20260404"
    / "summary.json"
)
STAGE520_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage520_noun_attribute_bridge_causal_four_model_20260404"
    / "summary.json"
)
STAGE518_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage518_four_model_cross_task_causal_synthesis_20260404"
    / "summary.json"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage519 = load_json(STAGE519_PATH)
    stage520 = load_json(STAGE520_PATH)
    stage518 = load_json(STAGE518_PATH)
    band_rows = {row["model_key"]: row for row in stage519["model_rows"]}
    bridge_rows = {row["model_key"]: row for row in stage520["model_rows"]}
    causal_rows = {row["model_key"]: row for row in stage518["model_rows"]}

    model_rows = []
    for model_key, band in band_rows.items():
        bridge = bridge_rows[model_key]["final_result"]
        causal = causal_rows[model_key]
        model_rows.append(
            {
                "model_key": model_key,
                "noun_peak_layers": band["noun_peak_layers"],
                "attribute_peak_layers": band["attribute_peak_layers"],
                "bridge_peak_layers": band["bridge_peak_layers"],
                "noun_centroid": band["noun_centroid"],
                "attribute_centroid": band["attribute_centroid"],
                "bridge_centroid": band["bridge_centroid"],
                "bridge_causal_target_drop": bridge["target_drop"],
                "bridge_causal_control_shift": bridge["control_abs_shift"],
                "cross_task_core_subset": causal["final_subset"],
                "cross_task_core_utility": causal["utility"],
            }
        )

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage521_language_layer_band_dynamics_synthesis",
        "title": "语言层带动力学综合摘要",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "source_summaries": {
            "stage518": str(STAGE518_PATH),
            "stage519": str(STAGE519_PATH),
            "stage520": str(STAGE520_PATH),
        },
        "model_rows": model_rows,
        "core_answer": (
            "语言相关有效神经元既不是单层点状，也不是整个网络均匀铺开，而更像若干宽带层区中的功能聚集。"
            "名词、属性和桥接通常具有相近但不完全重合的层带，部分模型中的桥接还能被压缩成小型因果子集。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# stage521 语言层带动力学综合摘要", "", "## 核心结论", summary["core_answer"], ""]
    for row in model_rows:
        lines.extend(
            [
                f"## {row['model_key']}",
                f"- 名词峰值层：`{row['noun_peak_layers']}`，质心：`{row['noun_centroid']:.2f}`",
                f"- 属性峰值层：`{row['attribute_peak_layers']}`，质心：`{row['attribute_centroid']:.2f}`",
                f"- 桥接峰值层：`{row['bridge_peak_layers']}`，质心：`{row['bridge_centroid']:.2f}`",
                f"- 桥接因果目标下降：`{row['bridge_causal_target_drop']:.6f}`，控制偏移：`{row['bridge_causal_control_shift']:.6f}`",
                f"- 跨任务核心子集：`{', '.join(row['cross_task_core_subset'])}`，综合效用：`{row['cross_task_core_utility']:.6f}`",
                "",
            ]
        )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
