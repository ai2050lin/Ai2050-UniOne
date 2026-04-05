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
    / "stage523_noun_predictive_encoding_synthesis_20260404"
)
STAGE522_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage522_noun_panorama_hierarchy_scan_20260404"
    / "summary.json"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage522 = json.loads(STAGE522_PATH.read_text(encoding="utf-8"))
    model_rows = []
    for row in stage522["model_rows"]:
        s = row["summary"]
        apple = s["apple_breakdown"]
        model_rows.append(
            {
                "model_key": row["model_key"],
                "family_prediction_accuracy": s["family_prediction_accuracy"],
                "family_core_margin_win_rate": s["family_core_margin_win_rate"],
                "global_core_count": s["global_core_count"],
                "apple_global_core_shared_count": apple["apple_global_core_shared_count"],
                "apple_fruit_core_shared_count": apple["apple_fruit_core_shared_count"],
                "apple_animal_core_shared_count": apple["apple_animal_core_shared_count"],
                "apple_celestial_core_shared_count": apple["apple_celestial_core_shared_count"],
                "apple_abstract_core_shared_count": apple["apple_abstract_core_shared_count"],
                "apple_unique_vs_fruit_count": apple["apple_unique_vs_fruit_count"],
            }
        )
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage523_noun_predictive_encoding_synthesis",
        "title": "名词编码预测规律综合摘要",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "source_summary": str(STAGE522_PATH),
        "model_rows": model_rows,
        "core_answer": (
            "大量名词的编码目前最像三层结构：全局共享名词骨干、家族共享骨干、名词独有残差。"
            "这条规律已经足以支持家族级编码预测，但还不足以精确预测单个新名词的全部独有细节。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# stage523 名词编码预测规律综合摘要", "", "## 核心结论", summary["core_answer"], ""]
    for row in model_rows:
        lines.extend(
            [
                f"## {row['model_key']}",
                f"- 家族预测准确率：`{row['family_prediction_accuracy']:.4f}`",
                f"- 家族核心胜率：`{row['family_core_margin_win_rate']:.4f}`",
                f"- 苹果与水果共享：`{row['apple_fruit_core_shared_count']}`",
                f"- 苹果与动物共享：`{row['apple_animal_core_shared_count']}`",
                f"- 苹果与天体共享：`{row['apple_celestial_core_shared_count']}`",
                f"- 苹果与抽象共享：`{row['apple_abstract_core_shared_count']}`",
                f"- 苹果相对水果独有：`{row['apple_unique_vs_fruit_count']}`",
                "",
            ]
        )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
