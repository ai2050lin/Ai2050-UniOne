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
    / "stage518_four_model_cross_task_causal_synthesis_20260404"
)
STAGE515_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage515_cross_task_minimal_causal_circuit_20260404"
    / "summary.json"
)
STAGE517_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage517_cross_task_minimal_causal_circuit_glm4_gemma4_20260404"
    / "summary.json"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["model_rows"]


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    model_rows = load_rows(STAGE515_PATH) + load_rows(STAGE517_PATH)
    compact_rows = []
    for row in model_rows:
        final_result = row["final_result"]
        compact_rows.append(
            {
                "model_key": row["model_key"],
                "model_name": row["model_name"],
                "final_subset": row["final_subset"],
                "subset_size": len(row["final_subset"]),
                "baseline_target_mean_correct_prob": row["baseline_target"]["mean_correct_prob"],
                "baseline_control_mean_correct_prob": row["baseline_control"]["mean_correct_prob"],
                "target_drop": final_result["target_drop"],
                "control_abs_shift": final_result["control_abs_shift"],
                "utility": final_result["utility"],
            }
        )
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage518_four_model_cross_task_causal_synthesis",
        "title": "四模型跨任务因果骨干综合摘要",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "source_summaries": {
            "stage515": str(STAGE515_PATH),
            "stage517": str(STAGE517_PATH),
        },
        "model_rows": compact_rows,
        "core_answer": (
            "共享概念骨干已经不只是结构复用假说，而是开始进入四模型最小因果比较框架。"
            "不同模型都能找到一小组跨任务共享神经元，但骨干厚度和专化程度明显不同。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "# stage518 四模型跨任务因果骨干综合摘要",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
    ]
    for row in compact_rows:
        report_lines.extend(
            [
                f"## {row['model_name']}",
                f"- 最终子集：`{', '.join(row['final_subset']) if row['final_subset'] else '空'}`",
                f"- 子集大小：`{row['subset_size']}`",
                f"- 目标下降：`{row['target_drop']:.6f}`",
                f"- 控制偏移：`{row['control_abs_shift']:.6f}`",
                f"- 综合效用：`{row['utility']:.6f}`",
                "",
            ]
        )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
