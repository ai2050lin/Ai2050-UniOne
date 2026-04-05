#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path

from stage515_cross_task_minimal_causal_circuit import (
    MODEL_KEYS,
    STAGE513_SUMMARY_PATH,
    build_report,
    ensure_dir,
    search_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage517_cross_task_minimal_causal_circuit_glm4_gemma4_20260404"
)
MODEL_KEYS = ["glm4", "gemma4"]


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    stage513_summary = json.loads(STAGE513_SUMMARY_PATH.read_text(encoding="utf-8"))
    stage513_rows = {row["model_key"]: row for row in stage513_summary["model_rows"]}
    started = time.time()
    model_rows = [search_model(model_key, stage513_rows[model_key]) for model_key in MODEL_KEYS]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage517_cross_task_minimal_causal_circuit_glm4_gemma4",
        "title": "GLM4 与 Gemma4 跨任务最小因果回路搜索",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "model_rows": model_rows,
        "core_answer": (
            "GLM4 和 Gemma4 的共享骨干也开始接受同一套跨任务因果检验，"
            "这样我们就能把‘共享概念骨干 + 任务适配器’从两模型工作假说，推进到四模型比较框架。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(model_rows, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
