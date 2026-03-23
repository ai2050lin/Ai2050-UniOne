#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage172_provenance_trace_probe_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE172_PROVENANCE_TRACE_PROBE_REPORT.md"

STAGE158_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage158_apple_result_binding_probe_20260323" / "summary.json"
STAGE160_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json"
STAGE163_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage163_result_binding_failure_atlas_20260323" / "summary.json"
STAGE165_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage165_apple_to_language_kernel_bridge_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> dict:
    s158 = load_json(STAGE158_SUMMARY_PATH)
    s160 = load_json(STAGE160_SUMMARY_PATH)
    s163 = load_json(STAGE163_SUMMARY_PATH)
    s165 = load_json(STAGE165_SUMMARY_PATH)

    case_count = int(s163["case_count"])
    stable_ratio = float(s163["stable_binding_count"]) / float(case_count)
    fragile_ratio = float(s163["fragile_binding_count"]) / float(case_count)
    hard_failure_ratio = float(s163["hard_failure_count"]) / float(case_count)
    raw_trace = float(s158["positive_binding_rate"])
    repair_trace = float(s160["positive_repair_rate"])
    recurrence_trace = next(
        float(row["score"]) for row in s165["proxy_rows"] if row["proxy_name"] == "r_proxy"
    )
    retained_trace = stable_ratio + 0.5 * fragile_ratio

    component_rows = [
        {"component_name": "raw_trace", "score": raw_trace},
        {"component_name": "retained_trace", "score": retained_trace},
        {"component_name": "repair_trace", "score": repair_trace},
        {"component_name": "recurrence_trace", "score": recurrence_trace},
        {"component_name": "anti_hard_failure", "score": 1.0 - hard_failure_ratio},
    ]
    provenance_trace_score = clamp01(
        0.25 * raw_trace
        + 0.15 * retained_trace
        + 0.25 * repair_trace
        + 0.20 * recurrence_trace
        + 0.15 * (1.0 - hard_failure_ratio)
    )
    ranked_rows = sorted(component_rows, key=lambda row: float(row["score"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage172_provenance_trace_probe",
        "title": "来源痕迹探针",
        "status_short": "provenance_trace_probe_ready",
        "case_count": case_count,
        "provenance_trace_score": provenance_trace_score,
        "hard_failure_ratio": hard_failure_ratio,
        "stable_ratio": stable_ratio,
        "fragile_ratio": fragile_ratio,
        "strongest_component_name": str(ranked_rows[-1]["component_name"]),
        "weakest_component_name": str(ranked_rows[0]["component_name"]),
        "component_rows": component_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage172: 来源痕迹探针",
        "",
        "## 核心结果",
        f"- 来源痕迹分数: {summary['provenance_trace_score']:.4f}",
        f"- 最强组件: {summary['strongest_component_name']}",
        f"- 最弱组件: {summary['weakest_component_name']}",
        f"- 强失绑占比: {summary['hard_failure_ratio']:.4f}",
        f"- 稳定绑定占比: {summary['stable_ratio']:.4f}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="来源痕迹探针")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
