#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage180_cross_object_puzzle_expansion_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE180_CROSS_OBJECT_PUZZLE_EXPANSION_REPORT.md"

STAGE154_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage154_apple_fruit_shared_core_20260323" / "summary.json"
STAGE166_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage166_category_fiber_map_20260323" / "summary.json"
STAGE169_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage169_category_delta_tensor_20260323" / "summary.json"
STAGE171_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage171_delta_route_recovery_bridge_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_readiness(score: float) -> str:
    if score >= 0.75:
        return "可扩张"
    if score >= 0.55:
        return "可推进"
    return "需补洞"


def build_summary() -> dict:
    s154 = load_json(STAGE154_SUMMARY_PATH)
    s166 = load_json(STAGE166_SUMMARY_PATH)
    s169 = load_json(STAGE169_SUMMARY_PATH)
    s171 = load_json(STAGE171_SUMMARY_PATH)

    object_rows = [
        {
            "domain_name": "fruit",
            "coverage_signal": float(s154["shared_core_score"]),
            "readiness": classify_readiness(float(s154["shared_core_score"])),
        },
        {
            "domain_name": "animal",
            "coverage_signal": float(s166["category_fiber_score"]),
            "readiness": classify_readiness(float(s166["category_fiber_score"])),
        },
        {
            "domain_name": "tool",
            "coverage_signal": float(s171["bridge_score"]),
            "readiness": classify_readiness(float(s171["bridge_score"])),
        },
        {
            "domain_name": "vehicle",
            "coverage_signal": float(s169["delta_tensor_score"]),
            "readiness": classify_readiness(float(s169["delta_tensor_score"])),
        },
        {
            "domain_name": "abstract",
            "coverage_signal": float(s166["cross_category_separation"]),
            "readiness": classify_readiness(float(s166["cross_category_separation"])),
        },
    ]
    ranked_rows = sorted(object_rows, key=lambda row: float(row["coverage_signal"]))
    expandable_domain_count = sum(1 for row in object_rows if str(row["readiness"]) == "可扩张")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage180_cross_object_puzzle_expansion",
        "title": "跨对象拼图扩张",
        "status_short": "cross_object_puzzle_expansion_ready",
        "domain_count": len(object_rows),
        "expandable_domain_count": expandable_domain_count,
        "strongest_domain_name": str(ranked_rows[-1]["domain_name"]),
        "weakest_domain_name": str(ranked_rows[0]["domain_name"]),
        "object_rows": object_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage180: 跨对象拼图扩张",
        "",
        "## 核心结果",
        f"- 对象域数量: {summary['domain_count']}",
        f"- 可扩张对象域数量: {summary['expandable_domain_count']}",
        f"- 当前最强对象域: {summary['strongest_domain_name']}",
        f"- 当前最弱对象域: {summary['weakest_domain_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨对象拼图扩张")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
