#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage185_boundary_crack_system_expansion_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE185_BOUNDARY_CRACK_SYSTEM_EXPANSION_REPORT.md"

STAGE182_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage182_boundary_crack_puzzle_expansion_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_gap(readiness: float) -> str:
    if readiness >= 0.75:
        return "稳定边界"
    if readiness >= 0.45:
        return "过渡边界"
    return "裂缝边界"


def build_summary() -> dict:
    s182 = load_json(STAGE182_SUMMARY_PATH)
    family_rows = []
    for row in s182["family_rows"]:
        readiness = float(row["boundary_readiness"])
        family_rows.append(
            {
                "family_name": str(row["family_name"]),
                "case_count": int(row["case_count"]),
                "boundary_readiness": readiness,
                "boundary_gap_level": classify_gap(readiness),
            }
        )
    ranked_rows = sorted(family_rows, key=lambda row: float(row["boundary_readiness"]))
    crack_family_count = sum(1 for row in family_rows if str(row["boundary_gap_level"]) == "裂缝边界")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage185_boundary_crack_system_expansion",
        "title": "边界裂缝系统扩张",
        "status_short": "boundary_crack_system_expansion_ready",
        "family_count": len(family_rows),
        "crack_family_count": crack_family_count,
        "worst_family_name": str(ranked_rows[0]["family_name"]),
        "best_family_name": str(ranked_rows[-1]["family_name"]),
        "family_rows": family_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage185: 边界裂缝系统扩张",
        "",
        "## 核心结果",
        f"- 家族数量: {summary['family_count']}",
        f"- 裂缝边界家族数量: {summary['crack_family_count']}",
        f"- 最弱边界家族: {summary['worst_family_name']}",
        f"- 最强边界家族: {summary['best_family_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="边界裂缝系统扩张")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
