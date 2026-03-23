#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage182_boundary_crack_puzzle_expansion_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE182_BOUNDARY_CRACK_PUZZLE_EXPANSION_REPORT.md"

STAGE155_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage155_apple_boundary_crack_map_20260323" / "summary.json"
STAGE166_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage166_category_fiber_map_20260323" / "summary.json"
STAGE169_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage169_category_delta_tensor_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> dict:
    s155 = load_json(STAGE155_SUMMARY_PATH)
    s166 = load_json(STAGE166_SUMMARY_PATH)
    s169 = load_json(STAGE169_SUMMARY_PATH)

    family_rows = []
    for family_name in sorted({row["family_name"] for row in s155["case_rows"]}):
        rows = [row for row in s155["case_rows"] if row["family_name"] == family_name]
        collision_rate = sum(1 for row in rows if bool(row["collision"])) / float(len(rows))
        family_rows.append(
            {
                "family_name": family_name,
                "case_count": len(rows),
                "collision_rate": collision_rate,
                "boundary_readiness": 1.0 - collision_rate,
            }
        )

    boundary_family_mean = sum(float(row["boundary_readiness"]) for row in family_rows) / float(len(family_rows))
    cross_category_separation = float(s166["cross_category_separation"])
    delta_tensor_score = float(s169["delta_tensor_score"])
    worst_family = min(family_rows, key=lambda row: float(row["boundary_readiness"]))
    best_family = max(family_rows, key=lambda row: float(row["boundary_readiness"]))
    boundary_expansion_score = clamp01(
        0.45 * boundary_family_mean
        + 0.25 * cross_category_separation
        + 0.30 * delta_tensor_score
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage182_boundary_crack_puzzle_expansion",
        "title": "边界裂缝拼图扩张",
        "status_short": "boundary_crack_puzzle_expansion_ready",
        "family_count": len(family_rows),
        "boundary_family_mean": boundary_family_mean,
        "cross_category_separation": cross_category_separation,
        "delta_tensor_score": delta_tensor_score,
        "boundary_expansion_score": boundary_expansion_score,
        "worst_family_name": str(worst_family["family_name"]),
        "best_family_name": str(best_family["family_name"]),
        "family_rows": family_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage182: 边界裂缝拼图扩张",
        "",
        "## 核心结果",
        f"- 家族数量: {summary['family_count']}",
        f"- 边界家族平均可用度: {summary['boundary_family_mean']:.4f}",
        f"- 跨类分离度: {summary['cross_category_separation']:.4f}",
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
    parser = argparse.ArgumentParser(description="边界裂缝拼图扩张")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
