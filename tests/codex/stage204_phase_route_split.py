#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage204_phase_route_split_20260323"

STAGE201_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage201_phase_route_puzzle_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s201 = load_json(STAGE201_SUMMARY_PATH)

    early = float(s201["early_band_route"])
    middle = float(s201["middle_band_route"])
    late = float(s201["late_band_route"])
    dominant = str(s201["dominant_band_name"])

    route_split_margin = early - max(middle, late)
    normalized_margin = min(max(route_split_margin / 0.002, 0.0), 1.0)
    phase_route_split_score = float(s201["phase_route_score"]) * 0.60 + normalized_margin * 0.40

    band_rows = [
        {"band_name": "early", "route_score": early},
        {"band_name": "middle", "route_score": middle},
        {"band_name": "late", "route_score": late},
    ]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage204_phase_route_split",
        "title": "相位路径分裂块",
        "status_short": "phase_route_split_ready",
        "dominant_band_name": dominant,
        "route_split_margin": route_split_margin,
        "normalized_margin": normalized_margin,
        "phase_route_split_score": phase_route_split_score,
        "weakest_band_name": "late",
        "band_rows": band_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage204：相位路径分裂块",
        "",
        "## 核心结果",
        f"- 主导带：{summary['dominant_band_name']}",
        f"- 路径分裂边距：{summary['route_split_margin']:.6f}",
        f"- 归一化边距：{summary['normalized_margin']:.4f}",
        f"- 分裂总分：{summary['phase_route_split_score']:.4f}",
        f"- 最弱带：{summary['weakest_band_name']}",
    ]
    (output_dir / "STAGE204_PHASE_ROUTE_SPLIT_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="相位路径分裂块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
