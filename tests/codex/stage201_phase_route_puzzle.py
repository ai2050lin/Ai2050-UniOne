#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage201_phase_route_puzzle_20260323"

STAGE123_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage123_route_shift_layer_localization_20260323" / "summary.json"
STAGE193_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage193_cross_model_invariant_3d_blocks_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s123 = load_json(STAGE123_SUMMARY_PATH)
    s193 = load_json(STAGE193_SUMMARY_PATH)

    early_band = next(float(row["verb_route_advantage_mean"]) for row in s123["band_rows"] if str(row["band_name"]) == "early")
    middle_band = next(float(row["verb_route_advantage_mean"]) for row in s123["band_rows"] if str(row["band_name"]) == "middle")
    late_band = next(float(row["verb_route_advantage_mean"]) for row in s123["band_rows"] if str(row["band_name"]) == "late")
    adverb_route_score = next(float(row["score"]) for row in s193["block_rows"] if str(row["block_name"]) == "副词动态选路")
    phase_route_score = s123["route_shift_layer_localization_score"] * 0.6 + adverb_route_score * 0.4
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage201_phase_route_puzzle",
        "title": "相位路径拼图",
        "status_short": "phase_route_puzzle_ready",
        "early_band_route": early_band,
        "middle_band_route": middle_band,
        "late_band_route": late_band,
        "dominant_band_name": str(s123["best_band_name"]),
        "cross_model_route_score": adverb_route_score,
        "phase_route_score": phase_route_score,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage201：相位路径拼图",
        "",
        "## 核心结果",
        f"- 主导带：{summary['dominant_band_name']}",
        f"- 早层路径值：{summary['early_band_route']:.6f}",
        f"- 中层路径值：{summary['middle_band_route']:.6f}",
        f"- 晚层路径值：{summary['late_band_route']:.6f}",
        f"- 相位路径总分：{summary['phase_route_score']:.4f}",
    ]
    (output_dir / "STAGE201_PHASE_ROUTE_PUZZLE_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="相位路径拼图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
