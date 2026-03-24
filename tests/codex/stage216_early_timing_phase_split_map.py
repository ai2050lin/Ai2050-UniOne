#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage216_early_timing_phase_split_map_20260324"

STAGE213_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage213_timing_phase_route_bridge_20260323" / "summary.json"
STAGE210_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage210_timing_phase_route_chain_20260323" / "summary.json"
STAGE204_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage204_phase_route_split_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s213 = load_json(STAGE213_SUMMARY_PATH)
    s210 = load_json(STAGE210_SUMMARY_PATH)
    s204 = load_json(STAGE204_SUMMARY_PATH)

    timing_score = next(float(row["score"]) for row in s210["chain_rows"] if str(row["piece_name"]) == "时序痕迹")
    phase_score = next(float(row["score"]) for row in s210["chain_rows"] if str(row["piece_name"]) == "相位路径")
    split_score = float(s204["phase_route_split_score"])
    bridge_score = float(s213["bridge_score"])

    split_rows = [
        {"piece_name": "时序触发", "score": timing_score},
        {"piece_name": "相位触发", "score": phase_score},
        {"piece_name": "早层路径分流", "score": split_score},
        {"piece_name": "时序-相位桥", "score": bridge_score},
    ]
    ranked_rows = sorted(split_rows, key=lambda row: float(row["score"]))
    split_map_score = (
        timing_score * 0.20
        + phase_score * 0.25
        + split_score * 0.35
        + bridge_score * 0.20
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage216_early_timing_phase_split_map",
        "title": "早层时序-相位分流图",
        "status_short": "early_timing_phase_split_map_ready",
        "piece_count": len(split_rows),
        "dominant_band_name": str(s213["dominant_band_name"]),
        "split_map_score": split_map_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "split_rows": split_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage216：早层时序-相位分流图",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 主导带：{summary['dominant_band_name']}",
        f"- 分流图总分：{summary['split_map_score']:.4f}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强部件：{summary['strongest_piece_name']}",
    ]
    (output_dir / "STAGE216_EARLY_TIMING_PHASE_SPLIT_MAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="早层时序-相位分流图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
