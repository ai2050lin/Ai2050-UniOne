#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage210_timing_phase_route_chain_20260323"

STAGE200_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage200_timing_trace_puzzle_20260323" / "summary.json"
STAGE201_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage201_phase_route_puzzle_20260323" / "summary.json"
STAGE204_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage204_phase_route_split_20260323" / "summary.json"
STAGE207_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage207_phase_timing_coupling_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s200 = load_json(STAGE200_SUMMARY_PATH)
    s201 = load_json(STAGE201_SUMMARY_PATH)
    s204 = load_json(STAGE204_SUMMARY_PATH)
    s207 = load_json(STAGE207_SUMMARY_PATH)

    chain_rows = [
        {
            "piece_name": "时序痕迹",
            "score": float(s200["timing_trace_score"]),
        },
        {
            "piece_name": "相位路径",
            "score": float(s201["phase_route_score"]),
        },
        {
            "piece_name": "路径分裂",
            "score": float(s204["phase_route_split_score"]),
        },
        {
            "piece_name": "相位-时序耦合",
            "score": float(s207["coupling_score"]),
        },
    ]
    ranked_rows = sorted(chain_rows, key=lambda row: float(row["score"]))
    chain_score = sum(float(row["score"]) for row in chain_rows) / float(len(chain_rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage210_timing_phase_route_chain",
        "title": "时序-相位-路径链",
        "status_short": "timing_phase_route_chain_ready",
        "piece_count": len(chain_rows),
        "chain_score": chain_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "dominant_band_name": str(s204["dominant_band_name"]),
        "chain_rows": chain_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage210：时序-相位-路径链",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 链总分：{summary['chain_score']:.4f}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 主导带：{summary['dominant_band_name']}",
    ]
    (output_dir / "STAGE210_TIMING_PHASE_ROUTE_CHAIN_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="时序-相位-路径链")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
