#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage213_timing_phase_route_bridge_20260323"

STAGE210_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage210_timing_phase_route_chain_20260323" / "summary.json"
STAGE207_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage207_phase_timing_coupling_20260323" / "summary.json"
STAGE204_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage204_phase_route_split_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s210 = load_json(STAGE210_SUMMARY_PATH)
    s207 = load_json(STAGE207_SUMMARY_PATH)
    s204 = load_json(STAGE204_SUMMARY_PATH)

    rows = [
        {"piece_name": "时序痕迹桥", "score": next(float(r["score"]) for r in s210["chain_rows"] if str(r["piece_name"]) == "时序痕迹")},
        {"piece_name": "相位路径桥", "score": next(float(r["score"]) for r in s210["chain_rows"] if str(r["piece_name"]) == "相位路径")},
        {"piece_name": "路径分裂桥", "score": next(float(r["score"]) for r in s207["relation_rows"] if str(r["piece_name"]) == "路径分裂")},
        {"piece_name": "相位-时序耦合桥", "score": float(s207["coupling_score"])},
    ]
    ranked_rows = sorted(rows, key=lambda item: float(item["score"]))
    bridge_score = (
        float(rows[0]["score"]) * 0.25
        + float(rows[1]["score"]) * 0.20
        + float(rows[2]["score"]) * 0.30
        + float(rows[3]["score"]) * 0.25
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage213_timing_phase_route_bridge",
        "title": "时序-相位-路径桥",
        "status_short": "timing_phase_route_bridge_ready",
        "piece_count": len(rows),
        "bridge_score": bridge_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "dominant_band_name": str(s204["dominant_band_name"]),
        "bridge_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage213：时序-相位-路径桥",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 桥总分：{summary['bridge_score']:.4f}",
        f"- 最弱桥段：{summary['weakest_piece_name']}",
        f"- 最强桥段：{summary['strongest_piece_name']}",
        f"- 主导带：{summary['dominant_band_name']}",
    ]
    (output_dir / "STAGE213_TIMING_PHASE_ROUTE_BRIDGE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="时序-相位-路径桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
