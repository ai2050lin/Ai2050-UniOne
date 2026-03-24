#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage211_source_carried_closure_bridge_20260323"

STAGE206_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage206_retained_trace_transfer_20260323" / "summary.json"
STAGE208_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage208_forward_carried_provenance_20260323" / "summary.json"
STAGE205_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage205_reentry_closure_bridge_20260323" / "summary.json"
STAGE202_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage202_reentry_closure_puzzle_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s206 = load_json(STAGE206_SUMMARY_PATH)
    s208 = load_json(STAGE208_SUMMARY_PATH)
    s205 = load_json(STAGE205_SUMMARY_PATH)
    s202 = load_json(STAGE202_SUMMARY_PATH)

    source_carried_score = float(s208["forward_carried_score"])
    closure_bridge_score = float(s205["bridge_score"])
    reentry_score = float(s202["reentry_closure_score"])
    transfer_score = float(s206["transfer_score"])

    bridge_rows = [
        {"piece_name": "天然痕迹传递", "score": transfer_score},
        {"piece_name": "前向携带来源", "score": source_carried_score},
        {"piece_name": "重入闭合桥", "score": closure_bridge_score},
        {"piece_name": "重入闭合拼图", "score": reentry_score},
    ]
    ranked_rows = sorted(bridge_rows, key=lambda row: float(row["score"]))
    closure_bridge_score_final = (
        transfer_score * 0.25
        + source_carried_score * 0.35
        + closure_bridge_score * 0.20
        + reentry_score * 0.20
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage211_source_carried_closure_bridge",
        "title": "来源携带闭合桥",
        "status_short": "source_carried_closure_bridge_ready",
        "piece_count": len(bridge_rows),
        "closure_bridge_score": closure_bridge_score_final,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "天然痕迹传递不足",
        "bridge_rows": bridge_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage211：来源携带闭合桥",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 闭合桥总分：{summary['closure_bridge_score']:.4f}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE211_SOURCE_CARRIED_CLOSURE_BRIDGE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="来源携带闭合桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
