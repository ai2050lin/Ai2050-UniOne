#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage214_source_carried_reentry_closure_20260323"

STAGE211_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage211_source_carried_closure_bridge_20260323" / "summary.json"
STAGE208_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage208_forward_carried_provenance_20260323" / "summary.json"
STAGE205_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage205_reentry_closure_bridge_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s211 = load_json(STAGE211_SUMMARY_PATH)
    s208 = load_json(STAGE208_SUMMARY_PATH)
    s205 = load_json(STAGE205_SUMMARY_PATH)

    rows = [
        {"piece_name": "天然痕迹传递", "score": next(float(r["score"]) for r in s211["bridge_rows"] if str(r["piece_name"]) == "天然痕迹传递")},
        {"piece_name": "前向携带来源", "score": float(s208["forward_carried_score"])},
        {"piece_name": "重入闭合桥", "score": float(s205["bridge_score"])},
        {"piece_name": "来源携带闭合桥", "score": float(s211["closure_bridge_score"])},
    ]
    ranked_rows = sorted(rows, key=lambda item: float(item["score"]))
    closure_score = (
        float(rows[0]["score"]) * 0.30
        + float(rows[1]["score"]) * 0.30
        + float(rows[2]["score"]) * 0.20
        + float(rows[3]["score"]) * 0.20
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage214_source_carried_reentry_closure",
        "title": "来源携带重入闭合",
        "status_short": "source_carried_reentry_closure_ready",
        "piece_count": len(rows),
        "closure_score": closure_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "天然痕迹传递不足",
        "closure_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage214：来源携带重入闭合",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 闭合总分：{summary['closure_score']:.4f}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE214_SOURCE_CARRIED_REENTRY_CLOSURE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="来源携带重入闭合")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
