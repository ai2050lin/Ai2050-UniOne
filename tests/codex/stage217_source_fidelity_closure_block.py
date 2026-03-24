#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage217_source_fidelity_closure_block_20260324"

STAGE214_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage214_source_carried_reentry_closure_20260323" / "summary.json"
STAGE212_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage212_natural_trace_decay_map_20260323" / "summary.json"
STAGE208_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage208_forward_carried_provenance_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s214 = load_json(STAGE214_SUMMARY_PATH)
    s212 = load_json(STAGE212_SUMMARY_PATH)
    s208 = load_json(STAGE208_SUMMARY_PATH)

    natural_retention = next(float(row["retention_score"]) for row in s212["decay_rows"] if str(row["segment_name"]) == "天然保留段")
    forward_carried = float(s208["forward_carried_score"])
    closure_score = float(s214["closure_score"])
    source_carried_bridge = next(float(row["score"]) for row in s214["closure_rows"] if str(row["piece_name"]) == "来源携带闭合桥")

    block_rows = [
        {"piece_name": "天然来源保真", "score": natural_retention},
        {"piece_name": "前向携带来源", "score": forward_carried},
        {"piece_name": "来源携带闭合桥", "score": source_carried_bridge},
        {"piece_name": "后段闭合", "score": closure_score},
    ]
    ranked_rows = sorted(block_rows, key=lambda row: float(row["score"]))
    block_score = (
        natural_retention * 0.30
        + forward_carried * 0.25
        + source_carried_bridge * 0.20
        + closure_score * 0.25
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage217_source_fidelity_closure_block",
        "title": "来源保真闭合块",
        "status_short": "source_fidelity_closure_block_ready",
        "piece_count": len(block_rows),
        "block_score": block_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "天然来源保真不足",
        "block_rows": block_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage217：来源保真闭合块",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 闭合块总分：{summary['block_score']:.4f}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE217_SOURCE_FIDELITY_CLOSURE_BLOCK_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="来源保真闭合块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
