#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage202_reentry_closure_puzzle_20260323"

STAGE198_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage198_provenance_trace_continuity_tracking_20260323" / "summary.json"
STAGE199_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage199_result_chain_closure_strengthening_20260323" / "summary.json"
STAGE196_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage196_cross_model_invariant_block_refinement_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s198 = load_json(STAGE198_SUMMARY_PATH)
    s199 = load_json(STAGE199_SUMMARY_PATH)
    s196 = load_json(STAGE196_SUMMARY_PATH)

    trace_score = float(s198["trace_continuity_score"])
    closure_strength = float(s199["strengthening_score"])
    stable_backbone = float(s196["stable_block_count"]) / 5.0
    reentry_closure_score = trace_score * 0.35 + closure_strength * 0.40 + stable_backbone * 0.25
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage202_reentry_closure_puzzle",
        "title": "重入闭合拼图",
        "status_short": "reentry_closure_puzzle_ready",
        "trace_continuity_score": trace_score,
        "closure_strength_score": closure_strength,
        "stable_backbone_score": stable_backbone,
        "reentry_closure_score": reentry_closure_score,
        "top_gap_name": "来源痕迹天然保留",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage202：重入闭合拼图",
        "",
        "## 核心结果",
        f"- 痕迹持续分数：{summary['trace_continuity_score']:.4f}",
        f"- 闭合补强分数：{summary['closure_strength_score']:.4f}",
        f"- 稳定骨架分数：{summary['stable_backbone_score']:.4f}",
        f"- 重入闭合总分：{summary['reentry_closure_score']:.4f}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE202_REENTRY_CLOSURE_PUZZLE_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="重入闭合拼图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
