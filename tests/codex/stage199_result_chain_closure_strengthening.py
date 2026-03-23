#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage199_result_chain_closure_strengthening_20260323"

STAGE184_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage184_cross_model_puzzle_expansion_20260323" / "summary.json"
STAGE194_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage194_bottom_block_intervention_priority_20260323" / "summary.json"
STAGE196_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage196_cross_model_invariant_block_refinement_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s184 = load_json(STAGE184_SUMMARY_PATH)
    s194 = load_json(STAGE194_SUMMARY_PATH)
    s196 = load_json(STAGE196_SUMMARY_PATH)

    result_chain_score = next(float(row["score"]) for row in s184["shared_rows"] if str(row["block_name"]) == "结果链")
    result_chain_priority = next(str(row["priority"]) for row in s194["target_rows"] if str(row["target_name"]) == "结果链")
    stable_count = int(s196["stable_block_count"])
    closure_gap = 1.0 - result_chain_score
    strengthening_score = result_chain_score * 0.45 + (1.0 - closure_gap) * 0.20 + (stable_count / 5.0) * 0.35
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage199_result_chain_closure_strengthening",
        "title": "结果链闭合补强块",
        "status_short": "result_chain_closure_strengthening_ready",
        "result_chain_score": result_chain_score,
        "result_chain_priority": result_chain_priority,
        "closure_gap": closure_gap,
        "stable_block_count": stable_count,
        "strengthening_score": strengthening_score,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage199：结果链闭合补强块",
        "",
        "## 核心结果",
        f"- 结果链分数：{summary['result_chain_score']:.4f}",
        f"- 结果链优先级：{summary['result_chain_priority']}",
        f"- 闭合缺口：{summary['closure_gap']:.4f}",
        f"- 补强总分：{summary['strengthening_score']:.4f}",
    ]
    (output_dir / "STAGE199_RESULT_CHAIN_CLOSURE_STRENGTHENING_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="结果链闭合补强块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
