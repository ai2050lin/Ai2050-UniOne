#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage196_cross_model_invariant_block_refinement_20260323"

STAGE193_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage193_cross_model_invariant_3d_blocks_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s193 = load_json(STAGE193_SUMMARY_PATH)
    stable_rows = [row for row in s193["block_rows"] if str(row["status"]) == "不变稳定块"]
    weak_rows = [row for row in s193["block_rows"] if str(row["status"]) == "不变薄弱块"]
    transition_rows = [row for row in s193["block_rows"] if str(row["status"]) == "不变过渡块"]
    refinement_score = (
        sum(float(row["score"]) for row in stable_rows) * 0.6
        + sum(float(row["score"]) for row in transition_rows) * 0.3
        + sum(float(row["score"]) for row in weak_rows) * 0.1
    ) / float(len(s193["block_rows"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage196_cross_model_invariant_block_refinement",
        "title": "跨模型不变拼块精炼",
        "status_short": "cross_model_invariant_block_refinement_ready",
        "stable_block_count": len(stable_rows),
        "transition_block_count": len(transition_rows),
        "weak_block_count": len(weak_rows),
        "stable_block_names": [str(row["block_name"]) for row in stable_rows],
        "transition_block_names": [str(row["block_name"]) for row in transition_rows],
        "weak_block_names": [str(row["block_name"]) for row in weak_rows],
        "refinement_score": refinement_score,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage196：跨模型不变拼块精炼",
        "",
        "## 核心结果",
        f"- 稳定块数量：{summary['stable_block_count']}",
        f"- 过渡块数量：{summary['transition_block_count']}",
        f"- 薄弱块数量：{summary['weak_block_count']}",
        f"- 精炼总分：{summary['refinement_score']:.4f}",
    ]
    (output_dir / "STAGE196_CROSS_MODEL_INVARIANT_BLOCK_REFINEMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型不变拼块精炼")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
