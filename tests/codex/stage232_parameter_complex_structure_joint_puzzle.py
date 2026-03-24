#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage232_parameter_complex_structure_joint_puzzle_20260324"

STAGE140_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323" / "summary.json"
STAGE229_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage229_cross_model_propagation_core_filter_20260324" / "summary.json"
STAGE230_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage230_parameter_boundary_trigger_pack_20260324" / "summary.json"
STAGE231_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage231_source_fidelity_parameter_structure_map_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s140 = load_json(STAGE140_SUMMARY_PATH)
    s229 = load_json(STAGE229_SUMMARY_PATH)
    s230 = load_json(STAGE230_SUMMARY_PATH)
    s231 = load_json(STAGE231_SUMMARY_PATH)

    deepseek_source_fidelity = float(s140["anaphora_summary"]["anaphora_ellipsis_score"])
    deepseek_result_chain = float(s140["result_summary"]["noun_verb_result_chain_score"])
    deepseek_joint_chain = float(s140["joint_summary"]["noun_verb_joint_propagation_score"])

    piece_rows = [
        {"piece_name": "参数级边界拼图", "score": float(s230["parameter_boundary_score"])},
        {"piece_name": "参数级来源保真拼图", "score": float(s231["parameter_structure_score"])},
        {"piece_name": "复杂处理结构拼图", "score": (deepseek_joint_chain + deepseek_result_chain) / 2.0},
        {"piece_name": "跨模型传播主核", "score": float(s229["filter_score"])},
        {"piece_name": "DeepSeek直测来源保真", "score": deepseek_source_fidelity},
    ]
    ranked_rows = sorted(piece_rows, key=lambda row: float(row["score"]), reverse=True)
    score = sum(float(row["score"]) for row in piece_rows) / len(piece_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage232_parameter_complex_structure_joint_puzzle",
        "title": "参数级编码机制与复杂处理结构联合拼图",
        "status_short": "parameter_complex_structure_joint_puzzle_ready",
        "model_name": str(s140["model_name"]),
        "piece_count": len(piece_rows),
        "joint_score": score,
        "strongest_piece_name": str(ranked_rows[0]["piece_name"]),
        "weakest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "复杂处理结构仍弱于参数级边界与来源保真拼图",
        "piece_rows": piece_rows,
        "deepseek_direct_verdict": str(s140["transfer_summary"]["transfer_verdict"]),
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage232：参数级编码机制与复杂处理结构联合拼图",
        "",
        "## 核心结果",
        f"- 直测模型：{summary['model_name']}",
        f"- 部件数量：{summary['piece_count']}",
        f"- 联合总分：{summary['joint_score']:.4f}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
        f"- DeepSeek直测判定：{summary['deepseek_direct_verdict']}",
    ]
    (output_dir / "STAGE232_PARAMETER_COMPLEX_STRUCTURE_JOINT_PUZZLE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="参数级编码机制与复杂处理结构联合拼图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
