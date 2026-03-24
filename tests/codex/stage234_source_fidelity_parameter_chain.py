#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage234_source_fidelity_parameter_chain_20260324"

STAGE217_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage217_source_fidelity_closure_block_20260324" / "summary.json"
STAGE228_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage228_natural_fidelity_gain_map_20260324" / "summary.json"
STAGE231_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage231_source_fidelity_parameter_structure_map_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pick_score(rows: list[dict], name: str) -> float:
    for row in rows:
        if str(row["piece_name"]) == name:
            return float(row["score"])
    raise KeyError(name)


def build_summary() -> dict:
    s217 = load_json(STAGE217_SUMMARY_PATH)
    s228 = load_json(STAGE228_SUMMARY_PATH)
    s231 = load_json(STAGE231_SUMMARY_PATH)

    block_rows = {str(row["piece_name"]): float(row["score"]) for row in s217["block_rows"]}
    gain_rows = {str(row["piece_name"]): float(row["score"]) for row in s228["gain_rows"]}
    structure_rows = {str(row["piece_name"]): float(row["score"]) for row in s231["piece_rows"]}

    chain_rows = [
        {"piece_name": "天然来源保真参数起点", "score": block_rows["天然来源保真"]},
        {"piece_name": "前向携带来源参数桥", "score": block_rows["前向携带来源"]},
        {"piece_name": "参数级复杂断裂约束", "score": structure_rows["复杂处理断裂约束"]},
        {"piece_name": "参数级理论提升空间", "score": gain_rows["理论提升空间"]},
        {"piece_name": "来源保真结构总底盘", "score": s231["parameter_structure_score"]},
    ]
    ranked_rows = sorted(chain_rows, key=lambda row: float(row["score"]), reverse=True)
    chain_score = sum(float(row["score"]) for row in chain_rows) / len(chain_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage234_source_fidelity_parameter_chain",
        "title": "来源保真参数链",
        "status_short": "source_fidelity_parameter_chain_ready",
        "piece_count": len(chain_rows),
        "chain_score": chain_score,
        "strongest_piece_name": str(ranked_rows[0]["piece_name"]),
        "weakest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "天然来源保真参数起点仍然过弱",
        "chain_rows": chain_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage234：来源保真参数链",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 参数链总分：{summary['chain_score']:.4f}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE234_SOURCE_FIDELITY_PARAMETER_CHAIN_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="来源保真参数链")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
