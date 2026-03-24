#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage225_attention_to_natural_fidelity_chain_20260324"

STAGE222_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage222_attention_source_fidelity_bridge_20260324" / "summary.json"
STAGE217_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage217_source_fidelity_closure_block_20260324" / "summary.json"
STAGE215_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage215_natural_trace_breakpoint_map_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def row_score(rows: list[dict], name: str) -> float:
    for row in rows:
        if str(row["piece_name"]) == name:
            return float(row["score"])
    raise KeyError(name)


def build_summary() -> dict:
    s222 = load_json(STAGE222_SUMMARY_PATH)
    s217 = load_json(STAGE217_SUMMARY_PATH)
    s215 = load_json(STAGE215_SUMMARY_PATH)

    rows = [
        {"piece_name": "品牌义注意力", "score": row_score(s222["bridge_rows"], "品牌义注意力")},
        {"piece_name": "前向携带来源", "score": row_score(s222["bridge_rows"], "前向携带来源")},
        {"piece_name": "天然来源保真", "score": row_score(s222["bridge_rows"], "天然来源保真")},
        {"piece_name": "天然断裂约束", "score": 1.0 - float(s215["breakpoint_score"])},
    ]
    ranked_rows = sorted(rows, key=lambda row: float(row["score"]))
    chain_score = (
        float(rows[0]["score"]) * 0.20
        + float(rows[1]["score"]) * 0.30
        + float(rows[2]["score"]) * 0.30
        + float(rows[3]["score"]) * 0.20
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage225_attention_to_natural_fidelity_chain",
        "title": "注意力取回到天然保真链",
        "status_short": "attention_to_natural_fidelity_chain_ready",
        "piece_count": len(rows),
        "chain_score": chain_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "天然来源保真不足",
        "chain_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage225：注意力取回到天然保真链",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 链总分：{summary['chain_score']:.4f}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE225_ATTENTION_TO_NATURAL_FIDELITY_CHAIN_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="注意力取回到天然保真链")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
