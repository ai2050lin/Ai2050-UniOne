#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage222_attention_source_fidelity_bridge_20260324"

STAGE220_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage220_attention_closure_bridge_20260324" / "summary.json"
STAGE217_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage217_source_fidelity_closure_block_20260324" / "summary.json"
STAGE218_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage218_apple_sense_attention_retrieval_map_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def row_score(rows: list[dict], name: str) -> float:
    for row in rows:
        if str(row["piece_name"]) == name:
            return float(row["score"])
    raise KeyError(name)


def build_summary() -> dict:
    s220 = load_json(STAGE220_SUMMARY_PATH)
    s217 = load_json(STAGE217_SUMMARY_PATH)
    s218 = load_json(STAGE218_SUMMARY_PATH)

    brand_attention = row_score(s218["retrieval_rows"], "品牌义取回")
    forward_carried = row_score(s217["block_rows"], "前向携带来源")
    source_fidelity = row_score(s217["block_rows"], "天然来源保真")
    closure_bridge = float(s220["bridge_score"])

    rows = [
        {"piece_name": "品牌义注意力", "score": brand_attention},
        {"piece_name": "前向携带来源", "score": forward_carried},
        {"piece_name": "天然来源保真", "score": source_fidelity},
        {"piece_name": "注意力-保真桥", "score": closure_bridge},
    ]
    ranked_rows = sorted(rows, key=lambda row: float(row["score"]))
    bridge_score = (
        brand_attention * 0.20
        + forward_carried * 0.30
        + source_fidelity * 0.25
        + closure_bridge * 0.25
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage222_attention_source_fidelity_bridge",
        "title": "注意力到来源保真桥",
        "status_short": "attention_source_fidelity_bridge_ready",
        "piece_count": len(rows),
        "bridge_score": bridge_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "天然来源保真不足",
        "bridge_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage222：注意力到来源保真桥",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 桥总分：{summary['bridge_score']:.4f}",
        f"- 最弱桥段：{summary['weakest_piece_name']}",
        f"- 最强桥段：{summary['strongest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE222_ATTENTION_SOURCE_FIDELITY_BRIDGE_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="注意力到来源保真桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
