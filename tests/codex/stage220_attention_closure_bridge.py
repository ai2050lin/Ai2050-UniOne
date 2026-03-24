#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage220_attention_closure_bridge_20260324"

STAGE218_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage218_apple_sense_attention_retrieval_map_20260324" / "summary.json"
STAGE214_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage214_source_carried_reentry_closure_20260323" / "summary.json"
STAGE217_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage217_source_fidelity_closure_block_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def row_score(rows: list[dict], name: str) -> float:
    for row in rows:
        if str(row["piece_name"]) == name:
            return float(row["score"])
    raise KeyError(name)


def build_summary() -> dict:
    s218 = load_json(STAGE218_SUMMARY_PATH)
    s214 = load_json(STAGE214_SUMMARY_PATH)
    s217 = load_json(STAGE217_SUMMARY_PATH)

    rows = [
        {"piece_name": "品牌义注意力取回", "score": row_score(s218["retrieval_rows"], "品牌义取回")},
        {"piece_name": "前向携带来源", "score": row_score(s217["block_rows"], "前向携带来源")},
        {"piece_name": "来源保真闭合", "score": float(s217["block_score"])},
        {"piece_name": "后段闭合", "score": float(s214["closure_score"])},
    ]
    ranked_rows = sorted(rows, key=lambda row: float(row["score"]))
    bridge_score = (
        float(rows[0]["score"]) * 0.20
        + float(rows[1]["score"]) * 0.30
        + float(rows[2]["score"]) * 0.25
        + float(rows[3]["score"]) * 0.25
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage220_attention_closure_bridge",
        "title": "注意力到回收闭合桥",
        "status_short": "attention_closure_bridge_ready",
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
        "# Stage220：注意力到回收闭合桥",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 桥总分：{summary['bridge_score']:.4f}",
        f"- 最弱桥段：{summary['weakest_piece_name']}",
        f"- 最强桥段：{summary['strongest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE220_ATTENTION_CLOSURE_BRIDGE_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="注意力到回收闭合桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
