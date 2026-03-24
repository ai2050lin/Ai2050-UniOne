#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage242_large_scale_noun_shared_delta_matrix import run_analysis as run_stage242


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage243_efficiency_reason_map_20260324"

STAGE237_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage237_noun_parameter_efficiency_map_20260324" / "summary.json"
STAGE240_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage240_structure_efficiency_candidate_map_20260324" / "summary.json"
STAGE242_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage242_large_scale_noun_shared_delta_matrix_20260324" / "summary.json"
STAGE157_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage157_apple_action_route_probe_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def piece_score(rows: list[dict], piece_name: str) -> float:
    for row in rows:
        if str(row["piece_name"]) == piece_name:
            return float(row["score"])
    raise KeyError(piece_name)


def build_summary() -> dict:
    s237 = load_json(STAGE237_SUMMARY_PATH)
    s240 = load_json(STAGE240_SUMMARY_PATH)
    if STAGE242_SUMMARY_PATH.exists():
        s242 = load_json(STAGE242_SUMMARY_PATH)
    else:
        s242 = run_stage242(force=True)
    s157 = load_json(STAGE157_SUMMARY_PATH)

    shared_reuse = piece_score(s237["efficiency_rows"], "共享复用效率")
    delta_sparsity = 1.0 - float(s242["local_delta_strength"])
    route_amplification = float(s157["apple_action_route_score"])
    structure_consensus = max(float(row["score"]) for row in s240["candidate_rows"] if row["candidate_name"] == "共享底盘 + 局部差分 + 路径放大")

    reason_rows = [
        {"reason_name": "共享复用强度", "score": shared_reuse},
        {"reason_name": "局部差分稀疏性", "score": delta_sparsity},
        {"reason_name": "路径放大强度", "score": route_amplification},
        {"reason_name": "候选结构一致性", "score": structure_consensus},
    ]
    ranked = sorted(reason_rows, key=lambda row: float(row["score"]), reverse=True)
    score = sum(float(row["score"]) for row in reason_rows) / len(reason_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage243_efficiency_reason_map",
        "title": "高效性原因图",
        "status_short": "efficiency_reason_map_ready",
        "reason_count": len(reason_rows),
        "reason_score": score,
        "strongest_reason_name": str(ranked[0]["reason_name"]),
        "weakest_reason_name": str(ranked[-1]["reason_name"]),
        "top_gap_name": "局部差分虽然稀疏高效，但天然边界仍不够锋利",
        "reason_rows": reason_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage243：高效性原因图",
        "",
        "## 核心结果",
        f"- 原因数量：{summary['reason_count']}",
        f"- 总分：{summary['reason_score']:.4f}",
        f"- 最强原因：{summary['strongest_reason_name']}",
        f"- 最弱原因：{summary['weakest_reason_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE243_EFFICIENCY_REASON_MAP_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="高效性原因图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
