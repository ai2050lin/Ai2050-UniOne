#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage206_retained_trace_transfer_20260323"

STAGE198_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage198_provenance_trace_continuity_tracking_20260323" / "summary.json"
STAGE203_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage203_retained_trace_hardening_20260323" / "summary.json"
STAGE205_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage205_reentry_closure_bridge_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s198 = load_json(STAGE198_SUMMARY_PATH)
    s203 = load_json(STAGE203_SUMMARY_PATH)
    s205 = load_json(STAGE205_SUMMARY_PATH)

    retained_trace = float(s203["retained_trace_score"])
    recurrence_trace = next(float(row["score"]) for row in s203["piece_rows"] if str(row["piece_name"]) == "复现痕迹")
    repair_trace = float(s203["repair_trace_score"])
    trace_bridge = next(float(row["score"]) for row in s205["bridge_rows"] if str(row["piece_name"]) == "时序痕迹桥")
    continuity_gap = float(s198["continuity_gap"])

    transfer_score = retained_trace * 0.35 + recurrence_trace * 0.25 + trace_bridge * 0.20 + (1.0 - continuity_gap) * 0.20
    transfer_rows = [
        {"piece_name": "天然保留", "score": retained_trace, "status": "薄弱传递点"},
        {"piece_name": "复现痕迹", "score": recurrence_trace, "status": "过渡传递点"},
        {"piece_name": "时序痕迹桥", "score": trace_bridge, "status": "薄弱传递点"},
        {"piece_name": "修复迁移", "score": repair_trace, "status": "外部补偿点"},
    ]
    weakest_piece_name = min(transfer_rows, key=lambda row: float(row["score"]))["piece_name"]
    strongest_piece_name = max(transfer_rows, key=lambda row: float(row["score"]))["piece_name"]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage206_retained_trace_transfer",
        "title": "天然痕迹传递块",
        "status_short": "retained_trace_transfer_ready",
        "continuity_gap": continuity_gap,
        "transfer_score": transfer_score,
        "weakest_piece_name": weakest_piece_name,
        "strongest_piece_name": strongest_piece_name,
        "transfer_rows": transfer_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage206：天然痕迹传递块",
        "",
        "## 核心结果",
        f"- 持续断层：{summary['continuity_gap']:.4f}",
        f"- 传递总分：{summary['transfer_score']:.4f}",
        f"- 最弱传递点：{summary['weakest_piece_name']}",
        f"- 最强补偿点：{summary['strongest_piece_name']}",
    ]
    (output_dir / "STAGE206_RETAINED_TRACE_TRANSFER_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="天然痕迹传递块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
