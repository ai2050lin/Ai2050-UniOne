#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage208_forward_carried_provenance_20260323"

STAGE206_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage206_retained_trace_transfer_20260323" / "summary.json"
STAGE207_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage207_phase_timing_coupling_20260323" / "summary.json"
STAGE205_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage205_reentry_closure_bridge_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s206 = load_json(STAGE206_SUMMARY_PATH)
    s207 = load_json(STAGE207_SUMMARY_PATH)
    s205 = load_json(STAGE205_SUMMARY_PATH)

    transfer_score = float(s206["transfer_score"])
    coupling_score = float(s207["coupling_score"])
    bridge_score = float(s205["bridge_score"])

    forward_carried_score = transfer_score * 0.40 + coupling_score * 0.35 + bridge_score * 0.25
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage208_forward_carried_provenance",
        "title": "前向携带来源块",
        "status_short": "forward_carried_provenance_ready",
        "transfer_score": transfer_score,
        "coupling_score": coupling_score,
        "bridge_score": bridge_score,
        "forward_carried_score": forward_carried_score,
        "top_gap_name": "天然保留不足",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage208：前向携带来源块",
        "",
        "## 核心结果",
        f"- 天然传递分数：{summary['transfer_score']:.4f}",
        f"- 相位-时序耦合分数：{summary['coupling_score']:.4f}",
        f"- 闭合桥分数：{summary['bridge_score']:.4f}",
        f"- 前向携带总分：{summary['forward_carried_score']:.4f}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE208_FORWARD_CARRIED_PROVENANCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="前向携带来源块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
