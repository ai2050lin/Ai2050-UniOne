#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage350_independent_amplification_core_hardening import run_analysis as run_stage350
from stage353_independent_amplification_core_isolation_review import run_analysis as run_stage353


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage357_independent_amplification_net_gain_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s350 = run_stage350(force=False)
    s353 = run_stage353(force=False)

    hardening_rows = {row["metric_name"]: row for row in s350["hardening_rows"]}
    review_rows = {row["metric_name"]: row for row in s353["review_rows"]}

    raw_gain = float(review_rows["独立放大核原始增益"]["strength"])
    net_gain = float(review_rows["残余耦合抑制后净增益"]["strength"])
    relay_ratio = float(review_rows["独立放大核 / 接力整体比值"]["strength"])
    residual_mean = float(hardening_rows["残余耦合均值"]["strength"])
    net_ratio = net_gain / max(raw_gain, 1e-6)
    review_score = net_gain * 0.45 + relay_ratio * 0.25 + net_ratio * 0.20 + max(0.0, 1.0 - residual_mean) * 0.10

    net_gain_rows = [
        {"metric_name": "独立放大核原始增益", "strength": raw_gain},
        {"metric_name": "残余耦合抑制后净增益", "strength": net_gain},
        {"metric_name": "净增益 / 原始增益比值", "strength": net_ratio},
        {"metric_name": "独立放大核 / 接力整体比值", "strength": relay_ratio},
    ]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage357_independent_amplification_net_gain_review",
        "title": "独立放大核净增益复核",
        "status_short": "independent_amplification_net_gain_review_ready",
        "review_score": float(review_score),
        "review_rows": net_gain_rows,
        "top_gap_name": "独立放大核当前最大问题不是完全没有增益，而是净增益仍然偏低，说明剥离后的独立强度还不够。",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="独立放大核净增益复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
