#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from stage166_category_fiber_map import run_analysis as run_stage166_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage169_category_delta_tensor_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE169_CATEGORY_DELTA_TENSOR_REPORT.md"
STAGE166_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage166_category_fiber_map_20260323" / "summary.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> Dict[str, object]:
    if not STAGE166_SUMMARY_PATH.exists():
        run_stage166_analysis(force=True)
    stage166 = load_json(STAGE166_SUMMARY_PATH)
    delta_rows: List[Dict[str, object]] = []
    for row in stage166["edge_rows"]:
        similarity = float(row["centroid_similarity"])
        delta = 1.0 - similarity
        delta_rows.append(
            {
                "category_a": str(row["category_a"]),
                "category_b": str(row["category_b"]),
                "centroid_similarity": similarity,
                "delta_strength": delta,
            }
        )
    delta_rows.sort(key=lambda row: float(row["delta_strength"]), reverse=True)
    strongest_delta = delta_rows[0]
    weakest_delta = delta_rows[-1]
    mean_delta = mean(float(row["delta_strength"]) for row in delta_rows)
    delta_tensor_score = clamp01(mean_delta * 10.0)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage169_category_delta_tensor",
        "title": "类别差分张量",
        "status_short": "category_delta_tensor_ready",
        "pair_count": len(delta_rows),
        "mean_delta_strength": mean_delta,
        "delta_tensor_score": delta_tensor_score,
        "strongest_delta_name": f"{strongest_delta['category_a']}->{strongest_delta['category_b']}",
        "weakest_delta_name": f"{weakest_delta['category_a']}->{weakest_delta['category_b']}",
        "delta_rows": delta_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage169: 类别差分张量",
        "",
        "## 核心结果",
        f"- 配对数: {summary['pair_count']}",
        f"- 平均差分强度: {summary['mean_delta_strength']:.4f}",
        f"- 差分张量分数: {summary['delta_tensor_score']:.4f}",
        f"- 最强差分边: {summary['strongest_delta_name']}",
        f"- 最弱差分边: {summary['weakest_delta_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="类别差分张量")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
