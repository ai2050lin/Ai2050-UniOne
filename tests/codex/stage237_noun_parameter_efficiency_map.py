#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage237_noun_parameter_efficiency_map_20260324"

STAGE189_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage189_family_neuron_bundle_map_20260323" / "summary.json"
STAGE124_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323" / "summary.json"
STAGE236_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage236_apple_pear_parameter_delta_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s189 = load_json(STAGE189_SUMMARY_PATH)
    s124 = load_json(STAGE124_SUMMARY_PATH)
    s236 = load_json(STAGE236_SUMMARY_PATH)

    bundle_strength_mean = sum(float(row["bundle_strength"]) for row in s189["bundle_rows"]) / len(s189["bundle_rows"])
    bundle_sep_mean = sum(float(row["bundle_separation"]) for row in s189["bundle_rows"]) / len(s189["bundle_rows"])
    delta_support_mean = sum(float(row["delta_support"]) for row in s189["bundle_rows"]) / len(s189["bundle_rows"])

    efficiency_rows = [
        {"piece_name": "共享复用效率", "score": bundle_strength_mean},
        {"piece_name": "差分分裂效率", "score": bundle_sep_mean + delta_support_mean},
        {"piece_name": "名词通用参数效率", "score": float(s124["noun_neuron_basic_probe_score"])},
        {"piece_name": "苹果-梨子差分效率", "score": float(s236["delta_score"])},
    ]
    ranked_rows = sorted(efficiency_rows, key=lambda row: float(row["score"]), reverse=True)
    efficiency_score = sum(float(row["score"]) for row in efficiency_rows) / len(efficiency_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage237_noun_parameter_efficiency_map",
        "title": "名词参数级高效编码图",
        "status_short": "noun_parameter_efficiency_map_ready",
        "piece_count": len(efficiency_rows),
        "efficiency_score": efficiency_score,
        "strongest_piece_name": str(ranked_rows[0]["piece_name"]),
        "weakest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "差分分裂效率仍然低于共享复用效率",
        "efficiency_rows": efficiency_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage237：名词参数级高效编码图",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 高效编码总分：{summary['efficiency_score']:.4f}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE237_NOUN_PARAMETER_EFFICIENCY_MAP_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="名词参数级高效编码图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
