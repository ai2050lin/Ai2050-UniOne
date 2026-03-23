#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage191_shared_neuron_overlap_tracing_20260323"

STAGE124_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323" / "summary.json"
STAGE127_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage127_noun_context_neuron_probe_20260323" / "summary.json"
STAGE131_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage131_l1_l3_l11_propagation_bridge_20260323" / "summary.json"
STAGE188_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage188_apple_neuron_role_card_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s124 = load_json(STAGE124_SUMMARY_PATH)
    s127 = load_json(STAGE127_SUMMARY_PATH)
    s131 = load_json(STAGE131_SUMMARY_PATH)
    s188 = load_json(STAGE188_SUMMARY_PATH)

    static_late = {
        int(row["neuron_index"])
        for row in s124["top_general_neurons"]
        if int(row["layer_index"]) == int(s124["dominant_general_layer_index"])
    }
    context_early = {
        int(row["neuron_index"])
        for row in s127["top_general_neurons"]
        if int(row["layer_index"]) == int(s127["dominant_general_layer_index"])
    }
    selected_early = {int(v) for v in s131["selected_early_neurons"]}
    selected_route = {int(v) for v in s131["selected_route_neurons"]}
    selected_late = {int(v) for v in s131["selected_late_neurons"]}

    early_anchor_overlap = len(context_early & selected_early)
    late_anchor_overlap = len(static_late & selected_late)
    cross_stage_overlap = len((context_early | static_late) & (selected_early | selected_late))
    direct_role_overlap_ratio = cross_stage_overlap / float(len(context_early | static_late | selected_early | selected_late))

    role_map = {str(row["role_name"]): float(row["score"]) for row in s188["role_rows"]}
    structural_continuity_score = (
        role_map["共享束"] * 0.20
        + role_map["差分束"] * 0.20
        + role_map["纤维束"] * 0.20
        + role_map["路径束"] * 0.20
        + role_map["来源痕迹束"] * 0.10
        + role_map["回收束"] * 0.10
    )
    same_block_candidate_score = direct_role_overlap_ratio * 0.40 + structural_continuity_score * 0.60
    overlap_rows = [
        {
            "overlap_name": "早层定锚重叠",
            "direct_overlap_count": early_anchor_overlap,
            "pool_a_count": len(context_early),
            "pool_b_count": len(selected_early),
        },
        {
            "overlap_name": "后层聚合重叠",
            "direct_overlap_count": late_anchor_overlap,
            "pool_a_count": len(static_late),
            "pool_b_count": len(selected_late),
        },
        {
            "overlap_name": "跨阶段总重叠",
            "direct_overlap_count": cross_stage_overlap,
            "pool_a_count": len(context_early | static_late),
            "pool_b_count": len(selected_early | selected_route | selected_late),
        },
    ]
    strongest_overlap_name = max(overlap_rows, key=lambda row: int(row["direct_overlap_count"]))["overlap_name"]
    weakest_overlap_name = min(overlap_rows, key=lambda row: int(row["direct_overlap_count"]))["overlap_name"]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage191_shared_neuron_overlap_tracing",
        "title": "共享神经元重叠追踪",
        "status_short": "shared_neuron_overlap_tracing_ready",
        "early_anchor_overlap_count": early_anchor_overlap,
        "late_anchor_overlap_count": late_anchor_overlap,
        "cross_stage_overlap_count": cross_stage_overlap,
        "direct_role_overlap_ratio": direct_role_overlap_ratio,
        "structural_continuity_score": structural_continuity_score,
        "same_block_candidate_score": same_block_candidate_score,
        "strongest_overlap_name": strongest_overlap_name,
        "weakest_overlap_name": weakest_overlap_name,
        "overlap_rows": overlap_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage191：共享神经元重叠追踪",
        "",
        "## 核心结果",
        f"- 早层定锚重叠数量：{summary['early_anchor_overlap_count']}",
        f"- 后层聚合重叠数量：{summary['late_anchor_overlap_count']}",
        f"- 跨阶段总重叠数量：{summary['cross_stage_overlap_count']}",
        f"- 直接重叠比例：{summary['direct_role_overlap_ratio']:.4f}",
        f"- 结构连续性分数：{summary['structural_continuity_score']:.4f}",
        f"- 同一底层拼块候选分数：{summary['same_block_candidate_score']:.4f}",
    ]
    (output_dir / "STAGE191_SHARED_NEURON_OVERLAP_TRACING_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="共享神经元重叠追踪")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
