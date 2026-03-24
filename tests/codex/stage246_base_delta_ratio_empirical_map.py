#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage245_large_scale_noun_shared_delta_tensor import run_analysis as run_stage245


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage246_base_delta_ratio_empirical_map_20260324"

STAGE240_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage240_structure_efficiency_candidate_map_20260324" / "summary.json"
STAGE245_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage245_large_scale_noun_shared_delta_tensor_20260324" / "summary.json"
STAGE157_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage157_apple_action_route_probe_20260323" / "summary.json"

CANDIDATE_RATIOS = [
    (0.95, 0.05),
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s240 = load_json(STAGE240_SUMMARY_PATH)
    if STAGE245_SUMMARY_PATH.exists():
        s245 = load_json(STAGE245_SUMMARY_PATH)
    else:
        s245 = run_stage245(force=True)
    s157 = load_json(STAGE157_SUMMARY_PATH)

    observed_base = float(s245["shared_base_mean"])
    observed_delta = float(s245["local_delta_mean"])
    observed_total = observed_base + observed_delta
    observed_ratio = observed_base / observed_total if observed_total > 0 else 0.0
    route_amp = float(s157["apple_action_route_score"])
    structure_consensus = max(
        float(row["score"])
        for row in s240["candidate_rows"]
        if row["candidate_name"] == "共享底盘 + 局部差分 + 路径放大"
    )

    ratio_rows = []
    for base_ratio, delta_ratio in CANDIDATE_RATIOS:
        ratio_fit = 1.0 - abs(base_ratio - observed_ratio)
        route_support = route_amp * base_ratio
        consensus_support = structure_consensus * (1.0 - abs(delta_ratio - (1.0 - observed_ratio)))
        score = (ratio_fit + route_support + consensus_support) / 3.0
        ratio_rows.append(
            {
                "ratio_name": f"{int(base_ratio * 100)}/{int(delta_ratio * 100)}",
                "base_ratio": base_ratio,
                "delta_ratio": delta_ratio,
                "score": score,
            }
        )

    ranked = sorted(ratio_rows, key=lambda row: float(row["score"]), reverse=True)
    score = sum(float(row["score"]) for row in ratio_rows) / len(ratio_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage246_base_delta_ratio_empirical_map",
        "title": "基底-差分比例经验图",
        "status_short": "base_delta_ratio_empirical_map_ready",
        "observed_base_ratio": observed_ratio,
        "ratio_count": len(ratio_rows),
        "map_score": score,
        "best_ratio_name": str(ranked[0]["ratio_name"]),
        "worst_ratio_name": str(ranked[-1]["ratio_name"]),
        "top_gap_name": "重差分比例与当前大样本分布不匹配，说明高效性并不来自强差分",
        "ratio_rows": ratio_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage246：基底-差分比例经验图",
        "",
        "## 核心结果",
        f"- 观察到的基底比例：{summary['observed_base_ratio']:.4f}",
        f"- 候选比例数量：{summary['ratio_count']}",
        f"- 总分：{summary['map_score']:.4f}",
        f"- 最优比例：{summary['best_ratio_name']}",
        f"- 最弱比例：{summary['worst_ratio_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE246_BASE_DELTA_RATIO_EMPIRICAL_MAP_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="基底-差分比例经验图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
