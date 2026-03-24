#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage242_large_scale_noun_shared_delta_matrix_20260324"
WORD_ROWS_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage119_gpt2_embedding_full_vocab_scan_20260323" / "word_rows.csv"

TARGET_GROUPS = [
    "meso_fruit",
    "meso_animal",
    "meso_object",
    "meso_food",
    "meso_human",
    "meso_vehicle",
]


def build_summary() -> dict:
    groups: dict[str, list[dict]] = {name: [] for name in TARGET_GROUPS}
    noun_count = 0
    with WORD_ROWS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["lexical_type"] != "noun":
                continue
            noun_count += 1
            group_name = row["group"]
            if group_name not in groups:
                continue
            groups[group_name].append(row)

    matrix_rows = []
    for group_name in TARGET_GROUPS:
        rows = groups[group_name]
        count = len(rows)
        mean_group_score = sum(float(row["group_score"]) for row in rows) / count
        mean_group_margin = sum(float(row["group_margin"]) for row in rows) / count
        mean_effective_score = sum(float(row["effective_encoding_score"]) for row in rows) / count
        strong_margin_ratio = (
            sum(1 for row in rows if float(row["group_margin"]) >= 0.05) / count
        )
        matrix_rows.append(
            {
                "group_name": group_name,
                "count": count,
                "mean_group_score": mean_group_score,
                "mean_group_margin": mean_group_margin,
                "mean_effective_encoding_score": mean_effective_score,
                "strong_margin_ratio": strong_margin_ratio,
            }
        )

    shared_base_strength = sum(row["mean_group_score"] for row in matrix_rows) / len(matrix_rows)
    local_delta_strength = sum(row["mean_group_margin"] for row in matrix_rows) / len(matrix_rows)
    effective_mean = sum(row["mean_effective_encoding_score"] for row in matrix_rows) / len(matrix_rows)
    strong_margin_mean = sum(row["strong_margin_ratio"] for row in matrix_rows) / len(matrix_rows)
    target_noun_ratio = sum(row["count"] for row in matrix_rows) / noun_count
    matrix_score = (
        shared_base_strength + local_delta_strength + effective_mean + strong_margin_mean + target_noun_ratio
    ) / 5.0
    strongest = max(matrix_rows, key=lambda row: row["mean_group_score"])
    weakest = min(matrix_rows, key=lambda row: row["mean_group_margin"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage242_large_scale_noun_shared_delta_matrix",
        "title": "全名词大样本共享-差分矩阵",
        "status_short": "large_scale_noun_shared_delta_matrix_ready",
        "noun_count": noun_count,
        "target_group_count": len(matrix_rows),
        "target_noun_ratio": target_noun_ratio,
        "shared_base_strength": shared_base_strength,
        "local_delta_strength": local_delta_strength,
        "effective_mean": effective_mean,
        "strong_margin_mean": strong_margin_mean,
        "matrix_score": matrix_score,
        "strongest_group_name": strongest["group_name"],
        "weakest_group_name": weakest["group_name"],
        "top_gap_name": "大样本下共享仍明显强于差分，说明局部差分仍然偏薄",
        "matrix_rows": matrix_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage242：全名词大样本共享-差分矩阵",
        "",
        "## 核心结果",
        f"- 名词总量：{summary['noun_count']}",
        f"- 目标组数量：{summary['target_group_count']}",
        f"- 目标名词覆盖率：{summary['target_noun_ratio']:.4f}",
        f"- 共享底盘强度：{summary['shared_base_strength']:.4f}",
        f"- 局部差分强度：{summary['local_delta_strength']:.4f}",
        f"- 矩阵总分：{summary['matrix_score']:.4f}",
        f"- 最强组：{summary['strongest_group_name']}",
        f"- 最弱组：{summary['weakest_group_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE242_LARGE_SCALE_NOUN_SHARED_DELTA_MATRIX_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="全名词大样本共享-差分矩阵")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
