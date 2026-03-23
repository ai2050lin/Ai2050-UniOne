#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage195_3d_block_slice_map_20260323"

STAGE188_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage188_apple_neuron_role_card_20260323" / "summary.json"
STAGE192_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage192_time_unfolded_role_slicing_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s188 = load_json(STAGE188_SUMMARY_PATH)
    s192 = load_json(STAGE192_SUMMARY_PATH)

    role_map = {str(row["role_name"]): float(row["score"]) for row in s188["role_rows"]}
    slice_map = {str(row["slice_name"]): float(row["score"]) for row in s192["slice_rows"]}
    block_rows = [
        {
            "block_name": "锚定-共享切面",
            "role_name": "共享束",
            "slice_name": "早层定锚切片",
            "role_score": role_map["共享束"],
            "slice_score": slice_map["早层定锚切片"],
        },
        {
            "block_name": "差分-路径切面",
            "role_name": "差分束",
            "slice_name": "中段选路切片",
            "role_score": role_map["差分束"],
            "slice_score": slice_map["中段选路切片"],
        },
        {
            "block_name": "纤维-聚合切面",
            "role_name": "纤维束",
            "slice_name": "后层聚合切片",
            "role_score": role_map["纤维束"],
            "slice_score": slice_map["后层聚合切片"],
        },
        {
            "block_name": "来源留痕切面",
            "role_name": "来源痕迹束",
            "slice_name": "来源痕迹切片",
            "role_score": role_map["来源痕迹束"],
            "slice_score": slice_map["来源痕迹切片"],
        },
        {
            "block_name": "回收闭合切面",
            "role_name": "回收束",
            "slice_name": "回收闭合切片",
            "role_score": role_map["回收束"],
            "slice_score": slice_map["回收闭合切片"],
        },
    ]
    for row in block_rows:
        row["block_score"] = (float(row["role_score"]) + float(row["slice_score"])) / 2.0
    strongest_block_name = max(block_rows, key=lambda row: float(row["block_score"]))["block_name"]
    weakest_block_name = min(block_rows, key=lambda row: float(row["block_score"]))["block_name"]
    slice_map_score = sum(float(row["block_score"]) for row in block_rows) / float(len(block_rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage195_3d_block_slice_map",
        "title": "三维拼块切面图",
        "status_short": "3d_block_slice_map_ready",
        "block_count": len(block_rows),
        "strongest_block_name": strongest_block_name,
        "weakest_block_name": weakest_block_name,
        "slice_map_score": slice_map_score,
        "block_rows": block_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage195：三维拼块切面图",
        "",
        "## 核心结果",
        f"- 切面数量：{summary['block_count']}",
        f"- 最强切面：{summary['strongest_block_name']}",
        f"- 最弱切面：{summary['weakest_block_name']}",
        f"- 切面图总分：{summary['slice_map_score']:.4f}",
    ]
    (output_dir / "STAGE195_3D_BLOCK_SLICE_MAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="三维拼块切面图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
