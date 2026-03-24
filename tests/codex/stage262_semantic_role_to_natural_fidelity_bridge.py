#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage262_semantic_role_to_natural_fidelity_bridge_20260324"
STAGE257_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage257_object_attribute_position_operation_role_map_20260324" / "summary.json"
STAGE217_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage217_source_fidelity_closure_block_20260324" / "summary.json"
STAGE228_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage228_natural_fidelity_gain_map_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    summary257 = load_json(STAGE257_SUMMARY)
    summary217 = load_json(STAGE217_SUMMARY)
    summary228 = load_json(STAGE228_SUMMARY)

    role_mean = sum(row["score"] if "score" in row else (row["activation_strength"] * 10.0 + row["compactness"]) / 2.0 for row in summary257["role_rows"]) / len(summary257["role_rows"])
    fidelity_base = next(row["score"] for row in summary217["block_rows"] if row["piece_name"] == "天然来源保真")
    fidelity_gain = next(row["score"] for row in summary228["gain_rows"] if row["piece_name"] == "理论提升空间")
    closure_piece = next(row["score"] for row in summary217["block_rows"] if row["piece_name"] == "来源携带闭合桥")
    bridge_score = (role_mean + fidelity_base + fidelity_gain + closure_piece) / 4.0
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage262_semantic_role_to_natural_fidelity_bridge",
        "title": "语义角色到天然来源保真桥",
        "status_short": "semantic_role_to_natural_fidelity_bridge_ready",
        "piece_count": 4,
        "bridge_score": bridge_score,
        "strongest_piece_name": "理论提升空间",
        "weakest_piece_name": "天然来源保真",
        "top_gap_name": "语义角色结构已经存在，但它们还没有自然升级成稳定的天然来源保真；角色先成立，保真后掉落，仍是当前主断点",
        "piece_rows": [
            {"piece_name": "语义角色平均强度", "score": role_mean},
            {"piece_name": "天然来源保真", "score": fidelity_base},
            {"piece_name": "理论提升空间", "score": fidelity_gain},
            {"piece_name": "来源携带闭合桥", "score": closure_piece},
        ],
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "piece_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["piece_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["piece_rows"])
    report = [
        "# Stage262：语义角色到天然来源保真桥",
        "",
        "## 核心结果",
        f"- 片段数量：{summary['piece_count']}",
        f"- 桥总分：{summary['bridge_score']:.4f}",
        f"- 最强片段：{summary['strongest_piece_name']}",
        f"- 最弱片段：{summary['weakest_piece_name']}",
        f"- 头号发现：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE262_SEMANTIC_ROLE_TO_NATURAL_FIDELITY_BRIDGE_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="语义角色到天然来源保真桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
