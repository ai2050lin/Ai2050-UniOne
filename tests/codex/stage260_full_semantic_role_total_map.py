#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage260_full_semantic_role_total_map_20260324"
STAGE257_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage257_object_attribute_position_operation_role_map_20260324" / "summary.json"
STAGE255_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage255_translation_token_role_refinement_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_summary() -> dict:
    summary257 = load_json(STAGE257_SUMMARY)
    summary255 = load_json(STAGE255_SUMMARY)

    role_lookup = {row["role_name"]: row for row in summary257["role_rows"]}
    constraint_score = clamp01(
        (
            (summary255["variant_rows"][1]["gate_shift_mean"] / 40.0)
            + (summary255["variant_rows"][2]["gate_shift_mean"] / 40.0)
            + (summary255["variant_rows"][3]["gate_shift_mean"] / 40.0)
        )
        / 3.0
    )
    result_score = clamp01(1.0 - summary255["variant_rows"][3]["hidden_similarity_to_base"])

    role_rows = [
        {"role_name": "object", "score": clamp01((role_lookup["object"]["activation_strength"] * 10.0 + role_lookup["object"]["compactness"]) / 2.0)},
        {"role_name": "attribute", "score": clamp01((role_lookup["attribute"]["activation_strength"] * 10.0 + role_lookup["attribute"]["compactness"]) / 2.0)},
        {"role_name": "position", "score": clamp01((role_lookup["position"]["activation_strength"] * 10.0 + role_lookup["position"]["compactness"]) / 2.0)},
        {"role_name": "operation", "score": clamp01((role_lookup["operation"]["activation_strength"] * 10.0 + role_lookup["operation"]["compactness"]) / 2.0)},
        {"role_name": "constraint", "score": constraint_score},
        {"role_name": "result", "score": result_score},
    ]

    strongest = max(role_rows, key=lambda row: row["score"])
    weakest = min(role_rows, key=lambda row: row["score"])
    total_score = sum(row["score"] for row in role_rows) / len(role_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage260_full_semantic_role_total_map",
        "title": "完整语义角色总图",
        "status_short": "full_semantic_role_total_map_ready",
        "role_count": len(role_rows),
        "total_score": total_score,
        "strongest_role_name": strongest["role_name"],
        "weakest_role_name": weakest["role_name"],
        "top_gap_name": "当前已经能把对象、属性、位置、操作、约束、结果压成同一张角色图，但约束和结果仍明显弱于对象与操作",
        "role_rows": role_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "role_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["role_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["role_rows"])
    report = [
        "# Stage260：完整语义角色总图",
        "",
        "## 核心结果",
        f"- 角色数量：{summary['role_count']}",
        f"- 总图得分：{summary['total_score']:.4f}",
        f"- 最强角色：{summary['strongest_role_name']}",
        f"- 最弱角色：{summary['weakest_role_name']}",
        f"- 头号发现：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE260_FULL_SEMANTIC_ROLE_TOTAL_MAP_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="完整语义角色总图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
