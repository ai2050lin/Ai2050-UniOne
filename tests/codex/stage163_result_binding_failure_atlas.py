#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage163_result_binding_failure_atlas_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE163_RESULT_BINDING_FAILURE_ATLAS_REPORT.md"
STAGE158_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage158_apple_result_binding_probe_20260323" / "summary.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def failure_label(binding_shift: float) -> str:
    if binding_shift <= -2.0:
        return "hard_misbinding"
    if binding_shift < 0.0:
        return "soft_misbinding"
    if binding_shift < 1.0:
        return "fragile_binding"
    return "stable_binding"


def build_summary() -> Dict[str, object]:
    stage158 = load_json(STAGE158_SUMMARY_PATH)
    pair_rows = stage158["pair_rows"]
    atlas_rows: List[Dict[str, object]] = []
    for row in pair_rows:
        shift = float(row["binding_shift"])
        atlas_rows.append(
            {
                "family_name": str(row["family_name"]),
                "difficulty": str(row["difficulty"]),
                "binding_shift": shift,
                "failure_label": failure_label(shift),
                "failure_severity": max(0.0, -shift),
            }
        )

    family_rows: List[Dict[str, object]] = []
    family_names = sorted({row["family_name"] for row in atlas_rows})
    for family_name in family_names:
        subset = [row for row in atlas_rows if row["family_name"] == family_name]
        label_counts: Dict[str, int] = {}
        for row in subset:
            label = str(row["failure_label"])
            label_counts[label] = label_counts.get(label, 0) + 1
        family_rows.append(
            {
                "family_name": family_name,
                "mean_binding_shift": mean(float(row["binding_shift"]) for row in subset),
                "mean_failure_severity": mean(float(row["failure_severity"]) for row in subset),
                "hard_misbinding_rate": label_counts.get("hard_misbinding", 0) / len(subset),
                "label_counts": label_counts,
            }
        )
    family_rows.sort(key=lambda row: float(row["mean_failure_severity"]), reverse=True)

    difficulty_rows: List[Dict[str, object]] = []
    difficulty_names = sorted({row["difficulty"] for row in atlas_rows})
    for difficulty in difficulty_names:
        subset = [row for row in atlas_rows if row["difficulty"] == difficulty]
        difficulty_rows.append(
            {
                "difficulty": difficulty,
                "mean_binding_shift": mean(float(row["binding_shift"]) for row in subset),
                "mean_failure_severity": mean(float(row["failure_severity"]) for row in subset),
            }
        )
    difficulty_rows.sort(key=lambda row: float(row["mean_failure_severity"]), reverse=True)

    hard_failure_count = sum(1 for row in atlas_rows if row["failure_label"] == "hard_misbinding")
    soft_failure_count = sum(1 for row in atlas_rows if row["failure_label"] == "soft_misbinding")
    fragile_count = sum(1 for row in atlas_rows if row["failure_label"] == "fragile_binding")
    stable_count = sum(1 for row in atlas_rows if row["failure_label"] == "stable_binding")

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage163_result_binding_failure_atlas",
        "title": "结果绑定失效图谱",
        "status_short": "result_binding_failure_atlas_ready",
        "case_count": len(atlas_rows),
        "hard_failure_count": hard_failure_count,
        "soft_failure_count": soft_failure_count,
        "fragile_binding_count": fragile_count,
        "stable_binding_count": stable_count,
        "worst_family_name": str(family_rows[0]["family_name"]),
        "worst_difficulty_name": str(difficulty_rows[0]["difficulty"]),
        "family_rows": family_rows,
        "difficulty_rows": difficulty_rows,
        "atlas_rows": atlas_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage163: 结果绑定失效图谱",
        "",
        "## 核心结果",
        f"- 案例数: {summary['case_count']}",
        f"- 强失效数: {summary['hard_failure_count']}",
        f"- 软失效数: {summary['soft_failure_count']}",
        f"- 脆弱绑定数: {summary['fragile_binding_count']}",
        f"- 稳定绑定数: {summary['stable_binding_count']}",
        f"- 最坏家族: {summary['worst_family_name']}",
        f"- 最坏难度: {summary['worst_difficulty_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="结果绑定失效图谱")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
