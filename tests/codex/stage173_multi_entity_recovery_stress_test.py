#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage173_multi_entity_recovery_stress_test_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE173_MULTI_ENTITY_RECOVERY_STRESS_TEST_REPORT.md"

STAGE160_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json"
STAGE163_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage163_result_binding_failure_atlas_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> dict:
    s160 = load_json(STAGE160_SUMMARY_PATH)
    s163 = load_json(STAGE163_SUMMARY_PATH)

    case_count = int(s163["case_count"])
    stable_ratio = float(s163["stable_binding_count"]) / float(case_count)
    fragile_ratio = float(s163["fragile_binding_count"]) / float(case_count)
    hard_ratio = float(s163["hard_failure_count"]) / float(case_count)
    soft_ratio = float(s163["soft_failure_count"]) / float(case_count)
    repair_rescue = float(s160["positive_repair_rate"])

    adversarial_rows = [row for row in s163["atlas_rows"] if row["difficulty"] == "adversarial"]
    adversarial_stable = sum(1 for row in adversarial_rows if row["failure_label"] == "stable_binding")
    adversarial_fragile = sum(1 for row in adversarial_rows if row["failure_label"] == "fragile_binding")
    adversarial_rescue_ratio = (
        float(adversarial_stable) + 0.5 * float(adversarial_fragile)
    ) / float(len(adversarial_rows))

    raw_stability = stable_ratio + 0.5 * fragile_ratio
    stress_survival_score = clamp01(
        0.45 * raw_stability + 0.35 * repair_rescue + 0.20 * adversarial_rescue_ratio
    )

    family_rows = []
    for row in s163["family_rows"]:
        family_name = str(row["family_name"])
        severity = float(row["mean_failure_severity"])
        hard_misbinding_rate = float(row["hard_misbinding_rate"])
        repair_family = next(
            item for item in s160["family_rows"] if str(item["family_name"]) == family_name
        )
        recovery_tolerance = clamp01(
            0.45 * float(repair_family["positive_repair_rate"])
            + 0.35 * (1.0 - min(1.0, severity / 4.0))
            + 0.20 * (1.0 - hard_misbinding_rate)
        )
        family_rows.append(
            {
                "family_name": family_name,
                "recovery_tolerance": recovery_tolerance,
                "mean_failure_severity": severity,
                "hard_misbinding_rate": hard_misbinding_rate,
            }
        )
    ranked_rows = sorted(family_rows, key=lambda row: float(row["recovery_tolerance"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage173_multi_entity_recovery_stress_test",
        "title": "多实体回收压力测试",
        "status_short": "multi_entity_recovery_stress_ready",
        "case_count": case_count,
        "stress_survival_score": stress_survival_score,
        "raw_stability": raw_stability,
        "repair_rescue_ratio": repair_rescue,
        "adversarial_rescue_ratio": adversarial_rescue_ratio,
        "weakest_family_name": str(ranked_rows[0]["family_name"]),
        "strongest_family_name": str(ranked_rows[-1]["family_name"]),
        "family_rows": family_rows,
        "hard_ratio": hard_ratio,
        "soft_ratio": soft_ratio,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage173: 多实体回收压力测试",
        "",
        "## 核心结果",
        f"- 多实体压力生存分数: {summary['stress_survival_score']:.4f}",
        f"- 最弱家族: {summary['weakest_family_name']}",
        f"- 最强家族: {summary['strongest_family_name']}",
        f"- 原始稳定度: {summary['raw_stability']:.4f}",
        f"- 修复救援率: {summary['repair_rescue_ratio']:.4f}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="多实体回收压力测试")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
