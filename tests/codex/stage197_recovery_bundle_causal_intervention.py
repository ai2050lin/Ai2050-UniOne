#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage197_recovery_bundle_causal_intervention_20260323"

STAGE186_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage186_result_binding_system_expansion_20260323" / "summary.json"
STAGE188_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage188_apple_neuron_role_card_20260323" / "summary.json"
STAGE194_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage194_bottom_block_intervention_priority_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_intervention(score: float) -> str:
    if score < 0.4:
        return "强干预"
    if score < 0.6:
        return "中干预"
    return "观察优先"


def find_score(rows: list[dict], key: str, value: str, score_key: str = "score") -> float:
    for row in rows:
        if str(row[key]) == value:
            return float(row[score_key])
    raise KeyError(value)


def build_summary() -> dict:
    s186 = load_json(STAGE186_SUMMARY_PATH)
    s188 = load_json(STAGE188_SUMMARY_PATH)
    s194 = load_json(STAGE194_SUMMARY_PATH)

    native_binding = find_score(s186["piece_rows"], "piece_name", "原生绑定")
    recovery_bundle = find_score(s188["role_rows"], "role_name", "回收束")
    repair_bundle = find_score(s188["role_rows"], "role_name", "修复束")
    intervention_priority = find_score(s194["target_rows"], "target_name", "回收束")

    target_rows = [
        {
            "target_name": "回收束",
            "score": recovery_bundle,
            "priority_level": "一级",
            "intervention_mode": classify_intervention(recovery_bundle),
        },
        {
            "target_name": "原生绑定",
            "score": native_binding,
            "priority_level": "一级",
            "intervention_mode": classify_intervention(native_binding),
        },
        {
            "target_name": "修复束",
            "score": repair_bundle,
            "priority_level": "三级",
            "intervention_mode": classify_intervention(repair_bundle),
        },
    ]
    causal_intervention_score = (1.0 - recovery_bundle) * 0.45 + (1.0 - native_binding) * 0.45 + intervention_priority * 0.10
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage197_recovery_bundle_causal_intervention",
        "title": "回收束因果干预块",
        "status_short": "recovery_bundle_causal_intervention_ready",
        "target_count": len(target_rows),
        "top_target_name": "回收束",
        "second_target_name": "原生绑定",
        "causal_intervention_score": causal_intervention_score,
        "target_rows": target_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage197：回收束因果干预块",
        "",
        "## 核心结果",
        f"- 目标数量：{summary['target_count']}",
        f"- 头号目标：{summary['top_target_name']}",
        f"- 第二目标：{summary['second_target_name']}",
        f"- 因果干预紧迫度：{summary['causal_intervention_score']:.4f}",
    ]
    (output_dir / "STAGE197_RECOVERY_BUNDLE_CAUSAL_INTERVENTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="回收束因果干预块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
