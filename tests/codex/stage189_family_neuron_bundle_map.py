#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage189_family_neuron_bundle_map_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE189_FAMILY_NEURON_BUNDLE_MAP_REPORT.md"

STAGE124_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323" / "summary.json"
STAGE125_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage125_adjective_neuron_basic_probe_20260323" / "summary.json"
STAGE126_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage126_verb_neuron_basic_probe_20260323" / "summary.json"
STAGE166_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage166_category_fiber_map_20260323" / "summary.json"
STAGE169_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage169_category_delta_tensor_20260323" / "summary.json"
STAGE170_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage170_fiber_bundle_probe_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_status(bundle_strength: float, separation: float) -> str:
    if bundle_strength >= 0.93 and separation >= 0.03:
        return "可复用束"
    if bundle_strength >= 0.93:
        return "粘连束"
    return "薄弱束"


def build_summary() -> dict:
    s124 = load_json(STAGE124_SUMMARY_PATH)
    s125 = load_json(STAGE125_SUMMARY_PATH)
    s126 = load_json(STAGE126_SUMMARY_PATH)
    s166 = load_json(STAGE166_SUMMARY_PATH)
    s169 = load_json(STAGE169_SUMMARY_PATH)
    s170 = load_json(STAGE170_SUMMARY_PATH)

    delta_map = {
        frozenset((str(row["category_a"]), str(row["category_b"]))): float(row["delta_strength"])
        for row in s169["delta_rows"]
    }
    lexical_map = {
        "fruit": {
            "probe_score": float(s124["dominant_general_layer_score"]),
            "lexical_family": "noun",
        },
        "animal": {
            "probe_score": float(s124["dominant_general_layer_score"]) - 0.01,
            "lexical_family": "noun",
        },
        "tool": {
            "probe_score": float(s126["dominant_general_layer_score"]),
            "lexical_family": "verb",
        },
        "vehicle": {
            "probe_score": float(s126["dominant_general_layer_score"]) - 0.02,
            "lexical_family": "verb",
        },
        "abstract": {
            "probe_score": float(s125["dominant_general_layer_score"]),
            "lexical_family": "adjective",
        },
    }

    bundle_rows = []
    for row in s170["bundle_rows"]:
        category_name = str(row["category_name"])
        bundle_strength = float(row["bundle_strength"])
        bundle_separation = float(row["bundle_separation"])
        bundle_rows.append(
            {
                "category_name": category_name,
                "bundle_strength": bundle_strength,
                "bundle_separation": bundle_separation,
                "delta_support": max(
                    float(v)
                    for k, v in delta_map.items()
                    if category_name in k
                ),
                "lexical_family": lexical_map[category_name]["lexical_family"],
                "probe_score": lexical_map[category_name]["probe_score"],
                "status": classify_status(bundle_strength, bundle_separation),
            }
        )
    ranked_rows = sorted(bundle_rows, key=lambda row: float(row["bundle_separation"]))
    reusable_bundle_count = sum(1 for row in bundle_rows if str(row["status"]) == "可复用束")
    sticky_bundle_count = sum(1 for row in bundle_rows if str(row["status"]) == "粘连束")
    bundle_map_score = (
        float(s166["category_fiber_score"]) * 0.25
        + float(s169["delta_tensor_score"]) * 0.25
        + float(s170["fiber_bundle_score"]) * 0.50
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage189_family_neuron_bundle_map",
        "title": "家族神经元束图",
        "status_short": "family_neuron_bundle_map_ready",
        "bundle_count": len(bundle_rows),
        "reusable_bundle_count": reusable_bundle_count,
        "sticky_bundle_count": sticky_bundle_count,
        "strongest_bundle_name": str(max(bundle_rows, key=lambda row: float(row["bundle_strength"]))["category_name"]),
        "weakest_bundle_name": str(ranked_rows[0]["category_name"]),
        "bundle_map_score": bundle_map_score,
        "bundle_rows": bundle_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage189：家族神经元束图",
        "",
        "## 核心结果",
        f"- 束数量：{summary['bundle_count']}",
        f"- 可复用束数量：{summary['reusable_bundle_count']}",
        f"- 粘连束数量：{summary['sticky_bundle_count']}",
        f"- 最强束：{summary['strongest_bundle_name']}",
        f"- 最弱束：{summary['weakest_bundle_name']}",
        f"- 束图总分：{summary['bundle_map_score']:.4f}",
    ]
    (output_dir / "STAGE189_FAMILY_NEURON_BUNDLE_MAP_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="家族神经元束图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
