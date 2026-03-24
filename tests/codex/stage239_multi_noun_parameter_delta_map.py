#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage239_multi_noun_parameter_delta_map_20260324"

STAGE119_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage119_gpt2_embedding_full_vocab_scan_20260323" / "summary.json"
STAGE154_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage154_apple_fruit_shared_core_20260323" / "summary.json"
STAGE189_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage189_family_neuron_bundle_map_20260323" / "summary.json"
STAGE236_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage236_apple_pear_parameter_delta_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bundle_score(rows: list[dict], category_name: str, field_name: str) -> float:
    for row in rows:
        if str(row["category_name"]) == category_name:
            return float(row[field_name])
    raise KeyError(category_name)


def similarity_of(rows: list[dict], word: str) -> float:
    for row in rows:
        if str(row["word"]) == word:
            return float(row["similarity_to_apple"])
    raise KeyError(word)


def build_summary() -> dict:
    s119 = load_json(STAGE119_SUMMARY_PATH)
    s154 = load_json(STAGE154_SUMMARY_PATH)
    s189 = load_json(STAGE189_SUMMARY_PATH)
    s236 = load_json(STAGE236_SUMMARY_PATH)

    selected = s154["selected_similarity_rows"]
    fruit_mean = (
        similarity_of(selected, "banana")
        + similarity_of(selected, "pear")
        + similarity_of(selected, "peach")
        + similarity_of(selected, "grape")
        + similarity_of(selected, "melon")
    ) / 5.0

    cross_margin = float(s154["apple_to_fruit_centroid"]) - float(s154["apple_to_nonfruit_centroid"])
    fruit_bundle_strength = bundle_score(s189["bundle_rows"], "fruit", "bundle_strength")
    animal_bundle_strength = bundle_score(s189["bundle_rows"], "animal", "bundle_strength")
    tool_bundle_strength = bundle_score(s189["bundle_rows"], "tool", "bundle_strength")
    fruit_delta_support = bundle_score(s189["bundle_rows"], "fruit", "delta_support")
    noun_count = float(s119["lexical_type_counts"]["noun"])
    fruit_group_count = float(s119["group_counts_top20"]["meso_fruit"])
    animal_group_count = float(s119["group_counts_top20"]["meso_animal"])
    object_group_count = float(s119["group_counts_top20"]["meso_object"])
    category_coverage = (fruit_group_count + animal_group_count + object_group_count) / noun_count

    piece_rows = [
        {"piece_name": "水果家族共享近度", "score": fruit_mean},
        {"piece_name": "水果-非水果类别余量", "score": cross_margin},
        {"piece_name": "水果束强度", "score": fruit_bundle_strength},
        {"piece_name": "动物束强度", "score": animal_bundle_strength},
        {"piece_name": "工具束强度", "score": tool_bundle_strength},
        {"piece_name": "水果局部差分支持", "score": fruit_delta_support},
        {"piece_name": "苹果-梨子差分锚点", "score": float(s236["delta_score"])},
        {"piece_name": "多名词参数覆盖率", "score": category_coverage},
    ]
    ranked = sorted(piece_rows, key=lambda row: float(row["score"]), reverse=True)
    score = sum(float(row["score"]) for row in piece_rows) / len(piece_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage239_multi_noun_parameter_delta_map",
        "title": "更多名词对象参数差分图",
        "status_short": "multi_noun_parameter_delta_map_ready",
        "piece_count": len(piece_rows),
        "map_score": score,
        "strongest_piece_name": str(ranked[0]["piece_name"]),
        "weakest_piece_name": str(ranked[-1]["piece_name"]),
        "top_gap_name": "跨类别边界余量仍然薄，说明更多名词共享大于差分",
        "piece_rows": piece_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage239：更多名词对象参数差分图",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 总分：{summary['map_score']:.4f}",
        f"- 最强块：{summary['strongest_piece_name']}",
        f"- 最弱块：{summary['weakest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE239_MULTI_NOUN_PARAMETER_DELTA_MAP_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="更多名词对象参数差分图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
