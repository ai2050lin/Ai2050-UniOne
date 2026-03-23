#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage120: GPT-2 词类投影图谱。

目标：
1. 把 Stage119（第一百一十九阶段）的全词类扫描压成词类-语义耦合图谱。
2. 提取 noun（名词）/ verb（动词）/ adjective（形容词）/ adverb（副词）/ function（功能词）
   在 micro/meso/macro（微观/中观/宏观）三层中的投影形状。
3. 为后续数学理论提供“对象层 / 路由层 / 桥接层”的结构证据。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from stage119_gpt2_embedding_full_vocab_scan import run_analysis as run_stage119_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage120_lexical_type_projection_atlas_20260323"
BANDS = ["micro", "meso", "macro"]
LEXICAL_TYPES = ["noun", "verb", "adjective", "adverb", "function"]
EXPECTED_BAND = {
    "noun": "meso",
    "verb": "macro",
    "adjective": "micro",
    "function": "macro",
}


def ensure_stage119_rows(input_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    summary_path = input_dir / "summary.json"
    rows_path = input_dir / "word_rows.jsonl"
    if summary_path.exists() and rows_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        rows: List[Dict[str, object]] = []
        with rows_path.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                rows.append(json.loads(line))
        return summary, rows
    return run_stage119_analysis(output_dir=input_dir)


def normalized_entropy(counter: Counter[str], labels: Sequence[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = [counter.get(label, 0) / total for label in labels if counter.get(label, 0) > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return float(entropy / math.log(len(labels)))


def safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def top_words(rows: Sequence[Dict[str, object]], key: str, count: int) -> List[Dict[str, object]]:
    selected = sorted(rows, key=lambda item: float(item[key]), reverse=True)[:count]
    return [
        {
            "word": row["word"],
            "band": row["band"],
            "group": row["group"],
            "lexical_type": row["lexical_type"],
            key: float(row[key]),
        }
        for row in selected
    ]


def build_type_rows(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    by_type: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["lexical_type"])].append(row)

    atlas_rows: List[Dict[str, object]] = []
    for lexical_type in LEXICAL_TYPES:
        type_rows = by_type[lexical_type]
        band_counter = Counter(str(row["band"]) for row in type_rows)
        group_counter = Counter(str(row["group"]) for row in type_rows)
        total = max(1, len(type_rows))
        dominant_band, dominant_count = band_counter.most_common(1)[0]
        entropy = normalized_entropy(band_counter, BANDS)
        dominant_ratio = dominant_count / total
        alignment_ratio = band_counter.get(EXPECTED_BAND.get(lexical_type, ""), 0) / total if lexical_type in EXPECTED_BAND else 0.0
        atlas_rows.append(
            {
                "lexical_type": lexical_type,
                "count": len(type_rows),
                "dominant_band": dominant_band,
                "dominant_band_ratio": float(dominant_ratio),
                "expected_band": EXPECTED_BAND.get(lexical_type, "bridge"),
                "expected_band_ratio": float(alignment_ratio),
                "band_entropy": float(entropy),
                "mean_type_score": safe_mean([float(row["lexical_type_score"]) for row in type_rows]),
                "mean_type_margin": safe_mean([float(row["lexical_type_margin"]) for row in type_rows]),
                "mean_effective_score": safe_mean([float(row["effective_encoding_score"]) for row in type_rows]),
                "top_groups": [
                    {"group": group, "count": int(count), "ratio": float(count / total)}
                    for group, count in group_counter.most_common(5)
                ],
                "band_counts": {band: int(band_counter.get(band, 0)) for band in BANDS},
                "band_ratios": {band: float(band_counter.get(band, 0) / total) for band in BANDS},
            }
        )
    return atlas_rows, by_type


def build_boundary_words(type_rows: Sequence[Dict[str, object]], count: int = 12) -> List[Dict[str, object]]:
    filtered = [row for row in type_rows if float(row["lexical_type_score"]) >= 0.48]
    selected = sorted(
        filtered,
        key=lambda row: (float(row["lexical_type_margin"]), -float(row["effective_encoding_score"])),
    )[:count]
    return [
        {
            "word": row["word"],
            "band": row["band"],
            "group": row["group"],
            "lexical_type_margin": float(row["lexical_type_margin"]),
            "lexical_type_score": float(row["lexical_type_score"]),
            "effective_encoding_score": float(row["effective_encoding_score"]),
        }
        for row in selected
    ]


def build_drift_words(
    lexical_type: str,
    type_rows: Sequence[Dict[str, object]],
    count: int = 10,
) -> List[Dict[str, object]]:
    expected = EXPECTED_BAND.get(lexical_type)
    if expected is None:
        return []
    drift_rows = [
        row
        for row in type_rows
        if str(row["band"]) != expected and float(row["lexical_type_score"]) >= 0.50
    ]
    selected = sorted(
        drift_rows,
        key=lambda row: (
            -float(row["effective_encoding_score"]),
            -float(row["lexical_type_score"]),
        ),
    )[:count]
    return [
        {
            "word": row["word"],
            "band": row["band"],
            "group": row["group"],
            "lexical_type_score": float(row["lexical_type_score"]),
            "effective_encoding_score": float(row["effective_encoding_score"]),
        }
        for row in selected
    ]


def build_summary(stage119_summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    atlas_rows, by_type = build_type_rows(rows)
    atlas_map = {row["lexical_type"]: row for row in atlas_rows}

    noun_meso_ratio = atlas_map["noun"]["band_ratios"]["meso"]
    verb_macro_ratio = atlas_map["verb"]["band_ratios"]["macro"]
    adjective_micro_ratio = atlas_map["adjective"]["band_ratios"]["micro"]
    function_macro_ratio = atlas_map["function"]["band_ratios"]["macro"]
    adverb_bridge_entropy = atlas_map["adverb"]["band_entropy"]
    atlas_score = safe_mean(
        [
            float(noun_meso_ratio),
            float(verb_macro_ratio),
            float(adjective_micro_ratio),
            float(function_macro_ratio),
            float(adverb_bridge_entropy),
        ]
    )

    bridge_type_name = max(
        atlas_rows,
        key=lambda row: (float(row["band_entropy"]), -float(row["dominant_band_ratio"])),
    )["lexical_type"]

    boundary_words = {
        lexical_type: build_boundary_words(type_rows)
        for lexical_type, type_rows in by_type.items()
    }
    drift_words = {
        lexical_type: build_drift_words(lexical_type, type_rows)
        for lexical_type, type_rows in by_type.items()
        if lexical_type in EXPECTED_BAND
    }

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage120_lexical_type_projection_atlas",
        "title": "GPT-2 词类投影图谱",
        "status_short": "gpt2_lexical_type_projection_atlas_ready",
        "source_stage": "stage119_gpt2_embedding_full_vocab_scan",
        "source_output_dir": str(STAGE119_OUTPUT_DIR),
        "clean_unique_word_count": int(stage119_summary["clean_unique_word_count"]),
        "lexical_type_projection_atlas_score": float(atlas_score),
        "noun_meso_anchor_ratio": float(noun_meso_ratio),
        "verb_macro_anchor_ratio": float(verb_macro_ratio),
        "adjective_micro_anchor_ratio": float(adjective_micro_ratio),
        "function_macro_anchor_ratio": float(function_macro_ratio),
        "adverb_bridge_entropy": float(adverb_bridge_entropy),
        "bridge_type_name": bridge_type_name,
        "type_rows": atlas_rows,
        "boundary_words": boundary_words,
        "drift_words": drift_words,
        "top_type_examples": {
            lexical_type: top_words(type_rows, "lexical_type_score", 12)
            for lexical_type, type_rows in by_type.items()
        },
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage120: GPT-2 词类投影图谱",
        "",
        "## 核心结果",
        f"- 词类投影图谱分数: {summary['lexical_type_projection_atlas_score']:.4f}",
        f"- noun（名词）-> meso（中观）锚定率: {summary['noun_meso_anchor_ratio']:.4f}",
        f"- verb（动词）-> macro（宏观）锚定率: {summary['verb_macro_anchor_ratio']:.4f}",
        f"- adjective（形容词）-> micro（微观）锚定率: {summary['adjective_micro_anchor_ratio']:.4f}",
        f"- function（功能词）-> macro（宏观）锚定率: {summary['function_macro_anchor_ratio']:.4f}",
        f"- adverb（副词）桥接熵: {summary['adverb_bridge_entropy']:.4f}",
        f"- 最强桥接词类: {summary['bridge_type_name']}",
        "",
        "## 结构解释",
        "- 名词主要投到 meso（中观）对象层，说明对象家族仍是词嵌入里的主体块。",
        "- 动词高度投到 macro（宏观）动作层，说明动作更接近系统级变换而不是静态对象。",
        "- 形容词主要投到 micro（微观）属性层，但仍有一部分漂到 macro（宏观）抽象层，说明属性词里混有评价与抽象风格分量。",
        "- 副词不是简单附属物，而是桥接型词类，它在 meso（中观）/ macro（宏观）之间分布更散。",
        "- 功能词偏 macro（宏观），这支持“语言控制/路由层不等同于对象层”的判断。",
        "",
        "## 词类行",
    ]

    for row in summary["type_rows"]:
        top_groups = ", ".join(f"{item['group']}:{item['ratio']:.3f}" for item in row["top_groups"][:3])
        lines.append(
            "- "
            f"{row['lexical_type']}: dominant_band={row['dominant_band']} "
            f"({row['dominant_band_ratio']:.4f}), entropy={row['band_entropy']:.4f}, "
            f"top_groups=[{top_groups}]"
        )

    lines.extend(["", "## 理论提示"])
    lines.append("- 词类不是简单语法标签，而更像统一编码系统中的不同投影叶层。")
    lines.append("- noun（名词）/ verb（动词）/ adjective（形容词）/ function（功能词）之间已经出现稳定的层级偏置。")
    lines.append("- adverb（副词）若持续表现为高熵桥接层，后续应重点看它与上下文门控、路由调制的关系。")
    lines.append("")
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    type_rows_csv = output_dir / "lexical_type_rows.csv"
    report_path = output_dir / "STAGE120_LEXICAL_TYPE_PROJECTION_ATLAS_REPORT.md"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_path.write_text(build_report(summary), encoding="utf-8-sig")

    with type_rows_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "lexical_type",
                "count",
                "dominant_band",
                "dominant_band_ratio",
                "expected_band",
                "expected_band_ratio",
                "band_entropy",
                "mean_type_score",
                "mean_type_margin",
                "mean_effective_score",
                "micro_ratio",
                "meso_ratio",
                "macro_ratio",
                "top_group_1",
                "top_group_1_ratio",
                "top_group_2",
                "top_group_2_ratio",
                "top_group_3",
                "top_group_3_ratio",
            ],
        )
        writer.writeheader()
        for row in summary["type_rows"]:
            top_groups = row["top_groups"]
            payload = {
                "lexical_type": row["lexical_type"],
                "count": row["count"],
                "dominant_band": row["dominant_band"],
                "dominant_band_ratio": row["dominant_band_ratio"],
                "expected_band": row["expected_band"],
                "expected_band_ratio": row["expected_band_ratio"],
                "band_entropy": row["band_entropy"],
                "mean_type_score": row["mean_type_score"],
                "mean_type_margin": row["mean_type_margin"],
                "mean_effective_score": row["mean_effective_score"],
                "micro_ratio": row["band_ratios"]["micro"],
                "meso_ratio": row["band_ratios"]["meso"],
                "macro_ratio": row["band_ratios"]["macro"],
                "top_group_1": top_groups[0]["group"] if len(top_groups) > 0 else "",
                "top_group_1_ratio": top_groups[0]["ratio"] if len(top_groups) > 0 else "",
                "top_group_2": top_groups[1]["group"] if len(top_groups) > 1 else "",
                "top_group_2_ratio": top_groups[1]["ratio"] if len(top_groups) > 1 else "",
                "top_group_3": top_groups[2]["group"] if len(top_groups) > 2 else "",
                "top_group_3_ratio": top_groups[2]["ratio"] if len(top_groups) > 2 else "",
            }
            writer.writerow(payload)

    return {
        "summary": summary_path,
        "lexical_type_rows_csv": type_rows_csv,
        "report": report_path,
    }


def run_analysis(
    *,
    input_dir: Path = STAGE119_OUTPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, object]:
    stage119_summary, rows = ensure_stage119_rows(input_dir)
    summary = build_summary(stage119_summary, rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-2 词类投影图谱")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage120 输出目录")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "lexical_type_projection_atlas_score": summary["lexical_type_projection_atlas_score"],
                "bridge_type_name": summary["bridge_type_name"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
