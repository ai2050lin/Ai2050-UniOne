#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage528_wordclass_encoding_structure_synthesis_20260404"
)
STAGE527_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage527_wordclass_panorama_layer_scan_20260404"
    / "summary.json"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pairwise_lookup(rows: list[dict], left: str, right: str) -> float:
    for row in rows:
        if {row["left"], row["right"]} == {left, right}:
            return float(row["top256_jaccard"])
    return 0.0


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage527 = load_json(STAGE527_PATH)

    model_rows = []
    for model_row in stage527["model_rows"]:
        class_map = {row["class_name"]: row for row in model_row["class_rows"]}
        pair_rows = model_row["pairwise_rows"]
        model_rows.append(
            {
                "model_key": model_row["model_key"],
                "noun_signature": class_map["noun"],
                "adjective_signature": class_map["adjective"],
                "verb_signature": class_map["verb"],
                "adverb_signature": class_map["adverb"],
                "pronoun_signature": class_map["pronoun"],
                "preposition_signature": class_map["preposition"],
                "noun_verb_overlap": pairwise_lookup(pair_rows, "noun", "verb"),
                "noun_adjective_overlap": pairwise_lookup(pair_rows, "noun", "adjective"),
                "pronoun_preposition_overlap": pairwise_lookup(pair_rows, "pronoun", "preposition"),
                "adverb_verb_overlap": pairwise_lookup(pair_rows, "adverb", "verb"),
            }
        )

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage528_wordclass_encoding_structure_synthesis",
        "title": "六类词性编码结构综合摘要",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "source_summary": str(STAGE527_PATH),
        "model_rows": model_rows,
        "core_answer": (
            "六类词性现在已经能被看成六种不同的编码层带风格："
            "名词更像对象骨干带，形容词更像修饰双峰带，动词更像早中层动作带，"
            "副词更像中晚层调制带，代词和介词更像功能路由带。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# stage528 六类词性编码结构综合摘要",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
    ]
    for row in model_rows:
        lines.append(f"## {row['model_key']}")
        lines.append(
            f"- 名词：`{row['noun_signature']['structure_label']}`，质心 `{row['noun_signature']['weighted_layer_center']:.2f}`，主峰 `{row['noun_signature']['peak_layers']}`"
        )
        lines.append(
            f"- 形容词：`{row['adjective_signature']['structure_label']}`，质心 `{row['adjective_signature']['weighted_layer_center']:.2f}`，主峰 `{row['adjective_signature']['peak_layers']}`"
        )
        lines.append(
            f"- 动词：`{row['verb_signature']['structure_label']}`，质心 `{row['verb_signature']['weighted_layer_center']:.2f}`，主峰 `{row['verb_signature']['peak_layers']}`"
        )
        lines.append(
            f"- 副词：`{row['adverb_signature']['structure_label']}`，质心 `{row['adverb_signature']['weighted_layer_center']:.2f}`，主峰 `{row['adverb_signature']['peak_layers']}`"
        )
        lines.append(
            f"- 代词：`{row['pronoun_signature']['structure_label']}`，质心 `{row['pronoun_signature']['weighted_layer_center']:.2f}`，主峰 `{row['pronoun_signature']['peak_layers']}`"
        )
        lines.append(
            f"- 介词：`{row['preposition_signature']['structure_label']}`，质心 `{row['preposition_signature']['weighted_layer_center']:.2f}`，主峰 `{row['preposition_signature']['peak_layers']}`"
        )
        lines.append(
            f"- 词类重叠：名词-动词 `{row['noun_verb_overlap']:.4f}`，名词-形容词 `{row['noun_adjective_overlap']:.4f}`，"
            f"代词-介词 `{row['pronoun_preposition_overlap']:.4f}`，副词-动词 `{row['adverb_verb_overlap']:.4f}`"
        )
        lines.append("")
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
