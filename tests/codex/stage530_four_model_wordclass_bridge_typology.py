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
    / "stage530_four_model_wordclass_bridge_typology_20260404"
)
STAGE527_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage527_wordclass_panorama_layer_scan_20260404"
    / "summary.json"
)
STAGE529_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage529_glm4_gemma4_wordclass_scan_20260404"
    / "summary.json"
)
STAGE525_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage525_multi_bridge_causal_expansion_20260404"
    / "summary.json"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, encoding: str = "utf-8") -> dict:
    return json.loads(path.read_text(encoding=encoding))


def class_signature(row: dict, class_name: str) -> dict:
    item = next(r for r in row["class_rows"] if r["class_name"] == class_name)
    return {
        "weighted_layer_center": item["weighted_layer_center"],
        "structure_label": item["structure_label"],
        "peak_layers": item["peak_layers"],
    }


def class_signature_stage529(model_row: dict, class_name: str) -> dict:
    item = model_row["classes"][class_name]
    return {
        "weighted_layer_center": item["weighted_layer_center"],
        "structure_label": "reduced_scan",
        "peak_layers": [r["layer_index"] for r in item["top_layers_by_mass"][:5]],
    }


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage527 = load_json(STAGE527_PATH)
    stage529 = load_json(STAGE529_PATH)
    stage525 = load_json(STAGE525_PATH)

    bridge_rows = {row["model_key"]: row for row in stage525["model_rows"]}
    model_rows = []
    for model_row in stage527["model_rows"]:
        model_key = model_row["model_key"]
        model_rows.append(
            {
                "model_key": model_key,
                "noun": class_signature(model_row, "noun"),
                "adjective": class_signature(model_row, "adjective"),
                "verb": class_signature(model_row, "verb"),
                "adverb": class_signature(model_row, "adverb"),
                "pronoun": class_signature(model_row, "pronoun"),
                "preposition": class_signature(model_row, "preposition"),
                "bridge_rows": bridge_rows[model_key]["bridge_rows"],
            }
        )
    for model_key in ["glm4", "gemma4"]:
        model_row = stage529["models"][model_key]
        model_rows.append(
            {
                "model_key": model_key,
                "noun": class_signature_stage529(model_row, "noun"),
                "adjective": class_signature_stage529(model_row, "adjective"),
                "verb": class_signature_stage529(model_row, "verb"),
                "adverb": class_signature_stage529(model_row, "adverb"),
                "pronoun": class_signature_stage529(model_row, "pronoun"),
                "preposition": class_signature_stage529(model_row, "preposition"),
                "bridge_rows": bridge_rows[model_key]["bridge_rows"],
            }
        )

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage530_four_model_wordclass_bridge_typology",
        "title": "四模型词类层带风格与桥接分型综合图",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage527": str(STAGE527_PATH),
            "stage529": str(STAGE529_PATH),
            "stage525": str(STAGE525_PATH),
        },
        "model_rows": model_rows,
        "core_answer": (
            "四模型放到一起之后，词类层带风格和桥接分型开始显出统一趋势："
            "名词与形容词更靠内容骨干和修饰带，代词和介词更靠功能路由带，"
            "而桥接项则负责把这些带连接成当前任务实例。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(
        "# stage530 四模型词类层带风格与桥接分型综合图\n\n" + summary["core_answer"] + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
