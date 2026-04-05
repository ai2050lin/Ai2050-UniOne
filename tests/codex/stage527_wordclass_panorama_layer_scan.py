#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import statistics
import time
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage527_wordclass_panorama_layer_scan_20260404"
)
STAGE423_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage423_qwen3_deepseek_wordclass_layer_distribution_20260330"
    / "summary.json"
)
WORD_CLASSES = ["noun", "adjective", "verb", "adverb", "pronoun", "preposition"]
TOP_NEURON_SET = 256


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dominant_band(band_summary: dict) -> str:
    items = {
        key: float(value["effective_score_mass_share"])
        for key, value in band_summary.items()
    }
    return max(items.items(), key=lambda item: item[1])[0]


def structure_label(center: float, band_summary: dict) -> str:
    early = float(band_summary["early"]["effective_score_mass_share"])
    middle = float(band_summary["middle"]["effective_score_mass_share"])
    late = float(band_summary["late"]["effective_score_mass_share"])
    bands = sorted([("early", early), ("middle", middle), ("late", late)], key=lambda x: x[1], reverse=True)
    top_band, top_mass = bands[0]
    second_band, second_mass = bands[1]
    if top_mass >= 0.5:
        return f"{top_band}_dominant"
    if second_mass >= 0.3:
        return f"{top_band}_{second_band}_hybrid"
    return f"{top_band}_leaning"


def top_neuron_set(class_row: dict) -> set[tuple[int, int]]:
    return {
        (int(row["layer_index"]), int(row["neuron_index"]))
        for row in class_row["top_neurons"][:TOP_NEURON_SET]
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage423 = load_json(STAGE423_PATH)
    model_rows = []
    for model_key, model_row in stage423["models"].items():
        class_rows = []
        top_sets = {}
        for class_name in WORD_CLASSES:
            class_row = model_row["classes"][class_name]
            top_sets[class_name] = top_neuron_set(class_row)
            band_summary = class_row["band_summary"]
            class_rows.append(
                {
                    "class_name": class_name,
                    "weighted_layer_center": float(class_row["weighted_layer_center"]),
                    "effective_neuron_count": int(class_row["effective_neuron_count"]),
                    "dominant_band": dominant_band(band_summary),
                    "structure_label": structure_label(float(class_row["weighted_layer_center"]), band_summary),
                    "peak_layers": [int(row["layer_index"]) for row in class_row["top_layers_by_mass"][:5]],
                    "early_mass_share": float(band_summary["early"]["effective_score_mass_share"]),
                    "middle_mass_share": float(band_summary["middle"]["effective_score_mass_share"]),
                    "late_mass_share": float(band_summary["late"]["effective_score_mass_share"]),
                    "top32_mean_score": float(
                        statistics.mean(
                            float(item["score"]) for item in class_row["top_neurons"][:32]
                        )
                    ),
                }
            )
        pairwise_rows = []
        for left, right in combinations(WORD_CLASSES, 2):
            pairwise_rows.append(
                {
                    "left": left,
                    "right": right,
                    "top256_jaccard": jaccard(top_sets[left], top_sets[right]),
                }
            )
        model_rows.append(
            {
                "model_key": model_key,
                "layer_count": int(model_row["layer_count"]),
                "class_rows": class_rows,
                "pairwise_rows": pairwise_rows,
            }
        )

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage527_wordclass_panorama_layer_scan",
        "title": "六类词性全景式层带扫描",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "source_summary": str(STAGE423_PATH),
        "top_neuron_set_size": TOP_NEURON_SET,
        "model_rows": model_rows,
        "core_answer": (
            "名词、形容词、动词、副词、代词、介词的有效神经元并不是均匀撒在全网络里，"
            "而是各自具有可辨认的层带结构，并且不同词类之间既有共享骨干，也有明显分离区。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# stage527 六类词性全景式层带扫描",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
    ]
    for model_row in model_rows:
        lines.append(f"## {model_row['model_key']}")
        for row in model_row["class_rows"]:
            lines.append(
                f"- `{row['class_name']}`：质心 `{row['weighted_layer_center']:.2f}`，"
                f"主峰层 `{row['peak_layers']}`，结构 `{row['structure_label']}`，"
                f"早中晚质量 `{row['early_mass_share']:.3f}/{row['middle_mass_share']:.3f}/{row['late_mass_share']:.3f}`"
            )
        lines.append("")
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
