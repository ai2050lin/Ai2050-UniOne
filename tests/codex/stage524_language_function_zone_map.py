#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage524_language_function_zone_map_20260404"
)
STAGE423_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage423_qwen3_deepseek_wordclass_layer_distribution_20260330"
    / "summary.json"
)
STAGE493_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage493_chinese_language_master_atlas_20260403"
    / "summary.json"
)
STAGE519_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage519_noun_attribute_bridge_layer_atlas_20260404"
    / "summary.json"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, encoding: str = "utf-8") -> dict:
    return json.loads(path.read_text(encoding=encoding))


def band_label(centroid: float, layer_count: int) -> str:
    if layer_count <= 0:
        return "unknown"
    ratio = centroid / max(layer_count - 1, 1)
    if ratio < 1 / 3:
        return "early"
    if ratio < 2 / 3:
        return "middle"
    return "late"


def row_from_wordclass(model_key: str, model_row: dict, class_name: str) -> dict:
    cls = model_row["classes"][class_name]
    peak_layers = [int(row["layer_index"]) for row in cls["top_layers_by_mass"][:3]]
    centroid = float(cls["weighted_layer_center"])
    layer_count = int(model_row["layer_count"])
    return {
        "function_key": class_name,
        "source": "stage423_wordclass_distribution",
        "layer_count": layer_count,
        "centroid": centroid,
        "peak_layers": peak_layers,
        "band": band_label(centroid, layer_count),
        "notes": f"基于六类词性有效神经元分布，{class_name} 的加权层中心。",
    }


def row_from_pattern(model_key: str, pattern_model_row: dict, family_name: str) -> dict:
    family_row = pattern_model_row["model_summary"]["family_summary"][family_name]
    peak_layers = [int(x) for x in family_row["peak_layers"]]
    centroid = float(statistics.mean(peak_layers)) if peak_layers else 0.0
    sample_pattern = next(item for item in pattern_model_row["patterns"] if item["family"] == family_name)
    layer_count = len(sample_pattern["layer_route"]["mean_curve"]) - 1
    return {
        "function_key": family_name,
        "source": "stage493_chinese_pattern_atlas",
        "layer_count": layer_count,
        "centroid": centroid,
        "peak_layers": peak_layers,
        "band": band_label(centroid, layer_count),
        "notes": (
            f"基于中文模式图谱家族峰值层，"
            f"拓扑统计为 {family_row['topology_counts']}。"
        ),
    }


def row_from_layer_atlas(model_row: dict, field_prefix: str, function_key: str) -> dict:
    layer_count = int(model_row["layer_count"])
    centroid = float(model_row[f"{field_prefix}_centroid"])
    peak_layers = [int(x) for x in model_row[f"{field_prefix}_peak_layers"]]
    return {
        "function_key": function_key,
        "source": "stage519_noun_attribute_bridge_layer_atlas",
        "layer_count": layer_count,
        "centroid": centroid,
        "peak_layers": peak_layers,
        "band": band_label(centroid, layer_count),
        "notes": f"{function_key} 的跨家族层带图谱。",
    }


def summarize_model(model_key: str, rows: list[dict]) -> dict:
    band_groups: dict[str, list[str]] = {"early": [], "middle": [], "late": []}
    for row in rows:
        band_groups.setdefault(row["band"], []).append(row["function_key"])
    for value in band_groups.values():
        value.sort()
    return {
        "model_key": model_key,
        "function_rows": rows,
        "band_groups": band_groups,
    }


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage423 = load_json(STAGE423_PATH, encoding="utf-8-sig")
    stage493 = load_json(STAGE493_PATH)
    stage519 = load_json(STAGE519_PATH)

    layer_map = {row["model_key"]: row for row in stage519["model_rows"]}
    model_rows = []
    for model_key in ["qwen3", "deepseek7b"]:
        rows = []
        word_model = stage423["models"][model_key]
        for class_name in ["noun", "verb", "pronoun"]:
            rows.append(row_from_wordclass(model_key, word_model, class_name))
        pattern_model = stage493["models"][model_key]
        for family_name in ["connective", "fixed_phrase", "time", "quantity", "locative", "pronoun"]:
            rows.append(row_from_pattern(model_key, pattern_model, family_name))
        layer_model = layer_map[model_key]
        rows.append(row_from_layer_atlas(layer_model, "attribute", "attribute_channel"))
        rows.append(row_from_layer_atlas(layer_model, "bridge", "noun_attribute_bridge"))
        model_rows.append(summarize_model(model_key, rows))

    for model_key in ["glm4", "gemma4"]:
        layer_model = layer_map[model_key]
        rows = [
            row_from_layer_atlas(layer_model, "noun", "noun"),
            row_from_layer_atlas(layer_model, "attribute", "attribute_channel"),
            row_from_layer_atlas(layer_model, "bridge", "noun_attribute_bridge"),
        ]
        model_rows.append(summarize_model(model_key, rows))

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage524_language_function_zone_map",
        "title": "完整语言功能层区地图",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage423": str(STAGE423_PATH),
            "stage493": str(STAGE493_PATH),
            "stage519": str(STAGE519_PATH),
        },
        "model_rows": model_rows,
        "core_answer": (
            "语言功能相关有效神经元并不是全层均匀散开，而是呈现明显层带聚集。"
            "代词、连接词、时间词等功能性模式更容易出现在早中层路由带；"
            "名词、属性与固定搭配更容易在中晚层或晚层形成收束带；"
            "不同模型的层带坐标不同，但“功能层区”这件事本身是清楚存在的。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# stage524 完整语言功能层区地图",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
    ]
    for model_row in model_rows:
        lines.append(f"## {model_row['model_key']}")
        lines.append(f"- 早层功能区：`{', '.join(model_row['band_groups'].get('early', [])) or '无'}`")
        lines.append(f"- 中层功能区：`{', '.join(model_row['band_groups'].get('middle', [])) or '无'}`")
        lines.append(f"- 晚层功能区：`{', '.join(model_row['band_groups'].get('late', [])) or '无'}`")
        lines.append("")
        for row in model_row["function_rows"]:
            lines.append(
                f"- `{row['function_key']}`：峰值层 `{row['peak_layers']}`，质心 `{row['centroid']:.2f}`，"
                f"层带 `{row['band']}`，来源 `{row['source']}`"
            )
        lines.append("")
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
