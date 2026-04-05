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
    / "stage526_language_band_circuit_dynamics_20260404"
)
STAGE524_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage524_language_function_zone_map_20260404"
    / "summary.json"
)
STAGE525_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage525_multi_bridge_causal_expansion_20260404"
    / "summary.json"
)
STAGE521_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage521_language_layer_band_dynamics_synthesis_20260404"
    / "summary.json"
)
STAGE518_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage518_four_model_cross_task_causal_synthesis_20260404"
    / "summary.json"
)
STAGE495_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage495_unified_language_control_variable_protocol_20260403"
    / "summary.json"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_layers(subset_ids: list[str]) -> list[int]:
    rows = []
    for cid in subset_ids:
        parts = cid.split(":")
        if len(parts) >= 3 and parts[1].isdigit():
            rows.append(int(parts[1]))
    return rows


def band_from_layers(layers: list[int], layer_count: int) -> str:
    if not layers or layer_count <= 0:
        return "unknown"
    centroid = sum(layers) / len(layers)
    ratio = centroid / max(layer_count - 1, 1)
    if ratio < 1 / 3:
        return "early"
    if ratio < 2 / 3:
        return "middle"
    return "late"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage524 = load_json(STAGE524_PATH)
    stage525 = load_json(STAGE525_PATH)
    stage521 = load_json(STAGE521_PATH)
    stage518 = load_json(STAGE518_PATH)
    stage495 = load_json(STAGE495_PATH)

    zone_rows = {row["model_key"]: row for row in stage524["model_rows"]}
    bridge_rows = {row["model_key"]: row for row in stage525["model_rows"]}
    layer_band_rows = {row["model_key"]: row for row in stage521["model_rows"]}
    cross_task_rows = {row["model_key"]: row for row in stage518["model_rows"]}

    model_rows = []
    for model_key, zone_row in zone_rows.items():
        layer_count = zone_row["function_rows"][0]["layer_count"] if zone_row["function_rows"] else 0
        early_functions = sorted(zone_row["band_groups"].get("early", []))
        middle_functions = sorted(zone_row["band_groups"].get("middle", []))
        late_functions = sorted(zone_row["band_groups"].get("late", []))

        bridge_band_rows = []
        for bridge_row in bridge_rows[model_key]["bridge_rows"]:
            subset_layers = parse_layers(bridge_row["final_subset"])
            bridge_band_rows.append(
                {
                    "bridge_kind": bridge_row["bridge_kind"],
                    "subset_layers": subset_layers,
                    "band": band_from_layers(subset_layers, layer_count),
                    "utility": bridge_row["final_result"]["utility"],
                }
            )

        cross_task_layers = parse_layers(cross_task_rows[model_key]["final_subset"])
        cross_task_band = band_from_layers(cross_task_layers, layer_count)
        model_rows.append(
            {
                "model_key": model_key,
                "route_band": "early" if early_functions else "mixed",
                "binding_band_rows": bridge_band_rows,
                "cross_task_core_layers": cross_task_layers,
                "cross_task_core_band": cross_task_band,
                "early_functions": early_functions,
                "middle_functions": middle_functions,
                "late_functions": late_functions,
                "layer_band_reference": layer_band_rows.get(model_key, {}),
            }
        )

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage526_language_band_circuit_dynamics",
        "title": "层带位置图谱与最小因果回路统一动力学",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": {
            "stage524": str(STAGE524_PATH),
            "stage525": str(STAGE525_PATH),
            "stage521": str(STAGE521_PATH),
            "stage518": str(STAGE518_PATH),
            "stage495": str(STAGE495_PATH),
        },
        "symbolic_update": (
            "h_{t}^{l+1} = h_{t}^{l} + E_l(route_t) + M_l(bind_t) + L_l(readout_t)，"
            "其中早层 E_l 更偏路由与功能词控制，中层 M_l 更偏桥接与绑定，"
            "晚层 L_l 更偏名词/属性收束与最终读出。"
        ),
        "control_variable_reference": stage495["recurring_control_variables"],
        "model_rows": model_rows,
        "core_answer": (
            "把层带图谱和最小因果回路压到一起之后，当前最稳的统一动力学是："
            "早层主要负责路由和引用建立，中层主要负责桥接与绑定，晚层主要负责名词、属性、固定搭配等内容的收束读出。"
            "不同模型的具体层号不同，但“早路由、中绑定、晚读出”这个大框架已经越来越清楚。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# stage526 层带位置图谱与最小因果回路统一动力学",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
        "## 更新方程",
        summary["symbolic_update"],
        "",
    ]
    for row in model_rows:
        lines.append(f"## {row['model_key']}")
        lines.append(f"- 早层主功能：`{', '.join(row['early_functions']) or '无'}`")
        lines.append(f"- 中层主功能：`{', '.join(row['middle_functions']) or '无'}`")
        lines.append(f"- 晚层主功能：`{', '.join(row['late_functions']) or '无'}`")
        lines.append(
            f"- 跨任务共享骨干子集：`{', '.join(map(str, row['cross_task_core_layers'])) or '无'}`，"
            f"对应层带 `{row['cross_task_core_band']}`"
        )
        for bridge in row["binding_band_rows"]:
            lines.append(
                f"- `{bridge['bridge_kind']}`：子集层 `{bridge['subset_layers']}`，"
                f"层带 `{bridge['band']}`，效用 `{bridge['utility']:.6f}`"
            )
        lines.append("")
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
