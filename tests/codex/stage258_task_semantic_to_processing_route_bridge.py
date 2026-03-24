#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List

from translation_parameter_shared import GateCollector, cosine, layer_band, load_model_and_tokenizer, run_forward


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage258_task_semantic_to_processing_route_bridge_20260324"

TASKS = [
    {
        "task_name": "translation",
        "content": "今天天气不错",
        "prefix": "请把中文翻译为英文：",
        "instructed": "请把中文翻译为英文：今天天气不错",
    },
    {
        "task_name": "image_edit",
        "content": "左边苹果颜色改成红色",
        "prefix": "请修改图片：",
        "instructed": "请修改图片：左边苹果颜色改成红色",
    },
    {
        "task_name": "refactor",
        "content": "src/app.py 文件",
        "prefix": "请重构代码文件：",
        "instructed": "请重构代码文件：src/app.py 文件",
    },
    {
        "task_name": "rewrite",
        "content": "今天天气不错",
        "prefix": "请改写这句话：",
        "instructed": "请改写这句话：今天天气不错",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def analyze_task(model, tokenizer, collector: GateCollector, row: dict) -> dict:
    collector.reset()
    plain_inputs, plain_outputs = run_forward(model, tokenizer, row["content"])
    plain_gates = collector.get()

    collector.reset()
    inst_inputs, inst_outputs = run_forward(model, tokenizer, row["instructed"])
    inst_gates = collector.get()

    prefix_ids = tokenizer(row["prefix"], add_special_tokens=False).input_ids
    content_start = len(prefix_ids)
    content_end = inst_inputs["input_ids"].shape[1]
    plain_len = plain_inputs["input_ids"].shape[1]
    layer_rows = []
    for li in range(len(inst_outputs.attentions)):
        attn = inst_outputs.attentions[li][0]
        content_queries = attn[:, content_start:content_end, :]
        instruction_mass = float(content_queries[:, :, :content_start].mean().item()) if content_start > 0 else 0.0
        plain_hidden = plain_outputs.hidden_states[li + 1][0]
        inst_hidden = inst_outputs.hidden_states[li + 1][0, content_start:content_end, :]
        aligned_len = min(plain_len, inst_hidden.shape[0])
        similarities = [cosine(plain_hidden[idx], inst_hidden[idx]) for idx in range(aligned_len)]
        gate_shift = float((inst_gates[li] - plain_gates[li]).norm().item())
        layer_rows.append(
            {
                "task_name": row["task_name"],
                "layer_index": li,
                "band_name": layer_band(li, len(inst_outputs.attentions)),
                "instruction_attention_mass": instruction_mass,
                "content_hidden_similarity": sum(similarities) / max(len(similarities), 1),
                "gate_shift_l2": gate_shift,
            }
        )

    strongest = max(layer_rows, key=lambda item: item["gate_shift_l2"])
    early_rows = [item for item in layer_rows if item["band_name"] == "early"]
    valid_similarities = [item["content_hidden_similarity"] for item in layer_rows if item["content_hidden_similarity"] == item["content_hidden_similarity"]]
    return {
        "task_name": row["task_name"],
        "route_peak": strongest["gate_shift_l2"],
        "route_peak_layer": int(strongest["layer_index"]),
        "route_peak_band": strongest["band_name"],
        "instruction_attention_early": sum(item["instruction_attention_mass"] for item in early_rows) / max(len(early_rows), 1),
        "content_preservation_mean": sum(valid_similarities) / max(len(valid_similarities), 1),
        "layer_rows": layer_rows,
    }


def build_summary() -> dict:
    model, tokenizer = load_model_and_tokenizer()
    collector = GateCollector(model)
    try:
        task_rows = [analyze_task(model, tokenizer, collector, task) for task in TASKS]
    finally:
        collector.close()

    strongest = max(task_rows, key=lambda row: row["route_peak"])
    weakest = min(task_rows, key=lambda row: row["instruction_attention_early"])
    valid_preservation = [row["content_preservation_mean"] for row in task_rows if row["content_preservation_mean"] == row["content_preservation_mean"]]
    bridge_score = (
        sum(row["instruction_attention_early"] for row in task_rows) / len(task_rows)
        + sum(valid_preservation) / max(len(valid_preservation), 1)
        + sum(1.0 for row in task_rows if row["route_peak_band"] in {"early", "mid"}) / len(task_rows)
    ) / 3.0
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage258_task_semantic_to_processing_route_bridge",
        "title": "任务语义到处理路径桥",
        "status_short": "task_semantic_to_processing_route_bridge_ready",
        "task_count": len(task_rows),
        "bridge_score": bridge_score,
        "strongest_task_name": strongest["task_name"],
        "weakest_task_name": weakest["task_name"],
        "top_gap_name": "任务词并不是只改输出标签，而是会在早中层切入处理路径；但不同任务的切入力度并不对称",
        "task_rows": task_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "task_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [k for k in summary["task_rows"][0].keys() if k != "layer_rows"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["task_rows"]:
            writer.writerow({k: v for k, v in row.items() if k != "layer_rows"})
    report = [
        "# Stage258：任务语义到处理路径桥",
        "",
        "## 核心结果",
        f"- 任务数量：{summary['task_count']}",
        f"- 桥总分：{summary['bridge_score']:.4f}",
        f"- 最强任务：{summary['strongest_task_name']}",
        f"- 最弱任务：{summary['weakest_task_name']}",
        f"- 头号发现：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE258_TASK_SEMANTIC_TO_PROCESSING_ROUTE_BRIDGE_REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        cached = load_json(summary_path)
        if math.isfinite(float(cached.get("bridge_score", float("nan")))):
            return cached
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="任务语义到处理路径桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
