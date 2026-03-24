#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from translation_parameter_shared import GateCollector, cosine, layer_band, load_model_and_tokenizer, run_forward


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage254_translation_multitemplate_route_map_20260324"

TEMPLATES = [
    {
        "name": "weather",
        "plain": "今天天气不错",
        "prefix": "请把中文翻译为英文：",
        "instructed": "请把中文翻译为英文：今天天气不错",
    },
    {
        "name": "apple",
        "plain": "苹果很甜",
        "prefix": "把下面中文译成英文，只输出结果：",
        "instructed": "把下面中文译成英文，只输出结果：苹果很甜",
    },
    {
        "name": "window",
        "plain": "请开窗",
        "prefix": "请翻译成英文，只输出翻译：",
        "instructed": "请翻译成英文，只输出翻译：请开窗",
    },
]


def analyze_template(model, tokenizer, collector: GateCollector, template: dict) -> dict:
    collector.reset()
    plain_inputs, plain_outputs = run_forward(model, tokenizer, template["plain"])
    plain_gates = collector.get()

    collector.reset()
    instructed_inputs, instructed_outputs = run_forward(model, tokenizer, template["instructed"])
    instructed_gates = collector.get()

    prefix_ids = tokenizer(template["prefix"], add_special_tokens=False).input_ids
    content_start = len(prefix_ids)
    content_end = instructed_inputs["input_ids"].shape[1]
    plain_len = plain_inputs["input_ids"].shape[1]
    layer_rows = []
    for li in range(len(instructed_outputs.attentions)):
        attn = instructed_outputs.attentions[li][0]
        content_queries = attn[:, content_start:content_end, :]
        instruction_mass = float(content_queries[:, :, :content_start].mean().item()) if content_start > 0 else 0.0
        plain_hidden = plain_outputs.hidden_states[li + 1][0]
        inst_hidden = instructed_outputs.hidden_states[li + 1][0, content_start:content_end, :]
        aligned_len = min(plain_len, inst_hidden.shape[0])
        similarities = [cosine(plain_hidden[idx], inst_hidden[idx]) for idx in range(aligned_len)]
        gate_shift = float(torch.norm(instructed_gates[li] - plain_gates[li], p=2).item())
        layer_rows.append(
            {
                "template_name": template["name"],
                "layer_index": li,
                "band_name": layer_band(li, len(instructed_outputs.attentions)),
                "instruction_attention_mass": instruction_mass,
                "content_hidden_similarity": sum(similarities) / max(len(similarities), 1),
                "gate_shift_l2": gate_shift,
            }
        )

    def finite_mean(rows: list[dict], key: str) -> float:
        vals = [row[key] for row in rows if isinstance(row[key], float) and row[key] == row[key]]
        return sum(vals) / max(len(vals), 1)

    early_rows = [row for row in layer_rows if row["band_name"] == "early"]
    mid_rows = [row for row in layer_rows if row["band_name"] == "mid"]
    late_rows = [row for row in layer_rows if row["band_name"] == "late"]
    finite_gate_rows = [row for row in layer_rows if row["gate_shift_l2"] == row["gate_shift_l2"]]
    top_row = max(finite_gate_rows, key=lambda row: row["gate_shift_l2"])
    return {
        "template_name": template["name"],
        "instruction_attention_early": finite_mean(early_rows, "instruction_attention_mass"),
        "instruction_attention_mid": finite_mean(mid_rows, "instruction_attention_mass"),
        "instruction_attention_late": finite_mean(late_rows, "instruction_attention_mass"),
        "content_preservation_mean": finite_mean(layer_rows, "content_hidden_similarity"),
        "top_gate_shift_layer": int(top_row["layer_index"]),
        "top_gate_shift_band": top_row["band_name"],
        "route_peak": top_row["gate_shift_l2"],
        "layer_rows": layer_rows,
    }


def build_summary() -> dict:
    model, tokenizer = load_model_and_tokenizer()
    collector = GateCollector(model)
    try:
        template_rows = [analyze_template(model, tokenizer, collector, template) for template in TEMPLATES]
    finally:
        collector.close()

    strongest = max(template_rows, key=lambda row: row["route_peak"])
    weakest = min(template_rows, key=lambda row: row["instruction_attention_mid"])
    stability = sum(1.0 for row in template_rows if row["top_gate_shift_band"] in {"early", "mid"}) / len(template_rows)
    route_score = (
        sum(row["instruction_attention_mid"] for row in template_rows) / len(template_rows)
        + sum(row["content_preservation_mean"] for row in template_rows) / len(template_rows)
        + stability
    ) / 3.0
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage254_translation_multitemplate_route_map",
        "title": "翻译指令多模板参数级路径图",
        "status_short": "translation_multitemplate_route_map_ready",
        "template_count": len(template_rows),
        "route_score": route_score,
        "early_peak_rate": stability,
        "strongest_template_name": strongest["template_name"],
        "weakest_template_name": weakest["template_name"],
        "template_rows": template_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "template_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [k for k in summary["template_rows"][0].keys() if k != "layer_rows"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["template_rows"]:
            writer.writerow({k: v for k, v in row.items() if k != "layer_rows"})
    report = [
        "# Stage254：翻译指令多模板参数级路径图",
        "",
        "## 核心结果",
        f"- 模板数量：{summary['template_count']}",
        f"- 路径图总分：{summary['route_score']:.4f}",
        f"- 早层峰值比例：{summary['early_peak_rate']:.4f}",
        f"- 最强模板：{summary['strongest_template_name']}",
        f"- 最弱模板：{summary['weakest_template_name']}",
    ]
    (output_dir / "STAGE254_TRANSLATION_MULTITEMPLATE_ROUTE_MAP_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8-sig"
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="翻译指令多模板参数级路径图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
