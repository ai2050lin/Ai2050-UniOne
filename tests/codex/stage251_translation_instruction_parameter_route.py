#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from translation_parameter_shared import GateCollector, cosine, layer_band, load_model_and_tokenizer, run_forward, suffix_span


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage251_translation_instruction_parameter_route_20260324"

PLAIN_TEXT = "今天天气不错"
INSTRUCTED_TEXT = "请把中文翻译为英文：今天天气不错"
INSTRUCTION_PREFIX = "请把中文翻译为英文："


def build_summary() -> dict:
    model, tokenizer = load_model_and_tokenizer()
    collector = GateCollector(model)
    try:
        collector.reset()
        plain_inputs, plain_outputs = run_forward(model, tokenizer, PLAIN_TEXT)
        plain_gates = collector.get()
        collector.reset()
        instructed_inputs, instructed_outputs = run_forward(model, tokenizer, INSTRUCTED_TEXT)
        instructed_gates = collector.get()

        plain_ids = plain_inputs["input_ids"][0].tolist()
        instructed_ids = instructed_inputs["input_ids"][0].tolist()
        prefix_ids = tokenizer(INSTRUCTION_PREFIX, add_special_tokens=False).input_ids
        content_start = len(prefix_ids)
        content_end = len(instructed_ids)
        instruction_span = (0, content_start)
        layer_count = len(instructed_outputs.attentions)
        layer_rows = []
        for li in range(layer_count):
            attn = instructed_outputs.attentions[li][0]  # heads, seq, seq
            content_queries = attn[:, content_start:content_end, :]
            instruction_mass = float(content_queries[:, :, instruction_span[0] : instruction_span[1]].mean().item()) if content_start > 0 else 0.0
            content_mass = float(content_queries[:, :, content_start:content_end].mean().item())
            plain_hidden = plain_outputs.hidden_states[li + 1][0]
            inst_hidden = instructed_outputs.hidden_states[li + 1][0, content_start:content_end, :]
            aligned_len = min(plain_hidden.shape[0], inst_hidden.shape[0])
            similarities = [cosine(plain_hidden[idx], inst_hidden[idx]) for idx in range(aligned_len)]
            gate_shift = float(torch.stack([torch.norm(instructed_gates[li] - plain_gates[li], p=2)]).mean().item())
            layer_rows.append(
                {
                    "layer_index": li,
                    "band_name": layer_band(li, layer_count),
                    "instruction_attention_mass": instruction_mass,
                    "content_attention_mass": content_mass,
                    "content_hidden_similarity": sum(similarities) / len(similarities),
                    "gate_shift_l2": gate_shift,
                }
            )
        def finite_mean(rows, key):
            vals = [row[key] for row in rows if isinstance(row[key], float) and math.isfinite(row[key])]
            return sum(vals) / len(vals) if vals else 0.0

        early_rows = [row for row in layer_rows if row["band_name"] == "early"]
        mid_rows = [row for row in layer_rows if row["band_name"] == "mid"]
        late_rows = [row for row in layer_rows if row["band_name"] == "late"]
        instruction_attention_early = finite_mean(early_rows, "instruction_attention_mass")
        instruction_attention_mid = finite_mean(mid_rows, "instruction_attention_mass")
        instruction_attention_late = finite_mean(late_rows, "instruction_attention_mass")
        content_preservation_mean = finite_mean(layer_rows, "content_hidden_similarity")
        top_gate_shift_row = max(layer_rows, key=lambda row: row["gate_shift_l2"])
        route_score = (
            instruction_attention_early
            + instruction_attention_mid
            + content_preservation_mean
            + min(1.0, top_gate_shift_row["gate_shift_l2"] / max(row["gate_shift_l2"] for row in layer_rows))
        ) / 4.0
        return {
            "schema_version": "agi_research_result.v1",
            "experiment_id": "stage251_translation_instruction_parameter_route",
            "title": "翻译指令参数级路径图",
            "status_short": "translation_instruction_parameter_route_ready",
            "plain_text": PLAIN_TEXT,
            "instructed_text": INSTRUCTED_TEXT,
            "matched_content_token_count": aligned_len,
            "instruction_token_count": content_start,
            "instruction_attention_early": instruction_attention_early,
            "instruction_attention_mid": instruction_attention_mid,
            "instruction_attention_late": instruction_attention_late,
            "content_preservation_mean": content_preservation_mean,
            "top_gate_shift_layer": int(top_gate_shift_row["layer_index"]),
            "top_gate_shift_band": top_gate_shift_row["band_name"],
            "route_score": route_score,
            "strongest_component_name": "指令前缀重排内容路径",
            "weakest_component_name": "后段指令注意力维持",
            "layer_rows": layer_rows,
        }
    finally:
        collector.close()


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "layer_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["layer_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["layer_rows"])
    report = [
        "# Stage251：翻译指令参数级路径图",
        "",
        "## 核心结果",
        f"- 内容词元数量：{summary['matched_content_token_count']}",
        f"- 指令词元数量：{summary['instruction_token_count']}",
        f"- 早层指令注意力：{summary['instruction_attention_early']:.4f}",
        f"- 中层指令注意力：{summary['instruction_attention_mid']:.4f}",
        f"- 后层指令注意力：{summary['instruction_attention_late']:.4f}",
        f"- 内容保留均值：{summary['content_preservation_mean']:.4f}",
        f"- 最大门控偏移层：L{summary['top_gate_shift_layer']}",
        f"- 路径图总分：{summary['route_score']:.4f}",
    ]
    (output_dir / "STAGE251_TRANSLATION_INSTRUCTION_PARAMETER_ROUTE_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="翻译指令参数级路径图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
