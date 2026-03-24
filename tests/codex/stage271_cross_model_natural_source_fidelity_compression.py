#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_language_shared import QWEN3_MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage271_cross_model_natural_source_fidelity_compression_20260324"

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "model_path": QWEN3_MODEL_PATH},
    {"model_tag": "deepseek7b", "display_name": "DeepSeek-R1-Distill-Qwen-7B", "model_path": DEEPSEEK7B_MODEL_PATH},
]

PROBE_ROWS = [
    {
        "family": "fruit",
        "natural_prompt": "The apple was sliced, while the pear stayed on the plate. Later, it was eaten. Here, it refers to the",
        "repair_prompt": "The apple was sliced, while the pear stayed on the plate. To be clear, the sliced one was the apple. Later, it was eaten. Here, it refers to the",
        "target": " apple",
        "alt": " pear",
    },
    {
        "family": "animal",
        "natural_prompt": "The dog barked at the cat, while the cat hid under the chair. Later, it barked again. Here, it refers to the",
        "repair_prompt": "The dog barked at the cat, while the cat hid under the chair. To be clear, the one that barked was the dog. Later, it barked again. Here, it refers to the",
        "target": " dog",
        "alt": " cat",
    },
    {
        "family": "brand",
        "natural_prompt": "The company Apple launched a laptop, while the fruit apple stayed in the bowl. Later, it updated the software. Here, it refers to the",
        "repair_prompt": "The company Apple launched a laptop, while the fruit apple stayed in the bowl. To be clear, the one updating the software was the company. Here, it refers to the",
        "target": " company",
        "alt": " fruit",
    },
]


def load_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    return model, tokenizer


def single_token_id(tokenizer, token_text: str) -> int:
    ids = tokenizer.encode(token_text, add_special_tokens=False)
    return int(ids[0])


def margin_for_prompt(model, tokenizer, prompt: str, target_token: str, alt_token: str) -> float:
    enc = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**enc, use_cache=False, return_dict=True).logits[0, -1, :].detach().float().cpu()
    target_id = single_token_id(tokenizer, target_token)
    alt_id = single_token_id(tokenizer, alt_token)
    return float((logits[target_id] - logits[alt_id]).item())


def run_model(model_spec: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    probe_rows = []
    for probe in PROBE_ROWS:
        natural_margin = margin_for_prompt(model, tokenizer, probe["natural_prompt"], probe["target"], probe["alt"])
        repair_margin = margin_for_prompt(model, tokenizer, probe["repair_prompt"], probe["target"], probe["alt"])
        probe_rows.append(
            {
                "family": probe["family"],
                "natural_margin": natural_margin,
                "repair_margin": repair_margin,
                "repair_gain": repair_margin - natural_margin,
                "natural_positive": natural_margin > 0.0,
                "repair_positive": repair_margin > 0.0,
            }
        )
    natural_score = sum(1.0 for row in probe_rows if row["natural_positive"]) / len(probe_rows)
    repair_score = sum(1.0 for row in probe_rows if row["repair_positive"]) / len(probe_rows)
    gap = repair_score - natural_score
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "natural_fidelity_score": natural_score,
        "repair_fidelity_score": repair_score,
        "repair_gain_score": gap,
        "strongest_family": max(probe_rows, key=lambda row: row["natural_margin"])["family"],
        "weakest_family": min(probe_rows, key=lambda row: row["natural_margin"])["family"],
        "probe_rows": probe_rows,
    }


def build_summary() -> dict:
    model_rows = [run_model(spec) for spec in MODEL_SPECS]
    common_natural = sum(row["natural_fidelity_score"] for row in model_rows) / len(model_rows)
    common_repair = sum(row["repair_fidelity_score"] for row in model_rows) / len(model_rows)
    strongest = max(model_rows, key=lambda row: row["natural_fidelity_score"])
    weakest = min(model_rows, key=lambda row: row["natural_fidelity_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage271_cross_model_natural_source_fidelity_compression",
        "title": "跨模型天然来源保真压缩",
        "status_short": "cross_model_natural_source_fidelity_compression_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "common_natural_fidelity_score": common_natural,
        "common_repair_fidelity_score": common_repair,
        "compression_gap": common_repair - common_natural,
        "model_rows": model_rows,
        "top_gap_name": "跨模型共同最弱的不是修复能力，而是天然状态下来源结构能不能自己带稳",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage271 跨模型天然来源保真压缩",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
        f"- 跨模型天然保真均值：{summary['common_natural_fidelity_score']:.4f}",
        f"- 跨模型修复保真均值：{summary['common_repair_fidelity_score']:.4f}",
        f"- 关键结论：{summary['top_gap_name']}",
    ]
    for row in summary["model_rows"]:
        lines.extend(
            [
                "",
                f"## {row['display_name']}",
                f"- 天然来源保真：{row['natural_fidelity_score']:.4f}",
                f"- 修复来源保真：{row['repair_fidelity_score']:.4f}",
                f"- 修复增益：{row['repair_gain_score']:.4f}",
            ]
        )
    (output_dir / "STAGE271_CROSS_MODEL_NATURAL_SOURCE_FIDELITY_COMPRESSION_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型天然来源保真压缩")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
