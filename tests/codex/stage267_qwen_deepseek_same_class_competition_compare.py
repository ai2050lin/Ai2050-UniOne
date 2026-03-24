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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage267_qwen_deepseek_same_class_competition_compare_20260324"

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "model_path": QWEN3_MODEL_PATH},
    {"model_tag": "deepseek7b", "display_name": "DeepSeek-R1-Distill-Qwen-7B", "model_path": DEEPSEEK7B_MODEL_PATH},
]

PROBES = [
    {
        "family": "fruit",
        "probe_name": "苹果_vs_梨子",
        "prompt": "只输出 apple 或 pear：Tom washed the apple beside the pear, sliced the fruit, and ate it. 最后的 it 更可能指 apple 还是 pear？",
        "target": "apple",
        "distractor": "pear",
    },
    {
        "family": "fruit",
        "probe_name": "香蕉_vs_桃子",
        "prompt": "只输出 banana 或 peach：Tom peeled the banana beside the peach, ate the fruit, and liked it. 最后的 it 更可能指 banana 还是 peach？",
        "target": "banana",
        "distractor": "peach",
    },
    {
        "family": "animal",
        "probe_name": "猫_vs_狗",
        "prompt": "只输出 cat 或 dog：The dog stood near the cat, the pet was washed, and it later slept. 最后的 it 更可能指 cat 还是 dog？",
        "target": "cat",
        "distractor": "dog",
    },
    {
        "family": "animal",
        "probe_name": "狮子_vs_老虎",
        "prompt": "只输出 lion 或 tiger：The tiger stood near the lion, the animal was fed, and it later roared. 最后的 it 更可能指 lion 还是 tiger？",
        "target": "lion",
        "distractor": "tiger",
    },
    {
        "family": "brand",
        "probe_name": "iPhone_vs_MacBook",
        "prompt": "只输出 iphone 或 macbook：Apple displayed the MacBook beside the iPhone, repaired the device, and sold it. 最后的 it 更可能指 iphone 还是 macbook？",
        "target": "iphone",
        "distractor": "macbook",
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


def candidate_id(tokenizer, word: str) -> int:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(word, add_special_tokens=False)
    return int(ids[0])


def margin_for_prompt(model, tokenizer, prompt: str, target: str, distractor: str) -> float:
    enc = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**enc, use_cache=False, return_dict=True)
    logits = outputs.logits[0, -1, :].detach().float().cpu()
    target_logit = float(logits[candidate_id(tokenizer, target)].item())
    distractor_logit = float(logits[candidate_id(tokenizer, distractor)].item())
    return target_logit - distractor_logit


def analyze_model(model_spec: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    probe_rows = []
    for probe in PROBES:
        margin = margin_for_prompt(model, tokenizer, probe["prompt"], probe["target"], probe["distractor"])
        probe_rows.append(
            {
                "family": probe["family"],
                "probe_name": probe["probe_name"],
                "target_margin": margin,
                "is_positive": margin > 0.0,
            }
        )
    family_rows = []
    for family in sorted({probe["family"] for probe in PROBES}):
        margins = [float(row["target_margin"]) for row in probe_rows if row["family"] == family]
        family_rows.append(
            {
                "family": family,
                "margin_mean": sum(margins) / len(margins),
                "positive_rate": sum(1 for margin in margins if margin > 0.0) / len(margins),
            }
        )
    family_score = sum(row["positive_rate"] for row in family_rows) / len(family_rows)
    strongest = max(family_rows, key=lambda row: row["margin_mean"])
    weakest = min(family_rows, key=lambda row: row["margin_mean"])
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "probe_count": len(probe_rows),
        "same_class_score": family_score,
        "strongest_family": strongest["family"],
        "weakest_family": weakest["family"],
        "family_rows": family_rows,
        "probe_rows": probe_rows,
    }


def build_summary() -> dict:
    model_rows = [analyze_model(model_spec) for model_spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda row: row["same_class_score"])
    weakest = min(model_rows, key=lambda row: row["same_class_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage267_qwen_deepseek_same_class_competition_compare",
        "title": "Qwen 与 DeepSeek 同类高竞争参数对照",
        "status_short": "qwen_deepseek_same_class_competition_compare_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage267：Qwen 与 DeepSeek 同类高竞争参数对照",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
    ]
    for row in summary["model_rows"]:
        lines.extend(
            [
                "",
                f"## {row['display_name']}",
                f"- 同类高竞争总分：{row['same_class_score']:.4f}",
                f"- 最强家族：{row['strongest_family']}",
                f"- 最弱家族：{row['weakest_family']}",
            ]
        )
    (output_dir / "STAGE267_QWEN_DEEPSEEK_SAME_CLASS_COMPETITION_COMPARE_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen 与 DeepSeek 同类高竞争参数对照")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
