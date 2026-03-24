#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_language_shared import QWEN3_MODEL_PATH, discover_layers
from stage278_translation_target_language_readout_position_map import run_analysis as run_stage278


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage285_translation_target_language_causal_map_20260324"

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "model_path": QWEN3_MODEL_PATH},
    {"model_tag": "deepseek7b", "display_name": "DeepSeek-R1-Distill-Qwen-7B", "model_path": DEEPSEEK7B_MODEL_PATH},
]

PROMPTS = [
    {
        "name": "weather",
        "to_english": "请把中文翻译为英文，只输出结果：今天天气不错。",
        "to_chinese": "请把英文翻译为中文，只输出结果：The weather is nice today.",
    },
    {
        "name": "apple",
        "to_english": "请把中文翻译为英文，只输出结果：苹果很甜。",
        "to_chinese": "请把英文翻译为中文，只输出结果：The apple is sweet.",
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


def anchor_layers(model) -> Dict[str, int]:
    layer_count = len(discover_layers(model))
    return {
        "late": layer_count - 1,
    }


def collect_payloads(model, tokenizer, text: str, layer_id: int, intervention_dims: List[int] | None = None) -> dict:
    layers = discover_layers(model)
    intervention_dims = intervention_dims or []
    buffer: torch.Tensor | None = None

    def hook(module, inputs, output):
        nonlocal buffer
        hidden = output
        buffer = hidden[0, -1, :].detach().float().cpu()
        if intervention_dims:
            patched = hidden.clone()
            patched[:, :, intervention_dims] = 0.0
            return patched
        return hidden

    handle = layers[layer_id].mlp.gate_proj.register_forward_hook(hook)
    enc = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        out = model(**enc, use_cache=False, return_dict=True)
    handle.remove()
    return {
        "late_hidden": buffer,
        "final_logits": out.logits[0, -1, :].detach().float().cpu(),
    }


def english_ratio(token_texts: List[str]) -> float:
    if not token_texts:
        return 0.0
    total = 0
    hits = 0
    letters = set(string.ascii_letters)
    for token in token_texts:
        for ch in token:
            total += 1
            if ch in letters:
                hits += 1
    return hits / max(1, total)


def chinese_ratio(token_texts: List[str]) -> float:
    if not token_texts:
        return 0.0
    total = 0
    hits = 0
    for token in token_texts:
        for ch in token:
            total += 1
            if "\u4e00" <= ch <= "\u9fff":
                hits += 1
    return hits / max(1, total)


def top_token_texts(tokenizer, logits: torch.Tensor, k: int = 50) -> List[str]:
    ids = torch.topk(logits, min(k, logits.numel())).indices.detach().cpu().tolist()
    return [tokenizer.decode([idx], skip_special_tokens=True) for idx in ids]


def run_model(model_spec: dict, readout_summary: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    late_id = anchor_layers(model)["late"]
    row = next(item for item in readout_summary["model_rows"] if item["model_tag"] == model_spec["model_tag"])
    english_dims = [int(item["dim_index"]) for item in row["english_readout_dim_rows"][:12]]
    chinese_dims = [int(item["dim_index"]) for item in row["chinese_readout_dim_rows"][:12]]
    prompt_rows = []

    for prompt in PROMPTS:
        base_en = collect_payloads(model, tokenizer, prompt["to_english"], late_id)
        base_zh = collect_payloads(model, tokenizer, prompt["to_chinese"], late_id)
        ablated_en = collect_payloads(model, tokenizer, prompt["to_english"], late_id, english_dims)
        ablated_zh = collect_payloads(model, tokenizer, prompt["to_chinese"], late_id, chinese_dims)

        base_en_top = top_token_texts(tokenizer, base_en["final_logits"])
        base_zh_top = top_token_texts(tokenizer, base_zh["final_logits"])
        ablated_en_top = top_token_texts(tokenizer, ablated_en["final_logits"])
        ablated_zh_top = top_token_texts(tokenizer, ablated_zh["final_logits"])

        english_drop = max(0.0, (english_ratio(base_en_top) - chinese_ratio(base_en_top)) - (english_ratio(ablated_en_top) - chinese_ratio(ablated_en_top)))
        chinese_drop = max(0.0, (chinese_ratio(base_zh_top) - english_ratio(base_zh_top)) - (chinese_ratio(ablated_zh_top) - english_ratio(ablated_zh_top)))
        prompt_rows.append(
            {
                "prompt_name": prompt["name"],
                "english_readout_drop": float(english_drop),
                "chinese_readout_drop": float(chinese_drop),
            }
        )

    score = sum(item["english_readout_drop"] + item["chinese_readout_drop"] for item in prompt_rows) / len(prompt_rows)
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "causal_score": float(score),
        "prompt_rows": prompt_rows,
    }


def build_summary() -> dict:
    readout_summary = run_stage278(force=False)
    model_rows = [run_model(spec, readout_summary) for spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda item: item["causal_score"])
    weakest = min(model_rows, key=lambda item: item["causal_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage285_translation_target_language_causal_map",
        "title": "翻译目标语言逐位因果图",
        "status_short": "translation_target_language_causal_map_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
        "top_gap_name": "目标语言高频读出位一旦被压低，最终目标语言偏置会同步回落，说明这些位置已经开始具备目标语言读出因果作用",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="翻译目标语言逐位因果图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
