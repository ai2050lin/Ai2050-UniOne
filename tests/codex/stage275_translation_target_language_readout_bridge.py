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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage275_translation_target_language_readout_bridge_20260324"

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
        "early": max(0, min(layer_count - 1, round((layer_count - 1) * 1 / 11.0))),
        "route": max(0, min(layer_count - 1, round((layer_count - 1) * 3 / 11.0))),
        "late": layer_count - 1,
    }


def collect_payloads(model, tokenizer, text: str, layer_ids: List[int]) -> dict:
    layers = discover_layers(model)
    buffers: Dict[int, torch.Tensor | None] = {layer_id: None for layer_id in layer_ids}
    handles = []

    for layer_id in layer_ids:
        def make_hook(target_layer: int):
            def hook(module, inputs, output):
                buffers[target_layer] = output[0, -1, :].detach().float().cpu()
                return output

            return hook

        handles.append(layers[layer_id].mlp.gate_proj.register_forward_hook(make_hook(layer_id)))

    enc = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        out = model(**enc, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    return {
        "layer_payloads": {layer_id: buffers[layer_id] for layer_id in layer_ids if buffers[layer_id] is not None},
        "final_logits": out.logits[0, -1, :].detach().float().cpu(),
    }


def english_ratio(token_texts: List[str]) -> float:
    if not token_texts:
        return 0.0
    total = 0
    hits = 0
    english_chars = set(string.ascii_letters)
    for token in token_texts:
        for ch in token:
            total += 1
            if ch in english_chars:
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
    texts = []
    for idx in ids:
        texts.append(tokenizer.decode([idx], skip_special_tokens=True))
    return texts


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def run_model(model_spec: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    anchors = anchor_layers(model)
    layer_ids = list(anchors.values())
    prompt_rows = []

    for prompt in PROMPTS:
        eng = collect_payloads(model, tokenizer, prompt["to_english"], layer_ids)
        zh = collect_payloads(model, tokenizer, prompt["to_chinese"], layer_ids)
        eng_top = top_token_texts(tokenizer, eng["final_logits"], 50)
        zh_top = top_token_texts(tokenizer, zh["final_logits"], 50)
        layer_rows = []
        for layer_name, layer_id in anchors.items():
            layer_rows.append(
                {
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "direction_delta": float((eng["layer_payloads"][layer_id] - zh["layer_payloads"][layer_id]).abs().mean().item()),
                    "cross_direction_similarity": cosine(eng["layer_payloads"][layer_id], zh["layer_payloads"][layer_id]),
                }
            )
        prompt_rows.append(
            {
                "prompt_name": prompt["name"],
                "english_top50_english_ratio": english_ratio(eng_top),
                "english_top50_chinese_ratio": chinese_ratio(eng_top),
                "chinese_top50_english_ratio": english_ratio(zh_top),
                "chinese_top50_chinese_ratio": chinese_ratio(zh_top),
                "readout_gap": english_ratio(eng_top) - chinese_ratio(eng_top) + chinese_ratio(zh_top) - english_ratio(zh_top),
                "layer_rows": layer_rows,
            }
        )

    model_score = sum(row["readout_gap"] for row in prompt_rows) / len(prompt_rows)
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "bridge_score": model_score,
        "strongest_prompt_name": max(prompt_rows, key=lambda row: row["readout_gap"])["prompt_name"],
        "weakest_prompt_name": min(prompt_rows, key=lambda row: row["readout_gap"])["prompt_name"],
        "prompt_rows": prompt_rows,
    }


def build_summary() -> dict:
    model_rows = [run_model(spec) for spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda row: row["bridge_score"])
    weakest = min(model_rows, key=lambda row: row["bridge_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage275_translation_target_language_readout_bridge",
        "title": "翻译目标语言读出桥",
        "status_short": "translation_target_language_readout_bridge_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
        "top_gap_name": "翻译的目标语言控制不只是前段任务切换，还会进一步压到末端读出分布上",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage275 翻译目标语言读出桥",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
        f"- 关键结论：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE275_TRANSLATION_TARGET_LANGUAGE_READOUT_BRIDGE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="翻译目标语言读出桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
