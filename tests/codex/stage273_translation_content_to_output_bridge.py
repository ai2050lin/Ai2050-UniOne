#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_language_shared import QWEN3_MODEL_PATH, discover_layers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage273_translation_content_to_output_bridge_20260324"

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "model_path": QWEN3_MODEL_PATH},
    {"model_tag": "deepseek7b", "display_name": "DeepSeek-R1-Distill-Qwen-7B", "model_path": DEEPSEEK7B_MODEL_PATH},
]

CONTENT_ROWS = [
    {
        "name": "weather",
        "raw": "今天天气不错。",
        "tasked": "请把中文翻译为英文，只输出结果：今天天气不错。",
    },
    {
        "name": "apple",
        "raw": "苹果很甜。",
        "tasked": "请把中文翻译为英文，只输出结果：苹果很甜。",
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


def collect_last_token_payloads(model, tokenizer, text: str, layer_ids: List[int]) -> Dict[int, torch.Tensor]:
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
        model(**enc, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    return {layer_id: buffers[layer_id] for layer_id in layer_ids if buffers[layer_id] is not None}


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def run_model(model_spec: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    anchors = anchor_layers(model)
    layer_ids = list(anchors.values())
    content_rows = []

    for row in CONTENT_ROWS:
        raw_payloads = collect_last_token_payloads(model, tokenizer, row["raw"], layer_ids)
        tasked_payloads = collect_last_token_payloads(model, tokenizer, row["tasked"], layer_ids)
        layer_rows = []
        for layer_name, layer_id in anchors.items():
            sim = cosine(raw_payloads[layer_id], tasked_payloads[layer_id])
            layer_rows.append(
                {
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "content_preservation_similarity": sim,
                }
            )
        content_rows.append(
            {
                "content_name": row["name"],
                "bridge_score": sum(r["content_preservation_similarity"] for r in layer_rows) / len(layer_rows),
                "strongest_layer_name": max(layer_rows, key=lambda x: x["content_preservation_similarity"])["layer_name"],
                "weakest_layer_name": min(layer_rows, key=lambda x: x["content_preservation_similarity"])["layer_name"],
                "layer_rows": layer_rows,
            }
        )

    model_score = sum(row["bridge_score"] for row in content_rows) / len(content_rows)
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "bridge_score": model_score,
        "strongest_content_name": max(content_rows, key=lambda row: row["bridge_score"])["content_name"],
        "weakest_content_name": min(content_rows, key=lambda row: row["bridge_score"])["content_name"],
        "content_rows": content_rows,
    }


def build_summary() -> dict:
    model_rows = [run_model(spec) for spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda row: row["bridge_score"])
    weakest = min(model_rows, key=lambda row: row["bridge_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage273_translation_content_to_output_bridge",
        "title": "翻译内容保留到输出语言桥",
        "status_short": "translation_content_to_output_bridge_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
        "top_gap_name": "翻译不是把原语义删除再重写，而更像保留内容骨架，再把输出路径切到目标语言",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage273 翻译内容保留到输出语言桥",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
        f"- 关键结论：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE273_TRANSLATION_CONTENT_TO_OUTPUT_BRIDGE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="翻译内容保留到输出语言桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
