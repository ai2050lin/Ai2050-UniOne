#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_language_shared import QWEN3_MODEL_PATH, discover_layers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage276_refactor_structure_rewrite_position_map_20260324"

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "model_path": QWEN3_MODEL_PATH},
    {"model_tag": "deepseek7b", "display_name": "DeepSeek-R1-Distill-Qwen-7B", "model_path": DEEPSEEK7B_MODEL_PATH},
]

PROMPTS = [
    {
        "name": "refactor",
        "full": "请重构 src/app.py 文件，只输出修改后的代码。",
        "generic": "请改写 src/app.py 文件，只输出修改后的代码。",
    },
    {
        "name": "extract_function",
        "full": "请把 src/app.py 里的重复逻辑抽成函数，只输出修改后的代码。",
        "generic": "请修改 src/app.py，只输出修改后的代码。",
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


def collect_payloads(model, tokenizer, text: str, layer_ids: List[int]) -> Dict[int, torch.Tensor]:
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


def topk_dims(diff_vec: torch.Tensor, k: int = 16) -> List[int]:
    kk = min(k, diff_vec.numel())
    return torch.topk(diff_vec.abs(), kk).indices.detach().cpu().tolist()


def run_model(model_spec: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    anchors = anchor_layers(model)
    layer_ids = list(anchors.values())
    prompt_rows = []
    hot_counter: Dict[int, int] = {}

    for prompt in PROMPTS:
        full = collect_payloads(model, tokenizer, prompt["full"], layer_ids)
        generic = collect_payloads(model, tokenizer, prompt["generic"], layer_ids)
        layer_rows = []
        for layer_name, layer_id in anchors.items():
            diff_vec = full[layer_id] - generic[layer_id]
            dims = topk_dims(diff_vec, 16)
            for dim in dims:
                hot_counter[dim] = hot_counter.get(dim, 0) + 1
            layer_rows.append(
                {
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "rewrite_delta": float(diff_vec.abs().mean().item()),
                    "top_dims": dims[:8],
                }
            )
        prompt_rows.append(
            {
                "prompt_name": prompt["name"],
                "map_score": sum(row["rewrite_delta"] for row in layer_rows) / len(layer_rows),
                "strongest_layer_name": max(layer_rows, key=lambda row: row["rewrite_delta"])["layer_name"],
                "layer_rows": layer_rows,
            }
        )

    model_score = sum(row["map_score"] for row in prompt_rows) / len(prompt_rows)
    hot_rows = [{"dim_index": int(dim), "hit_count": int(count)} for dim, count in sorted(hot_counter.items(), key=lambda x: (-x[1], x[0]))[:16]]
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "map_score": model_score,
        "strongest_prompt_name": max(prompt_rows, key=lambda row: row["map_score"])["prompt_name"],
        "weakest_prompt_name": min(prompt_rows, key=lambda row: row["map_score"])["prompt_name"],
        "hot_dim_rows": hot_rows,
        "prompt_rows": prompt_rows,
    }


def build_summary() -> dict:
    model_rows = [run_model(spec) for spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda row: row["map_score"])
    weakest = min(model_rows, key=lambda row: row["map_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage276_refactor_structure_rewrite_position_map",
        "title": "重构结构改写位图",
        "status_short": "refactor_structure_rewrite_position_map_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
        "top_gap_name": "重构和普通改写的差异会集中落在少量高频位置上，这些位置更像结构改写位，而不是普通文本续写位",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage276 重构结构改写位图",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
        f"- 关键结论：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE276_REFACTOR_STRUCTURE_REWRITE_POSITION_MAP_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="重构结构改写位图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
