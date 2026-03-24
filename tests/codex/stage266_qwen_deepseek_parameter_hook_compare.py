#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from qwen3_language_shared import QWEN3_MODEL_PATH, discover_layers, load_qwen3_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage266_qwen_deepseek_parameter_hook_compare_20260324"

PROMPT_PAIRS = [
    {
        "contrast_name": "苹果水果义_vs_品牌义",
        "left": "Tom ate the apple after washing it.",
        "right": "Apple released a new iPhone and updated it.",
    },
    {
        "contrast_name": "翻译_vs_重构",
        "left": "请把中文翻译为英文，只输出结果：今天天气不错",
        "right": "请重构 src/app.py 文件，只输出修改后的代码",
    },
    {
        "contrast_name": "图像编辑_vs_翻译",
        "left": "修改左边苹果颜色，只输出编辑结果说明",
        "right": "请把中文翻译为英文，只输出结果：苹果很甜",
    },
]

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "model_path": QWEN3_MODEL_PATH},
    {"model_tag": "deepseek7b", "display_name": "DeepSeek-R1-Distill-Qwen-7B", "model_path": DEEPSEEK7B_MODEL_PATH},
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
        torch_dtype=torch.bfloat16,
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


def collect_gate_last_token(model, tokenizer, text: str, layer_ids: List[int]) -> Dict[int, torch.Tensor]:
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


def mean_abs(vec: torch.Tensor) -> float:
    return float(vec.abs().mean().item())


def topk_energy(vec: torch.Tensor, k: int = 64) -> float:
    score = vec.float().pow(2)
    total = float(score.sum().item()) + 1e-8
    kk = min(k, score.numel())
    topv = torch.topk(score, kk).values
    return float(topv.sum().item() / total)


def run_model_compare(model_spec: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    anchors = anchor_layers(model)
    layer_ids = [anchors["early"], anchors["route"], anchors["late"]]
    contrast_rows = []
    for pair in PROMPT_PAIRS:
        left_payloads = collect_gate_last_token(model, tokenizer, pair["left"], layer_ids)
        right_payloads = collect_gate_last_token(model, tokenizer, pair["right"], layer_ids)
        layer_rows = []
        for layer_name, layer_id in anchors.items():
            left_vec = left_payloads[layer_id]
            right_vec = right_payloads[layer_id]
            diff_vec = right_vec - left_vec
            layer_rows.append(
                {
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "mean_abs_delta": mean_abs(diff_vec),
                    "top64_energy_ratio": topk_energy(diff_vec, 64),
                }
            )
        strongest = max(layer_rows, key=lambda row: row["mean_abs_delta"])
        contrast_score = sum(row["mean_abs_delta"] for row in layer_rows) / len(layer_rows)
        contrast_rows.append(
            {
                "contrast_name": pair["contrast_name"],
                "contrast_score": contrast_score,
                "strongest_layer_name": strongest["layer_name"],
                "layer_rows": layer_rows,
            }
        )
    summary_score = sum(row["contrast_score"] for row in contrast_rows) / len(contrast_rows)
    strongest = max(contrast_rows, key=lambda row: row["contrast_score"])
    weakest = min(contrast_rows, key=lambda row: row["contrast_score"])
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "layer_anchor_map": anchors,
        "contrast_count": len(contrast_rows),
        "parameter_hook_score": summary_score,
        "strongest_contrast_name": strongest["contrast_name"],
        "weakest_contrast_name": weakest["contrast_name"],
        "contrast_rows": contrast_rows,
    }


def build_summary() -> dict:
    model_rows = [run_model_compare(model_spec) for model_spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda row: row["parameter_hook_score"])
    weakest = min(model_rows, key=lambda row: row["parameter_hook_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage266_qwen_deepseek_parameter_hook_compare",
        "title": "Qwen 与 DeepSeek 同口径参数钩子对照",
        "status_short": "qwen_deepseek_parameter_hook_compare_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage266：Qwen 与 DeepSeek 同口径参数钩子对照",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
    ]
    for row in summary["model_rows"]:
        lines.extend(
            [
                "",
                f"## {row['display_name']}",
                f"- 参数钩子总分：{row['parameter_hook_score']:.4f}",
                f"- 最强对照：{row['strongest_contrast_name']}",
                f"- 最弱对照：{row['weakest_contrast_name']}",
            ]
        )
    (output_dir / "STAGE266_QWEN_DEEPSEEK_PARAMETER_HOOK_COMPARE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="Qwen 与 DeepSeek 同口径参数钩子对照")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

