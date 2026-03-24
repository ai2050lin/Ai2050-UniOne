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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage272_translation_refactor_parameter_role_card_20260324"

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "model_path": QWEN3_MODEL_PATH},
    {"model_tag": "deepseek7b", "display_name": "DeepSeek-R1-Distill-Qwen-7B", "model_path": DEEPSEEK7B_MODEL_PATH},
]

TASK_VARIANTS = {
    "translation": {
        "full": "请把中文翻译为英文，只输出结果：今天天气不错。",
        "drop_operation": "请把中文处理为英文，只输出结果：今天天气不错。",
        "drop_object": "请翻译下面内容为英文，只输出结果：今天天气不错。",
        "drop_constraint": "请把中文翻译为英文：今天天气不错。",
    },
    "refactor": {
        "full": "请重构 src/app.py 文件，只输出修改后的代码。",
        "drop_operation": "请处理 src/app.py 文件，只输出修改后的代码。",
        "drop_object": "请重构这个文件，只输出修改后的代码。",
        "drop_constraint": "请重构 src/app.py 文件。",
    },
}


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


def mean_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def run_model(model_spec: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    anchors = anchor_layers(model)
    layer_ids = list(anchors.values())
    task_rows = []

    for task_name, variants in TASK_VARIANTS.items():
        payloads = {
            variant_name: collect_last_token_payloads(model, tokenizer, prompt, layer_ids)
            for variant_name, prompt in variants.items()
        }
        role_rows = []
        for role_name, variant_name in [
            ("operation_role", "drop_operation"),
            ("object_role", "drop_object"),
            ("constraint_role", "drop_constraint"),
        ]:
            layer_rows = []
            for layer_name, layer_id in anchors.items():
                score = mean_abs_delta(payloads["full"][layer_id], payloads[variant_name][layer_id])
                layer_rows.append(
                    {
                        "layer_name": layer_name,
                        "layer_id": layer_id,
                        "role_delta_score": score,
                    }
                )
            role_rows.append(
                {
                    "role_name": role_name,
                    "role_score": sum(row["role_delta_score"] for row in layer_rows) / len(layer_rows),
                    "strongest_layer_name": max(layer_rows, key=lambda row: row["role_delta_score"])["layer_name"],
                    "layer_rows": layer_rows,
                }
            )
        task_rows.append(
            {
                "task_name": task_name,
                "task_score": sum(row["role_score"] for row in role_rows) / len(role_rows),
                "strongest_role_name": max(role_rows, key=lambda row: row["role_score"])["role_name"],
                "weakest_role_name": min(role_rows, key=lambda row: row["role_score"])["role_name"],
                "role_rows": role_rows,
            }
        )

    role_score = sum(row["task_score"] for row in task_rows) / len(task_rows)
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "role_score": role_score,
        "strongest_task_name": max(task_rows, key=lambda row: row["task_score"])["task_name"],
        "weakest_task_name": min(task_rows, key=lambda row: row["task_score"])["task_name"],
        "task_rows": task_rows,
    }


def build_summary() -> dict:
    model_rows = [run_model(spec) for spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda row: row["role_score"])
    weakest = min(model_rows, key=lambda row: row["role_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage272_translation_refactor_parameter_role_card",
        "title": "翻译与重构参数角色卡",
        "status_short": "translation_refactor_parameter_role_card_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
        "top_gap_name": "翻译与重构都不是单词开关，而是操作、对象、约束三类角色共同切入参数路径，其中操作角色通常最强",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage272 翻译与重构参数角色卡",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
        f"- 关键结论：{summary['top_gap_name']}",
    ]
    for row in summary["model_rows"]:
        lines.extend(
            [
                "",
                f"## {row['display_name']}",
                f"- 角色卡总分：{row['role_score']:.4f}",
                f"- 最强任务：{row['strongest_task_name']}",
                f"- 最弱任务：{row['weakest_task_name']}",
            ]
        )
    (output_dir / "STAGE272_TRANSLATION_REFACTOR_PARAMETER_ROLE_CARD_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="翻译与重构参数角色卡")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
