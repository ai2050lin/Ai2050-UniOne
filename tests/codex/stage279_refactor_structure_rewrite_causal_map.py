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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage279_refactor_structure_rewrite_causal_map_20260324"

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "model_path": QWEN3_MODEL_PATH},
    {"model_tag": "deepseek7b", "display_name": "DeepSeek-R1-Distill-Qwen-7B", "model_path": DEEPSEEK7B_MODEL_PATH},
]

PROMPTS = [
    {
        "name": "refactor",
        "refactor": "请重构 src/app.py 文件，只输出修改后的代码。",
        "generic": "请改写 src/app.py 文件，只输出修改后的代码。",
    },
    {
        "name": "extract",
        "refactor": "请把 src/app.py 里的重复逻辑抽成函数，只输出修改后的代码。",
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


def collect_payloads(model, tokenizer, text: str, layer_ids: List[int], intervention: dict | None = None) -> dict:
    layers = discover_layers(model)
    buffers: Dict[int, torch.Tensor | None] = {layer_id: None for layer_id in layer_ids}
    handles = []
    intervention = intervention or {}

    for layer_id in layer_ids:
        dims = intervention.get("dims", []) if intervention.get("layer_id") == layer_id else []

        def make_hook(target_layer: int, target_dims: List[int]):
            def hook(module, inputs, output):
                hidden = output
                buffers[target_layer] = hidden[0, -1, :].detach().float().cpu()
                if target_dims:
                    patched = hidden.clone()
                    patched[:, :, target_dims] = 0.0
                    return patched
                return hidden

            return hook

        handles.append(layers[layer_id].mlp.gate_proj.register_forward_hook(make_hook(layer_id, dims)))

    enc = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        out = model(**enc, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    return {
        "layer_payloads": {layer_id: buffers[layer_id] for layer_id in layer_ids if buffers[layer_id] is not None},
        "final_logits": out.logits[0, -1, :].detach().float().cpu(),
    }


def topk_dims(diff_vec: torch.Tensor, k: int = 24) -> List[int]:
    kk = min(k, diff_vec.numel())
    return torch.topk(diff_vec.abs(), kk).indices.detach().cpu().tolist()


def mean_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def run_model(model_spec: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    anchors = anchor_layers(model)
    late_id = anchors["late"]
    route_id = anchors["route"]
    prompt_rows = []

    for prompt in PROMPTS:
        refactor = collect_payloads(model, tokenizer, prompt["refactor"], [route_id, late_id])
        generic = collect_payloads(model, tokenizer, prompt["generic"], [route_id, late_id])
        late_diff = refactor["layer_payloads"][late_id] - generic["layer_payloads"][late_id]
        hot_dims = topk_dims(late_diff, 24)
        ablated = collect_payloads(
            model,
            tokenizer,
            prompt["refactor"],
            [route_id, late_id],
            intervention={"layer_id": late_id, "dims": hot_dims},
        )
        route_drop = max(0.0, mean_abs_delta(refactor["layer_payloads"][route_id], generic["layer_payloads"][route_id]) - mean_abs_delta(ablated["layer_payloads"][route_id], generic["layer_payloads"][route_id]))
        late_drop = max(0.0, mean_abs_delta(refactor["layer_payloads"][late_id], generic["layer_payloads"][late_id]) - mean_abs_delta(ablated["layer_payloads"][late_id], generic["layer_payloads"][late_id]))
        logits_shift = mean_abs_delta(refactor["final_logits"], ablated["final_logits"])
        prompt_rows.append(
            {
                "prompt_name": prompt["name"],
                "late_rewrite_delta": float(late_diff.abs().mean().item()),
                "route_drop_after_late_ablation": route_drop,
                "late_drop_after_late_ablation": late_drop,
                "logits_shift_after_late_ablation": logits_shift,
                "hot_dims": hot_dims[:12],
            }
        )

    score = sum(row["logits_shift_after_late_ablation"] + row["late_drop_after_late_ablation"] for row in prompt_rows) / len(prompt_rows)
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "causal_score": score,
        "strongest_prompt_name": max(prompt_rows, key=lambda row: row["logits_shift_after_late_ablation"])["prompt_name"],
        "weakest_prompt_name": min(prompt_rows, key=lambda row: row["logits_shift_after_late_ablation"])["prompt_name"],
        "prompt_rows": prompt_rows,
    }


def build_summary() -> dict:
    model_rows = [run_model(spec) for spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda row: row["causal_score"])
    weakest = min(model_rows, key=lambda row: row["causal_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage279_refactor_structure_rewrite_causal_map",
        "title": "重构结构改写因果图",
        "status_short": "refactor_structure_rewrite_causal_map_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
        "top_gap_name": "重构的关键结构位一旦被压低，后段改写链和最终读出都会一起回落，说明这些位置已经具备结构改写因果作用",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage279 重构结构改写因果图",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
        f"- 关键结论：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE279_REFACTOR_STRUCTURE_REWRITE_CAUSAL_MAP_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="重构结构改写因果图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
