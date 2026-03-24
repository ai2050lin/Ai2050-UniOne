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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage269_parameter_position_causal_intervention_map_20260324"

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "model_path": QWEN3_MODEL_PATH},
    {"model_tag": "deepseek7b", "display_name": "DeepSeek-R1-Distill-Qwen-7B", "model_path": DEEPSEEK7B_MODEL_PATH},
]

PROMPT_PAIRS = [
    {
        "contrast_name": "苹果水果义_vs_品牌义",
        "left": "Tom washed the apple and later ate it.",
        "right": "Apple released a new phone and later updated it.",
    },
    {
        "contrast_name": "翻译_vs_重构",
        "left": "请把中文翻译为英文，只输出结果：今天天气不错。",
        "right": "请重构 src/app.py 文件，只输出修改后的代码。",
    },
    {
        "contrast_name": "图像编辑_vs_翻译",
        "left": "修改左边苹果的颜色，只输出编辑结果说明。",
        "right": "请把中文翻译为英文，只输出结果：苹果很甜。",
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


def forward_collect(model, tokenizer, text: str, layer_ids: List[int], intervention: dict | None = None) -> dict:
    layers = discover_layers(model)
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    intervention = intervention or {}

    for layer_id in layer_ids:
        dims = intervention.get("dims", []) if intervention.get("layer_id") == layer_id else []

        def make_hook(target_layer: int, target_dims: List[int]):
            def hook(module, inputs, output):
                hidden = output
                captured[target_layer] = hidden[0, -1, :].detach().float().cpu()
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
        "layer_payloads": captured,
        "final_logits": out.logits[0, -1, :].detach().float().cpu(),
    }


def mean_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def topk_dims(diff_vec: torch.Tensor, k: int = 32) -> List[int]:
    kk = min(k, diff_vec.numel())
    return torch.topk(diff_vec.abs(), kk).indices.detach().cpu().tolist()


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 20) -> float:
    ia = set(torch.topk(a, k).indices.detach().cpu().tolist())
    ib = set(torch.topk(b, k).indices.detach().cpu().tolist())
    return float(len(ia & ib) / max(1, len(ia | ib)))


def run_model(model_spec: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    anchors = anchor_layers(model)
    layer_ids = list(anchors.values())
    contrast_rows = []
    hot_dim_counter: Dict[int, int] = {}

    for pair in PROMPT_PAIRS:
        left = forward_collect(model, tokenizer, pair["left"], layer_ids)
        right = forward_collect(model, tokenizer, pair["right"], layer_ids)
        layer_rows = []
        diff_store: Dict[int, torch.Tensor] = {}
        for layer_name, layer_id in anchors.items():
            diff_vec = right["layer_payloads"][layer_id] - left["layer_payloads"][layer_id]
            diff_store[layer_id] = diff_vec
            layer_rows.append(
                {
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "mean_abs_delta": float(diff_vec.abs().mean().item()),
                }
            )
        strongest_layer = max(layer_rows, key=lambda row: row["mean_abs_delta"])
        strongest_layer_id = strongest_layer["layer_id"]
        dims = topk_dims(diff_store[strongest_layer_id], 32)
        for dim in dims:
            hot_dim_counter[dim] = hot_dim_counter.get(dim, 0) + 1

        intervened = forward_collect(
            model,
            tokenizer,
            pair["right"],
            layer_ids,
            intervention={"layer_id": strongest_layer_id, "dims": dims},
        )
        logits_shift = mean_abs_delta(right["final_logits"], intervened["final_logits"])
        overlap_after = topk_overlap(right["final_logits"], intervened["final_logits"], 20)
        contrast_rows.append(
            {
                "contrast_name": pair["contrast_name"],
                "strongest_layer_name": strongest_layer["layer_name"],
                "strongest_layer_id": strongest_layer_id,
                "baseline_contrast_score": strongest_layer["mean_abs_delta"],
                "intervention_logits_shift": logits_shift,
                "top20_overlap_after_intervention": overlap_after,
                "intervention_dim_count": len(dims),
                "top_dims": dims[:12],
            }
        )

    model_score = sum(row["intervention_logits_shift"] for row in contrast_rows) / len(contrast_rows)
    hottest = sorted(hot_dim_counter.items(), key=lambda item: (-item[1], item[0]))[:12]
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "parameter_intervention_score": model_score,
        "contrast_count": len(contrast_rows),
        "strongest_contrast_name": max(contrast_rows, key=lambda row: row["intervention_logits_shift"])["contrast_name"],
        "weakest_contrast_name": min(contrast_rows, key=lambda row: row["intervention_logits_shift"])["contrast_name"],
        "hot_dim_rows": [{"dim_index": int(dim), "hit_count": int(count)} for dim, count in hottest],
        "contrast_rows": contrast_rows,
    }


def build_summary() -> dict:
    model_rows = [run_model(spec) for spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda row: row["parameter_intervention_score"])
    weakest = min(model_rows, key=lambda row: row["parameter_intervention_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage269_parameter_position_causal_intervention_map",
        "title": "参数位置因果干预图",
        "status_short": "parameter_position_causal_intervention_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
        "top_gap_name": "高频差分位一旦被直接压低，任务路径和义项边界的输出分布会显著改写，说明这些位置不仅相关，而且具备因果杠杆",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage269 参数位置因果干预图",
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
                f"- 干预总分：{row['parameter_intervention_score']:.4f}",
                f"- 最强对照：{row['strongest_contrast_name']}",
                f"- 最弱对照：{row['weakest_contrast_name']}",
            ]
        )
    (output_dir / "STAGE269_PARAMETER_POSITION_CAUSAL_INTERVENTION_MAP_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="参数位置因果干预图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
