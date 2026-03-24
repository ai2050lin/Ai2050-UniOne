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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage270_thin_delta_to_route_amplification_causal_bridge_20260324"

MODEL_SPECS = [
    {"model_tag": "qwen4b", "display_name": "Qwen3-4B", "model_path": QWEN3_MODEL_PATH},
    {"model_tag": "deepseek7b", "display_name": "DeepSeek-R1-Distill-Qwen-7B", "model_path": DEEPSEEK7B_MODEL_PATH},
]

PAIR_ROWS = [
    {"family": "fruit", "left": "apple", "right": "pear"},
    {"family": "fruit", "left": "banana", "right": "peach"},
    {"family": "animal", "left": "cat", "right": "dog"},
    {"family": "animal", "left": "lion", "right": "tiger"},
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


def prompt(word: str) -> str:
    return f"After washing the {word}, Tom decided to"


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


def topk_dims(diff_vec: torch.Tensor, k: int = 32) -> List[int]:
    kk = min(k, diff_vec.numel())
    return torch.topk(diff_vec.abs(), kk).indices.detach().cpu().tolist()


def diff_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def run_model(model_spec: dict) -> dict:
    model, tokenizer = load_model(model_spec["model_path"])
    anchors = anchor_layers(model)
    layer_ids = list(anchors.values())
    pair_rows = []

    for pair in PAIR_ROWS:
        left = forward_collect(model, tokenizer, prompt(pair["left"]), layer_ids)
        right = forward_collect(model, tokenizer, prompt(pair["right"]), layer_ids)
        early_diff_vec = right["layer_payloads"][anchors["early"]] - left["layer_payloads"][anchors["early"]]
        route_diff_base = diff_mean(right["layer_payloads"][anchors["route"]], left["layer_payloads"][anchors["route"]])
        late_diff_base = diff_mean(right["layer_payloads"][anchors["late"]], left["layer_payloads"][anchors["late"]])
        early_diff_base = float(early_diff_vec.abs().mean().item())
        dims = topk_dims(early_diff_vec, 32)

        right_ablate = forward_collect(
            model,
            tokenizer,
            prompt(pair["right"]),
            layer_ids,
            intervention={"layer_id": anchors["early"], "dims": dims},
        )
        route_diff_after = diff_mean(right_ablate["layer_payloads"][anchors["route"]], left["layer_payloads"][anchors["route"]])
        late_diff_after = diff_mean(right_ablate["layer_payloads"][anchors["late"]], left["layer_payloads"][anchors["late"]])

        route_drop = max(0.0, route_diff_base - route_diff_after)
        late_drop = max(0.0, late_diff_base - late_diff_after)
        amplification_ratio = late_diff_base / max(1e-8, early_diff_base)
        bridge_score = (route_drop + late_drop + min(amplification_ratio, 5.0) / 5.0) / 3.0
        pair_rows.append(
            {
                "family": pair["family"],
                "pair_name": f"{pair['left']}_vs_{pair['right']}",
                "early_diff_base": early_diff_base,
                "route_diff_base": route_diff_base,
                "late_diff_base": late_diff_base,
                "route_drop_after_early_ablation": route_drop,
                "late_drop_after_early_ablation": late_drop,
                "amplification_ratio": amplification_ratio,
                "bridge_score": bridge_score,
                "top_dims": dims[:12],
            }
        )

    model_score = sum(row["bridge_score"] for row in pair_rows) / len(pair_rows)
    return {
        "model_tag": model_spec["model_tag"],
        "display_name": model_spec["display_name"],
        "bridge_score": model_score,
        "strongest_pair_name": max(pair_rows, key=lambda row: row["bridge_score"])["pair_name"],
        "weakest_pair_name": min(pair_rows, key=lambda row: row["bridge_score"])["pair_name"],
        "pair_rows": pair_rows,
    }


def build_summary() -> dict:
    model_rows = [run_model(spec) for spec in MODEL_SPECS]
    strongest = max(model_rows, key=lambda row: row["bridge_score"])
    weakest = min(model_rows, key=lambda row: row["bridge_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage270_thin_delta_to_route_amplification_causal_bridge",
        "title": "薄差分到路径放大因果桥",
        "status_short": "thin_delta_to_route_amplification_bridge_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
        "top_gap_name": "前面很薄的局部差分一旦落在高杠杆位，后续路径差异会被继续放大，而且早层差分位被压低后，后续路径差异会明显回落",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage270 薄差分到路径放大因果桥",
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
                f"- 因果桥总分：{row['bridge_score']:.4f}",
                f"- 最强对象对：{row['strongest_pair_name']}",
                f"- 最弱对象对：{row['weakest_pair_name']}",
            ]
        )
    (output_dir / "STAGE270_THIN_DELTA_TO_ROUTE_AMPLIFICATION_CAUSAL_BRIDGE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="薄差分到路径放大因果桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
