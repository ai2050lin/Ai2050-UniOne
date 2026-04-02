#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import plotly.graph_objects as go
import torch

from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, discover_layers, load_qwen_like_model
from stage435_apple_feature_binding_neuron_channels import capture_case_flat_neuron_vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage438_apple_neuron_role_3d_view_20260402"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]
TOP_NEURONS_PER_ROLE = 220

ROLE_COLORS = {
    "noun_backbone": "#2E8B57",
    "sense_switch": "#1F77B4",
    "attribute_modifier": "#D62728",
    "mixed": "#7F7F7F",
}

GROUP_CASES = {
    "apple_noun": [
        {"target": "apple", "sentence": "This is an apple."},
        {"target": "apple", "sentence": "I bought an apple today."},
        {"target": "apple", "sentence": "The apple rolled across the table."},
    ],
    "banana_noun": [
        {"target": "banana", "sentence": "This is a banana."},
        {"target": "banana", "sentence": "I bought a banana today."},
        {"target": "banana", "sentence": "The banana rested on the table."},
    ],
    "fruit_family": [
        {"target": "orange", "sentence": "This is an orange."},
        {"target": "pear", "sentence": "This is a pear."},
        {"target": "grape", "sentence": "This is a grape."},
        {"target": "peach", "sentence": "This is a peach."},
    ],
    "generic_noun": [
        {"target": "car", "sentence": "This is a car."},
        {"target": "book", "sentence": "This is a book."},
        {"target": "chair", "sentence": "This is a chair."},
        {"target": "hammer", "sentence": "This is a hammer."},
    ],
    "apple_fruit_neutral": [
        {"target": "apple", "sentence": "I bought an apple this morning."},
        {"target": "apple", "sentence": "The apple is on the table."},
    ],
    "apple_brand_neutral": [
        {"target": "Apple", "sentence": "Apple announced a new product today."},
        {"target": "Apple", "sentence": "Apple released another update this year."},
    ],
    "color_attr": [
        {"target": "red", "sentence": "The color is red."},
        {"target": "yellow", "sentence": "The color is yellow."},
        {"target": "green", "sentence": "The color is green."},
    ],
    "taste_attr": [
        {"target": "sweet", "sentence": "The taste is sweet."},
        {"target": "sour", "sentence": "The taste is sour."},
        {"target": "juicy", "sentence": "The flavor feels juicy."},
    ],
    "size_anchor": [
        {"target": "fist", "sentence": "The size is about a fist."},
        {"target": "hand", "sentence": "The size is about a hand."},
        {"target": "palm", "sentence": "The size is about a palm."},
    ],
}


def free_model(model) -> None:
    try:
        del model
    except UnboundLocalError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def mean_tensors(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([vec.float() for vec in vectors], dim=0).mean(dim=0)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def collect_group_means(model, tokenizer) -> Dict[str, torch.Tensor]:
    out = {}
    for group_name, cases in GROUP_CASES.items():
        vectors = [capture_case_flat_neuron_vector(model, tokenizer, case["sentence"], case["target"]) for case in cases]
        out[group_name] = mean_tensors(vectors)
    return out


def standardize(vec: torch.Tensor) -> torch.Tensor:
    v = vec.float()
    denom = v.abs().mean().clamp_min(1e-8)
    return v / denom


def dominant_role(x: float, y: float, z: float) -> str:
    vals = {
        "noun_backbone": abs(x),
        "sense_switch": abs(y),
        "attribute_modifier": abs(z),
    }
    ordered = sorted(vals.items(), key=lambda item: item[1], reverse=True)
    top_name, top_val = ordered[0]
    second_val = ordered[1][1]
    if top_val <= 1e-8:
        return "mixed"
    if second_val > 0 and top_val / second_val < 1.15:
        return "mixed"
    return top_name


def build_neuron_table(group_means: Dict[str, torch.Tensor], neuron_count: int) -> List[Dict[str, object]]:
    fruit_common = mean_tensors(
        [
            group_means["apple_noun"],
            group_means["banana_noun"],
            group_means["fruit_family"],
        ]
    )
    generic = group_means["generic_noun"]
    fruit_neutral = group_means["apple_fruit_neutral"]
    brand_neutral = group_means["apple_brand_neutral"]
    attr_common = mean_tensors(
        [
            group_means["color_attr"],
            group_means["taste_attr"],
            group_means["size_anchor"],
        ]
    )

    noun_backbone_score = standardize(torch.clamp(fruit_common - generic, min=0.0))
    sense_switch_score = standardize(brand_neutral - fruit_neutral)
    attribute_modifier_score = standardize(torch.clamp(attr_common - generic, min=0.0))

    top_ids = set()
    for vec in [noun_backbone_score.abs(), sense_switch_score.abs(), attribute_modifier_score.abs()]:
        take_k = min(TOP_NEURONS_PER_ROLE, vec.numel())
        _, idxs = torch.topk(vec, k=take_k)
        top_ids.update(int(idx) for idx in idxs.tolist())

    rows = []
    for flat_idx in sorted(top_ids):
        layer_idx = int(flat_idx // neuron_count)
        neuron_idx = int(flat_idx % neuron_count)
        x = float(noun_backbone_score[flat_idx].item())
        y = float(sense_switch_score[flat_idx].item())
        z = float(attribute_modifier_score[flat_idx].item())
        rows.append(
            {
                "flat_index": flat_idx,
                "layer_index": layer_idx,
                "neuron_index": neuron_idx,
                "noun_backbone_score": x,
                "sense_switch_score": y,
                "attribute_modifier_score": z,
                "dominant_role": dominant_role(x, y, z),
            }
        )
    return rows


def build_plot(rows: Sequence[Dict[str, object]], model_label: str) -> go.Figure:
    traces = []
    for role_name in ["noun_backbone", "sense_switch", "attribute_modifier", "mixed"]:
        part = [row for row in rows if row["dominant_role"] == role_name]
        if not part:
            continue
        traces.append(
            go.Scatter3d(
                x=[row["noun_backbone_score"] for row in part],
                y=[row["sense_switch_score"] for row in part],
                z=[row["attribute_modifier_score"] for row in part],
                mode="markers",
                name=role_name,
                marker={
                    "size": 5 if role_name != "mixed" else 4,
                    "opacity": 0.78 if role_name != "mixed" else 0.42,
                    "color": ROLE_COLORS[role_name],
                },
                text=[
                    f"L{row['layer_index']} N{row['neuron_index']}<br>"
                    f"骨干={row['noun_backbone_score']:.3f}<br>"
                    f"切换={row['sense_switch_score']:.3f}<br>"
                    f"属性={row['attribute_modifier_score']:.3f}"
                    for row in part
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{model_label} 苹果机制神经元 3D 角色图",
        scene={
            "xaxis_title": "名词骨干分数",
            "yaxis_title": "词义切换分数",
            "zaxis_title": "属性修饰分数",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.2}},
        },
        legend={"orientation": "h"},
        margin={"l": 0, "r": 0, "b": 0, "t": 48},
    )
    return fig


def summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    role_counts: Dict[str, int] = {}
    for role_name in ["noun_backbone", "sense_switch", "attribute_modifier", "mixed"]:
        role_counts[role_name] = sum(1 for row in rows if row["dominant_role"] == role_name)
    top_backbone = sorted(rows, key=lambda row: abs(float(row["noun_backbone_score"])), reverse=True)[:12]
    top_switch = sorted(rows, key=lambda row: abs(float(row["sense_switch_score"])), reverse=True)[:12]
    top_attr = sorted(rows, key=lambda row: abs(float(row["attribute_modifier_score"])), reverse=True)[:12]
    return {
        "neuron_count_visualized": len(rows),
        "role_counts": role_counts,
        "top_backbone_neurons": top_backbone,
        "top_switch_neurons": top_switch,
        "top_attribute_neurons": top_attr,
    }


def analyze_model(model_key: str, *, prefer_cuda: bool, output_dir: Path) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
    try:
        layers = discover_layers(model)
        neuron_count = int(layers[0].mlp.gate_proj.out_features)
        group_means = collect_group_means(model, tokenizer)
        rows = build_neuron_table(group_means, neuron_count)
        figure = build_plot(rows, MODEL_SPECS[model_key]["model_name"])
        html_path = output_dir / f"{model_key}_neuron_role_3d.html"
        figure.write_html(str(html_path), include_plotlyjs="cdn")
        json_path = output_dir / f"{model_key}_neuron_role_3d_points.json"
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8-sig")
        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "layer_count": len(layers),
            "neurons_per_layer": neuron_count,
            "html_path": str(html_path),
            "points_path": str(json_path),
            "summary": summarize_rows(rows),
        }
    finally:
        free_model(model)


def build_index(model_results: Sequence[Dict[str, object]], output_dir: Path) -> Path:
    lines = [
        "# 苹果机制神经元 3D 可视化",
        "",
        "下列文件可直接在浏览器中打开：",
        "",
    ]
    for row in model_results:
        lines.append(f"- {row['model_name']}: `{Path(row['html_path']).name}`")
    path = output_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apple neuron role 3D visualization")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    prefer_cuda = not bool(args.cpu)
    model_results = [analyze_model(model_key, prefer_cuda=prefer_cuda, output_dir=output_dir) for model_key in MODEL_ORDER]
    index_path = build_index(model_results, output_dir)
    elapsed = time.time() - start_time
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage438_apple_neuron_role_3d_view",
        "title": "苹果名词骨干-词义切换-属性修饰三轴神经元 3D 可视化",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "model_results": model_results,
        "index_path": str(index_path),
        "top_neurons_per_role": TOP_NEURONS_PER_ROLE,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


if __name__ == "__main__":
    main()
