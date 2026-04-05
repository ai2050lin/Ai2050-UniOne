#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from multimodel_language_shared import (
    MODEL_SPECS,
    discover_layers,
    encode_to_device,
    evenly_spaced_layers,
    free_model,
    load_model_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage502_residual_fine_grained_propagation_triple_model_20260404"
)
MODEL_KEYS = ["qwen3", "deepseek7b", "glm4"]

PAIRS = [
    {
        "pair_id": "apple_polysemy",
        "name": "苹果多义切换",
        "text_a": "我把苹果洗干净后切成小块，准备做水果沙拉。",
        "text_b": "我买了一台苹果电脑，准备继续写代码。",
    },
    {
        "pair_id": "pronoun_route",
        "name": "代词路由差异",
        "text_a": "张三批评了李四，因为李四迟到了。后来经理让他写检查。",
        "text_b": "张三批评了李四，因为张三迟到了。后来经理让他写检查。",
    },
    {
        "pair_id": "connective_logic",
        "name": "连接词逻辑差异",
        "text_a": "如果明天下雨，我们就改在室内开会。",
        "text_b": "虽然明天下雨，我们还是在室外训练。",
    },
    {
        "pair_id": "quantity_change",
        "name": "数量变化差异",
        "text_a": "书架上原来有十本书，后来借走三本。",
        "text_b": "书架上原来有十本书，后来又新买三本。",
    },
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def capture_last_token_residual(model, tokenizer, text: str, layer_idx: int) -> np.ndarray:
    layers = discover_layers(model)
    captured = [None]

    def hook_fn(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured[0] = hidden[:, -1, :].detach().float().cpu().numpy()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        encoded = encode_to_device(model, tokenizer, text, max_length=128)
        with torch.inference_mode():
            model(**encoded)
    finally:
        handle.remove()

    if captured[0] is None:
        return np.zeros(1, dtype=np.float32)
    return captured[0][0]


def analyze_pair(model, tokenizer, pair: dict) -> dict:
    sample_layers = evenly_spaced_layers(model, count=5)
    diffs = {}
    for layer_idx in sample_layers:
        a = capture_last_token_residual(model, tokenizer, str(pair["text_a"]), layer_idx)
        b = capture_last_token_residual(model, tokenizer, str(pair["text_b"]), layer_idx)
        diffs[layer_idx] = a - b

    final_layer = sample_layers[-1]
    final_diff = diffs[final_layer]
    segment_rows = []
    faithful_count = 0
    rewrite_count = 0

    for left, right in zip(sample_layers[:-1], sample_layers[1:]):
        left_vec = diffs[left]
        right_vec = diffs[right]
        adjacent_cos = cosine(left_vec, right_vec)
        final_cos_left = cosine(left_vec, final_diff)
        final_cos_right = cosine(right_vec, final_diff)
        sign_flip = bool(final_cos_left * final_cos_right < 0)
        is_faithful = adjacent_cos >= 0.7 and not sign_flip
        if is_faithful:
            faithful_count += 1
        else:
            rewrite_count += 1
        segment_rows.append(
            {
                "from_layer": left,
                "to_layer": right,
                "adjacent_cosine": round(adjacent_cos, 6),
                "final_cosine_left": round(final_cos_left, 6),
                "final_cosine_right": round(final_cos_right, 6),
                "sign_flip": sign_flip,
                "segment_mode": "faithful" if is_faithful else "rewrite_or_rotate",
            }
        )

    return {
        "pair_id": pair["pair_id"],
        "pair_name": pair["name"],
        "sample_layers": sample_layers,
        "segment_rows": segment_rows,
        "faithful_segment_count": faithful_count,
        "rewrite_segment_count": rewrite_count,
        "faithful_segment_ratio": round(faithful_count / max(len(segment_rows), 1), 4),
        "mean_adjacent_cosine": round(statistics.mean(row["adjacent_cosine"] for row in segment_rows), 6),
        "mean_abs_final_cosine": round(
            statistics.mean(abs(row["final_cosine_right"]) for row in segment_rows),
            6,
        ),
    }


def run_model(model_key: str) -> dict:
    model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
    try:
        pair_rows = [analyze_pair(model, tokenizer, pair) for pair in PAIRS]
        return {
            "model_label": MODEL_SPECS[model_key]["label"],
            "layer_count": len(discover_layers(model)),
            "pair_rows": pair_rows,
            "aggregate": {
                "mean_faithful_segment_ratio": round(
                    statistics.mean(row["faithful_segment_ratio"] for row in pair_rows),
                    4,
                ),
                "mean_adjacent_cosine": round(
                    statistics.mean(row["mean_adjacent_cosine"] for row in pair_rows),
                    6,
                ),
                "mean_abs_final_cosine": round(
                    statistics.mean(row["mean_abs_final_cosine"] for row in pair_rows),
                    6,
                ),
            },
        }
    finally:
        free_model(model)


def build_report(summary: dict) -> str:
    lines = ["# stage502 更细粒度残差传播三模型测试", ""]
    for model_key, row in summary["models"].items():
        agg = row["aggregate"]
        lines.append(f"## {model_key}")
        lines.append("")
        lines.append(f"- 平均忠实片段比例: `{agg['mean_faithful_segment_ratio']}`")
        lines.append(f"- 平均相邻余弦: `{agg['mean_adjacent_cosine']}`")
        lines.append(f"- 平均最终方向绝对余弦: `{agg['mean_abs_final_cosine']}`")
        lines.append("")
    lines.append("## 综合结论")
    lines.append("")
    lines.append(f"- {summary['core_answer']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    start = time.time()
    models = {model_key: run_model(model_key) for model_key in MODEL_KEYS}
    summary = {
        "stage": "stage502_residual_fine_grained_propagation_triple_model",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "core_answer": (
            "如果大部分层间片段保持较高余弦并维持最终方向一致，那么残差传播更接近“忠实传播带局部旋转”，"
            "而不是“每过几层就完全重写一次”。"
        ),
        "elapsed_seconds": round(time.time() - start, 1),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
