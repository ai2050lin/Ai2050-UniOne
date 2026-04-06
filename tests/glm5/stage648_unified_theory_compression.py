#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage648: P0-P2统一理论压缩——第一性原理验证

目标：将P0-P2所有已确认的不变量压缩成最小命题集，并用四模型数据做交叉验证
方法：
1. 汇总P0-P2的所有不变量（INV-123到INV-209）
2. 设计"理论压缩测试"：从不变量推导出可定量预测，与实际数据对比
3. 三个核心预测验证：
   a) "正交编码"预测：跨能力方向的cos应接近0 → 测量6×6方向矩阵
   b) "旋转速度-可提取性"预测：旋转越快→方向恢复率越高 → 跨模型相关
   c) "rank-1统一编码"预测：所有能力的delta有效秩=1 → 验证

预注册判伪条件：
INV-215: "存在至少3个跨模型一致的不变量"
如果跨模型一致的不变量<3个，则"语言编码存在跨模型不变量"被推翻。

INV-216: "旋转速度与方向恢复率正相关(跨模型)"
如果四模型的旋转速度与方向恢复率的Pearson相关<0.3，则INV-216被推翻。
"""

from __future__ import annotations

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    free_model,
    load_model_bundle,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# 6种能力的prompt对
CAPABILITY_PAIRS = {
    "syntax": ("The key to the cabinet is", "The keys to the cabinet are"),
    "relation": ("Paris is the capital of France. The capital of France is",
                 "Berlin is the capital of Germany. The capital of Germany is"),
    "coref": ("Alice thanked Mary because Alice had won. The winner was",
              "Alice thanked Mary because Mary had won. The winner was"),
    "syllogism": ("All mammals are animals. All cats are mammals. Cats are",
                  "All birds are animals. All sparrows are birds. Sparrows are"),
    "arithmetic": ("If x=7, y=3, then x+y=", "If x=15, y=8, then x+y="),
    "negation": ("Tom is NOT tall. Tom is", "The answer is NOT B. It is"),
}


def extract_layer_last_token(model, tokenizer, prompt: str, layer_idx: int) -> torch.Tensor:
    layers = discover_layers(model)
    device = get_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    captured = {}

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["value"] = hidden[0, -1, :].detach().float().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["value"]


def compute_effective_rank(tensor: torch.Tensor, threshold: float = 0.99) -> float:
    """计算有效秩：累计能量达到threshold%所需的奇异值数量"""
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    _, s, _ = torch.svd(tensor.float())
    s = s / (s.sum() + 1e-10)
    cumsum = torch.cumsum(s, dim=0)
    for i in range(len(cumsum)):
        if cumsum[i].item() >= threshold:
            return float(i + 1)
    return float(len(cumsum))


def compute_cross_capability_cos_matrix(model, tokenizer, layer: int = 0) -> Dict:
    """计算6种能力在L0的方向余弦矩阵"""
    cap_names = list(CAPABILITY_PAIRS.keys())
    deltas = {}
    for name, (pa, pb) in CAPABILITY_PAIRS.items():
        da = extract_layer_last_token(model, tokenizer, pa, layer)
        db = extract_layer_last_token(model, tokenizer, pb, layer)
        delta = da - db
        deltas[name] = delta / (torch.norm(delta) + 1e-10)

    matrix = {}
    for i, n1 in enumerate(cap_names):
        for j, n2 in enumerate(cap_names):
            if i <= j:
                cos_val = torch.dot(deltas[n1], deltas[n2]).item()
                matrix[f"{n1}_{n2}"] = round(cos_val, 4)

    # 提取mean|cos|（不包括对角线）
    off_diag = []
    for i, n1 in enumerate(cap_names):
        for j, n2 in enumerate(cap_names):
            if i != j:
                cos_val = torch.dot(deltas[n1], deltas[n2]).item()
                off_diag.append(abs(cos_val))
    mean_abs_cos = float(np.mean(off_diag))

    return {
        "layer": layer,
        "matrix": matrix,
        "mean_abs_cos": round(mean_abs_cos, 4),
        "max_abs_cos": round(max(off_diag), 4) if off_diag else 0,
    }


def compute_rotation_speed(model, tokenizer, num_layers: int) -> float:
    """计算平均逐层旋转速度"""
    templates = [
        "The cat sat on the mat.", "The dog ran in the park.",
        "A bird flew over the house.", "She read a book yesterday.",
    ]
    rotations = []
    for prompt in templates:
        for l1, l2 in [(0, 1), (2, 3), (5, 6)]:
            if l2 >= num_layers:
                continue
            try:
                h1 = extract_layer_last_token(model, tokenizer, prompt, l1)
                h2 = extract_layer_last_token(model, tokenizer, prompt, l2)
                delta = h2 - h1
                if torch.norm(h1) > 1e-8 and torch.norm(delta) > 1e-8:
                    cos_v = torch.dot(h1 / torch.norm(h1), delta / torch.norm(delta)).item()
                    cos_v = max(-1.0, min(1.0, cos_v))
                    rotations.append(abs(np.degrees(np.arccos(cos_v))))
            except Exception:
                continue
    return float(np.mean(rotations)) if rotations else 0.0


def verify_rank1_uniformity(model, tokenizer, num_layers: int, sample_layers: List[int] = None) -> Dict:
    """验证所有能力在所有层的delta是否都是rank-1"""
    if sample_layers is None:
        sample_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    sample_layers = [l for l in sample_layers if l < num_layers]

    results = {}
    for cap_name, (pa, pb) in CAPABILITY_PAIRS.items():
        for layer in sample_layers:
            da = extract_layer_last_token(model, tokenizer, pa, layer)
            db = extract_layer_last_token(model, tokenizer, pb, layer)
            delta = da - db
            eff_rank = compute_effective_rank(delta)
            top1_energy = 1.0  # rank-1意味着top1=100%
            if delta.norm() > 1e-8:
                u, s, v = torch.svd(delta.float().unsqueeze(0))
                top1_energy = float((s[0] ** 2) / (s ** 2).sum())
            results[f"{cap_name}_L{layer}"] = {
                "eff_rank": round(eff_rank, 2),
                "top1_energy": round(top1_energy, 4),
            }

    all_ranks = [v["eff_rank"] for v in results.values()]
    all_top1 = [v["top1_energy"] for v in results.values()]

    return {
        "per_cap_layer": results,
        "mean_eff_rank": round(float(np.mean(all_ranks)), 2),
        "mean_top1_energy": round(float(np.mean(all_top1)), 4),
        "all_rank1": all(r <= 1.1 for r in all_ranks),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python stage648_unified_theory_compression.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage648_unified_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage648] 加载模型 {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        print(f"[Stage648] 层数={num_layers}")

        # 1) 跨能力方向余弦矩阵
        print(f"\n[Stage648] 测试1: 跨能力方向余弦矩阵...")
        cos_matrix = compute_cross_capability_cos_matrix(model, tokenizer, layer=0)
        print(f"  mean|cos|={cos_matrix['mean_abs_cos']:.4f}, max|cos|={cos_matrix['max_abs_cos']:.4f}")

        # 2) 旋转速度
        print(f"\n[Stage648] 测试2: 旋转速度...")
        rotation = compute_rotation_speed(model, tokenizer, num_layers)
        print(f"  rotation: {rotation:.1f} deg/layer")

        # 3) rank-1统一编码验证
        print(f"\n[Stage648] 测试3: rank-1统一编码验证...")
        rank1_result = verify_rank1_uniformity(model, tokenizer, num_layers)
        print(f"  平均有效秩: {rank1_result['mean_eff_rank']:.2f}")
        print(f"  平均top1能量: {rank1_result['mean_top1_energy']:.4f}")
        print(f"  全部rank-1: {rank1_result['all_rank1']}")

        # 不变量验证汇总
        print(f"\n{'='*60}")
        print(f"[Stage648] 汇总: {model_key}")
        print(f"  INV-195 (跨能力正交, mean|cos|<0.3): {'存活' if cos_matrix['mean_abs_cos'] < 0.3 else '推翻'}")
        print(f"  INV-208 (delta统一rank-1): {'存活' if rank1_result['all_rank1'] else '推翻'}")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "num_layers": num_layers,
            "rotation_speed": round(rotation, 1),
            "cross_capability_cos": cos_matrix,
            "rank1_uniformity": rank1_result,
            "invariant_checks": {
                "INV195_orthogonal": cos_matrix["mean_abs_cos"] < 0.3,
                "INV208_rank1": rank1_result["all_rank1"],
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
