#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage651: 统一旋转速度测量协议 + 残差流分形分析

目标：
1. 用三种不同的旋转速度公式统一测量，找到最稳定的公式
2. 分析残差流中方向衰减后的"信息保留形式"

三种旋转速度公式：
  A) cos(h_l, h_{l+1}) = h_l . h_{l+1} / (|h_l| |h_{l+1}|)
     翻译：相邻层hidden state的余弦相似度，旋转角 = arccos(cos)
  B) cos(h_l, delta_l) = h_l . (h_{l+1} - h_l) / (|h_l| |delta|)
     翻译：hidden state与层间变化量的夹角
  C) cos(delta_l, delta_{l+1}) = (h_{l+1}-h_l) . (h_{l+2}-h_{l+1}) / (|delta_l| |delta_{l+1}|)
     翻译：相邻层变化量之间的夹角（变化方向的旋转）

残差流分形分析：
  在方向cos衰减到≈0后，检查残差流的高阶统计量（范数、偏度、能量分布）
  是否仍然携带任务相关信息

预注册判伪条件：
INV-219: "三种公式给出的旋转速度排序一致"
如果四种模型在三种公式下的排序不同，则INV-219被推翻。

INV-220: "残差流范数在方向衰减后仍保留任务差异"
如果方向cos≈0的层中，A/B版本的残差流范数差异<5%，则INV-220被推翻。
"""

from __future__ import annotations

import sys
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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


def extract_all_layers_hs(model, tokenizer, prompt: str) -> Dict[int, torch.Tensor]:
    """提取所有层的最后一个token的hidden state"""
    layers = discover_layers(model)
    device = get_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    captured = {}

    def make_hook(lidx):
        def hook_fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[lidx] = hidden[0, -1, :].detach().float().cpu()
        return hook_fn

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(len(layers))]
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return captured


def compute_rotation_methods(hs_dict: Dict[int, torch.Tensor]) -> Dict[str, float]:
    """用三种公式计算旋转速度"""
    sorted_layers = sorted(hs_dict.keys())
    method_a_angles = []  # cos(h_l, h_{l+1})
    method_b_angles = []  # cos(h_l, delta_l)
    method_c_angles = []  # cos(delta_l, delta_{l+1})

    for i in range(len(sorted_layers) - 1):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i + 1]
        h1 = hs_dict[l1]
        h2 = hs_dict[l2]
        delta = h2 - h1

        # Method A: cos(h1, h2)
        if h1.norm() > 1e-8 and h2.norm() > 1e-8:
            cos_a = torch.dot(h1, h2) / (h1.norm() * h2.norm())
            cos_a = max(-1.0, min(1.0, cos_a.item()))
            method_a_angles.append(np.degrees(np.arccos(cos_a)))

        # Method B: cos(h1, delta)
        if h1.norm() > 1e-8 and delta.norm() > 1e-8:
            cos_b = torch.dot(h1, delta) / (h1.norm() * delta.norm())
            cos_b = max(-1.0, min(1.0, cos_b.item()))
            method_b_angles.append(abs(np.degrees(np.arccos(cos_b))))

        # Method C: cos(delta_l, delta_{l+1}) - 需要l2到l3
        if i < len(sorted_layers) - 2:
            l3 = sorted_layers[i + 2]
            h3 = hs_dict[l3]
            delta2 = h3 - h2
            if delta.norm() > 1e-8 and delta2.norm() > 1e-8:
                cos_c = torch.dot(delta, delta2) / (delta.norm() * delta2.norm())
                cos_c = max(-1.0, min(1.0, cos_c.item()))
                method_c_angles.append(np.degrees(np.arccos(cos_c)))

    return {
        "A_hl_hl1": round(float(np.mean(method_a_angles)), 1) if method_a_angles else None,
        "B_hl_delta": round(float(np.mean(method_b_angles)), 1) if method_b_angles else None,
        "C_delta_delta": round(float(np.mean(method_c_angles)), 1) if method_c_angles else None,
    }


def analyze_residual_fractal(hs_a: Dict[int, torch.Tensor], hs_b: Dict[int, torch.Tensor]) -> Dict:
    """分析残差流的分形/高阶特征"""
    sorted_layers = sorted(hs_a.keys())
    results = {}

    for l in sorted_layers:
        ha = hs_a[l]
        hb = hs_b[l]
        delta = ha - hb
        cos_ab = (torch.dot(ha, hb) / (ha.norm() * hb.norm() + 1e-10)).item()

        # 高阶统计
        norm_diff_pct = abs(ha.norm().item() - hb.norm().item()) / max(ha.norm().item(), 1e-8) * 100

        # 能量分布：前10个PCA成分的能量占比
        delta_2d = delta.unsqueeze(0)
        try:
            _, s, _ = torch.linalg.svd(delta_2d, full_matrices=False)
            total_e = s.sum().item()
            top1_e = (s[0] ** 2 / (s ** 2).sum()).item() if total_e > 1e-10 else 1.0
            top5_e = (s[:5].pow(2).sum() / s.pow(2).sum()).item() if total_e > 1e-10 else 1.0
        except Exception:
            top1_e, top5_e = 1.0, 1.0

        results[str(l)] = {
            "cos_ab": round(cos_ab, 4),
            "norm_diff_pct": round(norm_diff_pct, 2),
            "delta_norm": round(delta.norm().item(), 4),
            "top1_energy": round(top1_e, 4),
            "top5_energy": round(top5_e, 4),
            "ha_norm": round(ha.norm().item(), 2),
            "hb_norm": round(hb.norm().item(), 2),
        }

    # 在cos≈0的层中检查信息保留
    decayed_layers = [(int(k), v) for k, v in results.items() if abs(v["cos_ab"]) < 0.1 and int(k) > 3]
    if decayed_layers:
        avg_norm_diff = statistics.mean([v["norm_diff_pct"] for _, v in decayed_layers])
        avg_delta_norm = statistics.mean([v["delta_norm"] for _, v in decayed_layers])
    else:
        avg_norm_diff = 0
        avg_delta_norm = 0

    return {
        "per_layer": results,
        "decayed_layers_count": len(decayed_layers),
        "avg_norm_diff_in_decayed_pct": round(avg_norm_diff, 2),
        "avg_delta_norm_in_decayed": round(avg_delta_norm, 4),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python stage651_unified_rotation_fractal.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage651_rotation_fractal_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage651] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        print(f"[Stage651] layers={num_layers}")

        # 测试用例
        test_pairs = [
            ("syntax",
             "The key to the cabinet is",
             "The keys to the cabinet are"),
            ("syllogism",
             "All mammals are animals. All cats are mammals. Cats are animals",
             "All birds are animals. All sparrows are birds. Sparrows are insects"),
        ]

        rotation_results = {}
        fractal_results = {}

        for cap_name, pa, pb in test_pairs:
            print(f"\n[Stage651] {cap_name}...")

            # 提取所有层hidden state
            hs_a = extract_all_layers_hs(model, tokenizer, pa)
            hs_b = extract_all_layers_hs(model, tokenizer, pb)

            # 旋转速度（用A版本）
            rot = compute_rotation_methods(hs_a)
            rotation_results[cap_name] = rot
            print(f"  rot: A={rot['A_hl_hl1']}, B={rot['B_hl_delta']}, C={rot['C_delta_delta']}")

            # 分形分析
            fractal = analyze_residual_fractal(hs_a, hs_b)
            fractal_results[cap_name] = fractal
            decayed = fractal["decayed_layers_count"]
            norm_diff = fractal["avg_norm_diff_in_decayed_pct"]
            print(f"  decayed_layers={decayed}, norm_diff_in_decayed={norm_diff:.1f}%")

        # INV-219: 排序一致性
        models_order = {}
        for method in ["A_hl_hl1", "B_hl_delta", "C_delta_delta"]:
            vals = [(cap, rotation_results[cap].get(method, 0) or 0) for cap in rotation_results]
            vals.sort(key=lambda x: x[1])
            models_order[method] = [v[0] for v in vals]
        # 检查三种方法是否给出相同的cap排序（此处比较的是不同能力间的排序，而非模型间）
        # 对于单模型，我们比较三种方法是否一致
        avg_a = statistics.mean([rotation_results[c]["A_hl_hl1"] or 0 for c in rotation_results])
        avg_b = statistics.mean([rotation_results[c]["B_hl_delta"] or 0 for c in rotation_results])
        avg_c = statistics.mean([rotation_results[c]["C_delta_delta"] or 0 for c in rotation_results])
        print(f"\n  avg rotation: A={avg_a:.1f}, B={avg_b:.1f}, C={avg_c:.1f}")

        # INV-220: 残差流范数差异
        avg_norm_diff = statistics.mean([fractal_results[c]["avg_norm_diff_in_decayed_pct"] for c in fractal_results])
        inv220 = "SURVIVED" if avg_norm_diff > 5.0 else "FALSIFIED"
        print(f"  INV-220 (norm_diff>5%): {inv220} (avg={avg_norm_diff:.1f}%)")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "num_layers": num_layers,
            "rotation_methods": rotation_results,
            "avg_rotation": {
                "A": round(avg_a, 1),
                "B": round(avg_b, 1),
                "C": round(avg_c, 1),
            },
            "fractal_analysis": {k: {"summary": {
                "decayed_layers": v["decayed_layers_count"],
                "norm_diff_pct": v["avg_norm_diff_in_decayed_pct"],
                "delta_norm": v["avg_delta_norm_in_decayed"],
            }} for k, v in fractal_results.items()},
            "inv220_result": inv220,
            "avg_norm_diff_pct": round(avg_norm_diff, 2),
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
