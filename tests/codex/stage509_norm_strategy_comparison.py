#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage509: LayerNorm vs RMSNorm 深层表示对比
核心问题：Gemma4(RMSNorm)深层不坍缩，Qwen3/DeepSeek(LayerNorm)深层坍缩——
         归一化策略如何影响深层概念表示？

分析维度：
- N1: 四个模型的归一化策略检测
- N2: 归一化操作对残差流norm的实际影响
- N3: 归一化后hidden states的方向保留度（cos(a, normalize(a))）
- N4: 深层概念分离度——归一化前后对比
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex.qwen3_language_shared import (
    load_qwen3_model, load_glm4_model, load_gemma4_model, load_deepseek7b_model,
    discover_layers, get_model_device,
)


CONCEPT_WORDS = ["苹果", "水果", "猫", "动物", "城市", "地点"]


def get_sample_layers(model) -> List[int]:
    n = len(discover_layers(model))
    if n <= 12:
        return list(range(n))
    return [max(0, min(n - 1, round((n - 1) * i / 11))) for i in range(12)]


def get_hidden_at_last_token(model, tokenizer, text: str, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    device = get_model_device(model)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    result = {}
    for li in layer_indices:
        h = outputs.hidden_states[li + 1][:, -1, :].float().cpu()
        result[li] = h
    return result


def detect_norm_type(model) -> str:
    """检测模型使用的归一化类型"""
    layers = discover_layers(model)
    if not layers:
        return "unknown"
    layer0 = layers[0]

    # 检查RMSNorm
    for name, module in layer0.named_modules():
        cls_name = module.__class__.__name__
        if "RMSNorm" in cls_name:
            return "RMSNorm"
        if "LayerNorm" in cls_name and "RMS" not in cls_name:
            return "LayerNorm"

    # 检查model.config
    if hasattr(model, "config"):
        cfg = model.config
        for attr in ["hidden_act", "normalization"]:
            val = getattr(cfg, attr, None)
            if val:
                return f"config:{val}"

    # 检查model.model中的norm
    if hasattr(model, "model"):
        for name, module in model.model.named_modules():
            cls_name = module.__class__.__name__
            if "RMSNorm" in cls_name:
                return "RMSNorm"
            if "LayerNorm" in cls_name:
                return "LayerNorm"

    return "unknown"


def analyze_N1_norm_detection(model, model_name):
    """N1: 检测每个模型的归一化策略"""
    norm_type = detect_norm_type(model)

    # 额外信息：检查模型配置
    config_info = {}
    if hasattr(model, "config"):
        cfg = model.config
        for key in ["hidden_size", "num_hidden_layers", "intermediate_size",
                     "hidden_act", "torch_dtype", "norm_epsilon"]:
            if hasattr(cfg, key):
                val = getattr(cfg, key)
                config_info[key] = str(val) if not isinstance(val, (int, float)) else val

    return {
        "norm_type": norm_type,
        "config": config_info,
    }


def analyze_N2_norm_impact(model, tokenizer, layer_indices):
    """N2: 每层hidden state的norm分布——实际影响"""
    results = {}
    for w in CONCEPT_WORDS[:3]:
        hs = get_hidden_at_last_token(model, tokenizer, w, layer_indices)
        for li in layer_indices:
            h = hs[li]
            norm = torch.norm(h, p=2).item()
            # 模拟LayerNorm的效果
            h_layernormed = F.layer_norm(h, h.shape)
            # 模拟RMSNorm的效果
            rms = torch.sqrt(torch.mean(h ** 2))
            h_rmsnormed = h / (rms + 1e-8)

            ln_norm = torch.norm(h_layernormed, p=2).item()
            rn_norm = torch.norm(h_rmsnormed, p=2).item()

            key = f"L{li}"
            if key not in results:
                results[key] = {}
            results[key][w] = {
                "raw_norm": round(norm, 4),
                "after_layernorm": round(ln_norm, 4),
                "after_rmsnorm": round(rn_norm, 4),
                "norm_reduction_factor": round(norm / max(ln_norm, 1e-8), 4),
            }

    return results


def analyze_N3_direction_preservation(model, tokenizer, layer_indices):
    """N3: 归一化后方向保留度——cos(raw, normalized)"""
    results = {}
    for w in CONCEPT_WORDS[:4]:
        hs = get_hidden_at_last_token(model, tokenizer, w, layer_indices)
        for li in layer_indices:
            h = hs[li].squeeze(0)
            h_norm = F.normalize(h, p=2, dim=0)
            direction_cos = F.cosine_similarity(h.unsqueeze(0), h_norm.unsqueeze(0), dim=1).item()

            key = f"L{li}"
            if key not in results:
                results[key] = {}
            results[key][w] = round(direction_cos, 6)

    return results


def analyze_N4_normalized_separation(model, tokenizer, layer_indices):
    """N4: 归一化后的概念分离度——逐层"""
    # 具体vs大类
    specific = ["苹果", "猫", "城市"]
    general = ["食物", "生物", "地点"]
    unrelated = ["水", "书", "太阳"]

    all_words = specific + general + unrelated

    results = {}
    for li in layer_indices:
        # 收集并归一化
        vecs = {}
        for w in all_words:
            hs = get_hidden_at_last_token(model, tokenizer, w, layer_indices)
            vecs[w] = F.normalize(hs[li].squeeze(0), p=2, dim=0)

        # 三种关系的平均相似度
        intra_specific = []
        intra_general = []
        cross_level = []
        unrelated_pairs = []

        for i, w1 in enumerate(specific):
            for j, w2 in enumerate(specific):
                if i < j:
                    intra_specific.append(F.cosine_similarity(
                        vecs[w1].unsqueeze(0), vecs[w2].unsqueeze(0), dim=1).item())
        for i, w1 in enumerate(general):
            for j, w2 in enumerate(general):
                if i < j:
                    intra_general.append(F.cosine_similarity(
                        vecs[w1].unsqueeze(0), vecs[w2].unsqueeze(0), dim=1).item())
        for w1 in specific:
            for w2 in general:
                cross_level.append(F.cosine_similarity(
                    vecs[w1].unsqueeze(0), vecs[w2].unsqueeze(0), dim=1).item())
        for w1 in specific:
            for w2 in unrelated:
                unrelated_pairs.append(F.cosine_similarity(
                    vecs[w1].unsqueeze(0), vecs[w2].unsqueeze(0), dim=1).item())

        results[f"L{li}"] = {
            "intra_specific": round(float(np.mean(intra_specific)), 4) if intra_specific else 0,
            "intra_general": round(float(np.mean(intra_general)), 4) if intra_general else 0,
            "cross_level": round(float(np.mean(cross_level)), 4) if cross_level else 0,
            "unrelated": round(float(np.mean(unrelated_pairs)), 4) if unrelated_pairs else 0,
            "specific_vs_unrelated": round(
                (float(np.mean(intra_specific)) - float(np.mean(unrelated_pairs))), 4
            ) if intra_specific and unrelated_pairs else 0,
        }

    return results


def load_model(model_name: str):
    if model_name == "qwen3":
        return load_qwen3_model()
    elif model_name == "glm4":
        return load_glm4_model()
    elif model_name == "deepseek7b":
        return load_deepseek7b_model()
    elif model_name == "gemma4":
        return load_gemma4_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"[Stage509] LayerNorm vs RMSNorm | model: {model_name}")
    print(f"  time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    t0 = time.time()
    model, tokenizer = load_model(model_name)
    layer_indices = get_sample_layers(model)
    print(f"  layers: {len(discover_layers(model))}, sample: {layer_indices}")

    summary = {"model": model_name, "timestamp": datetime.now().isoformat()}

    # N1: 归一化检测
    print("  N1 norm detection...", end=" ", flush=True)
    n1 = analyze_N1_norm_detection(model, model_name)
    summary.update({"N1_norm_detection": n1})
    print(f"done ({n1['norm_type']})")

    # N2: norm影响
    print("  N2 norm impact...", end=" ", flush=True)
    t1 = time.time()
    n2 = analyze_N2_norm_impact(model, tokenizer, layer_indices)
    summary.update({"N2_norm_impact": n2})
    print(f"done ({time.time()-t1:.1f}s)")

    # N3: 方向保留
    print("  N3 direction preservation...", end=" ", flush=True)
    t1 = time.time()
    n3 = analyze_N3_direction_preservation(model, tokenizer, layer_indices)
    summary.update({"N3_direction_preservation": n3})
    print(f"done ({time.time()-t1:.1f}s)")

    # N4: 归一化分离度
    print("  N4 normalized separation...", end=" ", flush=True)
    t1 = time.time()
    n4 = analyze_N4_normalized_separation(model, tokenizer, layer_indices)
    summary.update({"N4_normalized_separation": n4})
    print(f"done ({time.time()-t1:.1f}s)")

    # 保存
    out_dir = Path("tests/codex_temp") / f"stage509_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{model_name}.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(convert(summary), f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Done! {elapsed:.1f}s -> {out_path}")

    # 打印关键发现
    print(f"\n=== Key Findings ({model_name}) ===")
    print(f"  Norm type: {n1['norm_type']}")
    if n1.get('config'):
        for k, v in n1['config'].items():
            print(f"    {k}: {v}")

    n2_first = n2.get("L0", {})
    n2_last_key = sorted(n2.keys())[-1] if n2 else "N/A"
    n2_last = n2.get(n2_last_key, {})
    if n2_first and n2_last:
        w0 = list(n2_first.keys())[0]
        print(f"  Norm: L0={n2_first[w0]['raw_norm']:.1f} -> L{layer_indices[-1]}={n2_last.get(w0,{}).get('raw_norm','N/A')}")

    # N4最佳分离层
    n4_best = max(n4.items(), key=lambda x: x[1].get("specific_vs_unrelated", 0))
    print(f"  N4 best normalized separation: {n4_best[0]} "
          f"(spec_vs_unrel={n4_best[1]['specific_vs_unrelated']:.4f})")


if __name__ == "__main__":
    main()
