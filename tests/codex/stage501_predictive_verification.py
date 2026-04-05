#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage501: 预测性验证协议
目标：基于Stage475-500积累的机制，做出可证伪的预测并验证
基于已积累拼图的理论预测：
  预测1：概念距离与其共现频率负相关（越常一起出现越近）
  预测2：同一概念家族的子概念在MLP神经元激活空间中比在embedding空间中更聚类
  预测3：反义词对在attention pattern中会互相"看到"对方（互注意力高）
  预测4：否定句中，被否定概念的激活会被"压制"（负方向偏移）
  预测5：因果句中，原因概念的激活会"引导"结果概念的激活方向
方法：对每个预测设计具体度量，在两个模型上验证
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "codex"))
from qwen3_language_shared import (
    discover_layers,
    load_qwen3_model,
    load_qwen3_tokenizer,
    PROJECT_ROOT,
    QWEN3_MODEL_PATH,
)


DEEPSEEK_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)


def load_deepseek_model(*, prefer_cuda: bool = True):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    kwargs = {
        "pretrained_model_name_or_path": str(DEEPSEEK_MODEL_PATH),
        "local_files_only": True, "trust_remote_code": True,
        "low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16,
    }
    if want_cuda:
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = "cpu"
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(DEEPSEEK_MODEL_PATH), local_files_only=True, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_embedding(model, tokenizer, word: str) -> np.ndarray:
    device = next(model.parameters()).device
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if not token_ids:
        return None
    with torch.inference_mode():
        emb = model.model.embed_tokens(torch.tensor([token_ids]).to(device))
    return emb[0, 0].detach().float().cpu().numpy()


def get_layer_activation(model, tokenizer, word: str, layer_idx: int) -> np.ndarray:
    device = next(model.parameters()).device
    layers = discover_layers(model)
    captured = [None]

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured[0] = output[0].detach().float().cpu()
        else:
            captured[0] = output.detach().float().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    encoded = tokenizer(word, return_tensors="pt", truncation=True, max_length=8)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        model(**encoded)
    handle.remove()
    if captured[0] is None:
        return None
    return captured[0][0, -1, :].numpy()


def get_attention_pattern(model, tokenizer, text: str, layer_idx: int, head_idx: int) -> np.ndarray:
    """获取指定层指定头的注意力矩阵"""
    device = next(model.parameters()).device
    layers = discover_layers(model)
    layer = layers[layer_idx]
    captured = [None]

    def attn_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1]  # (batch, num_heads, seq_len, seq_len)
            captured[0] = attn_weights.detach().float().cpu()

    handle = layer.self_attn.register_forward_hook(attn_hook)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        model(**encoded)
    handle.remove()

    if captured[0] is None:
        return None
    # shape: (1, num_heads, seq, seq) -> 取head_idx
    num_heads = captured[0].shape[1]
    hi = min(head_idx, num_heads - 1)
    return captured[0][0, hi, :, :].numpy()


def prediction1_cooccurrence_distance(model, tokenizer) -> Dict:
    """
    预测1：概念距离与其共现频率负相关
    用"高共现对"vs"低共现对"的embedding距离来验证
    """
    # 高共现词对（常一起出现）
    high_cooccurrence = [
        ("太阳", "月亮"), ("猫", "狗"), ("红", "蓝"), ("吃", "喝"),
        ("大", "小"), ("男", "女"), ("快", "慢"), ("冷", "热"),
        ("山", "水"), ("风", "雨"),
    ]
    # 低共现词对（不太一起出现）
    low_cooccurrence = [
        ("猫", "月亮"), ("太阳", "蓝色"), ("吃", "山"), ("快", "雨"),
        ("女", "月亮"), ("喝", "风"), ("慢", "红"), ("热", "狗"),
        ("水", "大"), ("蓝", "猫"),
    ]

    def avg_cosine_dist(pairs):
        dists = []
        for w1, w2 in pairs:
            e1 = get_embedding(model, tokenizer, w1)
            e2 = get_embedding(model, tokenizer, w2)
            if e1 is not None and e2 is not None:
                cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)
                dists.append(1 - cos)
        return dists

    high_dists = avg_cosine_dist(high_cooccurrence)
    low_dists = avg_cosine_dist(low_cooccurrence)

    high_mean = np.mean(high_dists) if high_dists else 0
    low_mean = np.mean(low_dists) if low_dists else 0
    prediction_holds = high_mean < low_mean

    return {
        "prediction": "高共现词对比低共现词对的embedding距离更近",
        "high_cooccurrence_mean_dist": round(float(high_mean), 6),
        "low_cooccurrence_mean_dist": round(float(low_mean), 6),
        "prediction_holds": prediction_holds,
        "effect_size": round(abs(high_mean - low_mean) / max(np.std(high_dists + low_dists), 1e-9), 4),
    }


def prediction2_family_clustering_deepens(model, tokenizer) -> Dict:
    """
    预测2：同一概念家族的子概念在MLP层比在embedding层更聚类
    """
    families = {
        "水果": ["苹果", "香蕉", "橙子", "葡萄"],
        "动物": ["猫", "狗", "牛", "马"],
        "颜色": ["红", "蓝", "绿", "黄"],
    }
    # 跨家族对照组
    cross_family = ["苹果", "猫", "红", "香蕉", "狗", "蓝", "橙子", "牛", "绿", "葡萄", "马", "黄"]

    layers = discover_layers(model)
    mid_layer = len(layers) // 2

    results_by_depth = {}

    for depth_name, get_vec in [("embedding", lambda w: get_embedding(model, tokenizer, w)),
                                  (f"layer{mid_layer}", lambda w: get_layer_activation(model, tokenizer, w, mid_layer))]:

        # 计算家族内距离 vs 家族间距离
        within_dists = []
        between_dists = []
        for family_words in families.values():
            vecs = []
            for w in family_words:
                v = get_vec(w)
                if v is not None:
                    vecs.append(v)
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    cos = np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-9)
                    within_dists.append(1 - cos)

        # 跨家族距离
        all_vecs = {}
        for fname, fwords in families.items():
            for w in fwords:
                v = get_vec(w)
                if v is not None:
                    all_vecs[w] = v

        family_names = list(families.keys())
        for i in range(len(family_names)):
            for j in range(i + 1, len(family_names)):
                for w1 in families[family_names[i]]:
                    for w2 in families[family_names[j]]:
                        v1 = all_vecs.get(w1)
                        v2 = all_vecs.get(w2)
                        if v1 is not None and v2 is not None:
                            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                            between_dists.append(1 - cos)

        within_mean = np.mean(within_dists) if within_dists else 0
        between_mean = np.mean(between_dists) if between_dists else 0
        clustering_score = (between_mean - within_mean) / max(between_mean, 1e-9)  # 正值=更聚类

        results_by_depth[depth_name] = {
            "within_mean": round(float(within_mean), 6),
            "between_mean": round(float(between_mean), 6),
            "clustering_score": round(float(clustering_score), 4),
        }

    emb_score = results_by_depth["embedding"]["clustering_score"]
    mid_score = results_by_depth[f"layer{mid_layer}"]["clustering_score"]
    prediction_holds = mid_score > emb_score

    return {
        "prediction": "同一概念家族在深层比在embedding层更聚类",
        "results_by_depth": results_by_depth,
        "embedding_clustering": round(emb_score, 4),
        f"layer{mid_layer}_clustering": round(mid_score, 4),
        "prediction_holds": prediction_holds,
    }


def prediction3_antonym_mutual_attention(model, tokenizer) -> Dict:
    """
    预测3：反义词对在attention pattern中互相"看到"对方
    """
    antonym_pairs = [
        ("大", "小"), ("热", "冷"), ("好", "坏"), ("黑", "白"), ("快", "慢"),
    ]
    control_pairs = [
        ("大", "猫"), ("热", "山"), ("好", "水"), ("黑", "风"), ("快", "雨"),
    ]

    layers = discover_layers(model)
    mid_layer = len(layers) // 2

    antonym_attentions = []
    control_attentions = []

    for w1, w2 in antonym_pairs:
        text = w1 + w2
        attn = get_attention_pattern(model, tokenizer, text, mid_layer, 0)
        if attn is not None and attn.shape[0] >= 2:
            # token1到token2的注意力
            antonym_attentions.append(attn[0, 1])

    for w1, w2 in control_pairs:
        text = w1 + w2
        attn = get_attention_pattern(model, tokenizer, text, mid_layer, 0)
        if attn is not None and attn.shape[0] >= 2:
            control_attentions.append(attn[0, 1])

    ant_mean = np.mean(antonym_attentions) if antonym_attentions else 0
    ctrl_mean = np.mean(control_attentions) if control_attentions else 0
    prediction_holds = ant_mean > ctrl_mean

    return {
        "prediction": "反义词对比无关词对有更高的互相注意力",
        "antonym_mean_attention": round(float(ant_mean), 6),
        "control_mean_attention": round(float(ctrl_mean), 6),
        "prediction_holds": prediction_holds,
    }


def prediction4_negation_suppression(model, tokenizer) -> Dict:
    """
    预测4：否定句中，被否定概念的激活被"压制"
    """
    positive_sentences = ["这是好的", "是大的", "很热", "很开心", "很快"]
    negative_sentences = ["这不是好的", "不是大的", "不热", "不开心", "不快"]

    layers = discover_layers(model)
    mid_layer = len(layers) // 2

    pos_activations = []
    neg_activations = []

    for pos_s, neg_s in zip(positive_sentences, negative_sentences):
        pos_act = get_layer_activation(model, tokenizer, pos_s, mid_layer)
        neg_act = get_layer_activation(model, tokenizer, neg_s, mid_layer)
        if pos_act is not None and neg_act is not None:
            pos_activations.append(pos_act)
            neg_activations.append(neg_act)

    # 计算否定vs肯定的激活偏移
    # 预测：否定句的激活范数更小（被压制）
    pos_norms = [np.linalg.norm(a) for a in pos_activations]
    neg_norms = [np.linalg.norm(a) for a in neg_activations]

    # 计算方向变化
    direction_changes = []
    for pa, na in zip(pos_activations, neg_activations):
        cos = np.dot(pa, na) / (np.linalg.norm(pa) * np.linalg.norm(na) + 1e-9)
        direction_changes.append(cos)

    pos_mean_norm = np.mean(pos_norms) if pos_norms else 0
    neg_mean_norm = np.mean(neg_norms) if neg_norms else 0
    mean_cos = np.mean(direction_changes) if direction_changes else 0

    # 否定应该使激活变小或方向反转
    suppression = pos_mean_norm > neg_mean_norm
    direction_reversal = mean_cos < 0.5

    return {
        "prediction": "否定句中被否定概念的激活被压制",
        "positive_mean_norm": round(float(pos_mean_norm), 4),
        "negative_mean_norm": round(float(neg_mean_norm), 4),
        "norm_suppression": suppression,
        "mean_direction_cosine": round(float(mean_cos), 4),
        "direction_reversal": direction_reversal,
        "prediction_holds": suppression or direction_reversal,
    }


def prediction5_causal_direction(model, tokenizer) -> Dict:
    """
    预测5：因果句中，原因概念的激活方向引导结果概念
    """
    cause_effect_pairs = [
        ("下雨", "地面湿"),
        ("没吃饭", "很饿"),
        ("用力推", "门开"),
        ("太阳出", "天亮"),
        ("下雪", "很冷"),
    ]

    layers = discover_layers(model)
    mid_layer = len(layers) // 2

    guidance_scores = []

    for cause, effect in cause_effect_pairs:
        cause_act = get_layer_activation(model, tokenizer, cause, mid_layer)
        cause_effect_act = get_layer_activation(model, tokenizer, cause + effect, mid_layer)
        standalone_effect_act = get_layer_activation(model, tokenizer, effect, mid_layer)

        if cause_act is None or cause_effect_act is None or standalone_effect_act is None:
            continue

        # 因果上下文vs单独effect的差异方向
        context_effect_delta = cause_effect_act - standalone_effect_act
        # 归一化
        if np.linalg.norm(context_effect_delta) > 1e-9 and np.linalg.norm(cause_act) > 1e-9:
            cos = np.dot(context_effect_delta, cause_act) / (
                np.linalg.norm(context_effect_delta) * np.linalg.norm(cause_act))
            guidance_scores.append(float(cos))

    if guidance_scores:
        mean_guidance = np.mean(guidance_scores)
        positive_guidance = sum(1 for g in guidance_scores if g > 0) / len(guidance_scores)
        prediction_holds = mean_guidance > 0.1 and positive_guidance > 0.6
    else:
        mean_guidance = 0
        positive_guidance = 0
        prediction_holds = False

    return {
        "prediction": "因果上下文使结果概念的激活偏向原因方向",
        "mean_guidance_cosine": round(float(mean_guidance), 4),
        "positive_guidance_ratio": round(float(positive_guidance), 4),
        "prediction_holds": prediction_holds,
    }


def run_experiment(model_name: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"Stage501: 预测性验证协议 — {model_name}")
    print(f"{'='*60}\n")

    if model_name == "qwen3":
        model, tokenizer = load_qwen3_model(prefer_cuda=True)
    else:
        model, tokenizer = load_deepseek_model(prefer_cuda=True)

    predictions = {}
    prediction_functions = [
        ("P1_cooccurrence_distance", prediction1_cooccurrence_distance),
        ("P2_family_clustering_deepens", prediction2_family_clustering_deepens),
        ("P3_antonym_attention", prediction3_antonym_mutual_attention),
        ("P4_negation_suppression", prediction4_negation_suppression),
        ("P5_causal_direction", prediction5_causal_direction),
    ]

    for name, func in prediction_functions:
        print(f"\n--- {name} ---")
        result = func(model, tokenizer)
        predictions[name] = result
        holds = result.get("prediction_holds", False)
        print(f"  预测: {result['prediction']}")
        print(f"  结果: {'[OK]' if holds else '[NO]'}")
        # 打印关键数据
        for k, v in result.items():
            if k not in ("prediction", "prediction_holds"):
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sv, (int, float)):
                            print(f"    {sk}: {sv}")
                elif isinstance(v, (int, float)):
                    print(f"  {k}: {v}")

    holds_count = sum(1 for r in predictions.values() if r.get("prediction_holds", False))

    summary = {
        "stage": "stage501_predictive_verification",
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": predictions,
        "aggregate": {
            "total_predictions": len(prediction_functions),
            "supported": holds_count,
            "refuted": len(prediction_functions) - holds_count,
            "support_rate": holds_count / len(prediction_functions),
        },
        "core_answer": (
            f"在{model_name}上，{len(prediction_functions)}个理论预测中，"
            f"{holds_count}个得到支持（支持率{holds_count/len(prediction_functions):.0%}）。"
        ),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage501: 预测性验证协议")
    parser.add_argument("model", choices=["qwen3", "deepseek"], help="模型选择")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "tests" / "codex_temp" / f"stage501_predictive_{time.strftime('%Y%m%d')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    summary = run_experiment(args.model)
    summary["elapsed_seconds"] = round(time.time() - start, 1)

    out_path = output_dir / f"summary_{args.model}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n结果保存到: {out_path}")
    print(f"\n核心结论: {summary['core_answer']}")


if __name__ == "__main__":
    main()
