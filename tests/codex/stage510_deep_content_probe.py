#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage510: 深层上下文编码内容探针分析
核心问题：深层(归一化后)到底在编码什么？
          语法角色？语义角色？位置信息？上下文依赖度？

分析维度：
- P1: 位置编码 vs 语义编码——同一词在不同位置
- P2: 语法角色探针——主语/宾语/表语中同一概念
- P3: 语义角色探针——同一概念在不同语义场景
- P4: 上下文依赖度——删词后hidden state变化
- P5: 深层"纯粹概念"提取——多上下文平均
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


# 用L2归一化后的cosine作为度量
def norm_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a1 = a.squeeze(0)
    b1 = b.squeeze(0)
    a_n = F.normalize(a1, p=2, dim=0)
    b_n = F.normalize(b1, p=2, dim=0)
    return F.cosine_similarity(a_n.unsqueeze(0), b_n.unsqueeze(0), dim=1).item()


def get_hidden_at_token(model, tokenizer, text: str, token_pos: int, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    """获取指定token位置在指定层的hidden states"""
    device = get_model_device(model)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    result = {}
    for li in layer_indices:
        pos = min(token_pos, outputs.hidden_states[li + 1].shape[1] - 1)
        h = outputs.hidden_states[li + 1][:, pos, :].float().cpu()
        result[li] = h
    return result


def get_sample_layers(model) -> List[int]:
    n = len(discover_layers(model))
    if n <= 12:
        return list(range(n))
    return [max(0, min(n - 1, round((n - 1) * i / 11))) for i in range(12)]


# ============================================================
# P1: 位置编码 vs 语义编码
# ============================================================
def analyze_P1_position_vs_semantics(model, tokenizer, layer_indices):
    """P1: 同一概念"苹果"在不同句子位置"""
    sentences = [
        "苹果很好吃",        # 位置0
        "我吃了一个苹果",    # 位置4
        "红色的苹果和橘子",  # 位置2
        "他喜欢苹果胜过橘子",# 位置3
    ]

    results = {}
    # 先确定"苹果"在各句子中的token位置
    positions = []
    for s in sentences:
        tokens = tokenizer.tokenize(s)
        # 找"苹果"的token位置
        apple_pos = None
        for i, t in enumerate(tokens):
            if "苹果" in t or t == "▁苹果" or t == "苹果":
                apple_pos = i
                break
        if apple_pos is None:
            apple_pos = len(tokens) // 2
        positions.append((s, apple_pos))

    # 逐层计算所有位置对之间的相似度
    for li in layer_indices:
        vecs = []
        for s, pos in positions:
            hs = get_hidden_at_token(model, tokenizer, s, pos, layer_indices)
            vecs.append(hs[li])

        # 所有位置对的归一化余弦
        pair_sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                pair_sims.append(norm_cos(vecs[i], vecs[j]))

        results[f"L{li}"] = {
            "mean_similarity": round(float(np.mean(pair_sims)), 4) if pair_sims else 0,
            "min_similarity": round(float(np.min(pair_sims)), 4) if pair_sims else 0,
            "max_similarity": round(float(np.max(pair_sims)), 4) if pair_sims else 0,
            "std_similarity": round(float(np.std(pair_sims)), 4) if pair_sims else 0,
        }

    return {"P1_position_vs_semantics": results}


# ============================================================
# P2: 语法角色探针
# ============================================================
def analyze_P2_grammar_role(model, tokenizer, layer_indices):
    """P2: 同一概念"猫"在不同语法角色中"""
    templates = {
        "subject": "猫在睡觉",           # 主语
        "object": "我喜欢猫",            # 宾语
        "topic": "猫是一种动物",         # 主题
        "modifier": "猫爪子很锋利",      # 修饰语
        "predicate": "那只动物是猫",     # 表语
    }

    results = {}
    # 找"猫"的位置
    role_positions = {}
    for role, s in templates.items():
        tokens = tokenizer.tokenize(s)
        cat_pos = None
        for i, t in enumerate(tokens):
            if "猫" in t:
                cat_pos = i
                break
        if cat_pos is None:
            cat_pos = 0
        role_positions[role] = (s, cat_pos)

    for li in layer_indices:
        vecs = {}
        for role, (s, pos) in role_positions.items():
            hs = get_hidden_at_token(model, tokenizer, s, pos, layer_indices)
            vecs[role] = hs[li]

        # 角色间两两相似度
        roles = list(vecs.keys())
        pair_sims = {}
        for i, r1 in enumerate(roles):
            for j, r2 in enumerate(roles):
                if i < j:
                    sim = norm_cos(vecs[r1], vecs[r2])
                    pair_sims[f"{r1}_vs_{r2}"] = round(sim, 4)

        # 角色间的平均差异
        all_sims = list(pair_sims.values())
        results[f"L{li}"] = {
            "mean_role_similarity": round(float(np.mean(all_sims)), 4) if all_sims else 0,
            "role_std": round(float(np.std(all_sims)), 4) if all_sims else 0,
            "pair_details": pair_sims,
        }

    return {"P2_grammar_role": results}


# ============================================================
# P3: 语义场景探针
# ============================================================
def analyze_P3_semantic_context(model, tokenizer, layer_indices):
    """P3: 同一概念"水"在不同语义场景"""
    templates = {
        "drinking": "我喝了一杯水",       # 饮用
        "weather": "今天下雨了",           # 天气（无"水"但有雨）
        "ocean": "大海里有水",             # 海洋
        "cooking": "加点水煮面",           # 烹饪
        "science": "水的化学式是H2O",     # 科学
        "fire": "用水灭火",               # 消防
    }

    # 改为都有"水"的句子
    templates = {
        "drinking": "我喝了一杯水",
        "ocean": "大海里有很多水",
        "cooking": "加点水煮面条",
        "science": "水的化学式是H2O",
        "fire": "用水来灭火",
        "clean": "用水洗干净衣服",
    }

    results = {}
    for li in layer_indices:
        vecs = {}
        for scene, s in templates.items():
            tokens = tokenizer.tokenize(s)
            water_pos = None
            for i, t in enumerate(tokens):
                if "水" in t:
                    water_pos = i
                    break
            if water_pos is None:
                water_pos = 0
            hs = get_hidden_at_token(model, tokenizer, s, water_pos, layer_indices)
            vecs[scene] = hs[li]

        scenes = list(vecs.keys())
        pair_sims = {}
        for i, s1 in enumerate(scenes):
            for j, s2 in enumerate(scenes):
                if i < j:
                    sim = norm_cos(vecs[s1], vecs[s2])
                    pair_sims[f"{s1}_vs_{s2}"] = round(sim, 4)

        all_sims = list(pair_sims.values())
        results[f"L{li}"] = {
            "mean_scene_similarity": round(float(np.mean(all_sims)), 4) if all_sims else 0,
            "scene_std": round(float(np.std(all_sims)), 4) if all_sims else 0,
            "max_scene_diff": round(1 - float(np.min(all_sims)), 4) if all_sims else 0,
        }

    return {"P3_semantic_context": results}


# ============================================================
# P4: 上下文依赖度
# ============================================================
def analyze_P4_context_dependency(model, tokenizer, layer_indices):
    """P4: 删除上下文后hidden state变化量"""
    base_words = ["苹果", "猫", "水"]
    context_templates = [
        lambda w: f"我昨天在超市买了新鲜的{w}",
        lambda w: f"{w}是这个世界上最美好的东西",
        lambda w: f"请给我一杯{w}",
    ]

    results = {}
    for w in base_words:
        # 单token baseline
        hs_base = get_hidden_at_token(model, tokenizer, w, 0, layer_indices)

        for ctx_idx, ctx_fn in enumerate(context_templates):
            sentence = ctx_fn(w)
            tokens = tokenizer.tokenize(sentence)
            # 找目标词位置
            target_pos = 0
            for i, t in enumerate(tokens):
                if w in t:
                    target_pos = i
                    break

            hs_ctx = get_hidden_at_token(model, tokenizer, sentence, target_pos, layer_indices)

            for li in layer_indices:
                # L2距离变化（归一化后）
                delta = norm_cos(hs_base[li], hs_ctx[li])
                key = f"L{li}"
                if key not in results:
                    results[key] = {}
                if w not in results[key]:
                    results[key][w] = {}
                results[key][w][f"ctx{ctx_idx}"] = {
                    "norm_cos_with_base": round(delta, 4),
                    "context_shift": round(1 - delta, 4),
                }

    # 汇总
    summary = {}
    for li_key, word_data in results.items():
        shifts = []
        for w, ctxs in word_data.items():
            for ctx, vals in ctxs.items():
                shifts.append(vals["context_shift"])
        summary[li_key] = {
            "mean_context_shift": round(float(np.mean(shifts)), 4) if shifts else 0,
            "max_context_shift": round(float(np.max(shifts)), 4) if shifts else 0,
        }

    return {"P4_context_dependency": summary, "P4_details": results}


# ============================================================
# P5: 深层"纯粹概念"提取
# ============================================================
def analyze_P5_pure_concept(model, tokenizer, layer_indices):
    """P5: 多上下文平均提取"概念核心"，测量其与各语义特征的关系"""
    words = {
        "apple": ["苹果很好吃", "我买了苹果", "苹果是红色的水果", "苹果手机很好用"],
        "cat": ["猫在睡觉", "我养了一只猫", "猫很可爱", "猫抓老鼠"],
    }

    results = {}
    for concept, sentences in words.items():
        for li in layer_indices:
            vecs = []
            for s in sentences:
                tokens = tokenizer.tokenize(s)
                target_pos = 0
                for i, t in enumerate(tokens):
                    if concept == "apple" and "苹果" in t:
                        target_pos = i
                        break
                    if concept == "cat" and "猫" in t:
                        target_pos = i
                        break
                hs = get_hidden_at_token(model, tokenizer, s, target_pos, layer_indices)
                vecs.append(hs[li])

            # 平均向量 = "纯粹概念"
            mean_vec = torch.stack(vecs).mean(dim=0)

            # 与每个上下文版本的相似度
            sims = [norm_cos(mean_vec, v) for v in vecs]
            # 归一化平均向量的norm
            mean_norm = torch.norm(mean_vec, p=2).item()

            key = f"{concept}_L{li}"
            results[key] = {
                "mean_similarity_to_contexts": round(float(np.mean(sims)), 4),
                "min_similarity": round(float(np.min(sims)), 4),
                "concept_norm": round(mean_norm, 2),
                "context_diversity": round(float(np.std(sims)), 4),
            }

    return {"P5_pure_concept": results}


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
    print(f"[Stage510] Deep layer content probe | model: {model_name}")
    print(f"  time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    t0 = time.time()
    model, tokenizer = load_model(model_name)
    layer_indices = get_sample_layers(model)
    print(f"  layers: {len(discover_layers(model))}, sample: {layer_indices}")

    summary = {"model": model_name, "timestamp": datetime.now().isoformat(), "layer_indices": layer_indices}

    analyses = [
        ("P1", lambda: analyze_P1_position_vs_semantics(model, tokenizer, layer_indices)),
        ("P2", lambda: analyze_P2_grammar_role(model, tokenizer, layer_indices)),
        ("P3", lambda: analyze_P3_semantic_context(model, tokenizer, layer_indices)),
        ("P4", lambda: analyze_P4_context_dependency(model, tokenizer, layer_indices)),
        ("P5", lambda: analyze_P5_pure_concept(model, tokenizer, layer_indices)),
    ]

    for name, fn in analyses:
        print(f"  {name}...", end=" ", flush=True)
        t1 = time.time()
        result = fn()
        summary.update(result)
        print(f"done ({time.time()-t1:.1f}s)")

    # 保存
    out_dir = Path("tests/codex_temp") / f"stage510_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    # 关键摘要
    print("\n=== Key Findings ===")
    p1 = summary.get("P1_position_vs_semantics", {})
    layers = sorted(p1.keys())
    if layers:
        print(f"  P1 位置不变性: L0 std={p1[layers[0]]['std_similarity']:.4f} -> "
              f"deep std={p1[layers[-1]]['std_similarity']:.4f}")

    p2 = summary.get("P2_grammar_role", {})
    layers2 = sorted(p2.keys())
    if layers2:
        print(f"  P2 语法角色影响: L0 std={p2[layers2[0]]['role_std']:.4f} -> "
              f"deep std={p2[layers2[-1]]['role_std']:.4f}")

    p4 = summary.get("P4_context_dependency", {})
    layers4 = sorted(p4.keys())
    if layers4:
        print(f"  P4 上下文依赖度: L0 shift={p4[layers4[0]]['mean_context_shift']:.4f} -> "
              f"deep shift={p4[layers4[-1]]['mean_context_shift']:.4f}")


if __name__ == "__main__":
    main()
