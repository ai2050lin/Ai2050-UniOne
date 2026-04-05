#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage511: 精细神经元级激活分析
核心问题：哪些具体神经元对特定概念最敏感？

方法：
- 在MLP层捕获每个神经元的激活值（pre-activation, 即SiLU/GeLU之前）
- 对每个token输入，记录所有中间MLP层的神经元激活
- 计算"苹果"相关词 vs 无关词的神经元激活差异
- 找到Top-K差异最大的神经元（概念敏感神经元）

分析维度：
- N1: 单token各层Top-20概念敏感神经元（苹果 vs 航天飞机）
- N2: 翻译对齐神经元——苹果 vs 苹果(中文)共享激活的神经元
- N3: 概念层级神经元——水果类共享 vs 动物类共享的神经元
- N4: 上下文依赖神经元——同一词在不同上下文中的激活差异
- N5: 跨模型一致性——同一概念在4个模型中的神经元激活模式对比
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex.qwen3_language_shared import (
    load_qwen3_model, load_glm4_model, load_gemma4_model, load_deepseek7b_model,
    discover_layers, get_model_device, move_batch_to_model_device,
    capture_qwen_mlp_payloads, remove_hooks,
)


LOADERS = {
    "qwen3": load_qwen3_model,
    "glm4": load_glm4_model,
    "gemma4": load_gemma4_model,
    "deepseek7b": load_deepseek7b_model,
}

# ============================================================
# 数据定义
# ============================================================
APPLE_WORDS = ["苹果", "banana", "orange", "葡萄", "芒果"]
FRUIT_WORDS = ["苹果", "banana", "orange", "葡萄", "芒果", "水果", "甜", "果汁"]
ANIMAL_WORDS = ["猫", "狗", "tiger", "lion", "大象", "动物"]
UNRELATED_WORDS = ["航天飞机", "计算机", "democracy", "宪法", "重力"]

# N4: 上下文句子
CONTEXT_SENTENCES = {
    "苹果_主语": "苹果很好吃",
    "苹果_宾语": "我喜欢苹果",
    "苹果_所属": "树上的苹果红了",
    "苹果_比喻": "她像苹果一样红润",
}

# 翻译对
TRANSLATION_PAIRS = [
    ("apple", "苹果"),
    ("cat", "猫"),
    ("banana", "香蕉"),
]


def capture_all_mlp_neurons(model, tokenizer, text: str, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    """捕获所有指定MLP层的神经元激活（pre-down_proj, 即SiLU激活后的值）"""
    device = get_model_device(model)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    encoded = move_batch_to_model_device(model, encoded)

    payload_map = {idx: "neuron_in" for idx in layer_indices}
    buffers, handles = capture_qwen_mlp_payloads(model, payload_map)

    with torch.no_grad():
        model(**encoded)

    remove_hooks(handles)
    result = {}
    for idx in layer_indices:
        v = buffers.get(idx)
        if v is not None:
            result[idx] = v.squeeze(0)  # [seq_len, neuron_dim]
    return result


def get_neuron_diff(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    """计算两个激活向量的逐神经元差异（绝对差值）"""
    # vec_a, vec_b: [neuron_dim]
    return (vec_a - vec_b).abs()


def top_k_neurons(diff: torch.Tensor, k: int = 20) -> Dict[str, list]:
    """找到差异最大的Top-K神经元"""
    if diff.dim() == 2:
        diff = diff.mean(dim=0)  # 平均跨序列位置
    vals, indices = torch.topk(diff, min(k, diff.shape[-1]))
    return {
        "indices": indices.tolist(),
        "values": [round(v, 6) for v in vals.tolist()],
    }


def neuron_overlap(set_a: List[int], set_b: List[int]) -> float:
    """两组神经元的重叠率（Jaccard）"""
    sa, sb = set(set_a), set(set_b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ============================================================
# 分析维度
# ============================================================

def analyze_n1(model, tokenizer, layer_indices: List[int], model_name: str) -> dict:
    """N1: 概念敏感神经元——苹果类 vs 无关类"""
    results = {}
    sample_apple = APPLE_WORDS[0]  # 苹果
    sample_unrel = UNRELATED_WORDS[0]  # 航天飞机

    for target_word, compare_words, label in [
        (sample_apple, UNRELATED_WORDS, "苹果_vs_无关"),
        (ANIMAL_WORDS[0], UNRELATED_WORDS, "猫_vs_无关"),
    ]:
        target_acts = capture_all_mlp_neurons(model, tokenizer, target_word, layer_indices)
        diffs_per_layer = {}
        for li in layer_indices:
            target_v = target_acts[li]  # [seq_len, neuron_dim]
            # 取最后一个token位置的激活
            target_last = target_v[-1] if target_v.shape[0] > 0 else target_v.mean(dim=0)
            all_diffs = []
            for cw in compare_words:
                cw_acts = capture_all_mlp_neurons(model, tokenizer, cw, layer_indices)
                cw_last = cw_acts[li][-1] if cw_acts[li].shape[0] > 0 else cw_acts[li].mean(dim=0)
                all_diffs.append(get_neuron_diff(target_last, cw_last))
            # 平均差异
            avg_diff = torch.stack(all_diffs).mean(dim=0)
            diffs_per_layer[str(li)] = top_k_neurons(avg_diff, k=20)
        results[label] = diffs_per_layer
    return results


def analyze_n2(model, tokenizer, layer_indices: List[int]) -> dict:
    """N2: 翻译对齐神经元——apple vs 苹果 共享激活的神经元"""
    results = {}
    for en_word, zh_word in TRANSLATION_PAIRS:
        en_acts = capture_all_mlp_neurons(model, tokenizer, en_word, layer_indices)
        zh_acts = capture_all_mlp_neurons(model, tokenizer, zh_word, layer_indices)
        per_layer = {}
        for li in layer_indices:
            en_v = en_acts[li][-1] if en_acts[li].shape[0] > 0 else en_acts[li].mean(dim=0)
            zh_v = zh_acts[li][-1] if zh_acts[li].shape[0] > 0 else zh_acts[li].mean(dim=0)
            cos = F.cosine_similarity(en_v.unsqueeze(0), zh_v.unsqueeze(0), dim=1).item()
            # 共享激活：两个神经元都高度激活的位置
            en_norm = (en_v - en_v.mean()) / (en_v.std() + 1e-8)
            zh_norm = (zh_v - zh_v.mean()) / (zh_v.std() + 1e-8)
            shared_strength = (en_norm * zh_norm).mean().item()
            per_layer[str(li)] = {
                "cos_sim": round(cos, 6),
                "shared_strength": round(shared_strength, 6),
            }
        results[f"{en_word}_{zh_word}"] = per_layer
    return results


def analyze_n3(model, tokenizer, layer_indices: List[int]) -> dict:
    """N3: 概念层级神经元——水果类共享 vs 动物类共享"""
    results = {}
    for li in layer_indices:
        # 水果类内部相似度
        fruit_acts = {}
        for w in FRUIT_WORDS:
            acts = capture_all_mlp_neurons(model, tokenizer, w, layer_indices)
            fruit_acts[w] = acts[li][-1] if acts[li].shape[0] > 0 else acts[li].mean(dim=0)

        # 动物类内部相似度
        animal_acts = {}
        for w in ANIMAL_WORDS:
            acts = capture_all_mlp_neurons(model, tokenizer, w, layer_indices)
            animal_acts[w] = acts[li][-1] if acts[li].shape[0] > 0 else acts[li].mean(dim=0)

        # 水果类内部平均cos
        fruit_cos = []
        fwords = list(fruit_acts.keys())
        for i in range(len(fwords)):
            for j in range(i+1, len(fwords)):
                c = F.cosine_similarity(fruit_acts[fwords[i]].unsqueeze(0),
                                        fruit_acts[fwords[j]].unsqueeze(0), dim=1).item()
                fruit_cos.append(c)

        # 动物类内部平均cos
        animal_cos = []
        awords = list(animal_acts.keys())
        for i in range(len(awords)):
            for j in range(i+1, len(awords)):
                c = F.cosine_similarity(animal_acts[awords[i]].unsqueeze(0),
                                        animal_acts[awords[j]].unsqueeze(0), dim=1).item()
                animal_cos.append(c)

        # 跨类平均cos
        cross_cos = []
        for fw in fwords:
            for aw in awords:
                c = F.cosine_similarity(fruit_acts[fw].unsqueeze(0),
                                        animal_acts[aw].unsqueeze(0), dim=1).item()
                cross_cos.append(c)

        results[str(li)] = {
            "fruit_intra_cos": round(np.mean(fruit_cos), 6) if fruit_cos else 0,
            "animal_intra_cos": round(np.mean(animal_cos), 6) if animal_cos else 0,
            "cross_cos": round(np.mean(cross_cos), 6) if cross_cos else 0,
            "category_separation": round(
                (np.mean(fruit_cos) + np.mean(animal_cos)) / 2 - np.mean(cross_cos), 6
            ) if fruit_cos and animal_cos and cross_cos else 0,
        }
    return results


def analyze_n4(model, tokenizer, layer_indices: List[int]) -> dict:
    """N4: 上下文依赖神经元——同一词在不同上下文中的激活差异"""
    results = {}
    contexts = list(CONTEXT_SENTENCES.values())
    context_labels = list(CONTEXT_SENTENCES.keys())

    # 获取所有上下文中的"苹果"激活
    context_acts = {}
    for label, sent in CONTEXT_SENTENCES.items():
        acts = capture_all_mlp_neurons(model, tokenizer, sent, layer_indices)
        context_acts[label] = acts

    for li in layer_indices:
        # 提取各上下文中"苹果"token的激活（取最后一个token）
        context_vecs = {}
        for label in context_labels:
            v = context_acts[label][li]
            context_vecs[label] = v[-1] if v.shape[0] > 0 else v.mean(dim=0)

        # 上下文间差异
        vecs = list(context_vecs.values())
        if len(vecs) < 2:
            continue
        # 计算标准差（衡量上下文依赖程度）
        stacked = torch.stack(vecs)
        context_std = stacked.std(dim=0)

        # Top-K上下文敏感神经元
        topk = top_k_neurons(context_std, k=15)

        # 上下文间平均cos
        pair_cos = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                c = F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0), dim=1).item()
                pair_cos.append(c)

        results[str(li)] = {
            "avg_pair_cos": round(np.mean(pair_cos), 6) if pair_cos else 0,
            "context_sensitivity_std": round(context_std.mean().item(), 6),
            "top_sensitive_neurons": topk,
        }
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    model_name = sys.argv[1].lower() if len(sys.argv) > 1 else "qwen3"
    if model_name not in LOADERS:
        print(f"未知模型: {model_name}, 可选: {list(LOADERS.keys())}")
        sys.exit(1)

    print(f"=== Stage511: 精细神经元级激活分析 [{model_name}] ===")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"tests/codex_temp/stage511_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"\n[加载模型] {model_name}...")
    t0 = time.time()
    model, tokenizer = LOADERS[model_name]()
    print(f"  加载耗时: {time.time()-t0:.1f}s")

    layers = discover_layers(model)
    n_layers = len(layers)
    print(f"  层数: {n_layers}")

    # 采样层索引
    layer_indices = list(range(0, n_layers, max(1, n_layers // 12)))
    if layer_indices[-1] != n_layers - 1:
        layer_indices.append(n_layers - 1)
    print(f"  采样层: {layer_indices}")

    device = get_model_device(model)
    print(f"  设备: {device}")

    summary = {"model": model_name, "timestamp": ts, "layers": n_layers, "sampled_layers": layer_indices}

    # ---- N1 ----
    print("\n[N1] 概念敏感神经元...")
    try:
        summary["N1"] = analyze_n1(model, tokenizer, layer_indices, model_name)
        print(f"  完成: {len(summary['N1'])} 组对比")
    except Exception as e:
        summary["N1"] = {"error": str(e)}
        print(f"  错误: {e}")

    # ---- N2 ----
    print("\n[N2] 翻译对齐神经元...")
    try:
        summary["N2"] = analyze_n2(model, tokenizer, layer_indices)
        print(f"  完成: {len(summary['N2'])} 组翻译对")
    except Exception as e:
        summary["N2"] = {"error": str(e)}
        print(f"  错误: {e}")

    # ---- N3 ----
    print("\n[N3] 概念层级神经元...")
    try:
        summary["N3"] = analyze_n3(model, tokenizer, layer_indices)
        print(f"  完成: {len(summary['N3'])} 层分析")
    except Exception as e:
        summary["N3"] = {"error": str(e)}
        print(f"  错误: {e}")

    # ---- N4 ----
    print("\n[N4] 上下文依赖神经元...")
    try:
        summary["N4"] = analyze_n4(model, tokenizer, layer_indices)
        print(f"  完成: {len(summary['N4'])} 层分析")
    except Exception as e:
        summary["N4"] = {"error": str(e)}
        print(f"  错误: {e}")

    # 保存结果
    out_path = out_dir / f"summary_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

    # 打印关键发现
    print("\n=== 关键发现 ===")
    if "N2" in summary and not isinstance(summary["N2"].get("error"), str):
        for pair_name, layers_data in summary["N2"].items():
            last_layer = str(layer_indices[-1])
            if last_layer in layers_data:
                print(f"  {pair_name} 深层shared_strength: {layers_data[last_layer]['shared_strength']}")
    if "N3" in summary and not isinstance(summary["N3"].get("error"), str):
        last_layer = str(layer_indices[-1])
        if last_layer in summary["N3"]:
            n3 = summary["N3"][last_layer]
            print(f"  深层category_separation: {n3['category_separation']}")


if __name__ == "__main__":
    main()
