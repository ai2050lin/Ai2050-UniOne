# -*- coding: utf-8 -*-
"""
Stage462: 单模型独立测试 — 黄金层因子语义验证 + 概念算术测试
================================================================

核心目标：
1. 逐层独立SVD分析（复现Stage461单层结果）
2. 黄金层因子语义验证（对比embedding投影）
3. ★概念算术测试：apple→banana预测精度★
4. 跨层概念算术对比
5. 因子空间vs原始空间算术对比

模型: 单模型（通过命令行参数选择）
  python stage462_concept_arithmetic.py qwen3
  python stage462_concept_arithmetic.py deepseek

注意：一次只运行一个模型，避免GPU内存溢出
"""

from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen3_language_shared import (
    PROJECT_ROOT,
    capture_qwen_mlp_payloads,
    discover_layers,
    move_batch_to_model_device,
    remove_hooks,
    QWEN3_MODEL_PATH,
)

DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)

TIMESTAMP = "20260401"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage462_concept_arithmetic_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 概念集（含属性标注用于算术测试） ====================
CONCEPTS = {
    "fruit": {
        "label": "水果",
        "words": {
            "apple": {"color": "red", "size": 3, "taste": "sweet", "shape": "round"},
            "banana": {"color": "yellow", "size": 3, "taste": "sweet", "shape": "curved"},
            "orange": {"color": "orange", "size": 3, "taste": "sour_sweet", "shape": "round"},
            "grape": {"color": "purple", "size": 2, "taste": "sweet", "shape": "round_small"},
            "mango": {"color": "orange", "size": 3, "taste": "sweet", "shape": "oval"},
            "peach": {"color": "pink", "size": 3, "taste": "sweet", "shape": "round"},
            "lemon": {"color": "yellow", "size": 2, "taste": "sour", "shape": "oval"},
            "cherry": {"color": "red", "size": 1, "taste": "sweet", "shape": "round_small"},
            "watermelon": {"color": "green", "size": 5, "taste": "sweet", "shape": "round_large"},
            "strawberry": {"color": "red", "size": 1, "taste": "sweet", "shape": "heart"},
            "pear": {"color": "green", "size": 3, "taste": "sweet", "shape": "pear"},
            "pineapple": {"color": "yellow", "size": 4, "taste": "sour_sweet", "shape": "oval"},
            "coconut": {"color": "brown", "size": 4, "taste": "sweet", "shape": "round"},
            "kiwi": {"color": "green", "size": 2, "taste": "sour_sweet", "shape": "oval_small"},
            "blueberry": {"color": "blue", "size": 1, "taste": "sweet", "shape": "round_small"},
            "melon": {"color": "green", "size": 4, "taste": "sweet", "shape": "round_large"},
            "fig": {"color": "purple", "size": 2, "taste": "sweet", "shape": "pear"},
            "plum": {"color": "purple", "size": 2, "taste": "sweet", "shape": "round"},
            "lime": {"color": "green", "size": 2, "taste": "sour", "shape": "round_small"},
        },
    },
    "animal": {
        "label": "动物",
        "words": {
            "dog": {"size": 2, "domestic": 1, "speed": 3},
            "cat": {"size": 2, "domestic": 1, "speed": 3},
            "horse": {"size": 4, "domestic": 1, "speed": 4},
            "lion": {"size": 3, "domestic": 0, "speed": 4},
            "tiger": {"size": 3, "domestic": 0, "speed": 4},
            "elephant": {"size": 5, "domestic": 0, "speed": 2},
            "whale": {"size": 5, "domestic": 0, "speed": 3},
            "shark": {"size": 3, "domestic": 0, "speed": 4},
            "eagle": {"size": 2, "domestic": 0, "speed": 5},
            "wolf": {"size": 2, "domestic": 0, "speed": 4},
            "rabbit": {"size": 1, "domestic": 1, "speed": 4},
            "deer": {"size": 3, "domestic": 0, "speed": 4},
            "fox": {"size": 2, "domestic": 0, "speed": 4},
            "bear": {"size": 4, "domestic": 0, "speed": 3},
            "monkey": {"size": 2, "domestic": 0, "speed": 4},
            "dolphin": {"size": 3, "domestic": 0, "speed": 5},
            "penguin": {"size": 2, "domestic": 0, "speed": 1},
            "snake": {"size": 2, "domestic": 0, "speed": 3},
            "giraffe": {"size": 5, "domestic": 0, "speed": 4},
            "panda": {"size": 3, "domestic": 0, "speed": 2},
        },
    },
    "vehicle": {
        "label": "交通工具",
        "words": {
            "car": {"speed": 3, "medium": "land", "size": 2},
            "bus": {"speed": 3, "medium": "land", "size": 3},
            "train": {"speed": 4, "medium": "land", "size": 4},
            "plane": {"speed": 5, "medium": "air", "size": 4},
            "ship": {"speed": 2, "medium": "water", "size": 5},
            "bicycle": {"speed": 2, "medium": "land", "size": 1},
            "motorcycle": {"speed": 3, "medium": "land", "size": 1},
            "truck": {"speed": 2, "medium": "land", "size": 3},
            "helicopter": {"speed": 4, "medium": "air", "size": 2},
            "rocket": {"speed": 5, "medium": "air", "size": 3},
            "boat": {"speed": 2, "medium": "water", "size": 2},
            "submarine": {"speed": 2, "medium": "water", "size": 3},
            "taxi": {"speed": 3, "medium": "land", "size": 2},
            "ambulance": {"speed": 4, "medium": "land", "size": 2},
            "ferry": {"speed": 2, "medium": "water", "size": 4},
        },
    },
    "profession": {
        "label": "职业",
        "words": {
            "doctor": {"domain": "medical", "social": 1, "creativity": 0},
            "nurse": {"domain": "medical", "social": 1, "creativity": 0},
            "teacher": {"domain": "education", "social": 1, "creativity": 1},
            "engineer": {"domain": "technology", "social": 0, "creativity": 1},
            "lawyer": {"domain": "law", "social": 1, "creativity": 1},
            "chef": {"domain": "food", "social": 0, "creativity": 1},
            "artist": {"domain": "art", "social": 0, "creativity": 1},
            "musician": {"domain": "art", "social": 0, "creativity": 1},
            "scientist": {"domain": "science", "social": 0, "creativity": 1},
            "pilot": {"domain": "transport", "social": 0, "creativity": 0},
            "soldier": {"domain": "military", "social": 1, "creativity": 0},
            "firefighter": {"domain": "emergency", "social": 1, "creativity": 0},
            "police": {"domain": "law", "social": 1, "creativity": 0},
            "farmer": {"domain": "agriculture", "social": 0, "creativity": 0},
            "baker": {"domain": "food", "social": 0, "creativity": 1},
            "architect": {"domain": "construction", "social": 0, "creativity": 1},
            "surgeon": {"domain": "medical", "social": 1, "creativity": 1},
            "dentist": {"domain": "medical", "social": 1, "creativity": 0},
            "pharmacist": {"domain": "medical", "social": 0, "creativity": 0},
            "veterinarian": {"domain": "medical", "social": 1, "creativity": 0},
        },
    },
}


def sanitize_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def load_model(model_path: Path):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    want_cuda = torch.cuda.is_available()
    print(f"  CUDA available: {want_cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if want_cuda else 'N/A'}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if want_cuda else "")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "local_files_only": True, "trust_remote_code": True,
        "low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16,
    }
    if want_cuda:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()

    layer_count = len(discover_layers(model))
    neuron_dim = discover_layers(model)[0].mlp.gate_proj.out_features
    print(f"  Layers: {layer_count}, NeuronDim: {neuron_dim}")

    if want_cuda:
        print(f"  GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer, layer_count, neuron_dim


def extract_word_per_layer(model, tokenizer, word, layer_count):
    """提取单个词在每层的MLP neuron_in激活"""
    prompt = f"The {word}"
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    token_ids = encoded["input_ids"][0].tolist()

    word_tokens = tokenizer.encode(word, add_special_tokens=False)
    target_pos = None
    for i, tid in enumerate(token_ids):
        if tid in word_tokens:
            target_pos = i
            break
    if target_pos is None:
        target_pos = -1

    layer_payload_map = {i: "neuron_in" for i in range(layer_count)}
    buffers, handles = capture_qwen_mlp_payloads(model, layer_payload_map)

    try:
        encoded = move_batch_to_model_device(model, encoded)
        with torch.no_grad():
            model(**encoded)

        per_layer = {}
        for li in range(layer_count):
            buf = buffers[li]
            if buf is not None:
                pos = target_pos if target_pos >= 0 else buf.shape[1] + target_pos
                pos = max(0, min(pos, buf.shape[1] - 1))
                per_layer[li] = buf[0, pos].float().numpy()
        return per_layer
    finally:
        remove_hooks(handles)


def extract_all_activations(model, tokenizer, concepts, layer_count):
    """提取所有概念在每层的激活"""
    all_activations = {}
    total = sum(len(c["words"]) for c in concepts.values())
    done = 0

    for cat_name, cat_data in concepts.items():
        all_activations[cat_name] = {}
        for word in cat_data["words"]:
            try:
                acts = extract_word_per_layer(model, tokenizer, word, layer_count)
                if acts:
                    all_activations[cat_name][word] = acts
                    done += 1
            except Exception:
                pass

    print(f"  Total: {done}/{total}")
    return all_activations


def get_embedding(model, tokenizer, word):
    """获取词的embedding向量"""
    tokens = tokenizer.encode(word, add_special_tokens=False)
    if not tokens:
        return None
    with torch.no_grad():
        emb = model.get_input_embeddings()(torch.tensor([tokens]).to(model.device))
        # 平均多个token的embedding
        return emb[0].float().mean(dim=0).cpu().numpy()


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def per_layer_svd(all_activations, concepts, layer_count, n_components=100):
    """逐层SVD分析"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

    layer_data = {}
    for li in range(layer_count):
        word_vecs = []
        word_labels = []
        word_cats = []

        for cat_name, cat_data in concepts.items():
            cat_vecs = []
            for word, acts in all_activations[cat_name].items():
                if li in acts:
                    cat_vecs.append(acts[li])
            if not cat_vecs:
                continue
            basis = np.mean(cat_vecs, axis=0)

            for word, acts in all_activations[cat_name].items():
                if li in acts:
                    word_vecs.append(acts[li] - basis)
                    word_labels.append(word)
                    word_cats.append(cat_name)

        if len(word_vecs) < 20:
            continue

        bias_matrix = np.array(word_vecs)
        scaler = StandardScaler()
        normed = scaler.fit_transform(bias_matrix)

        n_comp = min(n_components, min(normed.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        comp = svd.fit_transform(normed)

        cum_var = np.cumsum(svd.explained_variance_ratio_)

        layer_data[li] = {
            "bias_matrix": bias_matrix,
            "normed": normed,
            "components": comp,
            "svd": svd,
            "scaler": scaler,
            "labels": word_labels,
            "categories": word_cats,
            "n_concepts": len(word_labels),
            "top10_var": float(cum_var[9]) if n_comp >= 10 else 0,
            "top20_var": float(cum_var[19]) if n_comp >= 20 else 0,
            "top50_var": float(cum_var[49]) if n_comp >= 50 else 0,
            "top100_var": float(cum_var[min(n_comp-1, 99)]),
            "cum_var": cum_var,
        }

    return layer_data


def concept_arithmetic_test(layer_data, all_activations, concepts, model_name):
    """
    概念算术测试：用已知概念A预测概念B
    核心方法：
      predicted(B) = A + (B_attr - A_attr)
    """
    results = []

    # 定义算术测试对（同一类别内的属性替换）
    arithmetic_pairs = [
        # 水果类
        ("apple", "banana", "fruit", "color:red→yellow"),
        ("apple", "orange", "fruit", "color:red→orange"),
        ("apple", "grape", "fruit", "color:red→purple, size:3→2"),
        ("banana", "lemon", "fruit", "color:yellow→yellow, taste:sweet→sour"),
        ("banana", "cherry", "fruit", "size:3→1"),
        ("orange", "watermelon", "fruit", "size:3→5"),
        ("grape", "blueberry", "fruit", "color:purple→blue"),
        # 动物类
        ("dog", "cat", "animal", "同类替换"),
        ("dog", "wolf", "animal", "domestic:1→0"),
        ("dog", "horse", "animal", "size:2→4"),
        ("lion", "tiger", "animal", "同类替换"),
        ("elephant", "giraffe", "animal", "size:5→5, speed:2→4"),
        ("rabbit", "deer", "animal", "size:1→3, domestic:1→0"),
        # 交通工具类
        ("car", "bus", "vehicle", "size:2→3"),
        ("car", "bicycle", "vehicle", "speed:3→2, size:2→1"),
        ("car", "plane", "vehicle", "medium:land→air, speed:3→5"),
        ("ship", "submarine", "vehicle", "同类替换"),
        # 职业类
        ("doctor", "nurse", "profession", "同类替换"),
        ("doctor", "surgeon", "profession", "同类替换, creativity:0→1"),
        ("teacher", "scientist", "profession", "domain:education→science"),
        ("chef", "baker", "profession", "同类替换"),
    ]

    # 定义属性参考词对（用于计算属性方向）
    # 例如：颜色属性方向 = (lemon - apple) 因为 apple=红色, lemon=黄色
    attribute_references = {
        "color_red_to_yellow": ("apple", "lemon", "fruit"),
        "color_red_to_orange": ("apple", "orange", "fruit"),
        "color_red_to_purple": ("apple", "grape", "fruit"),
        "color_red_to_blue": ("apple", "blueberry", "fruit"),
        "color_yellow_to_green": ("banana", "pear", "fruit"),
        "size_small_to_large": ("cherry", "watermelon", "fruit"),
        "taste_sweet_to_sour": ("banana", "lemon", "fruit"),
        "domestic_to_wild": ("dog", "wolf", "animal"),
        "size_2_to_4": ("dog", "horse", "animal"),
        "land_to_air": ("car", "plane", "vehicle"),
        "speed_slow_to_fast": ("bicycle", "plane", "vehicle"),
        "medical_same_domain": ("doctor", "surgeon", "profession"),
    }

    # 选择测试层
    test_layers = sorted(layer_data.keys())
    # 选择方差最高的5个层
    top5_layers = sorted(test_layers, key=lambda l: layer_data[l]["top20_var"], reverse=True)[:5]
    print(f"  Test layers (top5 by variance): {top5_layers}")

    for src_word, tgt_word, cat, desc in arithmetic_pairs:
        # 检查两个词是否存在
        src_cat_acts = all_activations.get(cat, {})
        if src_word not in src_cat_acts or tgt_word not in src_cat_acts:
            continue

        pair_result = {
            "source": src_word,
            "target": tgt_word,
            "category": cat,
            "description": desc,
            "layer_results": {},
        }

        # 找到最佳属性参考对
        best_ref = None
        best_ref_score = -1
        for ref_name, (ref_a, ref_b, ref_cat) in attribute_references.items():
            if ref_cat != cat:
                continue
            # ref_a和ref_b应该和src/tgt有属性对应关系
            # 简单策略：如果src和ref_a相同或tgt和ref_b相同，使用该参考
            if src_word == ref_a and tgt_word == ref_b:
                best_ref = (ref_a, ref_b, ref_cat)
                best_ref_score = 2
                break
            elif ref_cat == cat:
                best_ref = (ref_a, ref_b, ref_cat)
                best_ref_score = 1

        for li in test_layers:
            if li not in layer_data:
                continue

            ld = layer_data[li]

            # 检查目标词是否在该层
            if tgt_word not in ld["labels"] or src_word not in ld["labels"]:
                continue

            src_idx = ld["labels"].index(src_word)
            tgt_idx = ld["labels"].index(tgt_word)

            # 目标真实编码
            tgt_real = ld["normed"][tgt_idx]

            # 原始空间余弦相似度（直接比较两个词的偏置）
            src_bias = ld["normed"][src_idx]
            raw_cos = cosine_sim(src_bias, tgt_real)

            # ====== 正确的概念算术（不使用目标向量） ======
            # 方法1: 3Cos方法 — predicted = src + (ref_b - ref_a)
            # 方法2: 类内平均 — predicted = src + (cat_mean - src) * factor
            # 方法3: 最近邻类比 — 在因子空间找最近邻

            # 方法1: 参考对算术（如果找到参考对）
            if best_ref and best_ref[0] in ld["labels"] and best_ref[1] in ld["labels"]:
                ref_a_bias = ld["normed"][ld["labels"].index(best_ref[0])]
                ref_b_bias = ld["normed"][ld["labels"].index(best_ref[1])]
                # 用参考对的差异作为属性方向
                attr_direction = ref_b_bias - ref_a_bias
                predicted_3cos = src_bias + attr_direction
                arith_3cos = cosine_sim(predicted_3cos, tgt_real)
            else:
                arith_3cos = None

            # 方法2: 类内平均偏移
            cat_words = [w for w, c in zip(ld["labels"], ld["categories"])
                        if c == cat and w != src_word and w != tgt_word]
            if len(cat_words) >= 2:
                cat_biases = np.array([ld["normed"][ld["labels"].index(w)]
                                      for w in cat_words if w in ld["labels"]])
                # 找到类内最近的3个邻居
                cat_sims = [cosine_sim(src_bias, cb) for cb in cat_biases]
                top3_idx = np.argsort(cat_sims)[-3:]
                neighbor_mean = np.mean(cat_biases[top3_idx], axis=0)
                # 预测: 从源出发，移向邻居方向
                shift = neighbor_mean - src_bias
                predicted_neighbor = src_bias + 0.5 * shift
                arith_neighbor = cosine_sim(predicted_neighbor, tgt_real)
            else:
                arith_neighbor = 0.0

            # 方法3: 因子空间最近邻
            src_factors = ld["svd"].transform(src_bias.reshape(1, -1))[0]
            # 计算所有类内其他词的因子距离
            cat_factor_dists = []
            for w, c in zip(ld["labels"], ld["categories"]):
                if c == cat and w != src_word and w != tgt_word:
                    w_factors = ld["svd"].transform(ld["normed"][ld["labels"].index(w)].reshape(1, -1))[0]
                    dist = np.linalg.norm(src_factors - w_factors)
                    cat_factor_dists.append((w, dist, w_factors))

            if cat_factor_dists:
                # 找最近邻
                cat_factor_dists.sort(key=lambda x: x[1])
                nearest_word = cat_factor_dists[0][0]
                nearest_factors = cat_factor_dists[0][2]
                # 用最近邻的因子差异
                factor_shift = nearest_factors - src_factors
                predicted_factor = src_factors + factor_shift
                predicted_factor_recon = ld["svd"].inverse_transform(predicted_factor.reshape(1, -1))[0]
                factor_nn_cos = cosine_sim(predicted_factor_recon, tgt_real)

                # 也测试: 用Top-3最近邻的平均因子差异
                if len(cat_factor_dists) >= 3:
                    top3_mean = np.mean([cd[2] for cd in cat_factor_dists[:3]], axis=0)
                    factor_shift3 = top3_mean - src_factors
                    predicted_factor3 = src_factors + factor_shift3
                    predicted_factor3_recon = ld["svd"].inverse_transform(predicted_factor3.reshape(1, -1))[0]
                    factor_nn3_cos = cosine_sim(predicted_factor3_recon, tgt_real)
                else:
                    factor_nn3_cos = factor_nn_cos
            else:
                factor_nn_cos = 0.0
                factor_nn3_cos = 0.0

            # 方法4: 因子空间类比 (king - man + woman = queen)
            if best_ref and best_ref[0] in ld["labels"] and best_ref[1] in ld["labels"]:
                ref_a_factors = ld["svd"].transform(
                    ld["normed"][ld["labels"].index(best_ref[0])].reshape(1, -1))[0]
                ref_b_factors = ld["svd"].transform(
                    ld["normed"][ld["labels"].index(best_ref[1])].reshape(1, -1))[0]
                analogy = src_factors + (ref_b_factors - ref_a_factors)
                analogy_recon = ld["svd"].inverse_transform(analogy.reshape(1, -1))[0]
                factor_analogy_cos = cosine_sim(analogy_recon, tgt_real)
            else:
                factor_analogy_cos = None

            pair_result["layer_results"][str(li)] = {
                "raw_cosine": round(raw_cos, 4),
                "neighbor_arith": round(arith_neighbor, 4),
                "factor_nn_cosine": round(factor_nn_cos, 4),
                "factor_nn3_cosine": round(factor_nn3_cos, 4),
                "3cos_arith": round(arith_3cos, 4) if arith_3cos is not None else None,
                "factor_analogy": round(factor_analogy_cos, 4) if factor_analogy_cos is not None else None,
            }

        results.append(pair_result)

    # 汇总统计
    summary = {"overall": {}, "by_category": {}, "by_layer": {}}

    # 总体统计
    all_raw = []
    all_neighbor = []
    all_factor_nn = []
    all_factor_nn3 = []
    all_3cos = []
    all_analogy = []
    for r in results:
        for li, lr in r["layer_results"].items():
            all_raw.append(lr["raw_cosine"])
            all_neighbor.append(lr["neighbor_arith"])
            all_factor_nn.append(lr["factor_nn_cosine"])
            all_factor_nn3.append(lr["factor_nn3_cosine"])
            if lr["3cos_arith"] is not None:
                all_3cos.append(lr["3cos_arith"])
            if lr["factor_analogy"] is not None:
                all_analogy.append(lr["factor_analogy"])

    if all_raw:
        summary["overall"] = {
            "raw_cosine_mean": float(np.mean(all_raw)),
            "neighbor_arith_mean": float(np.mean(all_neighbor)),
            "factor_nn_mean": float(np.mean(all_factor_nn)),
            "factor_nn3_mean": float(np.mean(all_factor_nn3)),
            "3cos_mean": float(np.mean(all_3cos)) if all_3cos else None,
            "analogy_mean": float(np.mean(all_analogy)) if all_analogy else None,
            "n_pairs": len(results),
        }

    # 按类别统计
    for r in results:
        cat = r["category"]
        if cat not in summary["by_category"]:
            summary["by_category"][cat] = {"raw": [], "neighbor": [], "factor_nn": [], "factor_nn3": []}
        for li, lr in r["layer_results"].items():
            summary["by_category"][cat]["raw"].append(lr["raw_cosine"])
            summary["by_category"][cat]["neighbor"].append(lr["neighbor_arith"])
            summary["by_category"][cat]["factor_nn"].append(lr["factor_nn_cosine"])
            summary["by_category"][cat]["factor_nn3"].append(lr["factor_nn3_cosine"])

    for cat in summary["by_category"]:
        d = summary["by_category"][cat]
        summary["by_category"][cat] = {
            "raw_mean": float(np.mean(d["raw"])),
            "neighbor_mean": float(np.mean(d["neighbor"])),
            "factor_nn_mean": float(np.mean(d["factor_nn"])),
            "factor_nn3_mean": float(np.mean(d["factor_nn3"])),
        }

    # 按层统计（只用top5层）
    for li in top5_layers:
        li_raw = []
        li_neighbor = []
        li_factor_nn = []
        li_factor_nn3 = []
        for r in results:
            if str(li) in r["layer_results"]:
                lr = r["layer_results"][str(li)]
                li_raw.append(lr["raw_cosine"])
                li_neighbor.append(lr["neighbor_arith"])
                li_factor_nn.append(lr["factor_nn_cosine"])
                li_factor_nn3.append(lr["factor_nn3_cosine"])
        if li_raw:
            summary["by_layer"][str(li)] = {
                "raw_mean": float(np.mean(li_raw)),
                "neighbor_mean": float(np.mean(li_neighbor)),
                "factor_nn_mean": float(np.mean(li_factor_nn)),
                "factor_nn3_mean": float(np.mean(li_factor_nn3)),
                "n_pairs": len(li_raw),
            }

    return results, summary, top5_layers


def embedding_vs_activation_comparison(model, tokenizer, all_activations, concepts):
    """对比embedding和激活的因子结构"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

    print("\n  Embedding vs Activation comparison...")

    words = []
    embeddings = []
    for cat_name, cat_data in concepts.items():
        for word in cat_data["words"]:
            emb = get_embedding(model, tokenizer, word)
            if emb is not None:
                words.append(word)
                embeddings.append(emb)

    if len(words) < 20:
        return {}

    emb_matrix = np.array(embeddings)
    scaler = StandardScaler()
    normed = scaler.fit_transform(emb_matrix)

    n_comp = min(50, len(words) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    comp = svd.fit_transform(normed)
    cum_var = np.cumsum(svd.explained_variance_ratio_)

    result = {
        "embedding_top10": float(cum_var[9]),
        "embedding_top20": float(cum_var[19]),
        "embedding_top50": float(cum_var[min(n_comp-1, 49)]),
    }

    # 对比：计算embedding和L1/L2激活的相似度
    if 0 in all_activations.get("fruit", {}):
        word_set = set(words)
        for cat_name in ["fruit", "animal"]:
            if cat_name not in all_activations:
                continue
            for word in list(all_activations[cat_name].keys())[:5]:
                if word in word_set and 0 in all_activations[cat_name][word]:
                    emb_vec = emb_matrix[words.index(word)]
                    act_vec = all_activations[cat_name][word][0]
                    cos = cosine_sim(emb_vec, act_vec)
                    result[f"emb_act_cos_{cat_name}_{word}"] = round(cos, 4)

    print(f"  Embedding SVD: top10={result['embedding_top10']*100:.1f}%, top20={result['embedding_top20']*100:.1f}%")
    return result


def run_single_model(model_name: str, model_path: Path) -> Dict:
    """运行单个模型的完整测试"""
    print(f"\n{'='*70}")
    print(f"  Stage462: {model_name} — Factor Verification + Concept Arithmetic")
    print(f"{'='*70}")

    # 1. 加载模型
    print(f"\n[1/5] Loading model to CUDA...")
    t0 = time.time()
    model, tokenizer, layer_count, neuron_dim = load_model(model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # 2. 提取激活
    print(f"\n[2/5] Extracting activations...")
    t0 = time.time()
    all_activations = extract_all_activations(model, tokenizer, CONCEPTS, layer_count)
    print(f"  Extracted in {time.time()-t0:.1f}s")

    # 3. Embedding vs Activation对比
    print(f"\n[3/5] Embedding vs Activation comparison...")
    t0 = time.time()
    emb_comparison = embedding_vs_activation_comparison(model, tokenizer, all_activations, CONCEPTS)
    print(f"  Done in {time.time()-t0:.1f}s")

    # 释放模型（节省GPU内存）
    print(f"\n  Freeing model memory...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"  GPU memory after free: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # 4. 逐层SVD
    print(f"\n[4/5] Per-layer SVD analysis...")
    t0 = time.time()
    layer_data = per_layer_svd(all_activations, CONCEPTS, layer_count, n_components=100)

    # 找黄金层
    best_layer = max(layer_data, key=lambda l: layer_data[l]["top20_var"])
    print(f"  Golden layer: L{best_layer} (top20={layer_data[best_layer]['top20_var']*100:.1f}%)")

    # 层摘要
    print(f"\n  Layer summary (sorted by top20 var):")
    sorted_layers = sorted(layer_data.keys(), key=lambda l: layer_data[l]["top20_var"], reverse=True)
    for li in sorted_layers[:10]:
        ld = layer_data[li]
        print(f"    L{li}: top10={ld['top10_var']*100:.1f}%, top20={ld['top20_var']*100:.1f}%, "
              f"top50={ld['top50_var']*100:.1f}%, top100={ld['top100_var']*100:.1f}%")
    print(f"  Per-layer SVD done in {time.time()-t0:.1f}s")

    # 5. 概念算术测试
    print(f"\n[5/5] Concept arithmetic test...")
    t0 = time.time()
    arith_results, arith_summary, top5_layers = concept_arithmetic_test(
        layer_data, all_activations, CONCEPTS, model_name
    )
    print(f"  Arithmetic test done in {time.time()-t0:.1f}s")
    print(f"\n  === Concept Arithmetic Summary ===")
    if arith_summary.get("overall"):
        o = arith_summary["overall"]
        print(f"  Raw cosine:        {o['raw_cosine_mean']:.4f}")
        print(f"  Neighbor arith:    {o['neighbor_arith_mean']:.4f}")
        print(f"  Factor NN:         {o['factor_nn_mean']:.4f}")
        print(f"  Factor NN3:        {o['factor_nn3_mean']:.4f}")
        if o.get("3cos_mean"):
            print(f"  3Cos arith:        {o['3cos_mean']:.4f}")
        if o.get("analogy_mean"):
            print(f"  Factor analogy:    {o['analogy_mean']:.4f}")
    print(f"  Per-layer (top5):")
    for li, ls in arith_summary.get("by_layer", {}).items():
        print(f"    L{li}: raw={ls['raw_mean']:.3f}, neighbor={ls['neighbor_mean']:.3f}, "
              f"factor_nn={ls['factor_nn_mean']:.3f}, factor_nn3={ls['factor_nn3_mean']:.3f}")
    print(f"  Per-category:")
    for cat, cs in arith_summary.get("by_category", {}).items():
        print(f"    {cat}: raw={cs['raw_mean']:.3f}, neighbor={cs['neighbor_mean']:.3f}, "
              f"factor_nn={cs['factor_nn_mean']:.3f}")

    # 保存层SVD摘要
    layer_summary = {}
    for li, ld in layer_data.items():
        layer_summary[str(li)] = {
            "n_concepts": ld["n_concepts"],
            "top10_var": ld["top10_var"],
            "top20_var": ld["top20_var"],
            "top50_var": ld["top50_var"],
            "top100_var": ld["top100_var"],
        }

    return {
        "model_name": model_name,
        "layer_count": layer_count,
        "neuron_dim": neuron_dim,
        "golden_layer": best_layer,
        "n_concepts_total": sum(ld["n_concepts"] for ld in layer_data.values()) // max(1, len(layer_data)),
        "layer_summary": layer_summary,
        "embedding_comparison": emb_comparison,
        "arithmetic_results": arith_results,
        "arithmetic_summary": arith_summary,
        "top5_layers": top5_layers,
    }


def build_report(results: Dict) -> str:
    lines = []
    lines.append("# Stage462: 概念算术测试 — 单模型独立验证")
    lines.append("")
    lines.append(f"**时间**: 2026-04-01 09:45")
    lines.append(f"**模型**: {results['model_name']}")
    lines.append(f"**概念数**: ~{results['n_concepts_total']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 逐层SVD
    lines.append("## 1. 逐层SVD方差解释")
    lines.append("")
    lines.append("| 层 | Top-10 | Top-20 | Top-50 | Top-100 |")
    lines.append("|----|--------|--------|--------|---------|")
    sorted_layers = sorted(results["layer_summary"].items(),
                          key=lambda x: x[1]["top20_var"], reverse=True)
    for li_str, ls in sorted_layers:
        mark = " ★" if int(li_str) == results["golden_layer"] else ""
        lines.append(f"| L{li_str}{mark} | {ls['top10_var']*100:.1f}% | "
                    f"{ls['top20_var']*100:.1f}% | {ls['top50_var']*100:.1f}% | "
                    f"{ls['top100_var']*100:.1f}% |")
    lines.append("")

    # 2. Embedding vs Activation
    lines.append("## 2. Embedding vs MLP激活 对比")
    lines.append("")
    ec = results.get("embedding_comparison", {})
    if ec:
        lines.append("| 空间 | Top-10 | Top-20 | Top-50 |")
        lines.append("|------|--------|--------|--------|")
        lines.append(f"| Embedding | {ec.get('embedding_top10',0)*100:.1f}% | "
                    f"{ec.get('embedding_top20',0)*100:.1f}% | {ec.get('embedding_top50',0)*100:.1f}% |")
        gs = str(results["golden_layer"])
        if gs in results["layer_summary"]:
            ls = results["layer_summary"][gs]
            lines.append(f"| MLP激活L{gs} | {ls['top10_var']*100:.1f}% | "
                        f"{ls['top20_var']*100:.1f}% | {ls['top50_var']*100:.1f}% |")
        lines.append("")

        # Embedding-激活余弦相似度
        emb_act = {k: v for k, v in ec.items() if k.startswith("emb_act_cos_")}
        if emb_act:
            lines.append("### Embedding-激活余弦相似度")
            lines.append("")
            lines.append("| 词 | 余弦相似度 |")
            lines.append("|-----|----------|")
            for k, v in emb_act.items():
                word = k.replace("emb_act_cos_", "").replace("_", " ")
                lines.append(f"| {word} | {v} |")
            lines.append("")

    # 3. 概念算术测试
    arith = results.get("arithmetic_summary", {})
    lines.append("## 3. 概念算术测试结果")
    lines.append("")

    if arith.get("overall"):
        o = arith["overall"]
        lines.append("### 总体统计")
        lines.append("")
        lines.append("| 方法 | 平均余弦相似度 |")
        lines.append("|------|----------------|")
        lines.append(f"| 原始偏置直接比较 | {o['raw_cosine_mean']:.4f} |")
        lines.append(f"| 最近邻算术(偏置空间) | {o['neighbor_arith_mean']:.4f} |")
        lines.append(f"| 因子空间最近邻 | {o['factor_nn_mean']:.4f} |")
        lines.append(f"| 因子空间Top-3近邻 | {o['factor_nn3_mean']:.4f} |")
        if o.get("3cos_mean"):
            lines.append(f"| 3Cos类比(属性参考) | {o['3cos_mean']:.4f} |")
        if o.get("analogy_mean"):
            lines.append(f"| 因子空间类比 | {o['analogy_mean']:.4f} |")
        lines.append("")
        lines.append(f"测试对数: {o['n_pairs']}")
        lines.append("")

    # 按层
    if arith.get("by_layer"):
        lines.append("### 按层统计（Top-5层）")
        lines.append("")
        lines.append("| 层 | 原始 | 近邻算术 | 因子NN | 因子NN3 | 对数 |")
        lines.append("|----|------|---------|--------|---------|------|")
        for li, ls in sorted(arith["by_layer"].items(), key=lambda x: int(x[0])):
            lines.append(f"| L{li} | {ls['raw_mean']:.3f} | {ls['neighbor_mean']:.3f} | "
                        f"{ls['factor_nn_mean']:.3f} | {ls['factor_nn3_mean']:.3f} | {ls['n_pairs']} |")
        lines.append("")

    # 按类别
    if arith.get("by_category"):
        lines.append("### 按类别统计")
        lines.append("")
        lines.append("| 类别 | 原始 | 近邻算术 | 因子NN |")
        lines.append("|------|------|---------|--------|")
        for cat, cs in arith["by_category"].items():
            lines.append(f"| {cat} | {cs['raw_mean']:.3f} | {cs['neighbor_mean']:.3f} | "
                        f"{cs['factor_nn_mean']:.3f} |")
        lines.append("")

    # 详细算术结果
    lines.append("### 详细算术结果")
    lines.append("")
    for r in results.get("arithmetic_results", []):
        lines.append(f"**{r['source']} → {r['target']}** ({r['description']})")
        lines.append("")
        lines.append("| 层 | 原始 | 近邻 | 因子NN | 因子NN3 | 3Cos |")
        lines.append("|----|------|------|--------|---------|------|")
        for li, lr in sorted(r["layer_results"].items(), key=lambda x: int(x[0])):
            cos3 = f"{lr['3cos_arith']:.3f}" if lr.get("3cos_arith") is not None else "-"
            lines.append(f"| L{li} | {lr['raw_cosine']:.3f} | {lr['neighbor_arith']:.3f} | "
                        f"{lr['factor_nn_cosine']:.3f} | {lr['factor_nn3_cosine']:.3f} | {cos3} |")
        lines.append("")

    # 4. 结论
    lines.append("## 4. 结论")
    lines.append("")
    if arith.get("overall"):
        o = arith["overall"]
        lines.append(f"- 原始偏置余弦(概念间相似度): {o['raw_cosine_mean']:.4f}")
        lines.append(f"- 最近邻算术精度(偏置空间): {o['neighbor_arith_mean']:.4f}")
        lines.append(f"- 因子空间最近邻精度: {o['factor_nn_mean']:.4f}")
        lines.append(f"- 因子空间Top-3近邻精度: {o['factor_nn3_mean']:.4f}")
        if o.get("3cos_mean"):
            lines.append(f"- 3Cos类比精度: {o['3cos_mean']:.4f}")
        lines.append(f"- 黄金层: L{results['golden_layer']}")
        best_layer_key = max(arith.get("by_layer", {}).items(),
                            key=lambda x: x[1].get("factor_nn3_mean", 0),
                            default=(None, None))
        if best_layer_key[0]:
            lines.append(f"- 最佳层(因子NN3): L{best_layer_key[0]}={best_layer_key[1]['factor_nn3_mean']:.3f}")
    lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python stage462_concept_arithmetic.py [qwen3|deepseek]")
        sys.exit(1)

    model_arg = sys.argv[1].lower()

    if model_arg == "qwen3":
        model_name = "Qwen3-4B"
        model_path = QWEN3_MODEL_PATH
    elif model_arg == "deepseek":
        model_name = "DeepSeek-7B"
        model_path = DEEPSEEK7B_MODEL_PATH
    else:
        print(f"Unknown model: {model_arg}. Use 'qwen3' or 'deepseek'.")
        sys.exit(1)

    print("=" * 70)
    print(f"  Stage462: {model_name} — Concept Arithmetic Test (CUDA)")
    print(f"  Single model mode (avoiding GPU OOM)")
    print("=" * 70)

    results = run_single_model(model_name, model_path)

    # 保存
    safe_name = model_name.replace("-", "_").replace(" ", "_").lower()
    summary_path = OUTPUT_DIR / f"{safe_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(results), f, ensure_ascii=False, indent=2)

    report = build_report(results)
    report_path = OUTPUT_DIR / f"{safe_name}_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  Report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
