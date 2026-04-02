# -*- coding: utf-8 -*-
"""
Stage458: 中英文双语概念编码一致性验证
======================================

核心假设：概念编码是语言无关的
  - "apple" 和 "苹果" 在神经网络中应该激活相似的语义因子
  - 如果成立，说明神经网络提取的是概念本身的数学结构，而非语言表层特征

实验设计：
1. 选取50个中英双语概念对（覆盖多个类别）
2. 提取中英文的MLP激活 → 偏置向量
3. 计算中英偏置的余弦相似度（直接对比）
4. SVD分解中英联合偏置矩阵 → 因子一致性
5. 因子-属性关联对比（英文因子vs中文因子）
6. 概念算术跨语言测试（dog-cat偏移能否预测猫-狗？）

模型：DeepSeek-7B (CUDA) — Qwen tokenizer原生支持中英文
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen3_language_shared import (
    PROJECT_ROOT,
    capture_qwen_mlp_payloads,
    discover_layers,
    move_batch_to_model_device,
    remove_hooks,
)

DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)

TIMESTAMP = "20260401"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage458_bilingual_concepts_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 中英双语概念对（含属性标注） ====================
BILINGUAL_CONCEPTS = {
    "fruit": {
        "label": "水果",
        "pairs": [
            {"en": "apple", "zh": "苹果", "color": "red", "size": 3},
            {"en": "banana", "zh": "香蕉", "color": "yellow", "size": 3},
            {"en": "orange", "zh": "橙子", "color": "orange", "size": 3},
            {"en": "grape", "zh": "葡萄", "color": "purple", "size": 2},
            {"en": "mango", "zh": "芒果", "color": "orange", "size": 3},
            {"en": "peach", "zh": "桃子", "color": "pink", "size": 3},
            {"en": "lemon", "zh": "柠檬", "color": "yellow", "size": 2},
            {"en": "cherry", "zh": "樱桃", "color": "red", "size": 1},
            {"en": "watermelon", "zh": "西瓜", "color": "green", "size": 5},
            {"en": "strawberry", "zh": "草莓", "color": "red", "size": 1},
        ],
    },
    "animal": {
        "label": "动物",
        "pairs": [
            {"en": "dog", "zh": "狗", "size": 2, "domestic": 1},
            {"en": "cat", "zh": "猫", "size": 2, "domestic": 1},
            {"en": "horse", "zh": "马", "size": 4, "domestic": 1},
            {"en": "lion", "zh": "狮子", "size": 3, "domestic": 0},
            {"en": "tiger", "zh": "老虎", "size": 3, "domestic": 0},
            {"en": "elephant", "zh": "大象", "size": 5, "domestic": 0},
            {"en": "whale", "zh": "鲸鱼", "size": 5, "domestic": 0},
            {"en": "snake", "zh": "蛇", "size": 2, "domestic": 0},
            {"en": "eagle", "zh": "鹰", "size": 2, "domestic": 0},
            {"en": "rabbit", "zh": "兔子", "size": 1, "domestic": 1},
        ],
    },
    "vehicle": {
        "label": "交通工具",
        "pairs": [
            {"en": "car", "zh": "汽车", "speed": 3, "medium": "land"},
            {"en": "bus", "zh": "公交车", "speed": 3, "medium": "land"},
            {"en": "train", "zh": "火车", "speed": 4, "medium": "land"},
            {"en": "plane", "zh": "飞机", "speed": 4, "medium": "air"},
            {"en": "ship", "zh": "船", "speed": 2, "medium": "water"},
            {"en": "bicycle", "zh": "自行车", "speed": 2, "medium": "land"},
            {"en": "motorcycle", "zh": "摩托车", "speed": 3, "medium": "land"},
            {"en": "truck", "zh": "卡车", "speed": 2, "medium": "land"},
            {"en": "helicopter", "zh": "直升机", "speed": 4, "medium": "air"},
            {"en": "rocket", "zh": "火箭", "speed": 5, "medium": "air"},
        ],
    },
    "natural": {
        "label": "自然",
        "pairs": [
            {"en": "mountain", "zh": "山", "size": 4},
            {"en": "river", "zh": "河流", "size": 3},
            {"en": "ocean", "zh": "海洋", "size": 5},
            {"en": "forest", "zh": "森林", "size": 4},
            {"en": "desert", "zh": "沙漠", "size": 4},
            {"en": "island", "zh": "岛屿", "size": 3},
            {"en": "lake", "zh": "湖泊", "size": 3},
            {"en": "cloud", "zh": "云", "size": 3},
            {"en": "fire", "zh": "火", "size": 1},
            {"en": "stone", "zh": "石头", "size": 1},
        ],
    },
    "furniture": {
        "label": "家具",
        "pairs": [
            {"en": "chair", "zh": "椅子", "softness": 1},
            {"en": "table", "zh": "桌子", "softness": 1},
            {"en": "bed", "zh": "床", "softness": 3},
            {"en": "sofa", "zh": "沙发", "softness": 4},
            {"en": "desk", "zh": "书桌", "softness": 1},
            {"en": "lamp", "zh": "灯", "softness": 1},
            {"en": "carpet", "zh": "地毯", "softness": 3},
            {"en": "pillow", "zh": "枕头", "softness": 5},
            {"en": "mirror", "zh": "镜子", "softness": 1},
            {"en": "curtain", "zh": "窗帘", "softness": 2},
        ],
    },
}

# 统计
TOTAL_PAIRS = sum(len(c["pairs"]) for c in BILINGUAL_CONCEPTS.values())
print(f"Total categories: {len(BILINGUAL_CONCEPTS)}")
print(f"Total concept pairs: {TOTAL_PAIRS}")

EPS = 1e-8


def sanitize_for_json(obj):
    """递归转换numpy类型"""
    if isinstance(obj, dict):
        return {str(int(k) if isinstance(k, (np.integer,)) else k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ==================== 模型加载 ====================
def load_model(model_path: Path):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    want_cuda = torch.cuda.is_available()
    print(f"  CUDA: {want_cuda}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True,
        trust_remote_code=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
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
    hidden_dim = discover_layers(model)[0].mlp.down_proj.out_features
    print(f"  Layers: {layer_count}, HiddenDim: {hidden_dim}")

    # 测试中文tokenization
    test_tokens = tokenizer.encode("苹果", add_special_tokens=False)
    print(f"  Test '苹果' tokens: {test_tokens} → {tokenizer.convert_ids_to_tokens(test_tokens)}")
    test_tokens_en = tokenizer.encode("apple", add_special_tokens=False)
    print(f"  Test 'apple' tokens: {test_tokens_en} → {tokenizer.convert_ids_to_tokens(test_tokens_en)}")

    return model, tokenizer, layer_count, hidden_dim


def extract_word_activation(model, tokenizer, word: str, layer_count: int) -> Optional[Dict[int, np.ndarray]]:
    """提取单个词在所有层的MLP激活"""
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

        activations = {}
        for layer_idx in range(layer_count):
            buf = buffers[layer_idx]
            if buf is not None:
                pos = target_pos if target_pos >= 0 else buf.shape[1] + target_pos
                pos = max(0, min(pos, buf.shape[1] - 1))
                activations[layer_idx] = buf[0, pos].float().numpy()
        return activations
    finally:
        remove_hooks(handles)


def extract_all_bilingual(model, tokenizer, concepts: Dict, layer_count: int) -> Dict:
    """提取所有双语概念激活"""
    results = {}
    total = sum(len(c["pairs"]) for c in concepts.values()) * 2  # ×2 for en+zh
    done = 0

    for cat_name, cat_info in concepts.items():
        print(f"  [{cat_name}]")
        results[cat_name] = []

        for pair in cat_info["pairs"]:
            entry = {"en": pair["en"], "zh": pair["zh"]}

            # 英文
            try:
                en_acts = extract_word_activation(model, tokenizer, pair["en"], layer_count)
                if en_acts:
                    entry["en_activations"] = en_acts
                    done += 1
            except Exception as e:
                print(f"    ERROR {pair['en']}: {e}")

            # 中文
            try:
                zh_acts = extract_word_activation(model, tokenizer, pair["zh"], layer_count)
                if zh_acts:
                    entry["zh_activations"] = zh_acts
                    done += 1
            except Exception as e:
                print(f"    ERROR {pair['zh']}: {e}")

            results[cat_name].append(entry)

        print(f"    Done: {done}/{total}")

    print(f"  Total extracted: {done}/{total}")
    return results


# ==================== 偏置矩阵 ====================
def build_bilingual_bias_matrices(
    all_data: Dict,
    concepts: Dict,
    target_layers: List[int],
) -> Dict:
    """分别为英文和中文构建偏置矩阵"""
    results = {"en": {"biases": [], "labels": [], "categories": []},
               "zh": {"biases": [], "labels": [], "categories": []}}

    for lang in ["en", "zh"]:
        for cat_name, pairs in all_data.items():
            cat_info = concepts[cat_name]
            # 收集该类别该语言的所有激活
            word_acts = {}
            for pair_data in pairs:
                key = f"{pair_data['en']}_{pair_data['zh']}"
                acts = pair_data.get(f"{lang}_activations")
                if acts is not None:
                    word_acts[key] = acts

            if len(word_acts) < 2:
                continue

            # 计算类别基底
            layer_vecs = defaultdict(list)
            for key, acts in word_acts.items():
                for l in target_layers:
                    if l in acts:
                        layer_vecs[l].append(acts[l])

            basis = {}
            for l in target_layers:
                if layer_vecs[l]:
                    basis[l] = np.mean(layer_vecs[l], axis=0)

            if not basis:
                continue

            # 计算偏置
            for key, acts in word_acts.items():
                bias_parts = []
                for l in sorted(target_layers):
                    if l in acts and l in basis:
                        bias = acts[l] - basis[l]
                        norm = np.linalg.norm(bias)
                        if norm > EPS:
                            bias_parts.append(bias / norm)

                if bias_parts:
                    combined = np.concatenate(bias_parts)
                    results[lang]["biases"].append(combined)
                    # 标签用英文统一
                    results[lang]["labels"].append(key)
                    results[lang]["categories"].append(cat_name)

        results[lang]["biases"] = np.array(results[lang]["biases"]) if results[lang]["biases"] else np.array([])

    return results


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


# ==================== 核心：中英一致性分析 ====================
def cross_lingual_similarity(en_biases, zh_biases, en_labels, zh_labels) -> Dict:
    """计算每个概念对的中英偏置相似度"""
    print(f"\n  Cross-lingual bias similarity analysis...")
    print(f"  EN concepts: {len(en_labels)}, ZH concepts: {len(zh_labels)}")

    # 构建标签到索引的映射
    en_map = {label: i for i, label in enumerate(en_labels)}
    zh_map = {label: i for i, label in enumerate(zh_labels)}

    results = []
    all_sims = []

    # 对每个概念对计算中英偏置相似度
    matched = 0
    for label in en_labels:
        if label in zh_map:
            sim = cosine_sim(en_biases[en_map[label]], zh_biases[zh_map[label]])
            en_word, zh_word = label.split("_")
            results.append({
                "en": en_word, "zh": zh_word, "label": label,
                "cosine_sim": sim,
            })
            all_sims.append(sim)
            matched += 1

    print(f"  Matched: {matched}")
    if all_sims:
        print(f"  Mean similarity: {np.mean(all_sims):.4f}")
        print(f"  Median similarity: {np.median(all_sims):.4f}")
        print(f"  Min: {np.min(all_sims):.4f}, Max: {np.max(all_sims):.4f}")

        # 按类别统计
        cat_sims = defaultdict(list)
        for r in results:
            # 找到类别
            for cat_name, cat_info in BILINGUAL_CONCEPTS.items():
                for p in cat_info["pairs"]:
                    if p["en"] == r["en"] and p["zh"] == r["zh"]:
                        cat_sims[cat_name].append(r["cosine_sim"])
                        break

        print(f"\n  By category:")
        for cat, sims in cat_sims.items():
            print(f"    {cat}: {np.mean(sims):.4f} (n={len(sims)})")

    return {
        "paired_similarities": results,
        "mean": float(np.mean(all_sims)) if all_sims else 0,
        "median": float(np.median(all_sims)) if all_sims else 0,
        "std": float(np.std(all_sims)) if all_sims else 0,
        "min": float(np.min(all_sims)) if all_sims else 0,
        "max": float(np.max(all_sims)) if all_sims else 0,
        "n_matched": matched,
    }


def joint_svd_analysis(en_biases, zh_biases, en_labels, zh_labels, en_categories, zh_categories, n_components=15):
    """联合SVD分析：中英文偏置矩阵合并后分解"""
    from sklearn.decomposition import TruncatedSVD

    print(f"\n  Joint SVD analysis...")

    # 合并中英文偏置（按对齐的概念对）
    en_map = {label: i for i, label in enumerate(en_labels)}
    zh_map = {label: i for i, label in enumerate(zh_labels)}

    joint_biases = []
    joint_labels = []
    joint_lang = []
    joint_cats = []

    for label in en_labels:
        if label in zh_map:
            # 英文偏置
            joint_biases.append(en_biases[en_map[label]])
            joint_labels.append(f"EN_{label}")
            joint_lang.append("en")
            joint_cats.append(en_categories[en_map[label]])
            # 中文偏置
            joint_biases.append(zh_biases[zh_map[label]])
            joint_labels.append(f"ZH_{label}")
            joint_lang.append("zh")
            joint_cats.append(zh_categories[zh_map[label]])

    joint_matrix = np.array(joint_biases)
    print(f"  Joint matrix: {joint_matrix.shape}")

    # SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    components = svd.fit_transform(joint_matrix)

    results = {
        "joint_matrix_shape": list(joint_matrix.shape),
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(svd.explained_variance_ratio_).tolist(),
        "n_matched_pairs": len(en_labels),
    }

    # 打印累计方差
    for k in [5, 10, 15]:
        if k <= n_components:
            print(f"    Top-{k} cumulative: {results['cumulative_variance'][k-1]*100:.1f}%")

    # 每个因子的top概念（标记语言）
    factor_info = {}
    for i in range(min(n_components, 12)):
        scores = components[:, i]
        top_idx = np.argsort(scores)[-10:]

        top_items = []
        for j in reversed(top_idx):
            top_items.append({
                "label": joint_labels[j],
                "lang": joint_lang[j],
                "score": float(scores[j]),
            })

        factor_info[f"factor_{i}"] = {
            "variance": float(svd.explained_variance_ratio_[i]),
            "top_concepts": top_items,
        }

        en_tops = [t for t in top_items[:5] if t["lang"] == "en"]
        zh_tops = [t for t in top_items[:5] if t["lang"] == "zh"]
        print(f"    Factor {i} ({svd.explained_variance_ratio_[i]*100:.1f}%): "
              f"EN[{', '.join(t['label'] for t in en_tops[:2])}] "
              f"ZH[{', '.join(t['label'] for t in zh_tops[:2])}]")

    results["factor_info"] = factor_info

    # 中英文在每个因子上的相关性
    en_components = []
    zh_components = []
    for label in en_labels:
        if label in zh_map:
            # 找到在joint中的位置
            for j, jl in enumerate(joint_labels):
                if jl == f"EN_{label}":
                    en_components.append(components[j])
                if jl == f"ZH_{label}":
                    zh_components.append(components[j])

    if en_components and zh_components:
        en_components = np.array(en_components)
        zh_components = np.array(zh_components)

        # 逐因子相关
        factor_correlations = []
        for i in range(n_components):
            corr = np.corrcoef(en_components[:, i], zh_components[:, i])[0, 1]
            factor_correlations.append(float(corr))
            print(f"    Factor {i} EN-ZH correlation: {corr:.4f}")

        results["factor_correlations"] = factor_correlations
        results["mean_factor_correlation"] = float(np.mean(factor_correlations))
        print(f"    Mean factor correlation: {np.mean(factor_correlations):.4f}")

    return results, components, joint_labels, joint_lang, joint_cats


def cross_lingual_arithmetic(en_biases, zh_biases, en_labels, zh_labels):
    """跨语言概念算术测试"""
    print(f"\n  Cross-lingual arithmetic tests...")

    en_map = {label: i for i, label in enumerate(en_labels)}
    zh_map = {label: i for i, label in enumerate(zh_labels)}

    # 测试对：用英文偏移预测中文偏置
    test_pairs = [
        ("dog_狗", "cat_猫"),    # animal→animal
        ("car_汽车", "bus_公交车"),  # vehicle→vehicle
        ("apple_苹果", "banana_香蕉"),  # fruit→fruit
        ("mountain_山", "ocean_海洋"),  # natural→natural
        ("chair_椅子", "table_桌子"),  # furniture→furniture
    ]

    results = []
    for src, tgt in test_pairs:
        if src not in en_map or tgt not in en_map or src not in zh_map or tgt not in zh_map:
            continue

        # 英文偏移: shift = bias_en(tgt) - bias_en(src)
        shift_en = en_biases[en_map[tgt]] - en_biases[en_map[src]]

        # 用英文偏移预测中文目标: pred_zh = bias_zh(src) + shift_en
        pred_zh = zh_biases[zh_map[src]] + shift_en
        actual_zh = zh_biases[zh_map[tgt]]

        sim = cosine_sim(pred_zh, actual_zh)
        results.append({
            "source": src, "target": tgt,
            "shift_lang": "en", "predict_lang": "zh",
            "cosine_sim": sim,
        })
        print(f"    EN shift '{src}'→'{tgt}' applied to ZH: {sim:.4f}")

    # 反向：用中文偏移预测英文
    for src, tgt in test_pairs:
        if src not in zh_map or tgt not in zh_map or src not in en_map or tgt not in en_map:
            continue

        shift_zh = zh_biases[zh_map[tgt]] - zh_biases[zh_map[src]]
        pred_en = en_biases[en_map[src]] + shift_zh
        actual_en = en_biases[en_map[tgt]]

        sim = cosine_sim(pred_en, actual_en)
        results.append({
            "source": src, "target": tgt,
            "shift_lang": "zh", "predict_lang": "en",
            "cosine_sim": sim,
        })
        print(f"    ZH shift '{src}'→'{tgt}' applied to EN: {sim:.4f}")

    en_to_zh = [r["cosine_sim"] for r in results if r["shift_lang"] == "en"]
    zh_to_en = [r["cosine_sim"] for r in results if r["shift_lang"] == "zh"]

    print(f"\n  EN→ZH shift avg: {np.mean(en_to_zh):.4f}" if en_to_zh else "")
    print(f"  ZH→EN shift avg: {np.mean(zh_to_en):.4f}" if zh_to_en else "")

    return results


# ==================== 报告 ====================
def generate_report(all_results, output_dir):
    lines = [
        "# Stage458: 中英文双语概念编码一致性验证",
        "",
        f"**时间**: 2026-04-01 02:10",
        f"**模型**: DeepSeek-7B (Qwen tokenizer, 中英双语)",
        f"**概念对**: {TOTAL_PAIRS}个（5个类别×10个）",
        "",
        "---",
    ]

    # 1. 中英偏置相似度
    sim = all_results.get("cross_lingual_sim", {})
    lines.append("\n## 1. 中英文偏置向量相似度")
    lines.append(f"- 匹配概念对: {sim.get('n_matched', 0)}")
    lines.append(f"- 平均余弦相似度: **{sim.get('mean', 0):.4f}**")
    lines.append(f"- 中位数: {sim.get('median', 0):.4f}")
    lines.append(f"- 标准差: {sim.get('std', 0):.4f}")
    lines.append(f"- 范围: [{sim.get('min', 0):.4f}, {sim.get('max', 0):.4f}]")

    # 按类别的相似度
    paired = sim.get("paired_similarities", [])
    if paired:
        cat_sims = defaultdict(list)
        for p in paired:
            for cat_name, cat_info in BILINGUAL_CONCEPTS.items():
                for pair in cat_info["pairs"]:
                    if pair["en"] == p["en"] and pair["zh"] == p["zh"]:
                        cat_sims[cat_name].append(p["cosine_sim"])
                        break

        lines.append(f"\n| 类别 | 平均相似度 | 概念数 |")
        lines.append(f"|------|-----------|--------|")
        for cat, sims in cat_sims.items():
            label = BILINGUAL_CONCEPTS[cat]["label"]
            lines.append(f"| {label} | {np.mean(sims):.4f} | {len(sims)} |")

        # Top/bottom 5
        sorted_paired = sorted(paired, key=lambda x: -x["cosine_sim"])
        lines.append(f"\n### 最相似Top5")
        lines.append("| 英文 | 中文 | 相似度 |")
        lines.append("|------|------|--------|")
        for p in sorted_paired[:5]:
            lines.append(f"| {p['en']} | {p['zh']} | {p['cosine_sim']:.4f} |")

        lines.append(f"\n### 最不相似Bottom5")
        lines.append("| 英文 | 中文 | 相似度 |")
        lines.append("|------|------|--------|")
        for p in sorted_paired[-5:]:
            lines.append(f"| {p['en']} | {p['zh']} | {p['cosine_sim']:.4f} |")

    # 2. 联合SVD
    jsvd = all_results.get("joint_svd", {})
    if jsvd:
        lines.append(f"\n## 2. 中英文联合SVD分解")
        cv = jsvd.get("cumulative_variance", [])
        if cv:
            lines.append(f"\n| K | 累计方差 |")
            lines.append(f"|---|---------|")
            for k in [5, 10, 15]:
                if k <= len(cv):
                    lines.append(f"| {k} | {cv[k-1]*100:.1f}% |")

        mean_corr = jsvd.get("mean_factor_correlation", 0)
        lines.append(f"\n- 因子EN-ZH平均相关: **{mean_corr:.4f}**")
        lines.append(f"- 解释: 若接近1.0，说明中英文共享完全相同的因子结构")

        fc = jsvd.get("factor_correlations", [])
        if fc:
            lines.append(f"\n| 因子 | EN-ZH相关 | 方差 |")
            lines.append(f"|------|-----------|------|")
            evr = jsvd.get("explained_variance_ratio", [])
            for i, corr in enumerate(fc):
                var = evr[i] * 100 if i < len(evr) else 0
                lines.append(f"| Factor {i} | {corr:.4f} | {var:.1f}% |")

    # 3. 跨语言算术
    arith = all_results.get("cross_lingual_arithmetic", [])
    if arith:
        lines.append(f"\n## 3. 跨语言概念算术")
        en_to_zh = [r for r in arith if r["shift_lang"] == "en"]
        zh_to_en = [r for r in arith if r["shift_lang"] == "zh"]

        lines.append(f"\n| 方向 | 对 | 相似度 |")
        lines.append(f"|------|-----|--------|")
        for r in en_to_zh:
            lines.append(f"| EN→ZH | {r['source']}→{r['target']} | {r['cosine_sim']:.4f} |")
        for r in zh_to_en:
            lines.append(f"| ZH→EN | {r['source']}→{r['target']} | {r['cosine_sim']:.4f} |")

        if en_to_zh:
            lines.append(f"\n- EN→ZH平均: {np.mean([r['cosine_sim'] for r in en_to_zh]):.4f}")
        if zh_to_en:
            lines.append(f"- ZH→EN平均: {np.mean([r['cosine_sim'] for r in zh_to_en]):.4f}")

    # 4. 结论
    lines.append(f"\n## 4. 结论")
    lines.append("")

    report_path = output_dir / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report: {report_path}")
    return report_path


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("Stage458: 中英文双语概念编码一致性验证")
    print(f"Concept pairs: {TOTAL_PAIRS}, Categories: {len(BILINGUAL_CONCEPTS)}")
    print("=" * 60)

    t0 = time.time()

    # 1. 加载模型
    print("\n[1/5] Loading DeepSeek-7B...")
    model, tokenizer, layer_count, hidden_dim = load_model(DEEPSEEK7B_MODEL_PATH)

    # 2. 提取双语激活
    print(f"\n[2/5] Extracting bilingual activations ({TOTAL_PAIRS * 2} words)...")
    all_data = extract_all_bilingual(model, tokenizer, BILINGUAL_CONCEPTS, layer_count)

    # 释放模型
    print("\n  Releasing model...")
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. 构建偏置矩阵
    target_layers = list(range(layer_count - 5, layer_count))
    print(f"\n[3/5] Building bilingual bias matrices (layers: {target_layers})...")
    bias_data = build_bilingual_bias_matrices(all_data, BILINGUAL_CONCEPTS, target_layers)

    en_b = bias_data["en"]["biases"]
    zh_b = bias_data["zh"]["biases"]
    print(f"  EN bias matrix: {en_b.shape}")
    print(f"  ZH bias matrix: {zh_b.shape}")

    if len(en_b) < 5 or len(zh_b) < 5:
        print("ERROR: Not enough concepts extracted")
        return

    # 4. 分析
    print(f"\n[4/5] Analysis...")

    # 4a. 中英偏置相似度
    cross_sim = cross_lingual_similarity(
        en_b, zh_b,
        bias_data["en"]["labels"], bias_data["zh"]["labels"]
    )

    # 4b. 联合SVD
    n_comp = min(15, en_b.shape[0] - 1, en_b.shape[1])
    jsvd_results, jsvd_components, jlabels, jlang, jcats = joint_svd_analysis(
        en_b, zh_b,
        bias_data["en"]["labels"], bias_data["zh"]["labels"],
        bias_data["en"]["categories"], bias_data["zh"]["categories"],
        n_components=n_comp,
    )

    # 4c. 跨语言算术
    arith_results = cross_lingual_arithmetic(
        en_b, zh_b,
        bias_data["en"]["labels"], bias_data["zh"]["labels"]
    )

    # 5. 保存
    all_results = {
        "cross_lingual_sim": cross_sim,
        "joint_svd": jsvd_results,
        "cross_lingual_arithmetic": arith_results,
        "meta": {
            "n_en_concepts": len(bias_data["en"]["labels"]),
            "n_zh_concepts": len(bias_data["zh"]["labels"]),
            "target_layers": target_layers,
            "elapsed": time.time() - t0,
        },
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(all_results), f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {summary_path}")

    # 报告
    generate_report(all_results, OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Stage458 COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
