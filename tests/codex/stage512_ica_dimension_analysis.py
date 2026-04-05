#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage512: Embedding矩阵ICA维度方向分析
核心问题：embedding矩阵的向量空间中，哪些正交方向对应哪些语义特征？

方法：
- 提取模型的embedding矩阵 [vocab_size, hidden_dim]
- 用PCA降维后做ICA（独立成分分析），找到统计独立的方向
- 对每个ICA成分，找到在该方向上投影最大的Top-K词
- 分析这些词是否聚集在某个语义类别（食物/动物/工具/抽象...）
- 对比4个模型的ICA成分——同一语义方向是否跨模型一致

分析维度：
- I1: ICA成分的Top-K词列表（每个成分代表一个"语义方向"）
- I2: 语义方向的语言对齐性——同一方向在中英文词中的投影
- I3: 语义方向的层级结构——食物方向内部的子结构
- I4: 跨模型方向一致性——4个模型的对应成分是否编码相同语义
- I5: ICA成分的解释方差比例——前K个成分能解释多少embedding变异
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
from safetensors import safe_open

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex.qwen3_language_shared import (
    load_qwen3_model, load_glm4_model, load_gemma4_model, load_deepseek7b_model,
    discover_layers, get_model_device,
)

LOADERS = {
    "qwen3": load_qwen3_model,
    "glm4": load_glm4_model,
    "gemma4": load_gemma4_model,
    "deepseek7b": load_deepseek7b_model,
}

MODEL_PATHS = {
    "qwen3": Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "glm4": Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
    "deepseek7b": Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"),
}

# 语义探测词
FRUIT_WORDS = ["apple", "苹果", "banana", "香蕉", "orange", "橙子", "grape", "葡萄", "mango", "芒果"]
ANIMAL_WORDS = ["cat", "猫", "dog", "狗", "tiger", "老虎", "lion", "狮子", "elephant", "大象"]
TOOL_WORDS = ["hammer", "锤子", "computer", "电脑", "phone", "手机", "car", "汽车", "knife", "刀"]
ABSTRACT_WORDS = ["love", "爱", "democracy", "民主", "freedom", "自由", "justice", "正义", "time", "时间"]
PLACE_WORDS = ["Beijing", "北京", "Paris", "巴黎", "Tokyo", "东京", "London", "伦敦", "mountain", "山"]

ALL_SEMANTIC_GROUPS = {
    "fruit": FRUIT_WORDS,
    "animal": ANIMAL_WORDS,
    "tool": TOOL_WORDS,
    "abstract": ABSTRACT_WORDS,
    "place": PLACE_WORDS,
}


def extract_embedding_matrix(model, tokenizer, model_name: str) -> Tuple[np.ndarray, List[str]]:
    """提取embedding矩阵 [vocab_size, hidden_dim]"""
    # 尝试从模型直接获取
    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
        weight = embed_layer.weight.detach().cpu().float().numpy()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        weight = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        weight = model.transformer.wte.weight.detach().cpu().float().numpy()
    else:
        raise RuntimeError(f"无法提取 {model_name} 的embedding矩阵")

    # 获取词表
    vocab = []
    if hasattr(tokenizer, "get_vocab"):
        vocab = list(tokenizer.get_vocab().keys())
    elif hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "get_vocab"):
        vocab = list(tokenizer.tokenizer.get_vocab().keys())
    else:
        vocab = list(range(weight.shape[0]))

    return weight, vocab


def find_token_id(tokenizer, word: str) -> int:
    """找到词的token id"""
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        return tokenizer.convert_tokens_to_ids(word)
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else -1


def compute_pca(data: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA降维，返回(变换后数据, 主成分, 解释方差比)"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, data.shape[1]))
    transformed = pca.fit_transform(data)
    return transformed, pca.components_, pca.explained_variance_ratio_


def compute_fast_ica(data: np.ndarray, n_components: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """FastICA独立成分分析，返回(变换后数据, 分离矩阵)"""
    try:
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=min(n_components, data.shape[1]), max_iter=500, random_state=42)
        transformed = ica.fit_transform(data)
        return transformed, ica.components_
    except ImportError:
        print("  警告: sklearn未安装，使用简化PCA替代ICA")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, data.shape[1]))
        transformed = pca.fit_transform(data)
        return transformed, pca.components_


def analyze_semantic_directions(ica_components: np.ndarray, vocab: List[str],
                                 tokenizer, n_components: int = 20) -> dict:
    """
    对每个ICA成分，找到投影最大的Top-K词，并分析其语义聚集性
    """
    results = {}
    # ica_components: [n_components, hidden_dim]
    # 需要embedding矩阵来做投影
    return results


def analyze_i1(ica_components: np.ndarray, pca_data: np.ndarray,
               tokenizer, n_top: int = 20) -> dict:
    """I1: 每个ICA成分的Top-K投影词
    pca_data: [vocab_size, pca_dim] — 已经PCA降维的embedding
    ica_components: [n_ica, pca_dim]
    """
    results = {}
    n_comp = ica_components.shape[0]

    for i in range(min(n_comp, 20)):
        direction = ica_components[i]  # [pca_dim]
        # 投影到PCA降维后的embedding矩阵
        projections = pca_data @ direction  # [vocab_size]
        # 找Top-K
        topk_vals, topk_ids = torch.topk(torch.from_numpy(projections), n_top)

        # 获取词
        top_words = []
        for idx in topk_ids.tolist():
            if hasattr(tokenizer, "decode"):
                word = tokenizer.decode([idx]).strip()
            elif hasattr(tokenizer, "tokenizer"):
                word = tokenizer.tokenizer.decode([idx]).strip()
            else:
                word = str(idx)
            top_words.append(word)

        results[f"component_{i}"] = {
            "top_words": top_words,
            "top_values": [round(v, 4) for v in topk_vals.tolist()],
        }

    return results


def analyze_i2(ica_components: np.ndarray, pca_data: np.ndarray,
               token_ids: dict, model_name: str) -> dict:
    """I2: 语义方向的语言对齐性
    pca_data: [vocab_size, pca_dim]
    token_ids: {word: token_id}
    """
    results = {}
    for group_name, words in ALL_SEMANTIC_GROUPS.items():
        group_projections = {}
        for word in words:
            tid = token_ids.get(word, -1)
            if tid < 0 or tid >= pca_data.shape[0]:
                continue
            word_emb = pca_data[tid]
            # 在各ICA成分上的投影
            proj = ica_components @ word_emb  # [n_comp]
            group_projections[word] = proj.tolist()

        results[group_name] = group_projections

    return results


def analyze_i3(ica_components: np.ndarray, pca_data: np.ndarray,
               token_ids: dict) -> dict:
    """I3: 概念层级结构——食物方向内部的子结构
    pca_data: [vocab_size, pca_dim]
    token_ids: {word: token_id}
    """
    results = {}
    n_comp = min(ica_components.shape[0], 20)

    # 计算水果类词在ICA空间中的两两距离
    fruit_embs = {}
    for word in FRUIT_WORDS:
        tid = token_ids.get(word, -1)
        if tid < 0 or tid >= pca_data.shape[0]:
            continue
        word_emb = pca_data[tid]
        fruit_embs[word] = (ica_components @ word_emb).tolist()

    fruit_words = list(fruit_embs.keys())
    pairwise_dist = {}
    for i in range(len(fruit_words)):
        for j in range(i+1, len(fruit_words)):
            v1 = np.array(fruit_embs[fruit_words[i]])
            v2 = np.array(fruit_embs[fruit_words[j]])
            dist = np.sqrt(np.sum((v1 - v2)**2))
            pairwise_dist[f"{fruit_words[i]}_{fruit_words[j]}"] = round(float(dist), 6)

    # 同样计算动物类
    animal_embs = {}
    for word in ANIMAL_WORDS:
        tid = token_ids.get(word, -1)
        if tid < 0 or tid >= pca_data.shape[0]:
            continue
        word_emb = pca_data[tid]
        animal_embs[word] = (ica_components @ word_emb).tolist()

    animal_words = list(animal_embs.keys())
    animal_pairwise = {}
    for i in range(len(animal_words)):
        for j in range(i+1, len(animal_words)):
            v1 = np.array(animal_embs[animal_words[i]])
            v2 = np.array(animal_embs[animal_words[j]])
            dist = np.sqrt(np.sum((v1 - v2)**2))
            animal_pairwise[f"{animal_words[i]}_{animal_words[j]}"] = round(float(dist), 6)

    # 跨类距离
    cross_pairwise = {}
    for fw in fruit_words:
        for aw in animal_words:
            v1 = np.array(fruit_embs[fw])
            v2 = np.array(animal_embs[aw])
            dist = np.sqrt(np.sum((v1 - v2)**2))
            cross_pairwise[f"{fw}_{aw}"] = round(float(dist), 6)

    results = {
        "fruit_intra_distances": pairwise_dist,
        "animal_intra_distances": animal_pairwise,
        "cross_distances_sample": dict(list(cross_pairwise.items())[:10]),
        "fruit_intra_mean": round(np.mean(list(pairwise_dist.values())), 6) if pairwise_dist else 0,
        "animal_intra_mean": round(np.mean(list(animal_pairwise.values())), 6) if animal_pairwise else 0,
        "cross_mean": round(np.mean(list(cross_pairwise.values())), 6) if cross_pairwise else 0,
    }
    return results


def analyze_i5(embedding_matrix: np.ndarray, n_components: int = 50) -> dict:
    """I5: PCA/ICA的解释方差比例"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, embedding_matrix.shape[1]))
    pca.fit(embedding_matrix)

    explained = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained)

    return {
        "top_1_variance": round(float(explained[0]), 6) if len(explained) > 0 else 0,
        "top_5_variance": round(float(cumsum[min(4, len(cumsum)-1)]), 6) if len(cumsum) > 0 else 0,
        "top_10_variance": round(float(cumsum[min(9, len(cumsum)-1)]), 6) if len(cumsum) > 0 else 0,
        "top_20_variance": round(float(cumsum[min(19, len(cumsum)-1)]), 6) if len(cumsum) > 0 else 0,
        "top_50_variance": round(float(cumsum[-1]), 6) if len(cumsum) > 0 else 0,
        "total_dim": int(embedding_matrix.shape[1]),
        "vocab_size": int(embedding_matrix.shape[0]),
    }


# ============================================================
# 主函数
# ============================================================

def main():
    model_name = sys.argv[1].lower() if len(sys.argv) > 1 else "qwen3"
    if model_name not in LOADERS:
        print(f"未知模型: {model_name}, 可选: {list(LOADERS.keys())}")
        sys.exit(1)

    print(f"=== Stage512: ICA维度方向分析 [{model_name}] ===")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"tests/codex_temp/stage512_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"\n[加载模型] {model_name}...")
    t0 = time.time()
    model, tokenizer = LOADERS[model_name]()
    print(f"  加载耗时: {time.time()-t0:.1f}s")

    # 提取embedding矩阵
    print("\n[提取embedding矩阵]...")
    t1 = time.time()
    embedding_matrix, vocab = extract_embedding_matrix(model, tokenizer, model_name)
    print(f"  形状: {embedding_matrix.shape}")
    print(f"  词表大小: {len(vocab)}")
    print(f"  耗时: {time.time()-t1:.1f}s")

    summary = {
        "model": model_name,
        "timestamp": ts,
        "embedding_shape": list(embedding_matrix.shape),
    }

    # ---- I5: 先做方差分析 ----
    print("\n[I5] 方差解释比例...")
    try:
        summary["I5"] = analyze_i5(embedding_matrix, n_components=50)
        print(f"  Top-1: {summary['I5']['top_1_variance']*100:.2f}%")
        print(f"  Top-10: {summary['I5']['top_10_variance']*100:.2f}%")
        print(f"  Top-50: {summary['I5']['top_50_variance']*100:.2f}%")
    except Exception as e:
        summary["I5"] = {"error": str(e)}
        print(f"  错误: {e}")

    # ---- PCA + ICA ----
    print("\n[PCA降维]...")
    try:
        n_ica = 20
        pca_data, pca_components, pca_var = compute_pca(embedding_matrix, n_components=50)
        print(f"  PCA完成, 前10成分解释方差: {sum(pca_var[:10])*100:.2f}%")

        print("\n[ICA独立成分分析]...")
        ica_data, ica_components = compute_fast_ica(pca_data, n_components=n_ica)
        print(f"  ICA完成, 成分数: {ica_components.shape[0]}")
    except Exception as e:
        print(f"  PCA/ICA错误: {e}")
        ica_components = None

    # ---- I1: Top-K词 ----
    if ica_components is not None:
        print("\n[I1] 每个ICA成分的Top-K词...")
        try:
            summary["I1"] = analyze_i1(ica_components, pca_data, tokenizer, n_top=15)
            print(f"  完成: {len(summary['I1'])} 个成分")
        except Exception as e:
            summary["I1"] = {"error": str(e)}
            print(f"  错误: {e}")

    # 构建token_ids映射
    token_ids = {}
    for w in FRUIT_WORDS + ANIMAL_WORDS + TOOL_WORDS + ABSTRACT_WORDS + PLACE_WORDS:
        token_ids[w] = find_token_id(tokenizer, w)

    # ---- I2: 语言对齐 ----
    if ica_components is not None:
        print("\n[I2] 语义方向的语言对齐性...")
        try:
            summary["I2"] = analyze_i2(ica_components, pca_data, token_ids, model_name)
            print(f"  完成: {len(summary['I2'])} 个语义组")
        except Exception as e:
            summary["I2"] = {"error": str(e)}
            print(f"  错误: {e}")

    # ---- I3: 概念层级 ----
    if ica_components is not None:
        print("\n[I3] 概念层级结构...")
        try:
            summary["I3"] = analyze_i3(ica_components, pca_data, token_ids)
            i3 = summary["I3"]
            print(f"  水果内部平均距离: {i3.get('fruit_intra_mean', 'N/A')}")
            print(f"  动物内部平均距离: {i3.get('animal_intra_mean', 'N/A')}")
            print(f"  跨类平均距离: {i3.get('cross_mean', 'N/A')}")
        except Exception as e:
            summary["I3"] = {"error": str(e)}
            print(f"  错误: {e}")

    # 保存结果
    out_path = out_dir / f"summary_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

    # 打印关键发现
    print("\n=== 关键发现 ===")
    if "I1" in summary and not isinstance(summary["I1"].get("error"), str):
        for comp_key in list(summary["I1"].keys())[:3]:
            comp = summary["I1"][comp_key]
            print(f"  {comp_key}: {comp['top_words'][:5]}")


if __name__ == "__main__":
    main()
