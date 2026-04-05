#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage500: 共享子空间分析实验
目标：验证概念层次是否通过"共享子空间+独立子空间"编码（而非几何距离）
设计原理：
  - Stage498发现层次距离不保持单调性
  - 假设：同一父类的子概念共享一个公共子空间，但各自有独立子空间
  - 如果正确：子概念之间的共享子空间相似度 > 跨分支概念的共享子空间相似度
方法：
  1. 对每对概念做PCA，提取公共子空间和独有子空间
  2. 计算公共子空间的重叠度（子空间夹角/典型相关）
  3. 比较：同分支vs跨分支的公共子空间重叠度
  4. 验证：层次关系是否由"共享子空间大小"决定
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.linalg import subspace_angles
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


# 概念家族（与Stage498相同的层次结构）
CONCEPT_FAMILIES = {
    "animal": {
        "name": "动物界",
        "tree": {
            "动物": ["猫", "狗", "牛", "马"],
            "猫": ["小猫", "花猫", "黑猫", "白猫"],
            "狗": ["小狗", "黄狗", "黑狗", "白狗"],
        },
    },
    "food": {
        "name": "食物分类",
        "tree": {
            "食物": ["水果", "蔬菜", "肉类"],
            "水果": ["苹果", "香蕉", "橙子", "葡萄"],
            "蔬菜": ["白菜", "萝卜", "番茄", "黄瓜"],
            "肉类": ["猪肉", "牛肉", "鸡肉", "鱼肉"],
        },
    },
    "color": {
        "name": "颜色谱",
        "tree": {
            "暖色": ["红", "橙", "黄"],
            "冷色": ["蓝", "绿", "紫"],
        },
    },
    "spatial": {
        "name": "空间关系",
        "tree": {
            "方向": ["上", "下", "左", "右"],
            "位置": ["里", "外", "中", "旁"],
        },
    },
    "abstract": {
        "name": "抽象概念",
        "tree": {
            "情感": ["开心", "难过", "生气", "害怕"],
            "认知": ["知道", "认为", "感觉", "理解"],
        },
    },
}


def get_embedding(model, tokenizer, word: str) -> np.ndarray:
    """获取词的embedding"""
    device = next(model.parameters()).device
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if not token_ids:
        return None
    with torch.inference_mode():
        emb = model.model.embed_tokens(torch.tensor([token_ids]).to(device))
    return emb[0, 0].detach().float().cpu().numpy()


def principal_angles_between(v1: np.ndarray, v2: np.ndarray, k: int = 10) -> float:
    """
    计算两个向量张成子空间之间的主角度
    返回：平均主角度的余弦值（越大表示子空间越相似）
    """
    # 将向量reshape为矩阵并做SVD
    U1, _, _ = np.linalg.svd(v1.reshape(1, -1) if v1.ndim == 1 else v1, full_matrices=False)
    U2, _, _ = np.linalg.svd(v2.reshape(1, -1) if v2.ndim == 1 else v2, full_matrices=False)

    # 取top-k主方向
    k = min(k, U1.shape[1], U2.shape[1])
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]

    # 计算子空间角度
    try:
        angles = subspace_angles(U1_k, U2_k)
        cos_angles = np.cos(angles)
        return float(np.mean(cos_angles))
    except Exception:
        return 0.0


def shared_subspace_overlap(v1: np.ndarray, v2: np.ndarray, k: int = 20) -> Dict:
    """
    分析两个概念embedding的共享子空间
    1. 拼接两个向量，做PCA
    2. 取top-k主成分作为"公共子空间基底"
    3. 分别投影两个向量到公共子空间
    4. 计算投影后余弦相似度
    """
    if v1 is None or v2 is None:
        return {"error": "null embedding"}

    dim = v1.shape[0]
    k = min(k, dim)

    # 构建联合矩阵并PCA
    joint = np.stack([v1, v2], axis=0)  # 2 x dim
    # 对单个向量做SVD
    U1, S1, _ = np.linalg.svd(v1.reshape(1, -1), full_matrices=False)
    U2, S2, _ = np.linalg.svd(v2.reshape(1, -1), full_matrices=False)

    # 取每个向量的top-k方向
    dirs1 = U1[0, :k]  # shape: (k,)
    dirs2 = U2[0, :k]  # shape: (k,)

    # 方向1和方向2的余弦相似度矩阵
    # 这里简单处理：两个1d向量的余弦
    cos_between = float(np.abs(np.dot(dirs1, dirs2) / (np.linalg.norm(dirs1) * np.linalg.norm(dirs2) + 1e-9)))

    # 更好的方法：用典型的典型相关分析(CCA)思路
    # 构建协方差矩阵
    C = np.outer(v1, v2)  # dim x dim
    U_c, S_c, V_c = np.linalg.svd(C, full_matrices=False)
    canonical_correlations = S_c[:min(k, len(S_c))]

    return {
        "full_cosine": float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)),
        "top_k_canonical_correlations": [round(float(x), 6) for x in canonical_correlations[:k]],
        "mean_canonical_correlation": round(float(np.mean(canonical_correlations[:k])), 6),
        "max_canonical_correlation": round(float(np.max(canonical_correlations[:k])), 6),
        "energy_in_top_k_1": round(float(np.sum(S1[:k]**2) / max(np.sum(S1**2), 1e-9)), 4),
        "energy_in_top_k_2": round(float(np.sum(S2[:k]**2) / max(np.sum(S2**2), 1e-9)), 4),
    }


def analyze_family(model, tokenizer, family_id: str, family_data: Dict) -> Dict:
    """分析一个概念家族的共享子空间结构"""
    tree = family_data["tree"]

    # 提取所有概念embedding
    embeddings = {}
    for parent, children in tree.items():
        if parent not in embeddings:
            embeddings[parent] = get_embedding(model, tokenizer, parent)
        for child in children:
            if child not in embeddings:
                embeddings[child] = get_embedding(model, tokenizer, child)

    # 同分支vs跨分支的共享子空间分析
    same_branch_correlations = []
    cross_branch_correlations = []
    parent_child_correlations = []

    all_pairs_same = []
    all_pairs_cross = []
    all_pairs_parent_child = []

    parents = list(tree.keys())
    for i, parent1 in enumerate(parents):
        children1 = [c for c in tree[parent1] if embeddings.get(c) is not None]
        p1_emb = embeddings.get(parent1)
        if p1_emb is None:
            continue

        # 父-子相关
        for child in children1:
            c_emb = embeddings.get(child)
            if c_emb is not None:
                overlap = shared_subspace_overlap(p1_emb, c_emb, k=20)
                parent_child_correlations.append(overlap["mean_canonical_correlation"])
                all_pairs_parent_child.append({
                    "pair": f"{parent1}-{child}",
                    "type": "parent_child",
                    **overlap,
                })

        # 同分支：同一父类的子节点之间
        for j in range(len(children1)):
            for k in range(j + 1, len(children1)):
                c1_emb = embeddings[children1[j]]
                c2_emb = embeddings[children1[k]]
                overlap = shared_subspace_overlap(c1_emb, c2_emb, k=20)
                same_branch_correlations.append(overlap["mean_canonical_correlation"])
                all_pairs_same.append({
                    "pair": f"{children1[j]}-{children1[k]}",
                    "type": "same_branch",
                    **overlap,
                })

        # 跨分支
        for j in range(i + 1, len(parents)):
            parent2 = parents[j]
            children2 = [c for c in tree[parent2] if embeddings.get(c) is not None]
            p2_emb = embeddings.get(parent2)
            for c1 in children1:
                for c2 in children2:
                    c2_emb = embeddings.get(c2)
                    if c2_emb is not None:
                        overlap = shared_subspace_overlap(embeddings[c1], c2_emb, k=20)
                        cross_branch_correlations.append(overlap["mean_canonical_correlation"])
                        all_pairs_cross.append({
                            "pair": f"{c1}-{c2}",
                            "type": "cross_branch",
                            **overlap,
                        })

    def stats(vals):
        if not vals:
            return {"mean": 0, "std": 0, "count": 0}
        return {"mean": round(float(np.mean(vals)), 6), "std": round(float(np.std(vals)), 6), "count": len(vals)}

    # 关键检验：same_branch > cross_branch?
    hierarchy_holds = False
    if same_branch_correlations and cross_branch_correlations:
        hierarchy_holds = np.mean(same_branch_correlations) > np.mean(cross_branch_correlations)

    return {
        "family_id": family_id,
        "family_name": family_data["name"],
        "concepts_found": len(embeddings),
        "parent_child": stats(parent_child_correlations),
        "same_branch": stats(same_branch_correlations),
        "cross_branch": stats(cross_branch_correlations),
        "hierarchy_holds_by_shared_subspace": hierarchy_holds,
        "samples": {
            "parent_child": all_pairs_parent_child[:3],
            "same_branch": all_pairs_same[:3],
            "cross_branch": all_pairs_cross[:3],
        },
    }


def run_experiment(model_name: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"Stage500: 共享子空间分析实验 — {model_name}")
    print(f"{'='*60}\n")

    if model_name == "qwen3":
        model, tokenizer = load_qwen3_model(prefer_cuda=True)
    else:
        model, tokenizer = load_deepseek_model(prefer_cuda=True)

    layers = discover_layers(model)
    print(f"层数: {len(layers)}")

    all_results = {}
    for fid, fdata in CONCEPT_FAMILIES.items():
        print(f"\n--- {fdata['name']} ---")
        result = analyze_family(model, tokenizer, fid, fdata)
        all_results[fid] = result
        print(f"  概念数: {result['concepts_found']}")
        print(f"  父子典型相关: {result['parent_child']}")
        print(f"  同分支典型相关: {result['same_branch']}")
        print(f"  跨分支典型相关: {result['cross_branch']}")
        print(f"  层次保持: {result['hierarchy_holds_by_shared_subspace']}")

    # 全局统计
    holds_count = sum(1 for r in all_results.values() if r["hierarchy_holds_by_shared_subspace"])

    summary = {
        "stage": "stage500_shared_subspace_analysis",
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_layers": len(layers),
        "results": all_results,
        "aggregate": {
            "hierarchy_holds_count": holds_count,
            "total_families": len(CONCEPT_FAMILIES),
            "holds_ratio": holds_count / len(CONCEPT_FAMILIES),
        },
        "core_answer": (
            f"在{model_name}上，{len(CONCEPT_FAMILIES)}个概念家族中，"
            f"{holds_count}个通过共享子空间典型相关保持了层次结构（占比{holds_count/len(CONCEPT_FAMILIES):.0%}）。"
            + ("共享子空间分析比几何距离更好地捕获了概念层次。"
               if holds_count > len(CONCEPT_FAMILIES) / 2
               else "共享子空间分析也未完全捕获层次结构，编码机制更复杂。")
        ),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage500: 共享子空间分析实验")
    parser.add_argument("model", choices=["qwen3", "deepseek"], help="模型选择")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "tests" / "codex_temp" / f"stage500_shared_subspace_{time.strftime('%Y%m%d')}"
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
