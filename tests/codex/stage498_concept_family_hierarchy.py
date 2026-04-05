#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage498: 概念家族层次结构实验
目标：分析概念间的层次关系（is-a, has-a, part-of）如何在网络中编码
方法：
  1. 构造多个层次概念树（动物界、颜色谱、空间关系）
  2. 在每个节点提取embedding和各层激活
  3. 分析：父子概念的距离模式、兄弟概念的距离模式、跨分支概念的距离模式
  4. 验证：层次距离是否在网络空间中保持单调性
核心问题：概念层次是"欧氏嵌入"还是"非欧结构"？
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial.distance import cosine, pdist, squareform
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "codex"))
from qwen3_language_shared import (
    discover_layers,
    load_qwen3_model,
    load_qwen3_tokenizer,
    load_qwen3_embedding_weight,
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
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
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


# ============================================================
# 概念层次树
# ============================================================

CONCEPT_HIERARCHIES = {
    "animal": {
        "name": "动物界",
        "tree": {
            "动物": ["哺乳动物", "鸟类", "鱼类", "昆虫"],
            "哺乳动物": ["猫", "狗", "牛", "马", "猪"],
            "鸟类": ["鸡", "鸭", "鹰", "麻雀"],
            "鱼类": ["鲨鱼", "金鱼", "鲤鱼", "三文鱼"],
            "昆虫": ["蚂蚁", "蜜蜂", "蝴蝶", "蜻蜓"],
        },
    },
    "color": {
        "name": "颜色谱",
        "tree": {
            "颜色": ["红色", "橙色", "黄色", "绿色", "蓝色", "紫色"],
            "红色": ["深红", "浅红", "粉红", "暗红"],
            "蓝色": ["深蓝", "浅蓝", "天蓝", "靛蓝"],
            "绿色": ["深绿", "浅绿", "草绿", "墨绿"],
        },
    },
    "spatial": {
        "name": "空间关系",
        "tree": {
            "空间": ["方向", "位置", "距离"],
            "方向": ["上", "下", "左", "右", "前", "后"],
            "位置": ["里", "外", "中", "旁"],
            "距离": ["远", "近", "高", "低"],
        },
    },
    "time": {
        "name": "时间层次",
        "tree": {
            "时间": ["过去", "现在", "未来"],
            "过去": ["昨天", "前天", "上周", "去年"],
            "现在": ["今天", "此刻", "现在", "刚才"],
            "未来": ["明天", "后天", "下周", "明年"],
        },
    },
    "food": {
        "name": "食物分类",
        "tree": {
            "食物": ["水果", "蔬菜", "肉类", "主食"],
            "水果": ["苹果", "香蕉", "橙子", "葡萄"],
            "蔬菜": ["白菜", "萝卜", "番茄", "黄瓜"],
            "肉类": ["猪肉", "牛肉", "鸡肉", "鱼肉"],
            "主食": ["米饭", "面条", "馒头", "面包"],
        },
    },
}


def get_embedding(tokenizer, word: str, model=None) -> np.ndarray:
    """获取词的embedding向量"""
    device = next(model.parameters()).device
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if not token_ids:
        return np.zeros(2560)
    with torch.inference_mode():
        emb = model.model.embed_tokens(torch.tensor([token_ids]).to(device))
    return emb[0, 0].detach().float().cpu().numpy()


def get_layer_activation(model, tokenizer, word: str, layer_idx: int) -> np.ndarray:
    """获取词在指定层的激活向量"""
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
        return np.zeros(2560)
    # 取最后一个token的激活
    return captured[0][0, -1, :].numpy()


def compute_hierarchy_distances(embeddings: Dict[str, np.ndarray], tree: Dict[str, List[str]]) -> Dict:
    """
    计算层次结构中的距离模式
    返回：父子距离、兄弟距离、跨分支距离的统计
    """
    # 收集所有概念
    all_concepts = set()
    parent_map = {}  # child -> parent
    for parent, children in tree.items():
        all_concepts.add(parent)
        for child in children:
            all_concepts.add(child)
            parent_map[child] = parent

    # 只保留有embedding的概念
    available = {c: embeddings[c] for c in all_concepts if c in embeddings}

    if len(available) < 3:
        return {"error": "概念太少", "available_count": len(available)}

    # 计算各类距离
    parent_child_dists = []
    sibling_dists = []
    cross_branch_dists = []

    for parent, children in tree.items():
        if parent not in available:
            continue
        parent_emb = available[parent]

        # 父子距离
        for child in children:
            if child in available:
                dist = cosine(parent_emb, available[child])
                parent_child_dists.append(dist)

        # 兄弟距离
        available_children = [c for c in children if c in available]
        for i in range(len(available_children)):
            for j in range(i + 1, len(available_children)):
                dist = cosine(available[available_children[i]], available[available_children[j]])
                sibling_dists.append(dist)

    # 跨分支距离：不同父节点的子节点之间的距离
    all_children_by_parent = {}
    for parent, children in tree.items():
        avail = [c for c in children if c in available]
        if avail:
            all_children_by_parent[parent] = avail

    parents_list = list(all_children_by_parent.keys())
    for i in range(len(parents_list)):
        for j in range(i + 1, len(parents_list)):
            for c1 in all_children_by_parent[parents_list[i]]:
                for c2 in all_children_by_parent[parents_list[j]]:
                    dist = cosine(available[c1], available[c2])
                    cross_branch_dists.append(dist)

    def stats(dists):
        if not dists:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
        return {
            "mean": round(np.mean(dists), 6),
            "std": round(np.std(dists), 6),
            "min": round(np.min(dists), 6),
            "max": round(np.max(dists), 6),
            "count": len(dists),
        }

    # 关键检验：层次单调性
    # 父子距离 < 兄弟距离 < 跨分支距离 ？
    monotonic = False
    if parent_child_dists and sibling_dists and cross_branch_dists:
        monotonic = (
            np.mean(parent_child_dists) < np.mean(sibling_dists) < np.mean(cross_branch_dists)
        )

    return {
        "parent_child": stats(parent_child_dists),
        "sibling": stats(sibling_dists),
        "cross_branch": stats(cross_branch_dists),
        "monotonicity_holds": monotonic,
        "available_concepts": len(available),
        "total_concepts": len(all_concepts),
    }


def analyze_geometry(embeddings: Dict[str, np.ndarray]) -> Dict:
    """分析嵌入空间的几何性质"""
    embs = np.array(list(embeddings.values()))
    concepts = list(embeddings.keys())

    # 全局距离矩阵
    dist_matrix = squareform(pdist(embs, metric='cosine'))

    # SVD分析
    U, S, Vt = np.linalg.svd(embs - embs.mean(axis=0), full_matrices=False)

    # 曲率估计：用最近邻距离 vs 全局距离比
    n = len(concepts)
    nn_dists = []
    for i in range(n):
        sorted_dists = sorted([dist_matrix[i][j] for j in range(n) if i != j])
        nn_dists.append(sorted_dists[0])

    avg_nn_dist = np.mean(nn_dists)
    avg_global_dist = np.mean(dist_matrix[np.triu_indices(n, k=1)])

    # 维度集中度：前k维占总方差的比例
    total_var = np.sum(S ** 2)
    dim_concentration = {
        f"top{k}_var_ratio": round(np.sum(S[:k] ** 2) / total_var, 4)
        for k in [1, 2, 3, 5, 10, 17, 20]
    }

    return {
        "concept_count": n,
        "embedding_dim": embs.shape[1],
        "avg_nearest_neighbor_dist": round(avg_nn_dist, 6),
        "avg_global_dist": round(avg_global_dist, 6),
        "local_to_global_ratio": round(avg_nn_dist / max(avg_global_dist, 1e-9), 4),
        "top_singular_values": [round(float(s), 2) for s in S[:10]],
        "dimension_concentration": dim_concentration,
    }


def run_experiment(model_name: str) -> Dict:
    """运行完整实验"""
    print(f"\n{'='*60}")
    print(f"Stage498: 概念家族层次结构实验 — {model_name}")
    print(f"{'='*60}\n")

    if model_name == "qwen3":
        model, tokenizer = load_qwen3_model(prefer_cuda=True)
    else:
        model, tokenizer = load_deepseek_model(prefer_cuda=True)

    layers = discover_layers(model)
    total_layers = len(layers)
    print(f"层数: {total_layers}")

    # 选取代表性层
    sample_layers = sorted(set([
        0,
        total_layers // 4,
        total_layers // 2,
        total_layers * 3 // 4,
        total_layers - 1,
    ]))

    all_results = {}

    for hier_id, hier_data in CONCEPT_HIERARCHIES.items():
        print(f"\n--- {hier_data['name']} ---")
        tree = hier_data["tree"]

        # 提取所有概念的embedding和层激活
        all_concepts = set()
        for parent, children in tree.items():
            all_concepts.add(parent)
            all_concepts.update(children)

        # Embedding层
        embeddings_emb = {}
        for concept in tqdm(all_concepts, desc=f"提取{hier_data['name']}embedding"):
            try:
                emb = get_embedding(tokenizer, concept, model)
                if np.any(emb != 0):
                    embeddings_emb[concept] = emb
            except Exception:
                pass

        # 层激活
        layer_embeddings = {}
        for layer_idx in sample_layers:
            layer_embs = {}
            for concept in tqdm(all_concepts, desc=f"L{layer_idx}"):
                try:
                    act = get_layer_activation(model, tokenizer, concept, layer_idx)
                    if np.any(act != 0):
                        layer_embs[concept] = act
                except Exception:
                    pass
            layer_embeddings[layer_idx] = layer_embs

        # 分析层次距离模式
        emb_hierarchy = compute_hierarchy_distances(embeddings_emb, tree)
        layer_hierarchies = {}
        for layer_idx, layer_embs in layer_embeddings.items():
            layer_hierarchies[layer_idx] = compute_hierarchy_distances(layer_embs, tree)

        # 分析几何性质
        emb_geometry = analyze_geometry(embeddings_emb) if len(embeddings_emb) >= 3 else {"error": "概念不足"}

        # 层次单调性总结
        monotonic_layers = {
            "embedding": emb_hierarchy.get("monotonicity_holds", False),
        }
        for li, lh in layer_hierarchies.items():
            monotonic_layers[f"layer_{li}"] = lh.get("monotonicity_holds", False)

        monotonic_count = sum(1 for v in monotonic_layers.values() if v)
        total_checks = len(monotonic_layers)

        result = {
            "hierarchy_id": hier_id,
            "hierarchy_name": hier_data["name"],
            "total_concepts": len(all_concepts),
            "embedding_hierarchy": emb_hierarchy,
            "layer_hierarchies": layer_hierarchies,
            "embedding_geometry": emb_geometry,
            "monotonicity_by_layer": monotonic_layers,
            "monotonicity_rate": round(monotonic_count / total_checks, 4) if total_checks > 0 else 0,
        }
        all_results[hier_id] = result

        print(f"  概念数: {len(all_concepts)}, 有效embedding: {len(embeddings_emb)}")
        print(f"  Embedding层单调性: {emb_hierarchy.get('monotonicity_holds', 'N/A')}")
        print(f"  层单调性: {monotonic_layers}")
        print(f"  单调性率: {result['monotonicity_rate']:.2%}")

    # 全局汇总
    all_monotonic_rates = [r["monotonicity_rate"] for r in all_results.values()]
    avg_monotonic = np.mean(all_monotonic_rates) if all_monotonic_rates else 0

    summary = {
        "stage": "stage498_concept_family_hierarchy",
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_layers": total_layers,
        "layers_analyzed": sample_layers,
        "hierarchies_tested": len(CONCEPT_HIERARCHIES),
        "results": all_results,
        "aggregate": {
            "avg_monotonicity_rate": round(avg_monotonic, 4),
            "hierarchies_with_monotonicity": sum(1 for r in all_results.values() if r["monotonicity_rate"] > 0.5),
            "hierarchies_without_monotonicity": sum(1 for r in all_results.values() if r["monotonicity_rate"] <= 0.5),
        },
        "core_answer": (
            f"在{model_name}上，{len(CONCEPT_HIERARCHIES)}个概念层次树中，"
            f"平均层次单调性率为{avg_monotonic:.2%}。"
            f"{'支持概念层次在嵌入空间中保持单调距离结构' if avg_monotonic > 0.5 else '概念层次在嵌入空间中不完全保持单调距离结构，可能需要非欧几何解释'}。"
        ),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage498: 概念家族层次结构实验")
    parser.add_argument("model", choices=["qwen3", "deepseek"], help="模型选择")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "tests" / "codex_temp" / f"stage498_hierarchy_{time.strftime('%Y%m%d')}"
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
