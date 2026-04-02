# -*- coding: utf-8 -*-
"""
Stage464-467: 非线性流形分析四合一实验
=================================================================

核心问题：偏置空间存在非线性结构（Stage463已证明），但：
  - 非线性结构具体长什么样？
  - AE的潜在因子是否可解释？
  - 概念在流形上的分布有什么规律？
  - 流形的弯曲程度有多大？

四个实验模块：

  Stage464 — 非线性因子解释
    目标：在AE潜在空间中寻找可解释的语义结构
    方法：训练充分AE(latent=10) → 提取z编码 → 计算z与属性的eta2
    判据：如果AE潜在编码与属性相关性高(eta2>0.5)→非线性因子可解释

  Stage465 — 流形可视化
    目标：可视化偏置空间的真实几何结构
    方法：UMAP/Isomap/t-SNE降维到2D → 按类别/属性着色
    判据：如果概念按类别聚集→空间有有意义的结构

  Stage466 — 非线性概念算术
    目标：测试在AE潜在空间做概念算术是否比SVD空间更有效
    方法：3Cos(A,B,C) in AE-space vs SVD-space vs raw-space
    判据：AE空间算术精度 > SVD空间 → 非线性编码更有利于概念操控

  Stage467 — 流形曲率分析
    目标：量化偏置空间的弯曲程度
    方法：局部PCA分析 + 高斯曲率估计 + 本征维估计
    判据：高斯曲率显著偏离0 → 流形确实弯曲

用法：
  python stage464_nonlinear_manifold_analysis.py qwen3
  python stage464_nonlinear_manifold_analysis.py deepseek
"""

from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage464_nonlinear_manifold_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8

# ==================== 概念集（与Stage462/463一致） ====================
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
        },
    },
}

# 属性-类别映射（用于eta2分析）
CATEGORY_ATTRIBUTES = {
    "fruit": ["color", "size", "taste", "shape"],
    "animal": ["size", "domestic", "speed"],
    "vehicle": ["speed", "medium", "size"],
    "profession": ["domain", "social", "creativity"],
}


def sanitize_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    if isinstance(obj, (bool,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


# ==================== 模型加载 ====================
def load_model(model_path: Path):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    want_cuda = torch.cuda.is_available()
    print(f"  CUDA: {want_cuda}")
    if want_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

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
    return model, tokenizer, layer_count, neuron_dim


# ==================== 激活提取 ====================
def extract_word_per_layer(model, tokenizer, word, layer_count):
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

    print(f"  Extracted: {done}/{total}")
    return all_activations


# ==================== AE模型 ====================
class ShallowAE(nn.Module):
    """浅AE（1H + ReLU），Stage463中表现最好的架构"""
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_ae(bias_scaled, latent_dim=10, hidden_dim=256, epochs=1000, lr=1e-3):
    """在CPU上训练AE（Stage463验证：CPU训练稳定可靠）"""
    device = torch.device("cpu")
    input_dim = bias_scaled.shape[1]

    model = ShallowAE(input_dim, latent_dim, hidden_dim).to(device)
    X = torch.tensor(bias_scaled, dtype=torch.float32).to(device)
    X = torch.nan_to_num(X, nan=0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        x_recon, z = model(X)
        loss = nn.functional.mse_loss(x_recon, X)
        if torch.isnan(loss):
            break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        _, z = model(X)
        z_np = z.cpu().numpy()
        x_recon_np = model(X)[0].cpu().numpy()

    z_np = np.nan_to_num(z_np, nan=0.0)
    x_recon_np = np.nan_to_num(x_recon_np, nan=0.0)
    evr = 1.0 - np.mean((bias_scaled - x_recon_np) ** 2) / max(np.var(bias_scaled), EPS)

    return z_np, evr


# ==================== eta2分析 ====================
def compute_eta_squared(z_codes, attribute_values):
    """计算潜在编码各维度与属性的eta2（关联强度）"""
    n_samples, n_latent = z_codes.shape
    n_groups = len(set(attribute_values))
    if n_groups < 2:
        return np.zeros(n_latent)

    grand_mean = np.mean(z_codes, axis=0)
    ss_between = 0.0
    ss_total = np.sum((z_codes - grand_mean) ** 2)
    if ss_total < EPS:
        return np.zeros(n_latent)

    group_values = np.array(attribute_values)
    eta_per_dim = np.zeros(n_latent)
    for d in range(n_latent):
        group_means = {}
        for g in np.unique(group_values):
            mask = group_values == g
            group_means[g] = np.mean(z_codes[mask, d])
        ss_b = sum(
            np.sum(mask) * (group_means[g] - grand_mean[d]) ** 2
            for g in group_means
            for mask in [group_values == g]
        )
        eta_per_dim[d] = ss_b / ss_total if ss_total > EPS else 0.0

    return eta_per_dim


# ==================== Stage464: 非线性因子解释 ====================
def run_stage464(bias_scaled, word_labels, concepts):
    """在AE潜在空间中寻找可解释的语义结构"""
    print("\n" + "=" * 60)
    print("  Stage464: 非线性因子解释")
    print("=" * 60)

    results = {"best_factors": {}}

    for cat_name, cat_data in concepts.items():
        cat_words = list(cat_data["words"].keys())
        cat_attrs = CATEGORY_ATTRIBUTES.get(cat_name, [])

        # 获取该类别的词和偏置向量
        cat_indices = [i for i, w in enumerate(word_labels) if w in cat_words]
        if len(cat_indices) < 5:
            continue

        cat_bias = bias_scaled[cat_indices]
        cat_labels = [word_labels[i] for i in cat_indices]

        # 训练AE
        z_codes, evr = train_ae(cat_bias, latent_dim=min(10, len(cat_indices) - 1),
                                 hidden_dim=256, epochs=1000)
        print(f"\n  [{cat_name}] AE EVR={evr:.4f}, latent={z_codes.shape[1]}")

        # 对每个属性计算eta2
        attr_results = {}
        for attr in cat_attrs:
            attr_values = []
            valid_indices = []
            for i, w in enumerate(cat_labels):
                if attr in cat_data["words"][w]:
                    attr_values.append(str(cat_data["words"][w][attr]))
                    valid_indices.append(i)

            if len(set(attr_values)) < 2:
                continue

            z_valid = z_codes[valid_indices]
            eta = compute_eta_squared(z_valid, attr_values)
            max_eta = float(np.max(eta))
            mean_eta = float(np.mean(eta))
            best_dim = int(np.argmax(eta))
            best_dim_val = float(eta[best_dim])

            # 与SVD的eta2对比
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=min(10, len(cat_indices) - 1), random_state=42)
            z_svd = svd.fit_transform(cat_bias[valid_indices])
            eta_svd = compute_eta_squared(z_svd, attr_values)
            max_eta_svd = float(np.max(eta_svd))

            attr_results[attr] = {
                "eta_max": round(max_eta, 4),
                "eta_mean": round(mean_eta, 4),
                "best_dim": best_dim,
                "best_dim_eta": round(best_dim_val, 4),
                "svd_eta_max": round(max_eta_svd, 4),
                "ae_advantage": round(max_eta - max_eta_svd, 4),
                "n_groups": len(set(attr_values)),
            }
            print(f"    {attr}: AE eta2_max={max_eta:.4f}, SVD eta2_max={max_eta_svd:.4f}, "
                  f"优势={max_eta - max_eta_svd:+.4f}")

        results[cat_name] = attr_results

    return results


# ==================== Stage465: 流形可视化（保存坐标数据） ====================
def run_stage465(bias_scaled, word_labels, concepts, layer_idx, model_name):
    """用UMAP/Isomap/t-SNE分析偏置空间的流形结构"""
    print("\n" + "=" * 60)
    print(f"  Stage465: 流形可视化 (Layer {layer_idx})")
    print("=" * 60)

    n = bias_scaled.shape[0]
    n_neighbors = min(15, n - 1)
    results = {}

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3,
                             n_components=2, random_state=42, metric='cosine')
        coords_umap = reducer.fit_transform(bias_scaled)
        results["umap"] = coords_umap.tolist()

        # 计算UMAP空间的类别分离度
        cat_coords = defaultdict(list)
        for i, w in enumerate(word_labels):
            for cat in concepts:
                if w in concepts[cat]["words"]:
                    cat_coords[cat].append(coords_umap[i])
                    break

        separation = {}
        cat_names = list(cat_coords.keys())
        for i, c1 in enumerate(cat_names):
            for j, c2 in enumerate(cat_names):
                if i >= j:
                    continue
                c1_arr = np.array(cat_coords[c1])
                c2_arr = np.array(cat_coords[c2])
                c1_center = np.mean(c1_arr, axis=0)
                c2_center = np.mean(c2_arr, axis=0)
                inter_dist = np.linalg.norm(c1_center - c2_center)
                intra_dist = (np.mean(np.linalg.norm(c1_arr - c1_center, axis=1)) +
                              np.mean(np.linalg.norm(c2_arr - c2_center, axis=1))) / 2
                ratio = inter_dist / max(intra_dist, EPS)
                separation[f"{c1}_vs_{c2}"] = {
                    "inter_dist": round(float(inter_dist), 4),
                    "intra_dist": round(float(intra_dist), 4),
                    "ratio": round(float(ratio), 4),
                }
        results["umap_separation"] = separation
        print(f"  UMAP: 类别间距离比 > 1.0 表示类别分离良好")
        for key, val in separation.items():
            print(f"    {key}: ratio={val['ratio']:.2f} ({'好' if val['ratio'] > 1 else '差'})")
    except Exception as e:
        print(f"  UMAP failed: {e}")
        results["umap_error"] = str(e)

    # Isomap
    try:
        from sklearn.manifold import Isomap
        n_neighbors_iso = min(10, n - 1)
        iso = Isomap(n_neighbors=n_neighbors_iso, n_components=2)
        coords_iso = iso.fit_transform(bias_scaled)
        results["isomap"] = coords_iso.tolist()
        print(f"  Isomap: done (n_neighbors={n_neighbors_iso})")
    except Exception as e:
        print(f"  Isomap failed: {e}")
        results["isomap_error"] = str(e)

    # t-SNE
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(30, n - 1),
                     random_state=42, learning_rate='auto', init='pca')
        coords_tsne = tsne.fit_transform(bias_scaled)
        results["tsne"] = coords_tsne.tolist()
        print(f"  t-SNE: done (perplexity=min(30,{n-1}))")
    except Exception as e:
        print(f"  t-SNE failed: {e}")
        results["tsne_error"] = str(e)

    # 保存可视化数据（包含标签信息，便于后续绘制）
    vis_data = {
        "model": model_name,
        "layer": layer_idx,
        "words": word_labels,
        "categories": [],
    }
    for i, w in enumerate(word_labels):
        for cat, cat_data in concepts.items():
            if w in cat_data["words"]:
                vis_data["categories"].append(cat)
                break

    vis_path = OUTPUT_DIR / f"{model_name}_L{layer_idx}_vis_data.json"
    with open(vis_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(vis_data), f, ensure_ascii=False, indent=2)

    # 保存坐标（单独文件）
    coords_path = OUTPUT_DIR / f"{model_name}_L{layer_idx}_coords.json"
    with open(coords_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(results), f, ensure_ascii=False, indent=2)

    return results


# ==================== Stage466: 非线性概念算术 ====================
def run_stage466(bias_scaled, word_labels, concepts):
    """测试在AE潜在空间中做概念算术是否更有效"""
    print("\n" + "=" * 60)
    print("  Stage466: 非线性概念算术 (3Cos)")
    print("=" * 60)

    results = {"per_space": {}}

    # 在三个空间中做算术：原始偏置空间、SVD空间、AE空间
    n = bias_scaled.shape[0]

    # SVD空间
    from sklearn.decomposition import TruncatedSVD
    svd_dim = min(20, n - 1)
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    z_svd = svd.fit_transform(bias_scaled)

    # AE空间
    z_ae, ae_evr = train_ae(bias_scaled, latent_dim=min(20, n - 1),
                              hidden_dim=512, epochs=1000)

    # 在三个空间中测试3Cos
    for space_name, z_space in [("raw", bias_scaled), ("svd", z_svd), ("ae", z_ae)]:
        word_to_idx = {w: i for i, w in enumerate(word_labels)}
        cos_results = []

        for cat_name, cat_data in concepts.items():
            cat_words = list(cat_data["words"].keys())
            cat_attrs = CATEGORY_ATTRIBUTES.get(cat_name, [])
            if not cat_attrs:
                continue

            # 选择数值型属性
            numeric_attrs = [a for a in cat_attrs if a in ["size", "speed", "social", "creativity", "domestic"]
                             and len(set(cat_data["words"][w].get(a, 0) for w in cat_words)) > 1]
            if not numeric_attrs:
                continue

            attr = numeric_attrs[0]
            # 按属性值分组
            values = {w: cat_data["words"][w].get(attr, 0) for w in cat_words}

            # 3Cos测试: 概念C和概念A在属性上不同，找B使得cos(C, A-B)最大
            cat_word_list = [w for w in cat_words if w in word_to_idx]
            if len(cat_word_list) < 5:
                continue

            correct = 0
            total = 0
            for w_c in cat_word_list:
                for w_a in cat_word_list:
                    if w_a == w_c:
                        continue
                    if abs(values[w_c] - values[w_a]) < 0.5:
                        continue

                    c_idx = word_to_idx[w_c]
                    a_idx = word_to_idx[w_a]

                    # 目标：找最接近 c + (c - a) 的概念
                    query = z_space[c_idx] + (z_space[c_idx] - z_space[a_idx])

                    best_score = -2
                    best_word = None
                    for w_b in cat_word_list:
                        if w_b == w_a:
                            continue
                        b_idx = word_to_idx[w_b]
                        cos_sim = np.dot(query, z_space[b_idx]) / max(
                            np.linalg.norm(query) * np.linalg.norm(z_space[b_idx]), EPS
                        )
                        if cos_sim > best_score:
                            best_score = cos_sim
                            best_word = w_b

                    # 判断是否预测正确：best_word应该和w_c有相同的属性值
                    if best_word and values.get(best_word, -1) == values[w_c]:
                        correct += 1
                    total += 1

            accuracy = correct / max(total, 1)
            cos_results.append({
                "category": cat_name,
                "attribute": attr,
                "accuracy": round(accuracy, 4),
                "n_tests": total,
            })
            print(f"    [{space_name}] {cat_name}/{attr}: acc={accuracy:.4f} ({correct}/{total})")

        avg_acc = np.mean([r["accuracy"] for r in cos_results]) if cos_results else 0
        results["per_space"][space_name] = {
            "tests": cos_results,
            "avg_accuracy": round(float(avg_acc), 4),
        }
        print(f"  [{space_name}] 平均精度: {avg_acc:.4f}")

    # 对比总结
    raw_acc = results["per_space"].get("raw", {}).get("avg_accuracy", 0)
    svd_acc = results["per_space"].get("svd", {}).get("avg_accuracy", 0)
    ae_acc = results["per_space"].get("ae", {}).get("avg_accuracy", 0)
    results["conclusion"] = {
        "raw_acc": round(raw_acc, 4),
        "svd_acc": round(svd_acc, 4),
        "ae_acc": round(ae_acc, 4),
        "ae_vs_raw": round(ae_acc - raw_acc, 4),
        "ae_vs_svd": round(ae_acc - svd_acc, 4),
        "ae_best": ae_acc > svd_acc and ae_acc > raw_acc,
    }

    return results


# ==================== Stage467: 流形曲率分析 ====================
def run_stage467(bias_scaled, word_labels):
    """量化偏置空间的弯曲程度"""
    print("\n" + "=" * 60)
    print("  Stage467: 流形曲率分析")
    print("=" * 60)

    results = {}
    n, dim = bias_scaled.shape

    # 1. 本征维估计（通过局部PCA）
    print("  [1] 本征维估计（局部PCA）...")
    from sklearn.decomposition import PCA

    # 归一化
    norms = np.linalg.norm(bias_scaled, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    bias_normed = bias_scaled / norms

    # 对每个点做局部PCA
    k_neighbors = min(20, n - 1)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
    nn.fit(bias_normed)
    distances, indices = nn.kneighbors(bias_normed)

    explained_ratios = []
    for i in range(n):
        local_points = bias_scaled[indices[i]]
        pca = PCA()
        pca.fit(local_points)
        # 找到累积解释95%方差所需的维度
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_95 = np.searchsorted(cumsum, 0.95) + 1
        explained_ratios.append({
            "top1": float(pca.explained_variance_ratio_[0]),
            "top5": float(np.sum(pca.explained_variance_ratio_[:5])),
            "top10": float(np.sum(pca.explained_variance_ratio_[:min(10, dim)])),
            "n_dim_95": int(n_95),
        })

    avg_top1 = np.mean([e["top1"] for e in explained_ratios])
    avg_top5 = np.mean([e["top5"] for e in explained_ratios])
    avg_n95 = np.mean([e["n_dim_95"] for e in explained_ratios])
    std_n95 = np.std([e["n_dim_95"] for e in explained_ratios])

    results["local_pca"] = {
        "avg_top1_explained": round(float(avg_top1), 4),
        "avg_top5_explained": round(float(avg_top5), 4),
        "avg_intrinsic_dim_95": round(float(avg_n95), 2),
        "std_intrinsic_dim_95": round(float(std_n95), 2),
        "global_dim": dim,
        "compression_ratio": round(dim / max(avg_n95, EPS), 2),
    }
    print(f"    本征维(95%): {avg_n95:.1f} ± {std_n95:.1f} (全局: {dim})")
    print(f"    压缩比: {dim/max(avg_n95, EPS):.1f}x")

    # 2. 高斯曲率近似（通过Hessian估计）
    print("  [2] 高斯曲率估计...")
    # 对每个点，用k近邻拟合局部二次曲面，估计曲率
    curvatures = []
    for i in range(min(n, 74)):  # 限制计算量
        neighbors = bias_scaled[indices[i]]
        center = neighbors[0]

        # 中心化
        centered = neighbors - center

        # PCA到主方向
        pca = PCA(n_components=min(3, dim))
        pca_coords = pca.fit_transform(centered)

        if pca_coords.shape[1] < 2:
            curvatures.append(0.0)
            continue

        # 在2D主平面上估计曲率
        x = pca_coords[1:, 0]
        y = pca_coords[1:, 1]

        # 用最小二乘拟合二次曲面: z = a*x^2 + b*x*y + c*y^2
        if len(x) < 4:
            curvatures.append(0.0)
            continue

        try:
            # 构建设计矩阵
            X_design = np.column_stack([x**2, x * y, y**2])
            # 假设高度函数为到中心的距离（投影到第3主成分）
            if pca_coords.shape[1] >= 3:
                z_vals = pca_coords[1:, 2]
            else:
                # 用到中心的残差作为高度
                z_vals = np.linalg.norm(centered[1:], axis=1) - np.linalg.norm(centered[0], axis=0)

            coeffs, _, _, _ = np.linalg.lstsq(X_design, z_vals, rcond=None)
            a, b, c = coeffs

            # 高斯曲率 K = 4ac - b^2
            K = 4 * a * c - b ** 2
            curvatures.append(float(K))
        except Exception:
            curvatures.append(0.0)

    curvatures = np.array(curvatures)
    results["gaussian_curvature"] = {
        "mean": round(float(np.mean(np.abs(curvatures))), 6),
        "std": round(float(np.std(np.abs(curvatures))), 6),
        "max_abs": round(float(np.max(np.abs(curvatures))), 6),
        "mean_signed": round(float(np.mean(curvatures)), 6),
        "frac_nonzero": round(float(np.mean(np.abs(curvatures) > 1e-6)), 4),
        "frac_positive": round(float(np.mean(curvatures > 1e-6)), 4),
        "frac_negative": round(float(np.mean(curvatures < -1e-6)), 4),
    }
    print(f"    |K|均值: {np.mean(np.abs(curvatures)):.6f}")
    print(f"    非零曲率比例: {np.mean(np.abs(curvatures) > 1e-6):.4f}")
    print(f"    正曲率: {np.mean(curvatures > 1e-6):.4f}, 负曲率: {np.mean(curvatures < -1e-6):.4f}")

    # 3. 线性度检测（全局PCA vs 局部PCA的维度差异）
    print("  [3] 线性度检测...")
    global_pca = PCA()
    global_pca.fit(bias_scaled)
    global_cumsum = np.cumsum(global_pca.explained_variance_ratio_)
    global_n95 = np.searchsorted(global_cumsum, 0.95) + 1

    linearity_score = avg_n95 / max(global_n95, EPS)
    results["linearity"] = {
        "global_n_dim_95": int(global_n95),
        "local_avg_n_dim_95": round(float(avg_n95), 2),
        "linearity_score": round(float(linearity_score), 4),
        "is_linear": linearity_score > 0.8,
    }
    print(f"    全局本征维(95%): {global_n95}")
    print(f"    局部本征维(95%): {avg_n95:.1f}")
    print(f"    线性度得分: {linearity_score:.4f} ({'线性' if linearity_score > 0.8 else '非线性'})")

    # 4. 非线性度综合评估
    nonlinear_evidence = 0
    if avg_n95 < global_n95 * 0.8:
        nonlinear_evidence += 1
        print(f"    [证据1] 局部维度 < 全局维度 → 流形弯曲")
    if np.mean(np.abs(curvatures)) > 1e-4:
        nonlinear_evidence += 1
        print(f"    [证据2] 高斯曲率显著非零 → 曲面弯曲")
    if np.std(curvatures) > 1e-4:
        nonlinear_evidence += 1
        print(f"    [证据3] 曲率方差大 → 弯曲不均匀")
    if linearity_score < 0.8:
        nonlinear_evidence += 1
        print(f"    [证据4] 线性度低 → 非线性结构")

    results["nonlinear_verdict"] = {
        "evidence_count": nonlinear_evidence,
        "total_checks": 4,
        "is_nonlinear": nonlinear_evidence >= 2,
        "strength": "强" if nonlinear_evidence >= 3 else ("中" if nonlinear_evidence >= 2 else "弱"),
    }

    return results


# ==================== 主流程 ====================
def main():
    if len(sys.argv) < 2:
        print("Usage: python stage464_nonlinear_manifold_analysis.py [qwen3|deepseek]")
        sys.exit(1)

    model_choice = sys.argv[1].lower()
    if model_choice == "qwen3":
        model_path = QWEN3_MODEL_PATH
        model_name = "qwen3_4b"
    elif model_choice == "deepseek":
        model_path = DEEPSEEK7B_MODEL_PATH
        model_name = "deepseek_7b"
    else:
        print(f"Unknown model: {model_choice}")
        sys.exit(1)

    print("=" * 70)
    print(f"Stage464-467: 非线性流形分析四合一")
    print(f"Model: {model_name}")
    print("=" * 70)

    t0 = time.time()

    # 1. 加载模型
    print("\n[1/5] Loading model...")
    model, tokenizer, layer_count, neuron_dim = load_model(model_path)

    # 2. 提取激活
    total_concepts = sum(len(c["words"]) for c in CONCEPTS.values())
    print(f"\n[2/5] Extracting activations ({total_concepts} concepts)...")
    all_activations = extract_all_activations(model, tokenizer, CONCEPTS, layer_count)

    # 释放模型
    print("\n  Releasing model...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. 构建黄金层偏置矩阵
    print(f"\n[3/5] Building bias matrices...")

    golden_layer = 2 if "qwen" in model_name else 1
    test_layers = [golden_layer]

    # 也测试一个中间层和一个后期层
    test_layers = list(set(test_layers + [0, min(10, layer_count - 1), layer_count - 1]))
    test_layers = sorted([l for l in test_layers if l < layer_count])
    print(f"  Test layers: {test_layers}")

    layer_bias_matrices = {}
    for li in test_layers:
        word_vecs = []
        word_labels = []
        for cat_name, cat_data in CONCEPTS.items():
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

        if len(word_vecs) < 20:
            continue

        bias_matrix = np.array(word_vecs)

        # 清理NaN
        nan_mask = np.any(np.isnan(bias_matrix), axis=1)
        inf_mask = np.any(np.isinf(bias_matrix), axis=1)
        valid_mask = ~(nan_mask | inf_mask)
        if not np.all(valid_mask):
            bias_matrix = bias_matrix[valid_mask]
            word_labels = [w for w, m in zip(word_labels, valid_mask) if m]

        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        bias_scaled = scaler.fit_transform(bias_matrix)
        bias_scaled = np.nan_to_num(bias_scaled, nan=0.0)

        layer_bias_matrices[li] = (bias_scaled, word_labels)
        print(f"  Layer {li}: {len(word_labels)} concepts, dim={bias_scaled.shape[1]}")

    # 4. 运行四个实验模块
    print(f"\n[4/5] Running Stage464-467 experiments...")

    all_results = {
        "model": model_name,
        "golden_layer": golden_layer,
        "layer_count": layer_count,
    }

    for li, (bias_scaled, word_labels) in layer_bias_matrices.items():
        layer_key = f"layer_{li}"
        all_results[layer_key] = {
            "n_concepts": len(word_labels),
            "dim": bias_scaled.shape[1],
        }

        print(f"\n{'#' * 60}")
        print(f"  Layer {li} ({'GOLDEN' if li == golden_layer else 'OTHER'})")
        print(f"{'#' * 60}")

        # Stage464: 非线性因子解释
        s464 = run_stage464(bias_scaled, word_labels, CONCEPTS)
        all_results[layer_key]["stage464_factor_interpretation"] = s464

        # Stage465: 流形可视化
        s465 = run_stage465(bias_scaled, word_labels, CONCEPTS, li, model_name)
        all_results[layer_key]["stage465_manifold_visualization"] = {
            "separation": s465.get("umap_separation", {}),
            "methods_ok": ["umap" if "umap" in s465 else "",
                           "isomap" if "isomap" in s465 else "",
                           "tsne" if "tsne" in s465 else ""],
        }

        # Stage466: 非线性概念算术
        s466 = run_stage466(bias_scaled, word_labels, CONCEPTS)
        all_results[layer_key]["stage466_nonlinear_arithmetic"] = s466

        # Stage467: 流形曲率分析
        s467 = run_stage467(bias_scaled, word_labels)
        all_results[layer_key]["stage467_curvature_analysis"] = s467

    # 5. 保存结果
    print(f"\n[5/5] Saving results...")

    output_path = OUTPUT_DIR / f"{model_name}_full_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(all_results), f, ensure_ascii=False, indent=2)
    print(f"  Results: {output_path}")

    # 生成报告
    generate_full_report(all_results, model_name)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Stage464-467 COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 70}")


def generate_full_report(all_data, model_name):
    """生成综合报告"""
    lines = [
        f"# Stage464-467: 非线性流形分析综合报告 — {model_name}",
        "",
        f"**时间**: 2026-04-01 13:30",
        f"**模型**: {model_name}",
        f"**黄金层**: {all_data['golden_layer']}",
        "",
        "---",
        "",
    ]

    for layer_key, layer_data in all_data.items():
        if not layer_key.startswith("layer_"):
            continue
        li_str = layer_key.replace("layer_", "")
        try:
            li = int(li_str)
        except ValueError:
            continue
        is_golden = li == all_data["golden_layer"]

        lines.append(f"## Layer {li} ({'GOLDEN' if is_golden else 'OTHER'})")
        lines.append(f"概念数: {layer_data['n_concepts']}, 维度: {layer_data['dim']}")
        lines.append("")

        # Stage464
        s464 = layer_data.get("stage464_factor_interpretation", {})
        if s464:
            lines.append("### Stage464: 非线性因子解释")
            lines.append("")
            lines.append("| 类别 | 属性 | AE eta2_max | SVD eta2_max | AE优势 |")
            lines.append("|------|------|------------|------------|--------|")
            for cat, attrs in s464.items():
                if isinstance(attrs, dict):
                    for attr, info in attrs.items():
                        if isinstance(info, dict):
                            lines.append(
                                f"| {cat} | {attr} | {info.get('eta_max', 0):.4f} | "
                                f"{info.get('svd_eta_max', 0):.4f} | {info.get('ae_advantage', 0):+.4f} |"
                            )
            lines.append("")

        # Stage465
        s465 = layer_data.get("stage465_manifold_visualization", {})
        if s465 and "separation" in s465:
            lines.append("### Stage465: 流形可视化 (UMAP)")
            lines.append("")
            lines.append("| 类别对 | 类间距离 | 类内距离 | 分离比 |")
            lines.append("|--------|---------|---------|--------|")
            for pair, info in s465["separation"].items():
                lines.append(
                    f"| {pair} | {info.get('inter_dist', 0):.4f} | "
                    f"{info.get('intra_dist', 0):.4f} | {info.get('ratio', 0):.2f} |"
                )
            lines.append("")

        # Stage466
        s466 = layer_data.get("stage466_nonlinear_arithmetic", {})
        if s466:
            lines.append("### Stage466: 非线性概念算术 (3Cos)")
            lines.append("")
            conc = s466.get("conclusion", {})
            lines.append(f"- 原始空间精度: {conc.get('raw_acc', 0):.4f}")
            lines.append(f"- SVD空间精度: {conc.get('svd_acc', 0):.4f}")
            lines.append(f"- AE空间精度: {conc.get('ae_acc', 0):.4f}")
            lines.append(f"- AE最优: {'是' if conc.get('ae_best', False) else '否'}")
            lines.append("")

        # Stage467
        s467 = layer_data.get("stage467_curvature_analysis", {})
        if s467:
            lines.append("### Stage467: 流形曲率分析")
            lines.append("")
            lpca = s467.get("local_pca", {})
            lines.append(f"- 本征维(95%): {lpca.get('avg_intrinsic_dim_95', 0):.1f} ± "
                         f"{lpca.get('std_intrinsic_dim_95', 0):.1f} (全局: {lpca.get('global_dim', 0)})")
            lines.append(f"- 压缩比: {lpca.get('compression_ratio', 0):.1f}x")
            gc = s467.get("gaussian_curvature", {})
            lines.append(f"- |K|均值: {gc.get('mean', 0):.6f}")
            lines.append(f"- 非零曲率比例: {gc.get('frac_nonzero', 0):.4f}")
            lin = s467.get("linearity", {})
            lines.append(f"- 线性度得分: {lin.get('linearity_score', 0):.4f} "
                         f"({'线性' if lin.get('is_linear', False) else '非线性'})")
            verdict = s467.get("nonlinear_verdict", {})
            lines.append(f"- 非线性证据: {verdict.get('evidence_count', 0)}/{verdict.get('total_checks', 4)} "
                         f"→ 强度: {verdict.get('strength', '?')}")
            lines.append("")

    # 总结论
    lines.append("---")
    lines.append("")
    lines.append("## 综合结论")
    lines.append("")
    lines.append("详见实验数据分析。")
    lines.append("")

    report_path = OUTPUT_DIR / f"{model_name}_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
