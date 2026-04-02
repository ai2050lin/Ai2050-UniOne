# -*- coding: utf-8 -*-
"""
Stage469: 测地线算术 — 在双曲空间中用测地线替代向量加法做概念操控
================================================================

核心目标：
  验证双曲空间的测地线操作是否能更精确地执行概念算术

理论背景：
  在欧氏空间中，概念算术用向量加法：king - man + woman = queen
  在双曲空间中，应该用测地线操作：
  - 测地线中点: geodesic_midpoint(king, queen)
  - 平行移动: parallel_transport(direction, source_point)
  - 指数映射: exp_map(tangent_vector)

实验设计：
  1. 欧氏算术 vs 双曲测地线算术对比
  2. 概念属性替换（颜色/大小等）
  3. 概念类别迁移
  4. 层次推理（IS-A关系）
  5. 精度对比（Top-1 / Top-3 / Top-5）

关键算子（Poincare Ball）：
  - Möbius加法: a ⊕ b = ((1+2<a,b>/c+||b||²/c)a + (1-||a||²)b) / (1+2<a,b>/c+||a||²||b||²/c²)
  - 测地线: γ(t) = a ⊕ tanh(t*d(a,b)/2) * (b ⊖ a) / ||b ⊖ a||
  - 测地线中点: midpoint(a,b) = a ⊕ (tanh(d/4)/tanh(d/2)) * (b ⊖ a) / ||b ⊖ a||

模型: 单模型（命令行选择）
  python stage469_geodesic_arithmetic.py qwen3
  python stage469_geodesic_arithmetic.py deepseek

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
from scipy.spatial.distance import cdist
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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage469_geodesic_arithmetic_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8

# ==================== 概念集（含属性标注） ====================
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
        },
    },
}


# ==================== Poincare Ball 几何操作 ====================

class PoincareBall:
    def __init__(self, dim: int, curvature: float = -1.0):
        self.dim = dim
        self.c = abs(curvature)

    def project(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        if norm >= 1.0 - 1e-5:
            return x * (1.0 - 1e-5) / norm * 0.99
        return x

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        x_norm_sq = np.sum(x ** 2)
        y_norm_sq = np.sum(y ** 2)
        diff_sq = np.sum((x - y) ** 2)
        denom = (1.0 - x_norm_sq) * (1.0 - y_norm_sq)
        if denom < EPS:
            return 10.0
        arg = 1.0 + 2.0 * self.c * diff_sq / denom
        if arg <= 1.0:
            return 0.0
        return np.arccosh(min(arg, 100.0)) / np.sqrt(self.c)

    def distance_matrix(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = self.distance(X[i], X[j])
                D[j, i] = D[i, j]
        return D

    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_norm_sq = np.sum(x ** 2)
        y_norm_sq = np.sum(y ** 2)
        xy = np.dot(x, y)
        num = ((1.0 + 2.0 * xy / self.c + y_norm_sq / self.c) * x
               + (1.0 - x_norm_sq) * y)
        denom = 1.0 + 2.0 * xy / self.c + x_norm_sq * y_norm_sq / (self.c ** 2)
        if abs(denom) < EPS:
            return np.zeros_like(x)
        return self.project(num / denom)

    def mobius_sub(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.mobius_add(x, -y)

    def scalar_mult(self, r: float, x: np.ndarray) -> np.ndarray:
        """标量乘法（沿测地线缩放）"""
        if abs(r) < EPS:
            return np.zeros_like(x)
        x_norm = np.linalg.norm(x)
        if x_norm < EPS:
            return np.zeros_like(x)
        tanh_r = np.tanh(r * np.arctanh(min(x_norm, 1.0 - 1e-7)))
        return self.project(tanh_r * x / x_norm)

    def geodesic(self, x: np.ndarray, y: np.ndarray, t: float = 0.5) -> np.ndarray:
        """测地线上的点：γ(t) where t ∈ [0,1]"""
        diff = self.mobius_sub(y, x)
        diff_norm = np.linalg.norm(diff)
        if diff_norm < EPS:
            return x.copy()
        direction = diff / diff_norm
        dist = self.distance(x, y)
        return self.mobius_add(x, self.scalar_mult(t * dist, direction))

    def geodesic_midpoint(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """测地线中点"""
        return self.geodesic(x, y, 0.5)

    def weighted_midpoint(self, points: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """加权测地线中点（Fréchet均值近似）"""
        # 迭代计算
        weights = np.array(weights)
        weights = weights / (weights.sum() + EPS)

        # 初始化为欧氏加权均值
        result = np.average(points, axis=0, weights=weights)
        result = self.project(result)

        for _ in range(20):
            new_result = np.zeros_like(result)
            for p, w in zip(points, weights):
                mid = self.geodesic(result, p, w)
                new_result += mid
            new_result = self.project(new_result / len(points))
            if np.linalg.norm(new_result - result) < 1e-8:
                break
            result = new_result

        return result

    def parallel_transport(self, v: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """平行移动：将src处的切向量v平行移动到dst"""
        # Poincare球中的平行移动公式
        diff = self.mobius_sub(dst, src)
        diff_norm = np.linalg.norm(diff)
        if diff_norm < EPS:
            return v.copy()

        src_norm_sq = np.sum(src ** 2)
        dst_norm_sq = np.sum(dst ** 2)
        sd = np.dot(src, diff)

        alpha = np.dot(v, diff) / (diff_norm ** 2 + EPS)
        beta = -np.dot(v, src) / (self.c + EPS)

        transported = v - 2.0 * sd / (self.c - dst_norm_sq + EPS) * (
            diff + dst
        )
        # 简化版：使用梯度修正
        transport_factor = (1.0 - dst_norm_sq) / (1.0 - src_norm_sq + EPS)
        return v * np.sqrt(max(transport_factor, 0))

    def reflect_across(self, point: np.ndarray, center: np.ndarray) -> np.ndarray:
        """关于center的对称点（在双曲空间中的"反射"操作）"""
        # 使用测地线延伸
        direction = self.mobius_sub(point, center)
        dir_norm = np.linalg.norm(direction)
        if dir_norm < EPS:
            return point.copy()
        unit_dir = direction / dir_norm
        return self.mobius_add(center, self.scalar_mult(-self.distance(center, point), unit_dir))


# ==================== 概念算术测试 ====================

def euclidean_analogy(a_vec, b_vec, c_vec, all_vecs, method='add'):
    """欧氏空间类比推理

    king - man + woman = ?
    a=king, b=man, c=woman → 找最接近 result 的词
    """
    if method == 'add':
        # 经典向量加法
        result = a_vec - b_vec + c_vec
    elif method == 'proportion':
        # 比例法: result = c + (a-b)
        result = c_vec + (a_vec - b_vec)
    else:
        result = a_vec - b_vec + c_vec

    dists = np.linalg.norm(all_vecs - result, axis=1)
    return dists, result


def hyperbolic_analogy(ball: PoincareBall, a_vec, b_vec, c_vec,
                        all_vecs, method='geodesic'):
    """双曲空间类比推理

    使用测地线操作替代向量加法
    """
    if method == 'geodesic':
        # 方法1: 沿测地线移动
        # 1. 计算从a到b的"方向"（切空间向量）
        # 2. 将这个方向从c出发做平行移动
        # 3. 沿移动后的方向走
        diff = ball.mobius_sub(b_vec, a_vec)  # 从a到b的"向量"
        diff_norm = np.linalg.norm(diff)
        if diff_norm < EPS:
            dists = np.array([ball.distance(c_vec, v) for v in all_vecs])
            return dists, c_vec.copy()

        direction = diff / diff_norm
        dist_ab = ball.distance(a_vec, b_vec)

        # 从c沿反方向（因为类比是 a-b+c，即反向）
        result = ball.mobius_sub(c_vec, diff)

    elif method == 'midpoint_shift':
        # 方法2: 中点偏移
        # target ≈ midpoint between c and reflection of a across b
        midpoint_ab = ball.geodesic_midpoint(a_vec, b_vec)
        # 计算从a到中点的偏移
        shift = ball.mobius_sub(midpoint_ab, a_vec)
        result = ball.mobius_add(c_vec, shift)

    elif method == 'transport':
        # 方法3: 平行移动
        diff = ball.mobius_sub(b_vec, a_vec)
        transported = ball.parallel_transport(diff, a_vec, c_vec)
        result = ball.mobius_add(c_vec, transported)

    else:
        result = ball.mobius_add(ball.mobius_sub(a_vec, b_vec), c_vec)

    # 计算到所有词的距离
    dists = np.array([ball.distance(result, v) for v in all_vecs])
    return dists, result


def find_nearest(dists, word_list, exclude_words=None, top_k=5):
    """找最近的k个词"""
    if exclude_words is None:
        exclude_words = set()
    indexed = sorted(enumerate(dists), key=lambda x: x[1])
    results = []
    for idx, d in indexed:
        if word_list[idx] in exclude_words:
            continue
        results.append((word_list[idx], float(d)))
        if len(results) >= top_k:
            break
    return results


# ==================== 模型和数据 ====================

def sanitize_for_json(obj):
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
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
        for word, attrs in cat_data["words"].items():
            try:
                acts = extract_word_per_layer(model, tokenizer, word, layer_count)
                if acts:
                    all_activations[cat_name][word] = {"acts": acts, "attrs": attrs}
                    done += 1
            except Exception:
                pass

    print(f"  Total extracted: {done}/{total}")
    return all_activations


def find_golden_layer(all_activations, concepts, layer_count):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

    best_layer = 0
    best_score = -1

    for li in range(layer_count):
        word_vecs = []
        word_cats = []

        for cat_name, words_data in all_activations.items():
            for word, data in words_data.items():
                if li in data["acts"]:
                    word_vecs.append(data["acts"][li])
                    word_cats.append(cat_name)

        if len(word_vecs) < 20:
            continue

        bias_matrix = np.array(word_vecs)
        scaler = StandardScaler()
        normed = scaler.fit_transform(bias_matrix)

        svd = TruncatedSVD(n_components=min(10, len(word_vecs) - 1), random_state=42)
        comp = svd.fit_transform(normed)

        cat_to_idx = defaultdict(list)
        for i, cat in enumerate(word_cats):
            cat_to_idx[cat].append(i)

        total_eta = 0
        for fi in range(min(5, comp.shape[1])):
            scores = comp[:, fi]
            grand_mean = np.mean(scores)
            ss_between = sum(len(idxs) * (np.mean(scores[idxs]) - grand_mean) ** 2
                           for idxs in cat_to_idx.values())
            ss_total = np.sum((scores - grand_mean) ** 2)
            total_eta += ss_between / (ss_total + 1e-10)

        if total_eta > best_score:
            best_score = total_eta
            best_layer = li

    return best_layer


def build_bias_space(all_activations, concepts, layer_count, golden_layer):
    word_vecs = []
    word_labels = []
    word_cats = []
    word_attrs = {}

    for cat_name, words_data in all_activations.items():
        cat_vecs = []
        for word, data in words_data.items():
            if golden_layer in data["acts"]:
                cat_vecs.append(data["acts"][golden_layer])
        if not cat_vecs:
            continue
        basis = np.mean(cat_vecs, axis=0)

        for word, data in words_data.items():
            if golden_layer in data["acts"]:
                bias = data["acts"][golden_layer] - basis
                word_vecs.append(bias)
                word_labels.append(word)
                word_cats.append(cat_name)
                word_attrs[word] = data["attrs"]

    return np.array(word_vecs), word_labels, word_cats, word_attrs


def embed_to_poincare(X_euclid: np.ndarray, curvature: float = -1.0) -> np.ndarray:
    """将偏置向量映射到Poincare球"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # 先降维到可管理的维度
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_euclid)

    # PCA降维
    n_comp = min(10, X_norm.shape[0] - 1, X_norm.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_norm)

    # 归一化到Poincare球内
    max_norm = np.max(np.linalg.norm(X_pca, axis=1))
    X_poincare = X_pca / (max_norm + EPS) * 0.85

    return X_poincare


# ==================== 类比测试设计 ====================

def generate_analogy_tests(concepts, word_attrs, word_labels):
    """生成概念类比测试集

    测试类型：
    1. 属性替换：apple(red,round) → banana(yellow,curved)
    2. 属性类比：apple:banana :: grape:?
    3. 类别内迁移
    """
    tests = []

    # 水果属性类比
    fruit_words = list(concepts["fruit"]["words"].keys())
    fruit_words_valid = [w for w in fruit_words if w in word_labels]

    for i, w1 in enumerate(fruit_words_valid):
        for w2 in fruit_words_valid[i + 1:]:
            # apple:banana :: grape:? → 找和grape同属性的词
            tests.append({
                "type": "fruit_attribute",
                "a": w1, "b": w2,
                "description": f"fruit analogy: {w1}:{w2} :: X:Y",
            })

    # 动物属性类比
    animal_words = list(concepts["animal"]["words"].keys())
    animal_words_valid = [w for w in animal_words if w in word_labels]

    for i, w1 in enumerate(animal_words_valid):
        for w2 in animal_words_valid[i + 1:]:
            tests.append({
                "type": "animal_attribute",
                "a": w1, "b": w2,
                "description": f"animal analogy: {w1}:{w2} :: X:Y",
            })

    # 跨类别推理
    profession_words = list(concepts["profession"]["words"].keys())
    profession_words_valid = [w for w in profession_words if w in word_labels]

    for i, w1 in enumerate(profession_words_valid):
        for w2 in profession_words_valid[i + 1:]:
            tests.append({
                "type": "profession_attribute",
                "a": w1, "b": w2,
                "description": f"profession analogy: {w1}:{w2} :: X:Y",
            })

    return tests


def generate_property_transfer_tests(concepts, word_attrs, word_labels):
    """生成属性转移测试

    测试：将概念A的某个属性替换为概念B的该属性
    例如：apple的颜色属性 → banana的颜色属性
    """
    tests = []

    # 水果颜色转移
    fruit_words = list(concepts["fruit"]["words"].keys())
    fruit_words_valid = [w for w in fruit_words if w in word_labels and "color" in word_attrs.get(w, {})]

    for i, w1 in enumerate(fruit_words_valid):
        for w2 in fruit_words_valid:
            if w1 == w2:
                continue
            tests.append({
                "type": "fruit_color_transfer",
                "source": w1, "target_attr_word": w2,
                "source_color": word_attrs[w1].get("color", ""),
                "target_color": word_attrs[w2].get("color", ""),
                "description": f"transfer color of {w2} to {w1}",
            })

    # 动物大小转移
    animal_words = list(concepts["animal"]["words"].keys())
    animal_words_valid = [w for w in animal_words if w in word_labels and "size" in word_attrs.get(w, {})]

    for i, w1 in enumerate(animal_words_valid):
        for w2 in animal_words_valid:
            if w1 == w2:
                continue
            tests.append({
                "type": "animal_size_transfer",
                "source": w1, "target_attr_word": w2,
                "source_size": word_attrs[w1].get("size", 0),
                "target_size": word_attrs[w2].get("size", 0),
                "description": f"transfer size of {w2} to {w1}",
            })

    return tests


def generate_midpoint_tests(concepts, word_labels):
    """生成测地线中点测试

    测试：两个概念的中间点应该接近一个"中间"概念
    例如：cat和dog的中点应该接近其他家养小动物
    """
    tests = []

    all_words = word_labels
    for i, w1 in enumerate(all_words):
        for j in range(i + 1, min(i + 5, len(all_words))):
            w2 = all_words[j]
            tests.append({
                "type": "midpoint",
                "a": w1, "b": w2,
                "description": f"midpoint({w1}, {w2})",
            })

    return tests


# ==================== 主实验 ====================

def run_experiment(model_name: str):
    print(f"\n{'='*60}")
    print(f"Stage469: 测地线算术 — {model_name}")
    print(f"{'='*60}")

    model_path = QWEN3_MODEL_PATH if model_name == "qwen3" else DEEPSEEK7B_MODEL_PATH

    # Step 1: 加载模型
    print("\n[Step 1] 加载模型...")
    t0 = time.time()
    model, tokenizer, layer_count, neuron_dim = load_model(model_path)
    print(f"  耗时: {time.time()-t0:.1f}s")

    # Step 2: 提取激活
    print("\n[Step 2] 提取概念激活...")
    t0 = time.time()
    all_activations = extract_all_activations(model, tokenizer, CONCEPTS, layer_count)
    print(f"  耗时: {time.time()-t0:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: 黄金层 + 偏置空间
    print("\n[Step 3] 构建偏置空间...")
    golden_layer = find_golden_layer(all_activations, CONCEPTS, layer_count)
    bias_matrix, word_labels, word_cats, word_attrs = build_bias_space(
        all_activations, CONCEPTS, layer_count, golden_layer
    )
    print(f"  黄金层: {golden_layer}, 概念数: {len(word_labels)}")

    # Step 4: 嵌入到Poincare空间
    print("\n[Step 4] 嵌入到Poincare空间...")
    X_poincare = embed_to_poincare(bias_matrix, curvature=-1.0)
    ball = PoincareBall(dim=X_poincare.shape[1], curvature=-1.0)

    # Step 5: 属性类比测试
    print("\n[Step 5] 属性类比测试...")
    analogy_tests = generate_analogy_tests(CONCEPTS, word_attrs, word_labels)

    # 限制测试数量
    if len(analogy_tests) > 200:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(analogy_tests), 200, replace=False)
        analogy_tests = [analogy_tests[i] for i in idx]

    euclidean_top1 = 0
    hyperbolic_top1 = 0
    euclidean_top3 = 0
    hyperbolic_top3 = 0
    euclidean_top5 = 0
    hyperbolic_top5 = 0
    total_tests = 0
    analogy_details = []

    word_to_idx = {w: i for i, w in enumerate(word_labels)}

    for test in analogy_tests:
        a_word = test["a"]
        b_word = test["b"]

        if a_word not in word_to_idx or b_word not in word_to_idx:
            continue

        a_idx = word_to_idx[a_word]
        b_idx = word_to_idx[b_word]

        # 找类比目标：与b同类别的其他词
        b_cat = None
        for cat_name, words_data in all_activations.items():
            if b_word in words_data:
                b_cat = cat_name
                break

        if b_cat is None:
            continue

        # 获取b类别的词（排除a和b）
        same_cat_words = [w for w in CONCEPTS.get(b_cat, {}).get("words", {})
                         if w in word_to_idx and w != a_word and w != b_word]

        if not same_cat_words:
            continue

        # 欧氏类比
        euclid_a = bias_matrix[a_idx]
        euclid_b = bias_matrix[b_idx]

        # 测试：a-b+c for each c in same_cat
        best_euclid_top1 = 0
        best_hyper_top1 = 0

        for c_word in same_cat_words[:3]:  # 每个b最多测3个c
            c_idx = word_to_idx[c_word]
            euclid_c = bias_matrix[c_idx]
            poinc_a = X_poincare[a_idx]
            poinc_b = X_poincare[b_idx]
            poinc_c = X_poincare[c_idx]

            # 排除词
            exclude = {a_word, b_word, c_word}

            # 欧氏
            e_dists, _ = euclidean_analogy(euclid_a, euclid_b, euclid_c, bias_matrix)
            e_nearest = find_nearest(e_dists, word_labels, exclude, top_k=5)

            # 双曲
            h_dists, _ = hyperbolic_analogy(ball, poinc_a, poinc_b, poinc_c, X_poincare)
            h_nearest = find_nearest(h_dists, word_labels, exclude, top_k=5)

            # 检查是否命中同类别
            e_top1_hit = any(w in same_cat_words for w, _ in [e_nearest[0]] if e_nearest)
            h_top1_hit = any(w in same_cat_words for w, _ in [h_nearest[0]] if h_nearest)
            e_top3_hit = any(w in same_cat_words for w, _ in e_nearest[:3] if e_nearest)
            h_top3_hit = any(w in same_cat_words for w, _ in h_nearest[:3] if h_nearest)

            euclidean_top1 += int(e_top1_hit)
            hyperbolic_top1 += int(h_top1_hit)
            euclidean_top3 += int(e_top3_hit)
            hyperbolic_top3 += int(h_top3_hit)
            total_tests += 1

            analogy_details.append({
                "test": test["description"],
                "euclid_nearest": [w for w, d in e_nearest[:3]],
                "hyper_nearest": [w for w, d in h_nearest[:3]],
                "expected_cat": b_cat,
                "e_hit": int(e_top1_hit),
                "h_hit": int(h_top1_hit),
            })

    print(f"  完成类比测试: {total_tests}")

    # Step 6: 属性转移测试
    print("\n[Step 6] 属性转移测试...")
    transfer_tests = generate_property_transfer_tests(CONCEPTS, word_attrs, word_labels)
    if len(transfer_tests) > 100:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(transfer_tests), 100, replace=False)
        transfer_tests = [transfer_tests[i] for i in idx]

    transfer_results = {"euclidean": 0, "hyperbolic": 0, "total": 0}

    for test in transfer_tests:
        src = test["source"]
        tgt = test["target_attr_word"]
        if src not in word_to_idx or tgt not in word_to_idx:
            continue

        src_idx = word_to_idx[src]
        tgt_idx = word_to_idx[tgt]

        # 找"具有tgt属性但属于src类别"的词
        src_cat = None
        for cat_name, words_data in all_activations.items():
            if src in words_data:
                src_cat = cat_name
                break
        if src_cat is None:
            continue

        # 欧氏：src + (attr_of_tgt - attr_of_src)
        src_vec = bias_matrix[src_idx]
        tgt_vec = bias_matrix[tgt_idx]
        result_e = src_vec + (tgt_vec - src_vec) * 0.3
        dists_e = np.linalg.norm(bias_matrix - result_e, axis=1)
        nearest_e = find_nearest(dists_e, word_labels, {src, tgt}, top_k=3)

        # 双曲：沿测地线从src向tgt移动一小步
        poinc_src = X_poincare[src_idx]
        poinc_tgt = X_poincare[tgt_idx]
        result_h = ball.geodesic(poinc_src, poinc_tgt, 0.3)
        dists_h = np.array([ball.distance(result_h, v) for v in X_poincare])
        nearest_h = find_nearest(dists_h, word_labels, {src, tgt}, top_k=3)

        transfer_results["total"] += 1
        # 检查最近邻是否和tgt共享属性
        tgt_attrs = word_attrs.get(tgt, {})
        for attr_key, attr_val in tgt_attrs.items():
            for w, d in nearest_e:
                w_attrs = word_attrs.get(w, {})
                if w_attrs.get(attr_key) == attr_val:
                    transfer_results["euclidean"] += 1
                    break
            for w, d in nearest_h:
                w_attrs = word_attrs.get(w, {})
                if w_attrs.get(attr_key) == attr_val:
                    transfer_results["hyperbolic"] += 1
                    break

    # Step 7: 测地线中点测试
    print("\n[Step 7] 测地线中点测试...")
    midpoint_tests = generate_midpoint_tests(CONCEPTS, word_labels)
    if len(midpoint_tests) > 100:
        midpoint_tests = midpoint_tests[:100]

    midpoint_results = []
    for test in midpoint_tests:
        a, b = test["a"], test["b"]
        if a not in word_to_idx or b not in word_to_idx:
            continue

        # 欧氏中点
        euclid_mid = (bias_matrix[word_to_idx[a]] + bias_matrix[word_to_idx[b]]) / 2
        dists_e = np.linalg.norm(bias_matrix - euclid_mid, axis=1)
        nearest_e = find_nearest(dists_e, word_labels, {a, b}, top_k=3)

        # 测地线中点
        poinc_mid = ball.geodesic_midpoint(
            X_poincare[word_to_idx[a]], X_poincare[word_to_idx[b]]
        )
        dists_h = np.array([ball.distance(poinc_mid, v) for v in X_poincare])
        nearest_h = find_nearest(dists_h, word_labels, {a, b}, top_k=3)

        midpoint_results.append({
            "a": a, "b": b,
            "euclid_nearest": [w for w, d in nearest_e[:3]],
            "hyper_nearest": [w for w, d in nearest_h[:3]],
        })

    # Step 8: 结果汇总
    print("\n[Step 8] 结果汇总...")

    total = max(total_tests, 1)
    results = {
        "model": model_name,
        "timestamp": TIMESTAMP,
        "golden_layer": int(golden_layer),
        "n_concepts": len(word_labels),
        "analogy_test": {
            "total": total_tests,
            "euclidean_top1_rate": float(euclidean_top1 / total) if total > 0 else 0,
            "hyperbolic_top1_rate": float(hyperbolic_top1 / total) if total > 0 else 0,
            "euclidean_top3_rate": float(euclidean_top3 / total) if total > 0 else 0,
            "hyperbolic_top3_rate": float(hyperbolic_top3 / total) if total > 0 else 0,
            "hyperbolic_improvement_top1": float(
                (hyperbolic_top1 - euclidean_top1) / max(euclidean_top1, 1) * 100
            ),
        },
        "transfer_test": transfer_results,
        "midpoint_samples": sanitize_for_json(midpoint_results[:20]),
    }

    # 打印关键结果
    print(f"\n{'='*60}")
    print(f"关键发现:")
    print(f"  属性类比 Top-1 精度:")
    print(f"    欧氏空间:  {euclidean_top1}/{total} = {euclidean_top1/total*100:.1f}%")
    print(f"    双曲空间:  {hyperbolic_top1}/{total} = {hyperbolic_top1/total*100:.1f}%")
    print(f"  属性类比 Top-3 精度:")
    print(f"    欧氏空间:  {euclidean_top3}/{total} = {euclidean_top3/total*100:.1f}%")
    print(f"    双曲空间:  {hyperbolic_top3}/{total} = {hyperbolic_top3/total*100:.1f}%")
    if total_tests > 0:
        improvement = (hyperbolic_top1 - euclidean_top1) / total * 100
        if improvement > 0:
            print(f"  ★ 双曲空间提升: +{improvement:.1f}% (Top-1)")
        else:
            print(f"  欧氏空间更优: {improvement:.1f}% (Top-1)")
    print(f"  属性转移命中:")
    print(f"    欧氏: {transfer_results['euclidean']}/{transfer_results['total']}")
    print(f"    双曲: {transfer_results['hyperbolic']}/{transfer_results['total']}")
    print(f"{'='*60}")

    # 保存结果
    output_path = OUTPUT_DIR / f"stage469_results_{model_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(results), f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存至: {output_path}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python stage469_geodesic_arithmetic.py [qwen3|deepseek]")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    if model_name not in ("qwen3", "deepseek"):
        print("错误: 请选择 qwen3 或 deepseek")
        sys.exit(1)

    run_experiment(model_name)
