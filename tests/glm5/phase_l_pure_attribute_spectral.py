"""
Phase L-P290/291/292/293/294: 去除名词影响的纯属性谱分析
======================================================================

Phase XLIX核心发现:
  1. 颜色子空间确认是torus(n_near_zero=3-8)
  2. 颜色φ编码家族而非颜色值 — 名词污染了属性子空间!
  3. 味道φ编码mild/salty/rich(r=0.3-0.6) — 部分可检测
  4. 大小φ0编码small/tiny(r=0.3-0.4) — 最易检测
  5. 子空间夹角47-63°, 远非正交

关键瓶颈:
  1. 颜色值检测失败: 用离散indicator无法检测连续编码(色相角度)
  2. 属性子空间被家族/名词污染: φ2-φ4编码家族而非属性值
  3. 环绕数<1: 12色未完全闭合环

Phase L核心改进:
  1. 名词中心化: G_centered(n,a) = G(n,a) - mean_n(G(n,a))
     去除每个名词的基线, 剩余纯属性信号
  2. 连续编码检测: 对颜色用色相角度θ的cos(θ)和sin(θ)替代indicator
     12色排列在色相环上: red→orange→yellow→green→...→purple→red
  3. Procrustes对齐: 对齐不同名词的Laplacian嵌入后取平均
     得到跨名词一致的属性编码
  4. 更大样本: 使用全部60个名词(5族×12) → 720三元组×30模板

五大实验:
  P290: 名词中心化后的属性子空间Laplacian谱
    - 名词中心化后, Laplacian特征值应该更集中
    - n_near_zero可能变化: 名词信号被去除后, 属性内部结构更清晰

  P291: 连续编码检测 — 色相角度cos/sin
    - 对颜色: 计算φ_i与cos(2π*k/12), sin(2π*k/12)的相关
    - 对味道: 计算φ_i与甜度/酸度/咸度的连续评分
    - 对大小: 计算φ_i与小→大轴的相关
    - 目标: 检测到颜色编码的连续方向!

  P292: Procrustes对齐后跨名词平均Laplacian
    - 对每个名词单独做Laplacian(12样本)
    - 用Procrustes对齐不同名词的φ空间
    - 对齐后平均: 得到跨名词一致的属性编码

  P293: 纯属性流形类型
    - 名词中心化后的颜色子空间是否仍是torus?
    - 闭合度是否提高?

  P294: 纯属性重构与语义公式
    - 用纯属性编码重构G项
    - 提出G项的显式数学公式:
      G(n,a) = f_base(n) + f_color(θ) + f_taste(φ) + f_size(s)

数据规模: 720三元组(60名词×12属性值) × 30模板 = 21600前向传播
实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免OOM)
"""

import torch
import numpy as np
import os, sys, gc, time, json, argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
try:
    from scipy.spatial.procrustes import procrustes
except ImportError:
    def procrustes(X, Y):
        """Simple Procrustes alignment"""
        from numpy.linalg import svd
        X_c = X - X.mean(axis=0)
        Y_c = Y - Y.mean(axis=0)
        U, _, Vt = svd(Y_c.T @ X_c)
        R = U @ Vt
        Y_aligned = Y_c @ R
        disparity = float(np.mean((X_c - Y_aligned)**2))
        return X_c, Y_aligned, disparity
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_l_log.txt"

class Logger:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', encoding='utf-8', buffering=1)
    def log(self, msg):
        ts = time.strftime('%H:%M:%S')
        self.f.write(f"{ts} {msg}\n")
        self.f.flush()
        print(f"  [{ts}] {msg}")
    def close(self): self.f.close()

L = Logger(LOG_FILE)

def get_model_path(model_name):
    paths = {
        "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    }
    return paths.get(model_name)

def load_model(model_name):
    p = get_model_path(model_name)
    p_abs = os.path.abspath(p)
    tok = AutoTokenizer.from_pretrained(p_abs, trust_remote_code=True, local_files_only=True, use_fast=False)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        p_abs, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map="cpu"
    )
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    mdl.eval()
    device = next(mdl.parameters()).device
    return mdl, tok, device

# ===================== 数据集定义(扩展!) =====================
STIMULI = {
    "fruit_family": ["apple","banana","pear","orange","grape","mango","strawberry","watermelon","cherry","peach","lemon","lime"],
    "animal_family": ["cat","dog","rabbit","horse","lion","eagle","elephant","dolphin","tiger","bear","fox","wolf"],
    "vehicle_family": ["car","bus","train","plane","boat","bicycle","truck","helicopter","motorcycle","ship","subway","taxi"],
    "furniture_family": ["chair","table","desk","sofa","bed","cabinet","shelf","bench","stool","dresser","couch","armchair"],
    "tool_family": ["hammer","wrench","screwdriver","saw","drill","pliers","knife","scissors","shovel","rake","axe","chisel"],
    "color_attrs": ["red","green","yellow","orange","brown","white","blue","black","pink","purple","gray","gold"],
    "taste_attrs": ["sweet","sour","bitter","salty","crisp","soft","spicy","fresh","tart","savory","rich","mild"],
    "size_attrs": ["big","small","tall","short","long","wide","thin","thick","heavy","light","huge","tiny"],
}

# 扩展到全部60个名词!
ALL_NOUNS = []
for fam in ["fruit_family","animal_family","vehicle_family","furniture_family","tool_family"]:
    ALL_NOUNS.extend(STIMULI[fam])

NOUN_TO_FAMILY = {}
for fam, words in STIMULI.items():
    if "family" in fam:
        for w in words:
            NOUN_TO_FAMILY[w] = fam

# 颜色属性的色相角度(按色相环排列!)
# red=0°, orange=30°, yellow=60°, green=120°, blue=240°, purple=270°, pink=330°
# 非光谱色(brown, white, black, gray, gold)用中间值
COLOR_HUE_ORDER = ["red","orange","yellow","green","blue","purple","pink",
                   "brown","white","black","gray","gold"]
COLOR_HUE_ANGLES = {
    "red": 0, "orange": 30, "yellow": 60, "green": 120,
    "blue": 240, "purple": 270, "pink": 330,
    "brown": 25, "white": 0, "black": 0, "gray": 0, "gold": 45,
}
# 更精细的色相排列(仅光谱色7个, 其余5个是亮度/饱和度变化)
SPECTRAL_COLORS = ["red","orange","yellow","green","blue","purple","pink"]
NONSPECTRAL_COLORS = ["brown","white","black","gray","gold"]
# 光谱色色相角
SPECTRAL_HUE_ANGLES = {"red": 0, "orange": 30, "yellow": 60, "green": 120, 
                       "blue": 240, "purple": 270, "pink": 330}

# 味道属性的连续维度(主观评分0-1)
# 甜度: sweet=1.0, mild=0.3, savory=0.2, fresh=0.1, sour=0.05, 其余=0
# 酸度: sour=1.0, tart=0.8, crisp=0.5, fresh=0.3, 其余=0
# 咸度: salty=1.0, savory=0.5, rich=0.3, 其余=0
# 苦度: bitter=1.0, tart=0.5, spicy=0.3, 其余=0
TASTE_DIMS = {
    "sweet":    {"sweetness": 1.0, "sourness": 0.05, "saltiness": 0.0, "bitterness": 0.0},
    "sour":     {"sweetness": 0.05, "sourness": 1.0, "saltiness": 0.0, "bitterness": 0.1},
    "bitter":   {"sweetness": 0.0, "sourness": 0.1, "saltiness": 0.0, "bitterness": 1.0},
    "salty":    {"sweetness": 0.0, "sourness": 0.0, "saltiness": 1.0, "bitterness": 0.0},
    "crisp":    {"sweetness": 0.3, "sourness": 0.5, "saltiness": 0.0, "bitterness": 0.1},
    "soft":     {"sweetness": 0.4, "sourness": 0.0, "saltiness": 0.0, "bitterness": 0.0},
    "spicy":    {"sweetness": 0.0, "sourness": 0.2, "saltiness": 0.1, "bitterness": 0.3},
    "fresh":    {"sweetness": 0.1, "sourness": 0.3, "saltiness": 0.0, "bitterness": 0.0},
    "tart":     {"sweetness": 0.0, "sourness": 0.8, "saltiness": 0.0, "bitterness": 0.5},
    "savory":   {"sweetness": 0.2, "sourness": 0.0, "saltiness": 0.5, "bitterness": 0.0},
    "rich":     {"sweetness": 0.5, "sourness": 0.0, "saltiness": 0.3, "bitterness": 0.0},
    "mild":     {"sweetness": 0.3, "sourness": 0.0, "saltiness": 0.0, "bitterness": 0.0},
}

# 大小属性的连续维度(主观评分0-1)
# 体积: big=1.0, huge=1.0, small=0.1, tiny=0.0, 其余中间
# 长度: long=1.0, short=0.0, tall=0.8, wide=0.5
# 重量: heavy=1.0, light=0.0, thick=0.7, thin=0.2
SIZE_DIMS = {
    "big":      {"volume": 1.0, "length": 0.5, "weight": 0.8},
    "small":    {"volume": 0.1, "length": 0.2, "weight": 0.1},
    "tall":     {"volume": 0.6, "length": 0.8, "weight": 0.5},
    "short":    {"volume": 0.2, "length": 0.1, "weight": 0.2},
    "long":     {"volume": 0.5, "length": 1.0, "weight": 0.5},
    "wide":     {"volume": 0.7, "length": 0.5, "weight": 0.6},
    "thin":     {"volume": 0.1, "length": 0.5, "weight": 0.1},
    "thick":    {"volume": 0.6, "length": 0.3, "weight": 0.7},
    "heavy":    {"volume": 0.7, "length": 0.3, "weight": 1.0},
    "light":    {"volume": 0.3, "length": 0.5, "weight": 0.0},
    "huge":     {"volume": 1.0, "length": 0.8, "weight": 1.0},
    "tiny":     {"volume": 0.0, "length": 0.0, "weight": 0.0},
}

# 三种属性类型各720个三元组(60名词×12属性值)
COLOR_TRIPLES = [(n, c, f"{c} {n}") for n in ALL_NOUNS for c in STIMULI["color_attrs"]]
TASTE_TRIPLES = [(n, t, f"{t} {n}") for n in ALL_NOUNS for t in STIMULI["taste_attrs"]]
SIZE_TRIPLES = [(n, s, f"{s} {n}") for n in ALL_NOUNS for s in STIMULI["size_attrs"]]

# 标签
COLOR_LABELS = [(n, "color", c, NOUN_TO_FAMILY.get(n,"unknown")) for n in ALL_NOUNS for c in STIMULI["color_attrs"]]
TASTE_LABELS = [(n, "taste", t, NOUN_TO_FAMILY.get(n,"unknown")) for n in ALL_NOUNS for t in STIMULI["taste_attrs"]]
SIZE_LABELS = [(n, "size", s, NOUN_TO_FAMILY.get(n,"unknown")) for n in ALL_NOUNS for s in STIMULI["size_attrs"]]

ALL_TRIPLES = COLOR_TRIPLES + TASTE_TRIPLES + SIZE_TRIPLES
ALL_LABELS = COLOR_LABELS + TASTE_LABELS + SIZE_LABELS

PROMPT_TEMPLATES_30 = [
    "The {word} is", "A {word} can be", "This {word} has",
    "I saw a {word}", "The {word} was", "My {word} is",
    "That {word} looks", "One {word} might", "Every {word} has",
    "Some {word} are", "Look at the {word}", "The {word} feels",
    "There is a {word}", "I like the {word}", "What a {word}",
    "The {word} seems", "A {word} always", "The {word} became",
    "Many {word} exist", "This {word} could", "The {word} appears",
    "A {word} usually", "The {word} remains", "I found a {word}",
    "The {word} shows", "A {word} makes", "The {word} gives",
    "Such {word} are", "The {word} holds", "A {word} provides",
]

def get_key_layers(n_layers):
    return sorted(set([0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))


# ==================== 数据收集 ====================

def collect_G_terms_large_scale(mdl, tok, device, key_layers, triples, templates):
    all_words = set()
    for noun, attr, combo in triples:
        all_words.add(noun)
        all_words.add(attr)
        all_words.add(combo)
    
    L.log(f"  预计算{len(all_words)}个词 × {len(templates)}个模板...")
    
    word_hs_avg = {}
    done = 0
    total = len(all_words) * len(templates)
    
    for word in sorted(all_words):
        layer_sums = {}
        layer_counts = {}
        for tpl in templates:
            text = tpl.format(word=word)
            ids = tok.encode(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(ids, output_hidden_states=True)
                hs = out.hidden_states
            for l in range(len(hs)):
                vec = hs[l][0, -1, :].detach().float().cpu()
                if l not in layer_sums:
                    layer_sums[l] = vec
                    layer_counts[l] = 1
                else:
                    layer_sums[l] += vec
                    layer_counts[l] += 1
            del out, hs
            done += 1
            if done % 500 == 0:
                L.log(f"    {done}/{total} 完成")
        
        word_hs_avg[word] = {l: layer_sums[l] / layer_counts[l] for l in layer_sums}
        if len(word_hs_avg) % 50 == 0:
            L.log(f"    {len(word_hs_avg)}/{len(all_words)} 词完成")
            gc.collect()
    
    G_dict = {l: [] for l in key_layers}
    for noun, attr, combo in triples:
        for layer in key_layers:
            if layer in word_hs_avg.get(combo, {}) and layer in word_hs_avg.get(noun, {}):
                G = word_hs_avg[combo][layer] - word_hs_avg[noun][layer]
                G_dict[layer].append(G)
    
    del word_hs_avg
    gc.collect()
    return G_dict


# ==================== 图Laplacian ====================

def compute_graph_laplacian_spectra(X_pca, n_neighbors=15, n_eigenvectors=30):
    N, d = X_pca.shape
    if N < 10:
        return None, None
    
    k = min(n_neighbors, N - 1)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_pca)
    distances, indices = nbrs.kneighbors(X_pca)
    
    sigma = np.median(distances[:, 1:]) + 1e-10
    W = np.zeros((N, N))
    for i in range(N):
        for j_idx in range(1, k):
            j = indices[i, j_idx]
            W[i, j] = np.exp(-distances[i, j_idx]**2 / (2 * sigma**2))
            W[j, i] = W[i, j]
    
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(W.sum(axis=1)) + 1e-10))
    L_norm = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    
    n_eig = min(n_eigenvectors, N - 1)
    try:
        eigenvalues, eigenvectors = eigh(L_norm, subset_by_index=[0, n_eig - 1])
    except Exception:
        eigenvalues, eigenvectors = eigh(L_norm)
        eigenvalues = eigenvalues[:n_eig]
        eigenvectors = eigenvectors[:, :n_eig]
    
    return eigenvalues, eigenvectors


# ==================== 名词中心化 ====================

def noun_centered_G(G_matrix, labels):
    """
    名词中心化: G_centered(n,a) = G(n,a) - mean_n(G(n,a))
    对每个名词, 减去该名词在所有属性值上的平均G向量
    """
    N, D = G_matrix.shape
    nouns = [l[0] for l in labels]
    unique_nouns = sorted(set(nouns))
    
    # 计算每个名词的均值
    noun_means = {}
    for n in unique_nouns:
        mask = np.array([nn == n for nn in nouns])
        noun_means[n] = G_matrix[mask].mean(axis=0)
    
    # 中心化
    G_centered = np.zeros_like(G_matrix)
    for i in range(N):
        G_centered[i] = G_matrix[i] - noun_means[nouns[i]]
    
    return G_centered, noun_means


# ==================== P290: 名词中心化后的属性子空间Laplacian谱 ====================

def run_p290(G_dict, key_layers):
    """P290: 名词中心化后的属性子空间Laplacian谱"""
    L.log("P290: 名词中心化后的属性子空间Laplacian谱")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)  # 720
    N_taste = len(TASTE_TRIPLES)  # 720
    N_size = len(SIZE_TRIPLES)    # 720
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_attr, attr_list, labels in [
            ("color", N_color, STIMULI["color_attrs"], COLOR_LABELS),
            ("taste", N_taste, STIMULI["taste_attrs"], TASTE_LABELS),
            ("size", N_size, STIMULI["size_attrs"], SIZE_LABELS),
        ]:
            if attr_type == "color":
                G_sub = G_all[:N_color]
                sub_labels = labels
            elif attr_type == "taste":
                G_sub = G_all[N_color:N_color+N_taste]
                sub_labels = labels
            else:
                G_sub = G_all[N_color+N_taste:]
                sub_labels = labels
            
            N, D = G_sub.shape
            if N < 60:
                continue
            
            # 1) 原始子空间Laplacian
            pca_dim = min(20, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_sub)
            
            eigenvalues_raw, _ = compute_graph_laplacian_spectra(G_pca, n_neighbors=15, n_eigenvectors=25)
            
            # 2) 名词中心化后的Laplacian
            G_centered, noun_means = noun_centered_G(G_sub, sub_labels)
            pca_c = PCA(n_components=pca_dim)
            G_pca_c = pca_c.fit_transform(G_centered)
            
            eigenvalues_c, eigenvectors_c = compute_graph_laplacian_spectra(G_pca_c, n_neighbors=15, n_eigenvectors=25)
            if eigenvalues_c is None:
                continue
            
            cumvar_c = np.cumsum(pca_c.explained_variance_ratio_)
            dim_95_c = int(np.searchsorted(cumvar_c, 0.95) + 1) if len(cumvar_c) > 0 else 0
            
            n_near_zero_raw = int(np.sum(eigenvalues_raw < 0.01)) if eigenvalues_raw is not None else 0
            n_near_zero_c = int(np.sum(eigenvalues_c < 0.01))
            
            # 环绕数
            winding_c = 0.0
            if eigenvectors_c.shape[1] >= 3:
                phi1 = eigenvectors_c[:, 1]
                phi2 = eigenvectors_c[:, 2]
                angles = np.arctan2(phi2, phi1)
                sorted_idx = np.argsort(angles)
                angle_diffs = np.diff(angles[sorted_idx])
                angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
                winding_c = float(np.sum(angle_diffs) / (2 * np.pi))
            
            # 名词方差占比(中心化前后的方差比)
            var_raw = float(np.mean(np.var(G_sub, axis=0)))
            var_centered = float(np.mean(np.var(G_centered, axis=0)))
            var_ratio = var_centered / (var_raw + 1e-10)
            
            layer_result = {
                "layer": layer,
                "N_samples": N,
                "pca_dim": pca_dim,
                "dim_95_centered": dim_95_c,
                "eigenvalues_raw_top5": [round(float(e), 6) for e in eigenvalues_raw[:5]] if eigenvalues_raw is not None else [],
                "eigenvalues_centered_top5": [round(float(e), 6) for e in eigenvalues_c[:5]],
                "n_near_zero_raw": n_near_zero_raw,
                "n_near_zero_centered": n_near_zero_c,
                "winding_centered": round(winding_c, 3),
                "var_ratio_centered_vs_raw": round(var_ratio, 4),
                "pca_cumvar_centered_top5": [round(float(v), 4) for v in cumvar_c[:5]],
            }
            
            results[attr_type].append(layer_result)
            L.log(f"    L{layer} {attr_type}: raw_near_zero={n_near_zero_raw}, "
                  f"centered_near_zero={n_near_zero_c}, winding={winding_c:.3f}, "
                  f"var_ratio={var_ratio:.3f}, dim_95_c={dim_95_c}")
    
    return results


# ==================== P291: 连续编码检测 ====================

def run_p291(G_dict, key_layers):
    """P291: 连续编码检测 — 色相角度cos/sin, 味道维度, 大小维度"""
    L.log("P291: 连续编码检测")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        # === 颜色子空间: 连续色相编码 ===
        G_color = G_all[:N_color]
        G_color_c, _ = noun_centered_G(G_color, COLOR_LABELS)
        
        N_c, D_c = G_color_c.shape
        if N_c >= 60:
            pca_dim = min(20, N_c - 1, D_c)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_color_c)
            
            eigenvalues, eigenvectors = compute_graph_laplacian_spectra(G_pca, n_neighbors=15, n_eigenvectors=25)
            if eigenvalues is not None:
                color_values = [l[2] for l in COLOR_LABELS]
                
                # 构建连续编码变量
                hue_angles_rad = np.array([COLOR_HUE_ANGLES.get(c, 0) * np.pi / 180 for c in color_values])
                cos_hue = np.cos(hue_angles_rad)
                sin_hue = np.sin(hue_angles_rad)
                
                # 亮度维度: white=1.0, black=0.0, 其余0.5
                brightness = np.array([1.0 if c == "white" else 0.0 if c == "black" else 0.5 for c in color_values])
                
                # 饱和度: gray=0, 其余=1
                saturation = np.array([0.0 if c == "gray" else 0.5 if c in ["white","black"] else 1.0 for c in color_values])
                
                # 是否光谱色
                is_spectral = np.array([1.0 if c in SPECTRAL_COLORS else 0.0 for c in color_values])
                
                # 计算每个φ与连续变量的相关
                n_eig = min(20, eigenvectors.shape[1])
                phi_continuous_corrs = []
                for i in range(n_eig):
                    phi = eigenvectors[:, i]
                    corrs = {}
                    for var_name, var_vals in [
                        ("cos_hue", cos_hue), ("sin_hue", sin_hue),
                        ("brightness", brightness), ("saturation", saturation),
                        ("is_spectral", is_spectral),
                    ]:
                        if np.std(phi) > 1e-10 and np.std(var_vals) > 1e-10:
                            corr = float(np.corrcoef(phi, var_vals)[0, 1])
                        else:
                            corr = 0.0
                        corrs[var_name] = round(corr, 4)
                    
                    # 也保留indicator相关(对比)
                    for c in SPECTRAL_COLORS:
                        indicator = np.array([1.0 if v == c else 0.0 for v in color_values])
                        if np.std(phi) > 1e-10 and np.std(indicator) > 1e-10:
                            corrs[f"ind_{c}"] = round(float(np.corrcoef(phi, indicator)[0, 1]), 4)
                    
                    phi_continuous_corrs.append({"phi_index": i, "correlations": corrs})
                
                # 找最强的连续编码
                best_cos_hue = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("cos_hue", 0)))
                best_sin_hue = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("sin_hue", 0)))
                best_brightness = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("brightness", 0)))
                best_saturation = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("saturation", 0)))
                
                color_result = {
                    "layer": layer,
                    "phi_continuous_corrs": phi_continuous_corrs[:10],
                    "best_cos_hue": {"phi": best_cos_hue["phi_index"], "r": best_cos_hue["correlations"]["cos_hue"]},
                    "best_sin_hue": {"phi": best_sin_hue["phi_index"], "r": best_sin_hue["correlations"]["sin_hue"]},
                    "best_brightness": {"phi": best_brightness["phi_index"], "r": best_brightness["correlations"]["brightness"]},
                    "best_saturation": {"phi": best_saturation["phi_index"], "r": best_saturation["correlations"]["saturation"]},
                }
                results["color"].append(color_result)
                L.log(f"    L{layer} color: best_cos_hue=φ{best_cos_hue['phi_index']}(r={best_cos_hue['correlations']['cos_hue']:.3f}), "
                      f"best_sin_hue=φ{best_sin_hue['phi_index']}(r={best_sin_hue['correlations']['sin_hue']:.3f}), "
                      f"best_brightness=φ{best_brightness['phi_index']}(r={best_brightness['correlations']['brightness']:.3f})")
        
        # === 味道子空间: 连续维度编码 ===
        G_taste = G_all[N_color:N_color+N_taste]
        G_taste_c, _ = noun_centered_G(G_taste, TASTE_LABELS)
        
        N_t, D_t = G_taste_c.shape
        if N_t >= 60:
            pca_dim = min(20, N_t - 1, D_t)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_taste_c)
            
            eigenvalues, eigenvectors = compute_graph_laplacian_spectra(G_pca, n_neighbors=15, n_eigenvectors=25)
            if eigenvalues is not None:
                taste_values = [l[2] for l in TASTE_LABELS]
                
                # 构建连续味道维度
                sweetness = np.array([TASTE_DIMS.get(t, {}).get("sweetness", 0) for t in taste_values])
                sourness = np.array([TASTE_DIMS.get(t, {}).get("sourness", 0) for t in taste_values])
                saltiness = np.array([TASTE_DIMS.get(t, {}).get("saltiness", 0) for t in taste_values])
                bitterness = np.array([TASTE_DIMS.get(t, {}).get("bitterness", 0) for t in taste_values])
                
                n_eig = min(15, eigenvectors.shape[1])
                phi_continuous_corrs = []
                for i in range(n_eig):
                    phi = eigenvectors[:, i]
                    corrs = {}
                    for var_name, var_vals in [
                        ("sweetness", sweetness), ("sourness", sourness),
                        ("saltiness", saltiness), ("bitterness", bitterness),
                    ]:
                        if np.std(phi) > 1e-10 and np.std(var_vals) > 1e-10:
                            corr = float(np.corrcoef(phi, var_vals)[0, 1])
                        else:
                            corr = 0.0
                        corrs[var_name] = round(corr, 4)
                    phi_continuous_corrs.append({"phi_index": i, "correlations": corrs})
                
                best_sweet = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("sweetness", 0)))
                best_sour = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("sourness", 0)))
                best_salty = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("saltiness", 0)))
                best_bitter = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("bitterness", 0)))
                
                taste_result = {
                    "layer": layer,
                    "phi_continuous_corrs": phi_continuous_corrs[:10],
                    "best_sweetness": {"phi": best_sweet["phi_index"], "r": best_sweet["correlations"]["sweetness"]},
                    "best_sourness": {"phi": best_sour["phi_index"], "r": best_sour["correlations"]["sourness"]},
                    "best_saltiness": {"phi": best_salty["phi_index"], "r": best_salty["correlations"]["saltiness"]},
                    "best_bitterness": {"phi": best_bitter["phi_index"], "r": best_bitter["correlations"]["bitterness"]},
                }
                results["taste"].append(taste_result)
                L.log(f"    L{layer} taste: sweet=φ{best_sweet['phi_index']}(r={best_sweet['correlations']['sweetness']:.3f}), "
                      f"sour=φ{best_sour['phi_index']}(r={best_sour['correlations']['sourness']:.3f}), "
                      f"salty=φ{best_salty['phi_index']}(r={best_salty['correlations']['saltiness']:.3f})")
        
        # === 大小子空间: 连续维度编码 ===
        G_size = G_all[N_color+N_taste:]
        G_size_c, _ = noun_centered_G(G_size, SIZE_LABELS)
        
        N_s, D_s = G_size_c.shape
        if N_s >= 60:
            pca_dim = min(20, N_s - 1, D_s)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_size_c)
            
            eigenvalues, eigenvectors = compute_graph_laplacian_spectra(G_pca, n_neighbors=15, n_eigenvectors=25)
            if eigenvalues is not None:
                size_values = [l[2] for l in SIZE_LABELS]
                
                volume = np.array([SIZE_DIMS.get(s, {}).get("volume", 0.5) for s in size_values])
                length = np.array([SIZE_DIMS.get(s, {}).get("length", 0.5) for s in size_values])
                weight = np.array([SIZE_DIMS.get(s, {}).get("weight", 0.5) for s in size_values])
                
                n_eig = min(15, eigenvectors.shape[1])
                phi_continuous_corrs = []
                for i in range(n_eig):
                    phi = eigenvectors[:, i]
                    corrs = {}
                    for var_name, var_vals in [
                        ("volume", volume), ("length", length), ("weight", weight),
                    ]:
                        if np.std(phi) > 1e-10 and np.std(var_vals) > 1e-10:
                            corr = float(np.corrcoef(phi, var_vals)[0, 1])
                        else:
                            corr = 0.0
                        corrs[var_name] = round(corr, 4)
                    phi_continuous_corrs.append({"phi_index": i, "correlations": corrs})
                
                best_vol = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("volume", 0)))
                best_len = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("length", 0)))
                best_wt = max(phi_continuous_corrs, key=lambda x: abs(x["correlations"].get("weight", 0)))
                
                size_result = {
                    "layer": layer,
                    "phi_continuous_corrs": phi_continuous_corrs[:10],
                    "best_volume": {"phi": best_vol["phi_index"], "r": best_vol["correlations"]["volume"]},
                    "best_length": {"phi": best_len["phi_index"], "r": best_len["correlations"]["length"]},
                    "best_weight": {"phi": best_wt["phi_index"], "r": best_wt["correlations"]["weight"]},
                }
                results["size"].append(size_result)
                L.log(f"    L{layer} size: volume=φ{best_vol['phi_index']}(r={best_vol['correlations']['volume']:.3f}), "
                      f"length=φ{best_len['phi_index']}(r={best_len['correlations']['length']:.3f}), "
                      f"weight=φ{best_wt['phi_index']}(r={best_wt['correlations']['weight']:.3f})")
    
    return results


# ==================== P292: Procrustes对齐后跨名词平均Laplacian ====================

def run_p292(G_dict, key_layers):
    """P292: Procrustes对齐后跨名词平均Laplacian"""
    L.log("P292: Procrustes对齐后跨名词平均Laplacian")
    results = []
    
    N_color = len(COLOR_TRIPLES)
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        G_color = G_all[:N_color]
        
        # 名词中心化
        G_color_c, noun_means = noun_centered_G(G_color, COLOR_LABELS)
        
        # 对每个名词, 提取其12个颜色样本
        nouns = [l[0] for l in COLOR_LABELS]
        unique_nouns = sorted(set(nouns))
        
        # 每个名词单独做PCA+Laplacian
        noun_embeddings = {}  # noun -> (eigenvalues, eigenvectors)
        noun_pcas = {}
        
        for noun in unique_nouns:
            mask = np.array([nn == noun for nn in nouns])
            G_noun = G_color_c[mask]
            
            if G_noun.shape[0] < 12:
                continue
            
            pca_dim = min(8, G_noun.shape[0] - 1, G_noun.shape[1])
            pca_n = PCA(n_components=pca_dim)
            G_pca_n = pca_n.fit_transform(G_noun)
            
            eigenvalues_n, eigenvectors_n = compute_graph_laplacian_spectra(G_pca_n, n_neighbors=5, n_eigenvectors=min(8, 11))
            if eigenvalues_n is not None:
                noun_embeddings[noun] = (eigenvalues_n, eigenvectors_n)
                noun_pcas[noun] = pca_n
        
        if len(noun_embeddings) < 5:
            L.log(f"    L{layer}: 仅有{len(noun_embeddings)}个名词有有效嵌入, 跳过")
            continue
        
        # Procrustes对齐: 以第一个名词为参考
        ref_noun = list(noun_embeddings.keys())[0]
        ref_eigvals, ref_eigvecs = noun_embeddings[ref_noun]
        
        aligned_eigvecs = [ref_eigvecs]
        alignment_errors = []
        
        for noun in list(noun_embeddings.keys())[1:]:
            _, eigvecs = noun_embeddings[noun]
            # 对齐到参考: Procrustes
            min_cols = min(ref_eigvecs.shape[1], eigvecs.shape[1])
            try:
                mtx1, mtx2, disparity = procrustes(ref_eigvecs[:, :min_cols], eigvecs[:, :min_cols])
                aligned_eigvecs.append(mtx2)
                alignment_errors.append(float(disparity))
            except Exception:
                pass
        
        if len(aligned_eigvecs) < 3:
            continue
        
        # 平均对齐后的特征向量
        min_rows = min(e.shape[0] for e in aligned_eigvecs)
        min_cols = min(e.shape[1] for e in aligned_eigvecs)
        avg_eigvecs = np.zeros((min_rows, min_cols))
        for e in aligned_eigvecs:
            avg_eigvecs += e[:min_rows, :min_cols]
        avg_eigvecs /= len(aligned_eigvecs)
        
        # 分析平均特征向量与色相角度的相关
        color_values = STIMULI["color_attrs"]
        hue_angles_rad = np.array([COLOR_HUE_ANGLES.get(c, 0) * np.pi / 180 for c in color_values])
        cos_hue = np.cos(hue_angles_rad)
        sin_hue = np.sin(hue_angles_rad)
        brightness = np.array([1.0 if c == "white" else 0.0 if c == "black" else 0.5 for c in color_values])
        
        phi_corrs = []
        for i in range(min(8, min_cols)):
            phi = avg_eigvecs[:, i]
            corrs = {}
            for var_name, var_vals in [("cos_hue", cos_hue), ("sin_hue", sin_hue), ("brightness", brightness)]:
                if np.std(phi) > 1e-10 and np.std(var_vals) > 1e-10:
                    corr = float(np.corrcoef(phi, var_vals)[0, 1])
                else:
                    corr = 0.0
                corrs[var_name] = round(corr, 4)
            phi_corrs.append({"phi_index": i, "correlations": corrs})
        
        best_cos = max(phi_corrs, key=lambda x: abs(x["correlations"].get("cos_hue", 0)))
        best_sin = max(phi_corrs, key=lambda x: abs(x["correlations"].get("sin_hue", 0)))
        best_bright = max(phi_corrs, key=lambda x: abs(x["correlations"].get("brightness", 0)))
        
        layer_result = {
            "layer": layer,
            "n_nouns_aligned": len(aligned_eigvecs),
            "mean_alignment_error": round(float(np.mean(alignment_errors)), 4),
            "phi_corrs": phi_corrs,
            "best_cos_hue": {"phi": best_cos["phi_index"], "r": best_cos["correlations"]["cos_hue"]},
            "best_sin_hue": {"phi": best_sin["phi_index"], "r": best_sin["correlations"]["sin_hue"]},
            "best_brightness": {"phi": best_bright["phi_index"], "r": best_bright["correlations"]["brightness"]},
        }
        results.append(layer_result)
        L.log(f"    L{layer}: {len(aligned_eigvecs)} nouns aligned, "
              f"alignment_err={np.mean(alignment_errors):.4f}, "
              f"cos_hue=φ{best_cos['phi_index']}(r={best_cos['correlations']['cos_hue']:.3f}), "
              f"sin_hue=φ{best_sin['phi_index']}(r={best_sin['correlations']['sin_hue']:.3f}), "
              f"brightness=φ{best_bright['phi_index']}(r={best_bright['correlations']['brightness']:.3f})")
    
    return results


# ==================== P293: 纯属性流形类型 ====================

def run_p293(G_dict, key_layers):
    """P293: 名词中心化后的属性子空间流形类型"""
    L.log("P293: 纯属性流形类型判断")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_attr, labels in [
            ("color", N_color, COLOR_LABELS),
            ("taste", N_taste, TASTE_LABELS),
            ("size", N_size, SIZE_LABELS),
        ]:
            if attr_type == "color":
                G_sub = G_all[:N_color]
            elif attr_type == "taste":
                G_sub = G_all[N_color:N_color+N_taste]
            else:
                G_sub = G_all[N_color+N_taste:]
            
            N, D = G_sub.shape
            if N < 60:
                continue
            
            # 名词中心化
            G_centered, _ = noun_centered_G(G_sub, labels)
            
            # 全局PCA
            pca_dim = min(20, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
            
            # Laplacian谱
            eigenvalues, eigenvectors = compute_graph_laplacian_spectra(G_pca, n_neighbors=15, n_eigenvectors=25)
            if eigenvalues is None:
                continue
            
            n_near_zero = int(np.sum(eigenvalues < 0.01))
            
            # 环绕数(φ1-φ2平面)
            winding = 0.0
            if eigenvectors.shape[1] >= 3:
                phi1 = eigenvectors[:, 1]
                phi2 = eigenvectors[:, 2]
                angles = np.arctan2(phi2, phi1)
                sorted_idx = np.argsort(angles)
                angle_diffs = np.diff(angles[sorted_idx])
                angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
                winding = float(np.sum(angle_diffs) / (2 * np.pi))
            
            # 固定名词后, 对每个名词单独计算环绕数
            nouns = [l[0] for l in labels]
            unique_nouns = sorted(set(nouns))
            
            per_noun_windings = []
            per_noun_dim95 = []
            per_noun_closing_ratios = []
            
            for noun in unique_nouns:
                mask = np.array([nn == noun for nn in nouns])
                G_noun = G_centered[mask]
                
                if G_noun.shape[0] < 12:
                    continue
                
                pca_n = PCA(n_components=min(8, G_noun.shape[0] - 1, D))
                G_pca_n = pca_n.fit_transform(G_noun)
                cumvar_n = np.cumsum(pca_n.explained_variance_ratio_)
                dim_95_n = int(np.searchsorted(cumvar_n, 0.95) + 1) if len(cumvar_n) > 0 else 0
                per_noun_dim95.append(dim_95_n)
                
                # 环绕数(按属性值排列)
                attr_values = [l[2] for l in labels if l[0] == noun]
                # 颜色按色相排列
                if attr_type == "color":
                    # 按色相角度排序
                    hue_order = {c: i for i, c in enumerate(COLOR_HUE_ORDER)}
                    sorted_indices = sorted(range(len(attr_values)), key=lambda j: hue_order.get(attr_values[j], 99))
                elif attr_type == "taste":
                    # 按甜度排序
                    sorted_indices = sorted(range(len(attr_values)), key=lambda j: -TASTE_DIMS.get(attr_values[j], {}).get("sweetness", 0))
                else:
                    # 按体积排序
                    sorted_indices = sorted(range(len(attr_values)), key=lambda j: -SIZE_DIMS.get(attr_values[j], {}).get("volume", 0.5))
                
                G_sorted = G_pca_n[sorted_indices]
                
                # 计算闭合比: 首尾距离 / 平均步长
                if G_sorted.shape[0] >= 3:
                    dists = [np.linalg.norm(G_sorted[i+1] - G_sorted[i]) for i in range(G_sorted.shape[0] - 1)]
                    mean_step = float(np.mean(dists)) if dists else 1.0
                    closing_dist = float(np.linalg.norm(G_sorted[-1] - G_sorted[0]))
                    closing_ratio = closing_dist / (mean_step + 1e-10)
                    per_noun_closing_ratios.append(round(closing_ratio, 3))
                
                # Laplacian环绕数(12个样本)
                if G_noun.shape[0] >= 12:
                    eigs_n, eigvecs_n = compute_graph_laplacian_spectra(G_pca_n, n_neighbors=5, n_eigenvectors=8)
                    if eigs_n is not None and eigvecs_n.shape[1] >= 3:
                        phi1_n = eigvecs_n[:, 1]
                        phi2_n = eigvecs_n[:, 2]
                        angles_n = np.arctan2(phi2_n, phi1_n)
                        sorted_idx_n = np.argsort(angles_n)
                        angle_diffs_n = np.diff(angles_n[sorted_idx_n])
                        angle_diffs_n = (angle_diffs_n + np.pi) % (2 * np.pi) - np.pi
                        w_n = float(np.sum(angle_diffs_n) / (2 * np.pi))
                        per_noun_windings.append(round(w_n, 3))
            
            # 判断流形类型
            if n_near_zero >= 3:
                manifold_type = "torus"
            elif n_near_zero >= 2 and winding > 0.5:
                manifold_type = "torus_like"
            elif winding > 0.8:
                manifold_type = "partial_torus"
            else:
                manifold_type = "euclidean"
            
            layer_result = {
                "layer": layer,
                "dim_95_centered": dim_95,
                "n_near_zero_centered": n_near_zero,
                "winding_centered": round(winding, 3),
                "manifold_type_centered": manifold_type,
                "per_noun_mean_dim95": round(float(np.mean(per_noun_dim95)), 2) if per_noun_dim95 else 0,
                "per_noun_mean_winding": round(float(np.mean(per_noun_windings)), 3) if per_noun_windings else 0,
                "per_noun_mean_closing_ratio": round(float(np.mean(per_noun_closing_ratios)), 3) if per_noun_closing_ratios else 0,
                "per_noun_dim95_list": per_noun_dim95[:10],
                "per_noun_winding_list": per_noun_windings[:10],
            }
            
            results[attr_type].append(layer_result)
            L.log(f"    L{layer} {attr_type}: type={manifold_type}, n_near_zero={n_near_zero}, "
                  f"winding={winding:.3f}, per_noun_dim95={np.mean(per_noun_dim95):.1f}, "
                  f"per_noun_winding={np.mean(per_noun_windings):.3f}, "
                  f"per_noun_closing_ratio={np.mean(per_noun_closing_ratios):.3f}")
    
    return results


# ==================== P294: 纯属性重构与语义公式 ====================

def run_p294(G_dict, key_layers):
    """P294: 纯属性重构与G项显式数学公式"""
    L.log("P294: 纯属性重构与语义公式")
    results = []
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_all = N_color + N_taste + N_size
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        # ====== 方法1: PCA30(参考) ======
        pca30 = PCA(n_components=30)
        G_pca30 = pca30.fit_transform(G_all)
        G_recon_pca30 = pca30.inverse_transform(G_pca30)
        
        cos_pca30 = []
        for i in range(N_all):
            g_orig = G_all[i]
            g_recon = G_recon_pca30[i]
            n1 = np.linalg.norm(g_orig)
            n2 = np.linalg.norm(g_recon)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_pca30.append(float(np.dot(g_orig, g_recon) / (n1 * n2)))
        mean_cos_pca30 = float(np.mean(cos_pca30))
        
        # ====== 方法2: 分层重构 ======
        # G(n,a) = f_base(n) + f_color(θ) + f_taste(φ) + f_size(s)
        # f_base(n) = mean over all attributes for noun n
        
        # 计算f_base
        color_labels = COLOR_LABELS
        taste_labels = TASTE_LABELS
        size_labels = SIZE_LABELS
        
        nouns_all = [l[0] for l in ALL_LABELS]
        attr_types = [l[1] for l in ALL_LABELS]
        attr_values = [l[2] for l in ALL_LABELS]
        families = [l[3] for l in ALL_LABELS]
        unique_nouns = sorted(set(nouns_all))
        
        # f_base: 每个名词的平均G
        noun_base = {}
        for noun in unique_nouns:
            mask = np.array([nn == noun for nn in nouns_all])
            noun_base[noun] = G_all[mask].mean(axis=0)
        
        # 残差: G - f_base
        G_residual = np.zeros_like(G_all)
        for i in range(N_all):
            G_residual[i] = G_all[i] - noun_base[nouns_all[i]]
        
        # 对每个属性类型的残差做PCA重构
        G_recon_hierarchical = np.zeros_like(G_all)
        
        for attr_type, n_start, n_end, labels in [
            ("color", 0, N_color, color_labels),
            ("taste", N_color, N_color + N_taste, taste_labels),
            ("size", N_color + N_taste, N_all, size_labels),
        ]:
            G_sub = G_residual[n_start:n_end]
            N_sub = G_sub.shape[0]
            
            # PCA重构残差
            pca_sub = PCA(n_components=min(15, N_sub - 1, G_sub.shape[1]))
            G_pca_sub = pca_sub.fit_transform(G_sub)
            G_recon_sub = pca_sub.inverse_transform(G_pca_sub)
            
            # 加回f_base
            sub_nouns = [l[0] for l in labels]
            for i in range(N_sub):
                G_recon_hierarchical[n_start + i] = noun_base[sub_nouns[i]] + G_recon_sub[i]
        
        cos_hierarchical = []
        for i in range(N_all):
            g_orig = G_all[i]
            g_recon = G_recon_hierarchical[i]
            n1 = np.linalg.norm(g_orig)
            n2 = np.linalg.norm(g_recon)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_hierarchical.append(float(np.dot(g_orig, g_recon) / (n1 * n2)))
        mean_cos_hierarchical = float(np.mean(cos_hierarchical))
        
        # ====== 方法3: 名词中心化+Laplacian近邻重构 ======
        G_centered, _ = noun_centered_G(G_all, ALL_LABELS)
        
        pca_c = PCA(n_components=min(30, N_all - 1, G_all.shape[1]))
        G_pca_c = pca_c.fit_transform(G_centered)
        
        eigenvalues_c, eigenvectors_c = compute_graph_laplacian_spectra(G_pca_c, n_neighbors=15, n_eigenvectors=30)
        
        cos_laplacian_centered = 0.0
        if eigenvalues_c is not None:
            k = 15
            if k < eigenvectors_c.shape[1]:
                Phi_k = eigenvectors_c[:, 1:k+1]
                nn_model = NearestNeighbors(n_neighbors=min(10, N_all - 1), algorithm='ball_tree').fit(Phi_k)
                _, nn_indices = nn_model.kneighbors(Phi_k)
                
                G_recon_c = np.zeros_like(G_pca_c)
                for j in range(N_all):
                    neighbors = nn_indices[j]
                    G_recon_c[j] = np.mean(G_pca_c[neighbors], axis=0)
                
                # 加回noun mean
                G_recon_laplacian = pca_c.inverse_transform(G_recon_c)
                for i in range(N_all):
                    G_recon_laplacian[i] += (G_all[i] - G_centered[i])  # 加回noun mean
                
                cos_lc = []
                for i in range(N_all):
                    g_orig = G_all[i]
                    g_recon = G_recon_laplacian[i]
                    n1 = np.linalg.norm(g_orig)
                    n2 = np.linalg.norm(g_recon)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_lc.append(float(np.dot(g_orig, g_recon) / (n1 * n2)))
                cos_laplacian_centered = float(np.mean(cos_lc))
        
        # ====== 方法4: 纯名词重构(只用f_base) ======
        cos_base_only = []
        for i in range(N_all):
            g_orig = G_all[i]
            g_recon = noun_base[nouns_all[i]]
            n1 = np.linalg.norm(g_orig)
            n2 = np.linalg.norm(g_recon)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_base_only.append(float(np.dot(g_orig, g_recon) / (n1 * n2)))
        mean_cos_base = float(np.mean(cos_base_only))
        
        # ====== 残差方差分析 ======
        var_total = float(np.mean(np.var(G_all, axis=0)))
        var_base = float(np.mean(np.var(np.stack([noun_base[n] for n in nouns_all]), axis=0)))
        var_residual = float(np.mean(np.var(G_residual, axis=0)))
        
        # 每种属性类型的残差方差
        var_residual_by_type = {}
        for attr_type, n_start, n_end in [("color", 0, N_color), ("taste", N_color, N_color+N_taste), ("size", N_color+N_taste, N_all)]:
            var_residual_by_type[attr_type] = round(float(np.mean(np.var(G_residual[n_start:n_end], axis=0))), 6)
        
        layer_result = {
            "layer": layer,
            "cos_pca30": round(mean_cos_pca30, 4),
            "cos_hierarchical": round(mean_cos_hierarchical, 4),
            "cos_laplacian_centered": round(cos_laplacian_centered, 4),
            "cos_base_only": round(mean_cos_base, 4),
            "var_total": round(var_total, 6),
            "var_base": round(var_base, 6),
            "var_residual": round(var_residual, 6),
            "var_ratio_base_vs_total": round(var_base / (var_total + 1e-10), 4),
            "var_residual_by_type": var_residual_by_type,
        }
        results.append(layer_result)
        L.log(f"    L{layer}: cos_pca30={mean_cos_pca30:.3f}, cos_hierarchical={mean_cos_hierarchical:.3f}, "
              f"cos_laplacian_centered={cos_laplacian_centered:.3f}, cos_base_only={mean_cos_base:.3f}")
        L.log(f"           var_ratio(base/total)={var_base/(var_total+1e-10):.3f}, "
              f"var_residual_by_type={var_residual_by_type}")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"======== Phase L: 去除名词影响的纯属性谱分析 ========")
    L.log(f"模型: {model_name}")
    
    # 加载模型
    L.log(f"加载模型 {model_name}...")
    mdl, tok, device = load_model(model_name)
    n_layers = mdl.config.num_hidden_layers
    key_layers = get_key_layers(n_layers)
    L.log(f"模型: {model_name}, {n_layers}层, 关键层={key_layers}")
    
    # 数据收集
    L.log(f"数据收集: {len(ALL_TRIPLES)}三元组 × {len(PROMPT_TEMPLATES_30)}模板...")
    G_dict = collect_G_terms_large_scale(mdl, tok, device, key_layers, ALL_TRIPLES, PROMPT_TEMPLATES_30)
    
    for layer in key_layers:
        if layer in G_dict:
            L.log(f"  L{layer}: {len(G_dict[layer])} G向量")
    
    # 释放模型内存
    del mdl, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    L.log("模型已释放, 开始分析...")
    
    # 运行五大实验
    all_results = {"model": model_name, "n_layers": n_layers, "key_layers": key_layers,
                   "N_triples": len(ALL_TRIPLES), "N_templates": len(PROMPT_TEMPLATES_30)}
    
    L.log("\n====== P290: 名词中心化后的属性子空间Laplacian谱 ======")
    all_results["P290"] = run_p290(G_dict, key_layers)
    
    L.log("\n====== P291: 连续编码检测 ======")
    all_results["P291"] = run_p291(G_dict, key_layers)
    
    L.log("\n====== P292: Procrustes对齐后跨名词平均Laplacian ======")
    all_results["P292"] = run_p292(G_dict, key_layers)
    
    L.log("\n====== P293: 纯属性流形类型 ======")
    all_results["P293"] = run_p293(G_dict, key_layers)
    
    L.log("\n====== P294: 纯属性重构与语义公式 ======")
    all_results["P294"] = run_p294(G_dict, key_layers)
    
    # 保存结果
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_file = OUT_DIR / f"phase_l_p290_294_{model_name}_{ts}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    L.log(f"\n结果已保存: {out_file}")
    
    # 打印摘要
    L.log("\n======== 摘要 ========")
    L.log(f"模型: {model_name}")
    
    # P290摘要
    for attr_type in ["color", "taste", "size"]:
        if all_results["P290"][attr_type]:
            last = all_results["P290"][attr_type][-1]
            L.log(f"P290 {attr_type}: var_ratio={last.get('var_ratio_centered_vs_raw','?')}, "
                  f"centered_near_zero={last.get('n_near_zero_centered','?')}, "
                  f"winding={last.get('winding_centered','?')}")
    
    # P291摘要
    for attr_type in ["color", "taste", "size"]:
        if all_results["P291"][attr_type]:
            last = all_results["P291"][attr_type][-1]
            if attr_type == "color":
                L.log(f"P291 {attr_type}: cos_hue=φ{last['best_cos_hue']['phi']}(r={last['best_cos_hue']['r']}), "
                      f"sin_hue=φ{last['best_sin_hue']['phi']}(r={last['best_sin_hue']['r']}), "
                      f"brightness=φ{last['best_brightness']['phi']}(r={last['best_brightness']['r']})")
            elif attr_type == "taste":
                L.log(f"P291 {attr_type}: sweet=φ{last['best_sweetness']['phi']}(r={last['best_sweetness']['r']}), "
                      f"sour=φ{last['best_sourness']['phi']}(r={last['best_sourness']['r']}), "
                      f"salty=φ{last['best_saltiness']['phi']}(r={last['best_saltiness']['r']})")
            else:
                L.log(f"P291 {attr_type}: volume=φ{last['best_volume']['phi']}(r={last['best_volume']['r']}), "
                      f"length=φ{last['best_length']['phi']}(r={last['best_length']['r']}), "
                      f"weight=φ{last['best_weight']['phi']}(r={last['best_weight']['r']})")
    
    # P292摘要
    if all_results["P292"]:
        last = all_results["P292"][-1]
        L.log(f"P292: {last['n_nouns_aligned']} nouns, alignment_err={last['mean_alignment_error']}, "
              f"cos_hue=φ{last['best_cos_hue']['phi']}(r={last['best_cos_hue']['r']}), "
              f"sin_hue=φ{last['best_sin_hue']['phi']}(r={last['best_sin_hue']['r']})")
    
    # P293摘要
    for attr_type in ["color", "taste", "size"]:
        if all_results["P293"][attr_type]:
            last = all_results["P293"][attr_type][-1]
            L.log(f"P293 {attr_type}: type={last['manifold_type_centered']}, "
                  f"per_noun_winding={last['per_noun_mean_winding']}, "
                  f"per_noun_closing_ratio={last['per_noun_mean_closing_ratio']}")
    
    # P294摘要
    if all_results["P294"]:
        last = all_results["P294"][-1]
        L.log(f"P294: pca30={last['cos_pca30']}, hierarchical={last['cos_hierarchical']}, "
              f"laplacian_centered={last['cos_laplacian_centered']}, base_only={last['cos_base_only']}")
    
    L.close()


if __name__ == "__main__":
    main()
