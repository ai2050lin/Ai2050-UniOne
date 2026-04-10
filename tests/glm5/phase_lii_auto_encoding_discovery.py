"""
Phase LII-P300/301/302/303/304: 自动发现编码维度 — CCA/Kernel-CCA/Sparse-CCA
======================================================================

Phase LI关键瓶颈:
  1. 人工语义维度R²仅0.28-0.52: 设计矩阵不完整
  2. 非线性改善仅0.003: 属性编码基本线性
  3. 弱信号(饱和度/苦度/umami)仍未检测到
  4. sin_hue不显著: 色相环编码不是标准cos/sin
  5. 属性在G项中是"小头": 纯属性cos仅0.40-0.56

核心洞察: R²低不是回归方法问题, 而是"人工编码偏见"——
  我们预设的sweetness/sourness/saltiness等维度只是模型编码的粗粒度近似。
  模型内部可能有完全不同的编码体系!

Phase LII核心改进: 从"外部映射"转向"内部对齐"
  1. 线性CCA (Canonical Correlation Analysis):
     - 不预设语义维度, 用G_centered与属性indicator的CCA
     - 发现模型内部的真实编码轴(本征语义轴)
     - X = G_centered (N×D), Y = indicator矩阵 (N×K)
     - CCA找X和Y的最大相关方向
  2. Kernel-CCA (非线性CCA):
     - 用RBF核将G_centered映射到高维空间
     - 发现非线性编码结构(拓扑干涉)
     - 如果kernel-CCA >> 线性CCA → 找到了"最后50%"
  3. Sparse-CCA (稀疏CCA):
     - L1约束使编码轴稀疏
     - 过滤随机扰动, 锁定核心编码轴
     - 可解释性: 每个CCA方向只依赖少数原始维度
  4. PLS (偏最小二乘):
     - 对协方差建模, 比回归更容易捕捉弱信号
     - 方差虽小但逻辑重要的维度(如苦度)

五大实验:
  P300: 线性CCA — 发现模型本征语义轴
    - X=G_centered, Y=属性indicator矩阵
    - 比较CCA维度 vs 人工定义维度的R²
    - 5折交叉验证泛化性

  P301: Kernel-CCA — 发现非线性编码结构
    - RBF核 + 线性核组合
    - 比较kernel-CCA vs 线性CCA的重构质量
    - 如果kernel-CCA >> 线性CCA → 非线性拓扑干涉存在

  P302: Sparse-CCA — 稀疏编码轴发现
    - L1正则化约束
    - 分析每个CCA方向的稀疏模式
    - 锁定核心编码维度, 过滤噪声

  P303: PLS回归 — 弱信号增强
    - PLS对协方差建模
    - 检测苦度/饱和度/umami等弱信号
    - 比较PLS vs CCA的弱信号检测能力

  P304: 综合比较 — 最优编码策略
    - 比较所有方法: Ridge/CCA/Kernel-CCA/Sparse-CCA/PLS
    - 选择最优策略, 给出G项最终公式
    - 分析模型本征语义轴与人类语义维度的关系

数据规模: 2160三元组(与Phase L/LI相同) × 30模板
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_lii_log.txt"

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

# ===================== 数据集定义 =====================
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

ALL_NOUNS = []
for fam in ["fruit_family","animal_family","vehicle_family","furniture_family","tool_family"]:
    ALL_NOUNS.extend(STIMULI[fam])

NOUN_TO_FAMILY = {}
for fam, words in STIMULI.items():
    if "family" in fam:
        for w in words:
            NOUN_TO_FAMILY[w] = fam

FAMILY_NAMES = ["fruit_family","animal_family","vehicle_family","furniture_family","tool_family"]
FAMILY_NOUNS = {fam: STIMULI[fam] for fam in FAMILY_NAMES}

# 三种属性类型各720个三元组
COLOR_TRIPLES = [(n, c, f"{c} {n}") for n in ALL_NOUNS for c in STIMULI["color_attrs"]]
TASTE_TRIPLES = [(n, t, f"{t} {n}") for n in ALL_NOUNS for t in STIMULI["taste_attrs"]]
SIZE_TRIPLES = [(n, s, f"{s} {n}") for n in ALL_NOUNS for s in STIMULI["size_attrs"]]

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


# ==================== 名词中心化 ====================

def noun_centered_G(G_matrix, labels):
    N, D = G_matrix.shape
    nouns = [l[0] for l in labels]
    unique_nouns = sorted(set(nouns))
    noun_means = {}
    for n in unique_nouns:
        mask = np.array([nn == n for nn in nouns])
        noun_means[n] = G_matrix[mask].mean(axis=0)
    G_centered = np.zeros_like(G_matrix)
    for i in range(N):
        G_centered[i] = G_matrix[i] - noun_means[nouns[i]]
    return G_centered, noun_means


# ==================== 构建属性Indicator矩阵 ====================

def build_indicator_matrix(labels, attr_type):
    """构建属性indicator矩阵 Y (N×K), K=12个属性值"""
    attr_values = [l[2] for l in labels]
    if attr_type == "color":
        attr_list = STIMULI["color_attrs"]
    elif attr_type == "taste":
        attr_list = STIMULI["taste_attrs"]
    else:
        attr_list = STIMULI["size_attrs"]
    
    N = len(attr_values)
    K = len(attr_list)
    Y = np.zeros((N, K))
    for i, v in enumerate(attr_values):
        if v in attr_list:
            Y[i, attr_list.index(v)] = 1.0
    return Y, attr_list


# ==================== P300: 线性CCA ====================

def run_p300(G_dict, labels_dict, key_layers, model_name):
    """线性CCA: 发现模型本征语义轴"""
    L.log("=== P300: 线性CCA — 发现模型本征语义轴 ===")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_total = N_color + N_taste + N_size
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_start, n_end, labels in [
            ("color", 0, N_color, COLOR_LABELS),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS),
            ("size", N_color + N_taste, N_total, SIZE_LABELS),
        ]:
            G_sub = G_all[n_start:n_end]
            N, D = G_sub.shape
            if N < 60:
                continue
            
            # 名词中心化
            G_centered, noun_means = noun_centered_G(G_sub, labels)
            
            # PCA降维到30维 (CCA要求dim < N)
            pca_dim = min(30, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # 构建indicator矩阵
            Y, attr_list = build_indicator_matrix(labels, attr_type)
            
            # CCA: 找G_pca和Y的最大相关方向
            n_cca = min(10, pca_dim - 1, Y.shape[1] - 1)
            cca = CCA(n_components=n_cca, max_iter=500)
            
            try:
                G_cca, Y_cca = cca.fit_transform(G_pca, Y)
            except Exception as e:
                L.log(f"    L{layer} {attr_type}: CCA failed: {e}")
                continue
            
            # CCA相关系数
            cca_corrs = []
            for i in range(n_cca):
                r = float(np.corrcoef(G_cca[:, i], Y_cca[:, i])[0, 1])
                cca_corrs.append(round(abs(r), 4))
            
            # 用CCA编码方向重构G_centered
            # G_centered ≈ G_cca @ cca.x_weights_.T @ pca.components_
            G_recon = cca.inverse_transform(G_cca)  # 对G_pca的重建
            G_centered_recon = pca.inverse_transform(G_recon)
            
            # 重构余弦
            cos_vals = []
            for i in range(N):
                n1 = np.linalg.norm(G_centered[i])
                n2 = np.linalg.norm(G_centered_recon[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_vals.append(float(np.dot(G_centered[i], G_centered_recon[i]) / (n1 * n2)))
            mean_cos_centered = float(np.mean(cos_vals)) if cos_vals else 0.0
            
            # R²(中心化后)
            ss_res = np.sum((G_centered - G_centered_recon)**2)
            ss_tot = np.sum((G_centered - G_centered.mean(axis=0))**2)
            r2_centered = 1 - ss_res / (ss_tot + 1e-10)
            
            # 包含f_base的完整重构
            G_full_pred = np.zeros_like(G_sub)
            nouns = [l[0] for l in labels]
            for i in range(N):
                G_full_pred[i] = noun_means[nouns[i]] + G_centered_recon[i]
            
            cos_full = []
            for i in range(N):
                n1 = np.linalg.norm(G_sub[i])
                n2 = np.linalg.norm(G_full_pred[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_full.append(float(np.dot(G_sub[i], G_full_pred[i]) / (n1 * n2)))
            mean_cos_full = float(np.mean(cos_full)) if cos_full else 0.0
            
            # 分析CCA权重: 哪些G_pca维度被CCA选中
            x_weights_abs = np.abs(cca.x_weights_)  # (pca_dim, n_cca)
            top_dims_per_cca = []
            for c in range(n_cca):
                top_idx = np.argsort(x_weights_abs[:, c])[::-1][:5]
                top_dims_per_cca.append([(int(idx), round(float(x_weights_abs[idx, c]), 4)) for idx in top_idx])
            
            # 分析CCA权重: 哪些属性维度被CCA选中
            y_weights_abs = np.abs(cca.y_weights_)  # (K, n_cca)
            top_attrs_per_cca = []
            for c in range(n_cca):
                top_idx = np.argsort(y_weights_abs[:, c])[::-1][:3]
                top_attrs_per_cca.append([(attr_list[int(idx)], round(float(y_weights_abs[idx, c]), 4)) for idx in top_idx])
            
            layer_result = {
                "layer": layer,
                "n_cca": n_cca,
                "cca_correlations": cca_corrs,
                "r2_centered": round(float(r2_centered), 4),
                "cos_centered": round(mean_cos_centered, 4),
                "cos_full": round(mean_cos_full, 4),
                "pca_cumvar_top5": [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)[:5]],
                "top_g_dims_per_cca": [[(d, w) for d, w in comp] for comp in top_dims_per_cca[:5]],
                "top_attrs_per_cca": [[(a, w) for a, w in comp] for comp in top_attrs_per_cca[:5]],
            }
            results[attr_type].append(layer_result)
            
            top_cca = cca_corrs[:3]
            L.log(f"    L{layer} {attr_type}: R²={r2_centered:.3f}, cos_c={mean_cos_centered:.3f}, "
                  f"cos_f={mean_cos_full:.3f}, cca_r={top_cca}")
    
    return results


# ==================== Kernel-CCA实现 ====================

def kernel_cca(X, Y, n_components=10, kernel_x='rbf', kernel_y='linear', 
               gamma_x=None, reg=1e-3):
    """
    Kernel CCA: 非线性CCA
    
    1. 计算核矩阵 Kx = RBF(X), Ky = Linear(Y)
    2. 正则化: Kx_r = Kx + reg*I, Ky_r = Ky + reg*I
    3. 广义特征问题: Kx_r^{-1} Ky_r Kx_r^{-1} Ky_r v = ρ² v
    4. 简化: 用SVD分解
    """
    N = X.shape[0]
    n_comp = min(n_components, N - 1)
    
    # 计算核矩阵
    if kernel_x == 'rbf':
        if gamma_x is None:
            dists = pdist(X, 'euclidean')
            gamma_x = 1.0 / (2 * float(np.median(dists))**2 + 1e-10)
        dist_matrix = squareform(pdist(X, 'euclidean'))
        Kx = np.exp(-gamma_x * dist_matrix**2)
    else:
        Kx = X @ X.T
    
    if kernel_y == 'linear':
        Ky = Y @ Y.T
    else:
        # RBF核 for Y
        if gamma_x is None:
            dists_y = pdist(Y, 'euclidean')
            gamma_y = 1.0 / (2 * float(np.median(dists_y))**2 + 1e-10) if len(dists_y) > 0 else 1.0
        else:
            gamma_y = gamma_x
        dist_y = squareform(pdist(Y, 'euclidean'))
        Ky = np.exp(-gamma_y * dist_y**2)
    
    # 中心化核矩阵
    H = np.eye(N) - np.ones((N, N)) / N
    Kx_c = H @ Kx @ H
    Ky_c = H @ Ky @ H
    
    # 正则化
    Kx_r = Kx_c + reg * np.eye(N)
    Ky_r = Ky_c + reg * np.eye(N)
    
    # 白化: Kx_r^{-1/2} Ky_r Kx_r^{-1/2}
    try:
        eigvals_x, eigvecs_x = eigh(Kx_r)
        eigvals_x = np.maximum(eigvals_x, 1e-10)
        Kx_inv_sqrt = eigvecs_x @ np.diag(1.0 / np.sqrt(eigvals_x)) @ eigvecs_x.T
        
        M = Kx_inv_sqrt @ Ky_r @ Kx_inv_sqrt
        
        eigvals, eigvecs = eigh(M)
        # 从大到小
        eigvals = eigvals[::-1][:n_comp]
        eigvecs = eigvecs[:, ::-1][:, :n_comp]
        
        # CCA相关 = sqrt(eigenvalue)
        correlations = np.sqrt(np.maximum(eigvals, 0))
        
        # 投影
        alpha = Kx_inv_sqrt @ eigvecs  # X空间的投影方向
        
        X_proj = Kx_c @ alpha  # 核空间投影
        Y_proj = Ky_c @ (Ky_r @ alpha / (eigvals[np.newaxis, :] + reg))  # 近似Y投影
        
    except Exception as e:
        L.log(f"    Kernel-CCA eigen decomposition failed: {e}")
        return None, None, None, None
    
    return correlations, X_proj, Y_proj, alpha


# ==================== P301: Kernel-CCA ====================

def run_p301(G_dict, labels_dict, key_layers, model_name):
    """Kernel-CCA: 发现非线性编码结构"""
    L.log("=== P301: Kernel-CCA — 发现非线性编码结构 ===")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_total = N_color + N_taste + N_size
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_start, n_end, labels in [
            ("color", 0, N_color, COLOR_LABELS),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS),
            ("size", N_color + N_taste, N_total, SIZE_LABELS),
        ]:
            G_sub = G_all[n_start:n_end]
            N, D = G_sub.shape
            if N < 60:
                continue
            
            # 名词中心化
            G_centered, noun_means = noun_centered_G(G_sub, labels)
            
            # PCA降维到30维
            pca_dim = min(30, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # 构建indicator矩阵
            Y, attr_list = build_indicator_matrix(labels, attr_type)
            
            # 线性CCA(基线)
            n_cca = min(10, pca_dim - 1, Y.shape[1] - 1)
            cca_linear = CCA(n_components=n_cca, max_iter=500)
            try:
                G_cca_lin, Y_cca_lin = cca_linear.fit_transform(G_pca, Y)
                lin_corrs = [round(abs(float(np.corrcoef(G_cca_lin[:, i], Y_cca_lin[:, i])[0, 1])), 4) 
                            for i in range(n_cca)]
            except:
                lin_corrs = [0.0] * n_cca
            
            # Kernel-CCA (RBF核 for X, 线性核 for Y)
            kcca_corrs, X_proj, Y_proj, alpha = kernel_cca(
                G_pca, Y, n_components=n_cca, 
                kernel_x='rbf', kernel_y='linear', reg=1e-3
            )
            
            if kcca_corrs is None:
                L.log(f"    L{layer} {attr_type}: Kernel-CCA failed")
                continue
            
            # Kernel-CCA重构质量
            # 用X_proj重构G_centered
            # G_centered ≈ PCA.inverse_transform(some_recon)
            # 简化: 用X_proj的余弦相似度作为重构指标
            if X_proj is not None:
                # 通过核回归重建
                Kx = np.exp(-1.0 / (2 * float(np.median(pdist(G_pca)))**2 + 1e-10) * squareform(pdist(G_pca))**2)
                H = np.eye(N) - np.ones((N, N)) / N
                Kx_c = H @ Kx @ H
                
                # 重建G_pca: G_recon = Kx_c @ alpha @ (Kx_c @ alpha)^+ @ G_pca
                proj = Kx_c @ alpha  # (N, n_cca)
                try:
                    proj_pinv = np.linalg.pinv(proj)
                    G_pca_recon = proj @ proj_pinv @ G_pca
                except:
                    G_pca_recon = np.zeros_like(G_pca)
                
                G_centered_recon = pca.inverse_transform(G_pca_recon)
                
                cos_vals = []
                for i in range(N):
                    n1 = np.linalg.norm(G_centered[i])
                    n2 = np.linalg.norm(G_centered_recon[i])
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_vals.append(float(np.dot(G_centered[i], G_centered_recon[i]) / (n1 * n2)))
                mean_cos_kcca = float(np.mean(cos_vals)) if cos_vals else 0.0
                
                ss_res = np.sum((G_centered - G_centered_recon)**2)
                ss_tot = np.sum((G_centered - G_centered.mean(axis=0))**2)
                r2_kcca = 1 - ss_res / (ss_tot + 1e-10)
            else:
                mean_cos_kcca = 0.0
                r2_kcca = 0.0
            
            # 比较线性 vs 非线性
            kcca_corr_list = [round(float(c), 4) for c in kcca_corrs]
            
            layer_result = {
                "layer": layer,
                "n_cca": n_cca,
                "linear_cca_corrs": lin_corrs,
                "kernel_cca_corrs": kcca_corr_list,
                "r2_kernel_cca": round(float(r2_kcca), 4),
                "cos_kernel_cca": round(mean_cos_kcca, 4),
                "improvement_over_linear": round(float(r2_kcca - (sum(lin_corrs) / len(lin_corrs))**2), 4) if lin_corrs else 0,
            }
            results[attr_type].append(layer_result)
            
            L.log(f"    L{layer} {attr_type}: lin_cca_r={lin_corrs[:3]}, "
                  f"kcca_r={kcca_corr_list[:3]}, R²_kcca={r2_kcca:.3f}, cos={mean_cos_kcca:.3f}")
    
    return results


# ==================== Sparse-CCA实现 ====================

def sparse_cca(X, Y, n_components=10, alpha_x=0.1, alpha_y=0.1, max_iter=100):
    """
    Sparse CCA: 带L1约束的CCA
    
    迭代算法:
    1. 初始化: 随机u, v
    2. 更新v: v = Y^T @ X @ u / ||Y^T @ X @ u||, soft_threshold(v, alpha_y)
    3. 更新u: u = X^T @ Y @ v / ||X^T @ Y @ v||, soft_threshold(u, alpha_x)
    4. 重复直到收敛
    """
    N, px = X.shape
    py = Y.shape[1]
    
    # 标准化
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    Y_std = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-10)
    
    # 协方差矩阵
    C = X_std.T @ Y_std / (N - 1)  # (px, py)
    
    components_u = []
    components_v = []
    correlations = []
    
    residual_X = X_std.copy()
    residual_Y = Y_std.copy()
    
    for k in range(n_components):
        # SVD初始化
        try:
            uk, sk, vtk = np.linalg.svd(residual_X.T @ residual_Y / (N-1), full_matrices=False)
            u = uk[:, 0]
            v = vtk[0, :]
        except:
            u = np.random.randn(px)
            v = np.random.randn(py)
            u /= np.linalg.norm(u) + 1e-10
            v /= np.linalg.norm(v) + 1e-10
        
        # 迭代优化
        for it in range(max_iter):
            # 更新v
            v_new = residual_Y.T @ residual_X @ u / (N - 1)
            v_new = soft_threshold(v_new, alpha_y * np.linalg.norm(v_new))
            v_norm = np.linalg.norm(v_new)
            if v_norm > 1e-10:
                v_new /= v_norm
            
            # 更新u
            u_new = residual_X.T @ residual_Y @ v_new / (N - 1)
            u_new = soft_threshold(u_new, alpha_x * np.linalg.norm(u_new))
            u_norm = np.linalg.norm(u_new)
            if u_norm > 1e-10:
                u_new /= u_norm
            
            # 检查收敛
            if np.linalg.norm(u_new - u) < 1e-6 and np.linalg.norm(v_new - v) < 1e-6:
                break
            u, v = u_new, v_new
        
        # 计算相关
        X_proj = residual_X @ u
        Y_proj = residual_Y @ v
        corr = float(np.corrcoef(X_proj, Y_proj)[0, 1])
        
        components_u.append(u)
        components_v.append(v)
        correlations.append(round(abs(corr), 4))
        
        # Deflation
        residual_X = residual_X - np.outer(X_proj, X_proj) @ residual_X / (N-1)
        residual_Y = residual_Y - np.outer(Y_proj, Y_proj) @ residual_Y / (N-1)
    
    return correlations, np.array(components_u).T, np.array(components_v).T


def soft_threshold(x, threshold):
    """软阈值算子"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


# ==================== P302: Sparse-CCA ====================

def run_p302(G_dict, labels_dict, key_layers, model_name):
    """Sparse-CCA: 稀疏编码轴发现"""
    L.log("=== P302: Sparse-CCA — 稀疏编码轴发现 ===")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_total = N_color + N_taste + N_size
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_start, n_end, labels in [
            ("color", 0, N_color, COLOR_LABELS),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS),
            ("size", N_color + N_taste, N_total, SIZE_LABELS),
        ]:
            G_sub = G_all[n_start:n_end]
            N, D = G_sub.shape
            if N < 60:
                continue
            
            # 名词中心化
            G_centered, noun_means = noun_centered_G(G_sub, labels)
            
            # PCA降维到30维
            pca_dim = min(30, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # 构建indicator矩阵
            Y, attr_list = build_indicator_matrix(labels, attr_type)
            
            # Sparse-CCA (多个稀疏度)
            n_cca = min(10, pca_dim - 1, Y.shape[1] - 1)
            
            best_result = None
            best_r2 = -1
            best_alpha = None
            
            for alpha in [0.05, 0.1, 0.2, 0.3]:
                try:
                    corrs, U, V = sparse_cca(G_pca, Y, n_components=n_cca, 
                                              alpha_x=alpha, alpha_y=alpha, max_iter=50)
                    
                    # 重构
                    X_proj = G_pca @ U  # (N, n_cca)
                    G_pca_recon = X_proj @ U.T  # 近似重建
                    G_centered_recon = pca.inverse_transform(G_pca_recon)
                    
                    ss_res = np.sum((G_centered - G_centered_recon)**2)
                    ss_tot = np.sum((G_centered - G_centered.mean(axis=0))**2)
                    r2 = 1 - ss_res / (ss_tot + 1e-10)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_alpha = alpha
                        best_result = (corrs, U, V, G_centered_recon)
                except Exception as e:
                    L.log(f"      alpha={alpha} failed: {e}")
                    continue
            
            if best_result is None:
                continue
            
            corrs, U, V, G_centered_recon = best_result
            
            # 重构余弦
            cos_vals = []
            for i in range(N):
                n1 = np.linalg.norm(G_centered[i])
                n2 = np.linalg.norm(G_centered_recon[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_vals.append(float(np.dot(G_centered[i], G_centered_recon[i]) / (n1 * n2)))
            mean_cos = float(np.mean(cos_vals)) if cos_vals else 0.0
            
            # 包含f_base
            G_full_pred = np.zeros_like(G_sub)
            nouns = [l[0] for l in labels]
            for i in range(N):
                G_full_pred[i] = noun_means[nouns[i]] + G_centered_recon[i]
            
            cos_full = []
            for i in range(N):
                n1 = np.linalg.norm(G_sub[i])
                n2 = np.linalg.norm(G_full_pred[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_full.append(float(np.dot(G_sub[i], G_full_pred[i]) / (n1 * n2)))
            mean_cos_full = float(np.mean(cos_full)) if cos_full else 0.0
            
            # 分析稀疏模式
            sparsity_per_comp = []
            top_g_dims_per_comp = []
            top_attrs_per_comp = []
            for c in range(min(5, n_cca)):
                u_c = U[:, c]
                v_c = V[:, c]
                
                # 稀疏度 = 零元素比例
                sp_x = float(np.mean(np.abs(u_c) < 0.01))
                sp_y = float(np.mean(np.abs(v_c) < 0.01))
                sparsity_per_comp.append((round(sp_x, 3), round(sp_y, 3)))
                
                # 顶部G_pca维度
                top_idx = np.argsort(np.abs(u_c))[::-1][:5]
                top_g_dims_per_comp.append([(int(idx), round(float(u_c[idx]), 4)) for idx in top_idx])
                
                # 顶部属性
                top_idx_y = np.argsort(np.abs(v_c))[::-1][:3]
                top_attrs_per_comp.append([(attr_list[int(idx)], round(float(v_c[idx]), 4)) for idx in top_idx_y])
            
            layer_result = {
                "layer": layer,
                "n_cca": n_cca,
                "best_alpha": best_alpha,
                "sparse_cca_corrs": corrs,
                "r2_centered": round(float(best_r2), 4),
                "cos_centered": round(mean_cos, 4),
                "cos_full": round(mean_cos_full, 4),
                "sparsity_per_comp": sparsity_per_comp,
                "top_g_dims_per_comp": top_g_dims_per_comp,
                "top_attrs_per_comp": top_attrs_per_comp,
            }
            results[attr_type].append(layer_result)
            
            L.log(f"    L{layer} {attr_type}(α={best_alpha}): R²={best_r2:.3f}, cos_c={mean_cos:.3f}, "
                  f"cos_f={mean_cos_full:.3f}, scca_r={corrs[:3]}")
    
    return results


# ==================== P303: PLS回归 ====================

def run_p303(G_dict, labels_dict, key_layers, model_name):
    """PLS回归: 弱信号增强"""
    L.log("=== P303: PLS回归 — 弱信号增强 ===")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_total = N_color + N_taste + N_size
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_start, n_end, labels in [
            ("color", 0, N_color, COLOR_LABELS),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS),
            ("size", N_color + N_taste, N_total, SIZE_LABELS),
        ]:
            G_sub = G_all[n_start:n_end]
            N, D = G_sub.shape
            if N < 60:
                continue
            
            # 名词中心化
            G_centered, noun_means = noun_centered_G(G_sub, labels)
            
            # PCA降维到30维
            pca_dim = min(30, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # 构建indicator矩阵
            Y, attr_list = build_indicator_matrix(labels, attr_type)
            
            # PLS (多个n_components)
            best_pls_result = None
            best_pls_r2 = -1
            best_n_comp = None
            
            for n_comp in [5, 8, 10, 12]:
                if n_comp >= min(G_pca.shape[1], Y.shape[1]):
                    continue
                try:
                    pls = PLSRegression(n_components=n_comp)
                    pls.fit(G_pca, Y)
                    Y_pred = pls.predict(G_pca)
                    
                    # 用PLS重构G_centered
                    # PLS给的是Y的预测, 但我们也要看G的重构
                    G_pca_recon = pls.inverse_transform(Y_pred) if hasattr(pls, 'inverse_transform') else G_pca
                    
                    # 直接用PLS x_weights重构
                    X_scores = pls.x_scores_  # (N, n_comp)
                    G_pca_recon = X_scores @ pls.x_loadings_.T
                    G_centered_recon = pca.inverse_transform(G_pca_recon)
                    
                    ss_res = np.sum((G_centered - G_centered_recon)**2)
                    ss_tot = np.sum((G_centered - G_centered.mean(axis=0))**2)
                    r2 = 1 - ss_res / (ss_tot + 1e-10)
                    
                    if r2 > best_pls_r2:
                        best_pls_r2 = r2
                        best_n_comp = n_comp
                        best_pls_result = (pls, G_centered_recon, Y_pred)
                except Exception as e:
                    continue
            
            if best_pls_result is None:
                continue
            
            pls, G_centered_recon, Y_pred = best_pls_result
            
            # 重构余弦
            cos_vals = []
            for i in range(N):
                n1 = np.linalg.norm(G_centered[i])
                n2 = np.linalg.norm(G_centered_recon[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_vals.append(float(np.dot(G_centered[i], G_centered_recon[i]) / (n1 * n2)))
            mean_cos = float(np.mean(cos_vals)) if cos_vals else 0.0
            
            # 包含f_base
            G_full_pred = np.zeros_like(G_sub)
            nouns = [l[0] for l in labels]
            for i in range(N):
                G_full_pred[i] = noun_means[nouns[i]] + G_centered_recon[i]
            
            cos_full = []
            for i in range(N):
                n1 = np.linalg.norm(G_sub[i])
                n2 = np.linalg.norm(G_full_pred[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_full.append(float(np.dot(G_sub[i], G_full_pred[i]) / (n1 * n2)))
            mean_cos_full = float(np.mean(cos_full)) if cos_full else 0.0
            
            # 弱信号检测: 每个属性维度的预测R²
            attr_r2 = {}
            for j, attr_name in enumerate(attr_list):
                ss_res_j = np.sum((Y[:, j] - Y_pred[:, j])**2)
                ss_tot_j = np.sum((Y[:, j] - Y[:, j].mean())**2)
                r2_j = 1 - ss_res_j / (ss_tot_j + 1e-10) if ss_tot_j > 1e-10 else 0.0
                attr_r2[attr_name] = round(float(r2_j), 4)
            
            # PLS x_weights分析
            x_weights_abs = np.abs(pls.x_weights_)  # (pca_dim, n_comp)
            top_dims_per_pls = []
            for c in range(min(5, best_n_comp)):
                top_idx = np.argsort(x_weights_abs[:, c])[::-1][:5]
                top_dims_per_pls.append([(int(idx), round(float(x_weights_abs[idx, c]), 4)) for idx in top_idx])
            
            layer_result = {
                "layer": layer,
                "best_n_comp": best_n_comp,
                "r2_centered": round(float(best_pls_r2), 4),
                "cos_centered": round(mean_cos, 4),
                "cos_full": round(mean_cos_full, 4),
                "attr_r2": attr_r2,
                "top_g_dims_per_pls": top_dims_per_pls,
            }
            results[attr_type].append(layer_result)
            
            # 按R²排序的属性
            sorted_attrs = sorted(attr_r2.items(), key=lambda x: x[1], reverse=True)
            top3 = [(a, r2) for a, r2 in sorted_attrs[:3]]
            weak3 = [(a, r2) for a, r2 in sorted_attrs[-3:]]
            L.log(f"    L{layer} {attr_type}(n={best_n_comp}): R²={best_pls_r2:.3f}, cos_c={mean_cos:.3f}")
            L.log(f"      top3={top3}, weak3={weak3}")
    
    return results


# ==================== P304: 综合比较 ====================

def run_p304(G_dict, labels_dict, key_layers, model_name, 
             p300_results, p301_results, p302_results, p303_results):
    """综合比较: 最优编码策略"""
    L.log("=== P304: 综合比较 — 最优编码策略 ===")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_total = N_color + N_taste + N_size
    
    # Ridge基线 (Phase LI的结果)
    ridge_results = {"color": [], "taste": [], "size": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_start, n_end, labels in [
            ("color", 0, N_color, COLOR_LABELS),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS),
            ("size", N_color + N_taste, N_total, SIZE_LABELS),
        ]:
            G_sub = G_all[n_start:n_end]
            N, D = G_sub.shape
            if N < 60:
                continue
            
            # 名词中心化
            G_centered, noun_means = noun_centered_G(G_sub, labels)
            
            # PCA降维到30维
            pca_dim = min(30, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # 构建indicator矩阵
            Y, attr_list = build_indicator_matrix(labels, attr_type)
            
            # --- 方法1: Ridge (Phase LI基线) ---
            ridge = Ridge(alpha=1.0)
            ridge.fit(Y, G_pca)
            G_pca_ridge = ridge.predict(Y)
            G_centered_ridge = pca.inverse_transform(G_pca_ridge)
            ss_res = np.sum((G_centered - G_centered_ridge)**2)
            ss_tot = np.sum((G_centered - G_centered.mean(axis=0))**2)
            r2_ridge = 1 - ss_res / (ss_tot + 1e-10)
            
            # --- 方法2: CCA (P300) ---
            n_cca = min(10, pca_dim - 1, Y.shape[1] - 1)
            cca = CCA(n_components=n_cca, max_iter=500)
            try:
                G_cca, Y_cca = cca.fit_transform(G_pca, Y)
                G_pca_cca = cca.inverse_transform(G_cca)
                G_centered_cca = pca.inverse_transform(G_pca_cca)
                ss_res_cca = np.sum((G_centered - G_centered_cca)**2)
                r2_cca = 1 - ss_res_cca / (ss_tot + 1e-10)
                cca_corrs = [round(abs(float(np.corrcoef(G_cca[:, i], Y_cca[:, i])[0, 1])), 4) for i in range(n_cca)]
            except:
                r2_cca = 0.0
                cca_corrs = [0.0] * n_cca
            
            # --- 方法3: PLS (P303最优n_comp) ---
            # 从P303结果找最优n_comp
            pls_best_r2 = 0.0
            pls_best_cos = 0.0
            for r in p303_results.get(attr_type, []):
                if r["layer"] == layer:
                    pls_best_r2 = r["r2_centered"]
                    pls_best_cos = r["cos_centered"]
                    break
            
            # --- 方法4: Sparse-CCA (P302) ---
            scca_best_r2 = 0.0
            scca_best_cos = 0.0
            for r in p302_results.get(attr_type, []):
                if r["layer"] == layer:
                    scca_best_r2 = r["r2_centered"]
                    scca_best_cos = r["cos_centered"]
                    break
            
            # --- 方法5: Kernel-CCA (P301) ---
            kcca_best_r2 = 0.0
            kcca_best_cos = 0.0
            for r in p301_results.get(attr_type, []):
                if r["layer"] == layer:
                    kcca_best_r2 = r["r2_kernel_cca"]
                    kcca_best_cos = r["cos_kernel_cca"]
                    break
            
            # Ridge余弦
            cos_ridge = []
            for i in range(N):
                n1 = np.linalg.norm(G_centered[i])
                n2 = np.linalg.norm(G_centered_ridge[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_ridge.append(float(np.dot(G_centered[i], G_centered_ridge[i]) / (n1 * n2)))
            mean_cos_ridge = float(np.mean(cos_ridge)) if cos_ridge else 0.0
            
            # CCA余弦
            cos_cca = []
            if r2_cca > 0:
                for i in range(N):
                    n1 = np.linalg.norm(G_centered[i])
                    n2 = np.linalg.norm(G_centered_cca[i])
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_cca.append(float(np.dot(G_centered[i], G_centered_cca[i]) / (n1 * n2)))
            mean_cos_cca = float(np.mean(cos_cca)) if cos_cca else 0.0
            
            # 5折交叉验证: Ridge vs CCA
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            cv_ridge_r2 = []
            cv_cca_r2 = []
            cv_pls_r2 = []
            
            for train_idx, test_idx in kf.split(G_pca):
                G_train, G_test = G_pca[train_idx], G_pca[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                
                # Ridge CV
                ridge_cv = Ridge(alpha=1.0)
                ridge_cv.fit(Y_train, G_train)
                G_pred_test = ridge_cv.predict(Y_test)
                ss_res_cv = np.sum((G_test - G_pred_test)**2)
                ss_tot_cv = np.sum((G_test - G_test.mean(axis=0))**2)
                cv_ridge_r2.append(1 - ss_res_cv / (ss_tot_cv + 1e-10))
                
                # CCA CV
                n_cca_cv = min(8, G_train.shape[1] - 1, Y_train.shape[1] - 1)
                if n_cca_cv > 0:
                    try:
                        cca_cv = CCA(n_components=n_cca_cv, max_iter=500)
                        G_tr_cca, Y_tr_cca = cca_cv.fit_transform(G_train, Y_train)
                        G_test_cca = cca_cv.transform(G_test, Y_test)[0]
                        G_pred_test_cca = cca_cv.inverse_transform(G_test_cca)
                        ss_res_cca_cv = np.sum((G_test - G_pred_test_cca)**2)
                        cv_cca_r2.append(1 - ss_res_cca_cv / (ss_tot_cv + 1e-10))
                    except:
                        cv_cca_r2.append(0.0)
                else:
                    cv_cca_r2.append(0.0)
                
                # PLS CV
                pls_n = min(8, G_train.shape[1], Y_train.shape[1])
                if pls_n > 0:
                    try:
                        pls_cv = PLSRegression(n_components=pls_n)
                        pls_cv.fit(G_train, Y_train)
                        Y_pred_test = pls_cv.predict(G_test)
                        # 用x_scores重构
                        X_scores_test = G_test @ pls_cv.x_weights_
                        G_test_recon = X_scores_test @ pls_cv.x_loadings_.T
                        ss_res_pls_cv = np.sum((G_test - G_test_recon)**2)
                        cv_pls_r2.append(1 - ss_res_pls_cv / (ss_tot_cv + 1e-10))
                    except:
                        cv_pls_r2.append(0.0)
                else:
                    cv_pls_r2.append(0.0)
            
            layer_result = {
                "layer": layer,
                "comparison": {
                    "ridge": {"r2": round(float(r2_ridge), 4), "cos": round(mean_cos_ridge, 4),
                              "cv_r2": round(float(np.mean(cv_ridge_r2)), 4)},
                    "cca": {"r2": round(float(r2_cca), 4), "cos": round(mean_cos_cca, 4),
                            "cv_r2": round(float(np.mean(cv_cca_r2)), 4), "top_corrs": cca_corrs[:5]},
                    "kernel_cca": {"r2": round(float(kcca_best_r2), 4), "cos": round(float(kcca_best_cos), 4)},
                    "sparse_cca": {"r2": round(float(scca_best_r2), 4), "cos": round(float(scca_best_cos), 4)},
                    "pls": {"r2": round(float(pls_best_r2), 4), "cos": round(float(pls_best_cos), 4),
                            "cv_r2": round(float(np.mean(cv_pls_r2)), 4)},
                },
                "best_method": "",
                "best_r2": 0,
            }
            
            # 确定最优方法
            method_r2s = {
                "ridge": r2_ridge,
                "cca": r2_cca,
                "kernel_cca": kcca_best_r2,
                "sparse_cca": scca_best_r2,
                "pls": pls_best_r2,
            }
            best_method = max(method_r2s, key=method_r2s.get)
            layer_result["best_method"] = best_method
            layer_result["best_r2"] = round(float(method_r2s[best_method]), 4)
            
            results[attr_type].append(layer_result)
            
            L.log(f"    L{layer} {attr_type}: ridge_R²={r2_ridge:.3f}, cca_R²={r2_cca:.3f}, "
                  f"kcca_R²={kcca_best_r2:.3f}, scca_R²={scca_best_r2:.3f}, pls_R²={pls_best_r2:.3f}")
            L.log(f"      best={best_method}(R²={method_r2s[best_method]:.3f}), "
                  f"cv_ridge={np.mean(cv_ridge_r2):.3f}, cv_cca={np.mean(cv_cca_r2):.3f}, cv_pls={np.mean(cv_pls_r2):.3f}")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"Phase LII: 自动发现编码维度 — {model_name}")
    L.log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载模型
    L.log("加载模型...")
    mdl, tok, device = load_model(model_name)
    n_layers = mdl.config.num_hidden_layers
    key_layers = get_key_layers(n_layers)
    L.log(f"模型: {model_name}, 层数: {n_layers}, 关键层: {key_layers}")
    
    # 数据收集
    L.log("数据收集(2160三元组×30模板)...")
    G_dict = collect_G_terms_large_scale(mdl, tok, device, key_layers, ALL_TRIPLES, PROMPT_TEMPLATES_30)
    
    for layer in key_layers:
        if layer in G_dict:
            L.log(f"  L{layer}: {len(G_dict[layer])} G向量")
    
    # 释放模型
    del mdl, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    L.log("模型已释放, 开始分析...")
    
    # P300: 线性CCA
    p300_results = run_p300(G_dict, None, key_layers, model_name)
    
    # P301: Kernel-CCA
    p301_results = run_p301(G_dict, None, key_layers, model_name)
    
    # P302: Sparse-CCA
    p302_results = run_p302(G_dict, None, key_layers, model_name)
    
    # P303: PLS回归
    p303_results = run_p303(G_dict, None, key_layers, model_name)
    
    # P304: 综合比较
    p304_results = run_p304(G_dict, None, key_layers, model_name,
                             p300_results, p301_results, p302_results, p303_results)
    
    # 保存结果
    all_results = {
        "model": model_name,
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M'),
        "p300_linear_cca": p300_results,
        "p301_kernel_cca": p301_results,
        "p302_sparse_cca": p302_results,
        "p303_pls": p303_results,
        "p304_comparison": p304_results,
    }
    
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_file = OUT_DIR / f"phase_lii_p300_304_{model_name}_{ts}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    L.log(f"结果已保存: {out_file}")
    
    # 最终总结
    L.log("\n===== Phase LII 总结 =====")
    for attr_type in ["color", "taste", "size"]:
        for r in p304_results.get(attr_type, []):
            layer = r["layer"]
            comp = r["comparison"]
            L.log(f"  {attr_type} L{layer}: "
                  f"ridge={comp['ridge']['r2']:.3f}, "
                  f"cca={comp['cca']['r2']:.3f}, "
                  f"kcca={comp['kernel_cca']['r2']:.3f}, "
                  f"scca={comp['sparse_cca']['r2']:.3f}, "
                  f"pls={comp['pls']['r2']:.3f} → "
                  f"best={r['best_method']}({r['best_r2']:.3f})")
    
    L.log(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.close()

if __name__ == "__main__":
    main()
