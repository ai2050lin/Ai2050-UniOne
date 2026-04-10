"""
Phase LI-P295/296/297/298/299: Diffusion Maps + G项精确参数化
======================================================================

Phase L核心突破:
  1. 连续编码检测成功: 亮度r=0.86, 色相r=0.74, 咸度r=0.84
  2. 名词中心化是关键: 去除名词基线后属性编码才显现
  3. 三属性子空间都是torus: 中心化后n_near_zero=5-12
  4. 分层重构超越PCA30: 0.90 > 0.82 ★★★
  5. G项显式公式: G = f_base(noun) + f_color(θ,bright) + f_taste(sweet,salty,sour) + f_size(vol,weight)

关键瓶颈:
  1. 饱和度(saturation)检测弱
  2. 苦度(bitterness)检测弱
  3. Per-noun winding<1: 环不完全闭合
  4. 连续编码变量是人工定义的: 可能不是模型真实编码
  5. 非光谱色角度标注不精确

Phase LI核心改进:
  1. Diffusion Maps: 用扩散距离替代k-近邻
     - 核矩阵 K = exp(-d²/2σ²), 扩散矩阵 P = D^{-1}K
     - P^t的特征向量给出多尺度嵌入
     - 优势: 考虑所有路径, 不只近邻; 多尺度t参数
  2. G项精确参数化: 用连续维度做线性/非线性回归
     - 线性: G_centered = A·cos(θ) + B·sin(θ) + C·brightness + ...
     - 非线性: 加入cos·sin交互项, 二次项
     - 目标: R²>0.9
  3. 交叉验证: 训练名词 vs 测试名词
     - 用5族名词中4族训练, 1族测试
     - 验证公式泛化性
  4. 弱信号增强: 用Diffusion Maps的更低频φ检测饱和度/苦度

五大实验:
  P295: Diffusion Maps多尺度嵌入
    - 对颜色/味道/大小子空间做Diffusion Maps
    - 比较不同t(扩散时间)的嵌入质量
    - 检测弱信号(饱和度/苦度)

  P296: G项精确线性回归
    - 设计矩阵: [cos(θ), sin(θ), brightness, sweetness, sourness, saltiness, bitterness, volume, weight]
    - 对G_centered做线性回归
    - 计算每个维度的t-statistic和p-value
    - 目标: R²>0.9

  P297: G项非线性回归
    - 加入交互项: cos·brightness, sweet·salty, vol·weight
    - 加入二次项: brightness², sweetness²
    - 对比线性vs非线性的R²提升

  P298: 交叉验证泛化性
    - 5折交叉验证: 每次留1族名词(12个)做测试
    - 测试集上的R²和余弦
    - 检验公式是否可泛化到新名词

  P299: G项公式验证与总结
    - 在3个模型×3个属性上验证公式
    - 总结G项的最终数学公式
    - 分析哪些维度是跨模型稳定的

数据规模: 2160三元组(与Phase L相同) × 30模板
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

LOG_FILE = OUT_DIR / "phase_li_log.txt"

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

# 颜色连续编码
COLOR_HUE_ANGLES = {
    "red": 0, "orange": 30, "yellow": 60, "green": 120,
    "blue": 240, "purple": 270, "pink": 330,
    "brown": 25, "white": 0, "black": 0, "gray": 0, "gold": 45,
}
SPECTRAL_COLORS = ["red","orange","yellow","green","blue","purple","pink"]

# 味道连续编码(扩展! 加入更多维度)
TASTE_DIMS = {
    "sweet":    {"sweetness": 1.0, "sourness": 0.05, "saltiness": 0.0, "bitterness": 0.0, "umami": 0.1, "spiciness": 0.0},
    "sour":     {"sweetness": 0.05, "sourness": 1.0, "saltiness": 0.0, "bitterness": 0.1, "umami": 0.0, "spiciness": 0.0},
    "bitter":   {"sweetness": 0.0, "sourness": 0.1, "saltiness": 0.0, "bitterness": 1.0, "umami": 0.0, "spiciness": 0.0},
    "salty":    {"sweetness": 0.0, "sourness": 0.0, "saltiness": 1.0, "bitterness": 0.0, "umami": 0.5, "spiciness": 0.0},
    "crisp":    {"sweetness": 0.3, "sourness": 0.5, "saltiness": 0.0, "bitterness": 0.1, "umami": 0.0, "spiciness": 0.0},
    "soft":     {"sweetness": 0.4, "sourness": 0.0, "saltiness": 0.0, "bitterness": 0.0, "umami": 0.0, "spiciness": 0.0},
    "spicy":    {"sweetness": 0.0, "sourness": 0.2, "saltiness": 0.1, "bitterness": 0.3, "umami": 0.0, "spiciness": 1.0},
    "fresh":    {"sweetness": 0.1, "sourness": 0.3, "saltiness": 0.0, "bitterness": 0.0, "umami": 0.0, "spiciness": 0.0},
    "tart":     {"sweetness": 0.0, "sourness": 0.8, "saltiness": 0.0, "bitterness": 0.5, "umami": 0.0, "spiciness": 0.0},
    "savory":   {"sweetness": 0.2, "sourness": 0.0, "saltiness": 0.5, "bitterness": 0.0, "umami": 1.0, "spiciness": 0.0},
    "rich":     {"sweetness": 0.5, "sourness": 0.0, "saltiness": 0.3, "bitterness": 0.0, "umami": 0.5, "spiciness": 0.0},
    "mild":     {"sweetness": 0.3, "sourness": 0.0, "saltiness": 0.0, "bitterness": 0.0, "umami": 0.0, "spiciness": 0.0},
}

# 大小连续编码
SIZE_DIMS = {
    "big":      {"volume": 1.0, "length": 0.5, "weight": 0.8, "thinness": 0.0},
    "small":    {"volume": 0.1, "length": 0.2, "weight": 0.1, "thinness": 0.3},
    "tall":     {"volume": 0.6, "length": 0.8, "weight": 0.5, "thinness": 0.3},
    "short":    {"volume": 0.2, "length": 0.1, "weight": 0.2, "thinness": 0.3},
    "long":     {"volume": 0.5, "length": 1.0, "weight": 0.5, "thinness": 0.3},
    "wide":     {"volume": 0.7, "length": 0.5, "weight": 0.6, "thinness": 0.0},
    "thin":     {"volume": 0.1, "length": 0.5, "weight": 0.1, "thinness": 1.0},
    "thick":    {"volume": 0.6, "length": 0.3, "weight": 0.7, "thinness": 0.0},
    "heavy":    {"volume": 0.7, "length": 0.3, "weight": 1.0, "thinness": 0.0},
    "light":    {"volume": 0.3, "length": 0.5, "weight": 0.0, "thinness": 0.5},
    "huge":     {"volume": 1.0, "length": 0.8, "weight": 1.0, "thinness": 0.0},
    "tiny":     {"volume": 0.0, "length": 0.0, "weight": 0.0, "thinness": 0.5},
}

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


# ==================== Diffusion Maps ====================

def compute_diffusion_maps(X_pca, t=1, n_components=20, sigma='median'):
    """
    Diffusion Maps: 
    1. 构建高斯核矩阵 K_ij = exp(-||x_i - x_j||² / 2σ²)
    2. 归一化: P = D^{-1} K (行随机矩阵, 每行和=1)
    3. P^t的特征向量给出多尺度嵌入
    4. 扩散时间t控制尺度: t小=局部, t大=全局
    """
    N, d = X_pca.shape
    if N < 10:
        return None, None, None
    
    # 计算距离矩阵
    dist_matrix = squareform(pdist(X_pca, 'euclidean'))
    
    # 选择σ
    if sigma == 'median':
        # 用非零距离的中位数
        nonzero_dists = dist_matrix[dist_matrix > 0]
        sigma_val = float(np.median(nonzero_dists)) if len(nonzero_dists) > 0 else 1.0
    else:
        sigma_val = sigma
    
    # 高斯核矩阵
    K = np.exp(-dist_matrix**2 / (2 * sigma_val**2))
    np.fill_diagonal(K, 0)  # 去除自连接
    
    # 归一化: P = D^{-1} K
    D = K.sum(axis=1)
    D_inv = np.diag(1.0 / (D + 1e-10))
    P = D_inv @ K
    
    # 对称化用于特征分解: P_sym = D^{1/2} P D^{-1/2}
    D_sqrt = np.diag(np.sqrt(D + 1e-10))
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(D + 1e-10)))
    P_sym = D_sqrt @ P @ D_inv_sqrt
    
    # 特征分解
    n_eig = min(n_components, N - 1)
    try:
        eigenvalues, eigenvectors = eigh(P_sym, subset_by_index=[max(0, N - n_eig), N - 1])
        # 从大到小排列
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
    except Exception:
        eigenvalues, eigenvectors = eigh(P_sym)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        eigenvalues = eigenvalues[:n_eig]
        eigenvectors = eigenvectors[:, :n_eig]
    
    # 转换回原始空间: φ_i = D^{-1/2} v_i (v是P_sym的特征向量)
    diffusion_vectors = D_inv_sqrt @ eigenvectors
    
    # 应用扩散时间t: ψ_i(t) = λ_i^t · φ_i
    eigenvalues_t = eigenvalues ** t
    
    # 扩散嵌入: Ψ = [λ_1^t φ_1, λ_2^t φ_2, ...]
    diffusion_embedding = diffusion_vectors * eigenvalues_t[np.newaxis, :]
    
    return eigenvalues, diffusion_vectors, diffusion_embedding


# ==================== 构建连续编码特征矩阵 ====================

def build_color_features(labels):
    """构建颜色子空间的连续特征矩阵"""
    color_values = [l[2] for l in labels]
    N = len(color_values)
    
    hue_rad = np.array([COLOR_HUE_ANGLES.get(c, 0) * np.pi / 180 for c in color_values])
    
    features = {
        "cos_hue": np.cos(hue_rad),
        "sin_hue": np.sin(hue_rad),
        "brightness": np.array([1.0 if c == "white" else 0.0 if c == "black" else 0.5 for c in color_values]),
        "saturation": np.array([0.0 if c == "gray" else 0.3 if c in ["white","black","brown"] else 1.0 for c in color_values]),
        "is_spectral": np.array([1.0 if c in SPECTRAL_COLORS else 0.0 for c in color_values]),
        "is_warm": np.array([1.0 if c in ["red","orange","yellow","pink","gold","brown"] else 0.0 for c in color_values]),
    }
    
    # 构建设计矩阵
    X = np.column_stack([features[name] for name in features])
    feature_names = list(features.keys())
    
    return X, feature_names, features


def build_taste_features(labels):
    """构建味道子空间的连续特征矩阵"""
    taste_values = [l[2] for l in labels]
    
    features = {}
    for dim_name in ["sweetness","sourness","saltiness","bitterness","umami","spiciness"]:
        features[dim_name] = np.array([TASTE_DIMS.get(t, {}).get(dim_name, 0) for t in taste_values])
    
    X = np.column_stack([features[name] for name in features])
    feature_names = list(features.keys())
    
    return X, feature_names, features


def build_size_features(labels):
    """构建大小子空间的连续特征矩阵"""
    size_values = [l[2] for l in labels]
    
    features = {}
    for dim_name in ["volume","length","weight","thinness"]:
        features[dim_name] = np.array([SIZE_DIMS.get(s, {}).get(dim_name, 0.5) for s in size_values])
    
    X = np.column_stack([features[name] for name in features])
    feature_names = list(features.keys())
    
    return X, feature_names, features


# ==================== P295: Diffusion Maps多尺度嵌入 ====================

def run_p295(G_dict, key_layers):
    """P295: Diffusion Maps多尺度嵌入"""
    L.log("P295: Diffusion Maps多尺度嵌入")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        N_total = G_all.shape[0]
        
        for attr_type, n_start, n_end, labels, build_func in [
            ("color", 0, N_color, COLOR_LABELS, build_color_features),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS, build_taste_features),
            ("size", N_color + N_taste, N_total, SIZE_LABELS, build_size_features),
        ]:
            G_sub = G_all[n_start:n_end]
            sub_labels = labels
            
            N, D = G_sub.shape
            if N < 60:
                continue
            
            # 名词中心化
            G_centered, _ = noun_centered_G(G_sub, sub_labels)
            
            # PCA降维
            pca_dim = min(20, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # Diffusion Maps with different t
            diffusion_results = {}
            for t in [1, 2, 4, 8]:
                eigvals, diff_vecs, diff_emb = compute_diffusion_maps(G_pca, t=t, n_components=20)
                if eigvals is None:
                    continue
                
                # 计算扩散特征向量与连续变量的相关
                X_feat, feat_names, feat_dict = build_func(sub_labels)
                
                phi_corrs = []
                for i in range(min(10, diff_vecs.shape[1])):
                    phi = diff_vecs[:, i]
                    corrs = {}
                    for j, fname in enumerate(feat_names):
                        if np.std(phi) > 1e-10 and np.std(X_feat[:, j]) > 1e-10:
                            corr = float(np.corrcoef(phi, X_feat[:, j])[0, 1])
                        else:
                            corr = 0.0
                        corrs[fname] = round(corr, 4)
                    phi_corrs.append({"phi_index": i, "correlations": corrs})
                
                # 找最强相关
                best_corrs = {}
                for fname in feat_names:
                    best_r = 0.0
                    best_phi = -1
                    for pc in phi_corrs:
                        r = pc["correlations"].get(fname, 0)
                        if abs(r) > abs(best_r):
                            best_r = r
                            best_phi = pc["phi_index"]
                    best_corrs[fname] = {"phi": best_phi, "r": round(best_r, 4)}
                
                # 用扩散嵌入做近邻重构
                if diff_emb is not None and diff_emb.shape[1] >= 3:
                    k_nn = min(10, N - 1)
                    nn_model = NearestNeighbors(n_neighbors=k_nn, algorithm='ball_tree').fit(diff_emb[:, :min(10, diff_emb.shape[1])])
                    _, nn_indices = nn_model.kneighbors(diff_emb[:, :min(10, diff_emb.shape[1])])
                    
                    G_recon = np.zeros_like(G_pca)
                    for j in range(N):
                        neighbors = nn_indices[j]
                        G_recon[j] = np.mean(G_pca[neighbors], axis=0)
                    
                    cos_vals = []
                    for j in range(N):
                        n1 = np.linalg.norm(G_pca[j])
                        n2 = np.linalg.norm(G_recon[j])
                        if n1 > 1e-10 and n2 > 1e-10:
                            cos_vals.append(float(np.dot(G_pca[j], G_recon[j]) / (n1 * n2)))
                    mean_cos = float(np.mean(cos_vals)) if cos_vals else 0.0
                else:
                    mean_cos = 0.0
                
                diffusion_results[f"t={t}"] = {
                    "eigenvalues_top5": [round(float(e), 6) for e in eigvals[:5]],
                    "phi_corrs": phi_corrs[:5],
                    "best_corrs": best_corrs,
                    "reconstruction_cos": round(mean_cos, 4),
                }
                
                top_feats = [(k, v["r"]) for k, v in list(best_corrs.items())[:3]]
                top_str = ", ".join(f"{k}(r={r})" for k, r in top_feats)
                L.log(f"    L{layer} {attr_type} t={t}: recon_cos={mean_cos:.3f}, best_feats={top_str}")
            
            layer_result = {
                "layer": layer,
                "diffusion_results": diffusion_results,
            }
            results[attr_type].append(layer_result)
    
    return results


# ==================== P296: G项精确线性回归 ====================

def run_p296(G_dict, key_layers):
    """P296: G项精确线性回归"""
    L.log("P296: G项精确线性回归")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_total = N_color + N_taste + N_size
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_start, n_end, labels, build_func in [
            ("color", 0, N_color, COLOR_LABELS, build_color_features),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS, build_taste_features),
            ("size", N_color + N_taste, N_total, SIZE_LABELS, build_size_features),
        ]:
            G_sub = G_all[n_start:n_end]
            sub_labels = labels
            
            N, D = G_sub.shape
            if N < 60:
                continue
            
            # 名词中心化
            G_centered, noun_means = noun_centered_G(G_sub, sub_labels)
            
            # 构建特征矩阵
            X_feat, feat_names, feat_dict = build_func(sub_labels)
            
            # 对每个PCA维度做回归
            pca_dim = min(15, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # 线性回归: G_pca = X_feat @ beta + error
            # 使用Ridge回归防止过拟合
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_feat, G_pca)
            G_pca_pred = ridge.predict(X_feat)
            
            # R² (每个PCA维度)
            r2_per_dim = []
            for d in range(pca_dim):
                ss_res = np.sum((G_pca[:, d] - G_pca_pred[:, d])**2)
                ss_tot = np.sum((G_pca[:, d] - G_pca[:, d].mean())**2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                r2_per_dim.append(round(float(r2), 4))
            
            # 总R²
            G_centered_pred = pca.inverse_transform(G_pca_pred)
            ss_res_total = np.sum((G_centered - G_centered_pred)**2)
            ss_tot_total = np.sum((G_centered - G_centered.mean(axis=0))**2)
            r2_total = 1 - ss_res_total / (ss_tot_total + 1e-10)
            
            # 余弦相似度
            cos_vals = []
            for i in range(N):
                n1 = np.linalg.norm(G_centered[i])
                n2 = np.linalg.norm(G_centered_pred[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_vals.append(float(np.dot(G_centered[i], G_centered_pred[i]) / (n1 * n2)))
            mean_cos = float(np.mean(cos_vals))
            
            # 每个特征的系数显著性 (对前5个PCA维度的平均)
            coef_significance = {}
            for j, fname in enumerate(feat_names):
                coefs = [ridge.coef_[d, j] for d in range(min(5, pca_dim))]
                mean_coef = float(np.mean(np.abs(coefs)))
                # t-test: coef / std(coef) 
                if np.std(coefs) > 1e-10:
                    t_stat = float(np.mean(coefs) / (np.std(coefs) / np.sqrt(len(coefs))))
                else:
                    t_stat = 0.0
                coef_significance[fname] = {
                    "mean_abs_coef": round(mean_coef, 6),
                    "t_stat": round(t_stat, 3),
                }
            
            # 预测包含名词基线的完整G
            G_full_pred = np.zeros_like(G_sub)
            nouns = [l[0] for l in sub_labels]
            for i in range(N):
                G_full_pred[i] = noun_means[nouns[i]] + G_centered_pred[i]
            
            cos_full = []
            for i in range(N):
                n1 = np.linalg.norm(G_sub[i])
                n2 = np.linalg.norm(G_full_pred[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_full.append(float(np.dot(G_sub[i], G_full_pred[i]) / (n1 * n2)))
            mean_cos_full = float(np.mean(cos_full))
            
            layer_result = {
                "layer": layer,
                "feature_names": feat_names,
                "r2_per_pca_dim": r2_per_dim[:10],
                "r2_total_centered": round(float(r2_total), 4),
                "cos_centered": round(mean_cos, 4),
                "cos_full": round(mean_cos_full, 4),
                "coef_significance": coef_significance,
                "pca_cumvar_top5": [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)[:5]],
            }
            results[attr_type].append(layer_result)
            L.log(f"    L{layer} {attr_type}: R²={r2_total:.3f}, cos_centered={mean_cos:.3f}, "
                  f"cos_full={mean_cos_full:.3f}")
            coef_str = ", ".join(f"{k}(t={v['t_stat']:.2f})" for k, v in list(coef_significance.items())[:5])
            L.log(f"      coefs: {coef_str}")
    
    return results


# ==================== P297: G项非线性回归 ====================

def run_p297(G_dict, key_layers):
    """P297: G项非线性回归 — 加入交互项和二次项"""
    L.log("P297: G项非线性回归")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_all = N_color + N_taste + N_size
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_start, n_end, labels, build_func in [
            ("color", 0, N_color, COLOR_LABELS, build_color_features),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS, build_taste_features),
            ("size", N_color + N_taste, N_color + N_taste + N_size, SIZE_LABELS, build_size_features),
        ]:
            G_sub = G_all[n_start:n_end]
            sub_labels = labels
            
            N, D = G_sub.shape
            if N < 60:
                continue
            
            G_centered, noun_means = noun_centered_G(G_sub, sub_labels)
            
            # 线性特征
            X_linear, feat_names_linear, feat_dict = build_func(sub_labels)
            
            # 扩展特征: 交互项 + 二次项
            X_extended = [X_linear]
            extended_names = list(feat_names_linear)
            
            # 二次项
            for i, name in enumerate(feat_names_linear):
                X_extended.append((X_linear[:, i] ** 2).reshape(-1, 1))
                extended_names.append(f"{name}²")
            
            # 关键交互项
            if attr_type == "color":
                # cos_hue * brightness (色相×亮度)
                if "cos_hue" in feat_dict and "brightness" in feat_dict:
                    X_extended.append((feat_dict["cos_hue"] * feat_dict["brightness"]).reshape(-1, 1))
                    extended_names.append("cos_hue×brightness")
                # sin_hue * brightness
                if "sin_hue" in feat_dict and "brightness" in feat_dict:
                    X_extended.append((feat_dict["sin_hue"] * feat_dict["brightness"]).reshape(-1, 1))
                    extended_names.append("sin_hue×brightness")
                # cos_hue * saturation
                if "cos_hue" in feat_dict and "saturation" in feat_dict:
                    X_extended.append((feat_dict["cos_hue"] * feat_dict["saturation"]).reshape(-1, 1))
                    extended_names.append("cos_hue×saturation")
            elif attr_type == "taste":
                # sweetness * saltiness
                if "sweetness" in feat_dict and "saltiness" in feat_dict:
                    X_extended.append((feat_dict["sweetness"] * feat_dict["saltiness"]).reshape(-1, 1))
                    extended_names.append("sweetness×saltiness")
                # sourness * bitterness
                if "sourness" in feat_dict and "bitterness" in feat_dict:
                    X_extended.append((feat_dict["sourness"] * feat_dict["bitterness"]).reshape(-1, 1))
                    extended_names.append("sourness×bitterness")
            else:  # size
                # volume * weight
                if "volume" in feat_dict and "weight" in feat_dict:
                    X_extended.append((feat_dict["volume"] * feat_dict["weight"]).reshape(-1, 1))
                    extended_names.append("volume×weight")
                # length * thinness
                if "length" in feat_dict and "thinness" in feat_dict:
                    X_extended.append((feat_dict["length"] * feat_dict["thinness"]).reshape(-1, 1))
                    extended_names.append("length×thinness")
            
            X_ext = np.column_stack(X_extended)
            
            # PCA
            pca_dim = min(15, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # 线性回归(仅线性项)
            ridge_lin = Ridge(alpha=1.0)
            ridge_lin.fit(X_linear, G_pca)
            G_pred_lin = ridge_lin.predict(X_linear)
            G_centered_pred_lin = pca.inverse_transform(G_pred_lin)
            ss_res_lin = np.sum((G_centered - G_centered_pred_lin)**2)
            ss_tot = np.sum((G_centered - G_centered.mean(axis=0))**2)
            r2_lin = 1 - ss_res_lin / (ss_tot + 1e-10)
            
            # 非线性回归(扩展特征)
            ridge_ext = Ridge(alpha=1.0)
            ridge_ext.fit(X_ext, G_pca)
            G_pred_ext = ridge_ext.predict(X_ext)
            G_centered_pred_ext = pca.inverse_transform(G_pred_ext)
            ss_res_ext = np.sum((G_centered - G_centered_pred_ext)**2)
            r2_ext = 1 - ss_res_ext / (ss_tot + 1e-10)
            
            # R²提升
            r2_improvement = r2_ext - r2_lin
            
            # 余弦
            cos_lin = []
            cos_ext = []
            for i in range(N):
                n1 = np.linalg.norm(G_centered[i])
                n2_lin = np.linalg.norm(G_centered_pred_lin[i])
                n2_ext = np.linalg.norm(G_centered_pred_ext[i])
                if n1 > 1e-10:
                    if n2_lin > 1e-10:
                        cos_lin.append(float(np.dot(G_centered[i], G_centered_pred_lin[i]) / (n1 * n2_lin)))
                    if n2_ext > 1e-10:
                        cos_ext.append(float(np.dot(G_centered[i], G_centered_pred_ext[i]) / (n1 * n2_ext)))
            
            # 显著的交互项
            significant_interactions = {}
            for j, name in enumerate(extended_names):
                if "×" in name or "²" in name:
                    coefs = [ridge_ext.coef_[d, j] for d in range(min(5, pca_dim))]
                    mean_abs = float(np.mean(np.abs(coefs)))
                    if mean_abs > 0.01:  # 阈值
                        significant_interactions[name] = round(mean_abs, 4)
            
            layer_result = {
                "layer": layer,
                "n_features_linear": len(feat_names_linear),
                "n_features_extended": len(extended_names),
                "r2_linear": round(float(r2_lin), 4),
                "r2_extended": round(float(r2_ext), 4),
                "r2_improvement": round(float(r2_improvement), 4),
                "cos_linear": round(float(np.mean(cos_lin)), 4) if cos_lin else 0,
                "cos_extended": round(float(np.mean(cos_ext)), 4) if cos_ext else 0,
                "significant_interactions": significant_interactions,
                "extended_feature_names": extended_names,
            }
            results[attr_type].append(layer_result)
            L.log(f"    L{layer} {attr_type}: R²_lin={r2_lin:.3f}, R²_ext={r2_ext:.3f}, "
                  f"improvement={r2_improvement:.4f}, cos_ext={np.mean(cos_ext):.3f}")
    
    return results


# ==================== P298: 交叉验证泛化性 ====================

def run_p298(G_dict, key_layers):
    """P298: 5折交叉验证 — 每折留1族名词做测试"""
    L.log("P298: 5折交叉验证泛化性")
    results = []
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_all = N_color + N_taste + N_size
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        cv_results = {}
        
        for attr_type, n_start, n_end, labels, build_func in [
            ("color", 0, N_color, COLOR_LABELS, build_color_features),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS, build_taste_features),
            ("size", N_color + N_taste, N_color + N_taste + N_size, SIZE_LABELS, build_size_features),
        ]:
            G_sub = G_all[n_start:n_end]
            sub_labels = labels
            N, D = G_sub.shape
            
            if N < 60:
                continue
            
            # 名词中心化
            G_centered, noun_means = noun_centered_G(G_sub, sub_labels)
            
            # 构建特征
            X_feat, feat_names, _ = build_func(sub_labels)
            
            # PCA
            pca_dim = min(15, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # 5折CV: 按家族划分
            nouns = [l[0] for l in sub_labels]
            families = [l[3] for l in sub_labels]
            
            fold_results = []
            for fold_idx, test_family in enumerate(FAMILY_NAMES):
                test_nouns = set(STIMULI[test_family])
                
                train_mask = np.array([n not in test_nouns for n in nouns])
                test_mask = np.array([n in test_nouns for n in nouns])
                
                if train_mask.sum() < 30 or test_mask.sum() < 10:
                    continue
                
                X_train = X_feat[train_mask]
                X_test = X_feat[test_mask]
                G_pca_train = G_pca[train_mask]
                G_pca_test = G_pca[test_mask]
                G_centered_train = G_centered[train_mask]
                G_centered_test = G_centered[test_mask]
                
                # 训练Ridge回归
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train, G_pca_train)
                
                # 测试集预测
                G_pca_test_pred = ridge.predict(X_test)
                G_centered_test_pred = pca.inverse_transform(G_pca_test_pred)
                
                # 测试集R²
                ss_res = np.sum((G_centered_test - G_centered_test_pred)**2)
                ss_tot = np.sum((G_centered_test - G_centered_test.mean(axis=0))**2)
                r2_test = 1 - ss_res / (ss_tot + 1e-10)
                
                # 训练集R²
                G_pca_train_pred = ridge.predict(X_train)
                G_centered_train_pred = pca.inverse_transform(G_pca_train_pred)
                ss_res_tr = np.sum((G_centered_train - G_centered_train_pred)**2)
                ss_tot_tr = np.sum((G_centered_train - G_centered_train.mean(axis=0))**2)
                r2_train = 1 - ss_res_tr / (ss_tot_tr + 1e-10)
                
                # 测试集余弦
                cos_test = []
                for i in range(G_centered_test.shape[0]):
                    n1 = np.linalg.norm(G_centered_test[i])
                    n2 = np.linalg.norm(G_centered_test_pred[i])
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_test.append(float(np.dot(G_centered_test[i], G_centered_test_pred[i]) / (n1 * n2)))
                
                fold_results.append({
                    "test_family": test_family,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "r2_train": round(float(r2_train), 4),
                    "r2_test": round(float(r2_test), 4),
                    "cos_test": round(float(np.mean(cos_test)), 4) if cos_test else 0,
                })
                
                L.log(f"      Fold {fold_idx}: test={test_family}, "
                      f"R²_train={r2_train:.3f}, R²_test={r2_test:.3f}, cos_test={np.mean(cos_test):.3f}")
            
            if fold_results:
                mean_r2_test = float(np.mean([f["r2_test"] for f in fold_results]))
                mean_cos_test = float(np.mean([f["cos_test"] for f in fold_results]))
                cv_results[attr_type] = {
                    "fold_results": fold_results,
                    "mean_r2_test": round(mean_r2_test, 4),
                    "mean_cos_test": round(mean_cos_test, 4),
                }
                L.log(f"    L{layer} {attr_type}: CV mean_R²_test={mean_r2_test:.3f}, "
                      f"mean_cos_test={mean_cos_test:.3f}")
        
        if cv_results:
            results.append({"layer": layer, "cv_results": cv_results})
    
    return results


# ==================== P299: G项公式验证与总结 ====================

def run_p299(G_dict, key_layers):
    """P299: G项公式验证与总结"""
    L.log("P299: G项公式验证与总结")
    results = []
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_all = N_color + N_taste + N_size
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        # ====== 全局重构质量 ======
        nouns_all = [l[0] for l in ALL_LABELS]
        unique_nouns = sorted(set(nouns_all))
        
        # f_base: 每个名词的均值G
        noun_base = {}
        for noun in unique_nouns:
            mask = np.array([nn == noun for nn in nouns_all])
            noun_base[noun] = G_all[mask].mean(axis=0)
        
        G_recon_base = np.stack([noun_base[n] for n in nouns_all])
        
        # ====== 对每个属性类型做回归 ======
        attr_residuals = np.zeros_like(G_all)
        attr_predictions = np.zeros_like(G_all)
        
        total_r2_by_attr = {}
        total_cos_by_attr = {}
        
        for attr_type, n_start, n_end, labels, build_func in [
            ("color", 0, N_color, COLOR_LABELS, build_color_features),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS, build_taste_features),
            ("size", N_color + N_taste, N_color + N_taste + N_size, SIZE_LABELS, build_size_features),
        ]:
            G_sub = G_all[n_start:n_end]
            sub_labels = labels
            N, D = G_sub.shape
            
            # 名词中心化
            G_centered, noun_means = noun_centered_G(G_sub, sub_labels)
            
            # 构建特征(含非线性项)
            X_feat, feat_names, feat_dict = build_func(sub_labels)
            
            # 扩展特征
            X_ext_list = [X_feat]
            ext_names = list(feat_names)
            
            for i, name in enumerate(feat_names):
                X_ext_list.append((X_feat[:, i] ** 2).reshape(-1, 1))
                ext_names.append(f"{name}²")
            
            if attr_type == "color" and "cos_hue" in feat_dict and "brightness" in feat_dict:
                X_ext_list.append((feat_dict["cos_hue"] * feat_dict["brightness"]).reshape(-1, 1))
                ext_names.append("cos_hue×brightness")
                X_ext_list.append((feat_dict["sin_hue"] * feat_dict["brightness"]).reshape(-1, 1))
                ext_names.append("sin_hue×brightness")
            
            X_ext = np.column_stack(X_ext_list)
            
            # PCA
            pca_dim = min(15, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # 回归
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_ext, G_pca)
            G_pca_pred = ridge.predict(X_ext)
            G_centered_pred = pca.inverse_transform(G_pca_pred)
            
            # 完整重构: f_base + 属性预测
            sub_nouns = [l[0] for l in sub_labels]
            G_full_pred = np.zeros_like(G_sub)
            for i in range(N):
                G_full_pred[i] = noun_means[sub_nouns[i]] + G_centered_pred[i]
            
            attr_predictions[n_start:n_end] = G_full_pred
            
            # R²
            ss_res = np.sum((G_centered - G_centered_pred)**2)
            ss_tot = np.sum((G_centered - G_centered.mean(axis=0))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            
            # 余弦
            cos_vals = []
            for i in range(N):
                n1 = np.linalg.norm(G_centered[i])
                n2 = np.linalg.norm(G_centered_pred[i])
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_vals.append(float(np.dot(G_centered[i], G_centered_pred[i]) / (n1 * n2)))
            
            total_r2_by_attr[attr_type] = round(float(r2), 4)
            total_cos_by_attr[attr_type] = round(float(np.mean(cos_vals)), 4)
            
            L.log(f"    L{layer} {attr_type}: R²={r2:.3f}, cos={np.mean(cos_vals):.3f}")
        
        # ====== 全局重构(合并) ======
        cos_global = []
        for i in range(N_all):
            n1 = np.linalg.norm(G_all[i])
            n2 = np.linalg.norm(attr_predictions[i])
            if n1 > 1e-10 and n2 > 1e-10:
                cos_global.append(float(np.dot(G_all[i], attr_predictions[i]) / (n1 * n2)))
        
        layer_result = {
            "layer": layer,
            "r2_by_attr": total_r2_by_attr,
            "cos_by_attr": total_cos_by_attr,
            "cos_global": round(float(np.mean(cos_global)), 4),
            "formula": "G(n,a) = f_base(n) + Ridge(X_attr(a), PCA15(G_centered))",
        }
        results.append(layer_result)
        L.log(f"    L{layer}: cos_global={np.mean(cos_global):.3f}, "
              f"by_attr={total_r2_by_attr}")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"======== Phase LI: Diffusion Maps + G项精确参数化 ========")
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
    
    L.log("\n====== P295: Diffusion Maps多尺度嵌入 ======")
    all_results["P295"] = run_p295(G_dict, key_layers)
    
    L.log("\n====== P296: G项精确线性回归 ======")
    all_results["P296"] = run_p296(G_dict, key_layers)
    
    L.log("\n====== P297: G项非线性回归 ======")
    all_results["P297"] = run_p297(G_dict, key_layers)
    
    L.log("\n====== P298: 交叉验证泛化性 ======")
    all_results["P298"] = run_p298(G_dict, key_layers)
    
    L.log("\n====== P299: G项公式验证与总结 ======")
    all_results["P299"] = run_p299(G_dict, key_layers)
    
    # 保存结果
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_file = OUT_DIR / f"phase_li_p295_299_{model_name}_{ts}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    L.log(f"\n结果已保存: {out_file}")
    
    # 打印摘要
    L.log("\n======== 摘要 ========")
    L.log(f"模型: {model_name}")
    
    # P295摘要
    for attr_type in ["color", "taste", "size"]:
        if all_results["P295"][attr_type]:
            last = all_results["P295"][attr_type][-1]
            for t_key in ["t=1", "t=4"]:
                if t_key in last.get("diffusion_results", {}):
                    dr = last["diffusion_results"][t_key]
                    L.log(f"P295 {attr_type} {t_key}: recon_cos={dr.get('reconstruction_cos','?')}")
    
    # P296摘要
    for attr_type in ["color", "taste", "size"]:
        if all_results["P296"][attr_type]:
            last = all_results["P296"][attr_type][-1]
            L.log(f"P296 {attr_type}: R²={last['r2_total_centered']}, cos_full={last['cos_full']}")
    
    # P297摘要
    for attr_type in ["color", "taste", "size"]:
        if all_results["P297"][attr_type]:
            last = all_results["P297"][attr_type][-1]
            L.log(f"P297 {attr_type}: R²_lin={last['r2_linear']}, R²_ext={last['r2_extended']}, "
                  f"improvement={last['r2_improvement']}")
    
    # P298摘要
    if all_results["P298"]:
        for cv_res in all_results["P298"]:
            for attr_type, cv_data in cv_res.get("cv_results", {}).items():
                L.log(f"P298 L{cv_res['layer']} {attr_type}: mean_R²_test={cv_data['mean_r2_test']}, "
                      f"mean_cos_test={cv_data['mean_cos_test']}")
    
    # P299摘要
    if all_results["P299"]:
        for r in all_results["P299"]:
            L.log(f"P299 L{r['layer']}: cos_global={r['cos_global']}, R²_by_attr={r['r2_by_attr']}")
    
    L.close()


if __name__ == "__main__":
    main()
