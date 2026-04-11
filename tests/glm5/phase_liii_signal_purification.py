"""
Phase LIII-P305/306/307/308/309: 正交空间分解与算子提纯
======================================================================

Phase LII核心结论:
  1. CCA前10方向相关>0.95: 属性编码与indicator近乎完美线性相关
  2. Kernel-CCA系统性差于线性CCA: 否定非线性假设
  3. Ridge交叉验证R²最高(0.48-0.79): 最优编码方法
  4. "最后50%"是高维噪声: 不是非线性拓扑干涉
  5. G项 = f_base(noun) + f_signal(attr) + f_noise

Phase LIII核心目标: 语义提纯手术 — 验证信号算子是否构成"白盒算子"

核心数学:
  信号子空间 S: 由CCA前k个方向 {u_1, ..., u_k} 张成
  噪声子空间 N: 与S正交的剩余维度
  
  投影矩阵: P_s = U(U^T U)^{-1} U^T
  信号分量: G_signal = P_s · G_centered
  噪声分量: G_noise = (I - P_s) · G_centered

五大实验:
  P305: 信号子空间拓扑回归
    - 在G_signal上重新计算颜色环的环绕数(winding number)
    - 预期: 环绕数从0.85趋近1.0 (去除噪声后环面更闭合)
    - 对味道/大小子空间也做同样的验证

  P306: 噪声独立性测试
    - 计算不同属性(颜色vs味道vs大小)的G_noise重叠度
    - 如果噪声是通用的: 不同属性噪声空间高度重合 → "语境背景"
    - 如果噪声是属性特异的: 不同属性噪声正交 → "高维残差"

  P307: 算子干预实验 — ★★★核心实验★★★
    - 构造纯净属性算子: Ĝ_red = mean(G_signal | red)
    - 注入到未见名词: h_new = h_object + Ĝ_red
    - 通过Unembedding层观察logits
    - 验证模型是否精准地将物体"涂"成红色
    - 成功 → 信号算子是可控的白盒算子
    - 失败 → 噪声是"语义粘合剂", 不可分离

  P308: 算子线性组合实验
    - 构造组合算子: Ĝ_combo = α·Ĝ_red + β·Ĝ_big
    - 测试是否可以通过调节α,β像调音台一样控制输出
    - 验证算子的可加性和可插拔性

  P309: 跨名词泛化实验
    - 用水果族训练的算子, 注入到动物/工具族
    - 验证算子是否跨名词族泛化
    - 如果泛化 → 算子是通用的"语义基石"
    - 如果不泛化 → 算子是名词族特异的

数据规模: 2160三元组(与Phase LII相同) × 30模板
实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免OOM)
"""

import torch
import numpy as np
import os, sys, gc, time, json, argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
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

LOG_FILE = OUT_DIR / "phase_liii_log.txt"

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


def collect_hidden_states(mdl, tok, device, key_layers, words, templates):
    """收集词的原始隐藏状态(用于干预实验)"""
    word_hs_avg = {}
    for word in sorted(words):
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
        word_hs_avg[word] = {l: layer_sums[l] / layer_counts[l] for l in layer_sums}
    return word_hs_avg


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


# ==================== 信号/噪声子空间投影 ====================

def compute_signal_noise_subspaces(G_pca, Y, n_components=10):
    """
    用CCA计算信号/噪声子空间
    返回: G_signal, G_noise, cca对象, 投影矩阵
    """
    N, D = G_pca.shape
    K = Y.shape[1]
    n_cca = min(n_components, D - 1, K - 1)
    
    cca = CCA(n_components=n_cca, max_iter=500)
    try:
        G_cca, Y_cca = cca.fit_transform(G_pca, Y)
    except Exception as e:
        # 处理NaN: 用Ridge替代
        L.log(f"    CCA failed ({e}), using Ridge fallback")
        ridge = Ridge(alpha=1.0)
        ridge.fit(Y, G_pca)
        G_signal = ridge.predict(Y)  # (N, D) — 信号预测
        G_noise = G_pca - G_signal   # (N, D) — 噪声残差
        # 构造投影矩阵: 用G_signal的SVD
        # G_signal = G_pca @ P_s → P_s = G_pca^+ @ G_signal
        try:
            G_pca_pinv = np.linalg.pinv(G_pca)
            P_s = G_pca_pinv @ G_signal  # (D, D)
        except:
            P_s = np.eye(D)  # fallback
        cca_corrs = [0.0] * n_cca
        return G_signal, G_noise, None, P_s, cca_corrs
    
    # CCA x_weights: (D, n_cca) — X空间的投影方向
    U = cca.x_weights_  # (D, n_cca)
    
    # 投影矩阵 P_s = U(U^T U)^{-1} U^T
    UtU_inv = np.linalg.inv(U.T @ U + 1e-8 * np.eye(n_cca))
    P_s = U @ UtU_inv @ U.T  # (D, D)
    
    # 信号分量: G_signal = G_pca @ P_s (在PCA空间投影)
    G_signal = G_pca @ P_s
    G_noise = G_pca - G_signal
    
    # CCA相关
    cca_corrs = []
    for i in range(n_cca):
        r = float(np.corrcoef(G_cca[:, i], Y_cca[:, i])[0, 1])
        cca_corrs.append(round(abs(r), 4))
    
    return G_signal, G_noise, cca, P_s, cca_corrs


# ==================== Laplacian环绕数计算 ====================

def compute_winding_number(X_pca, n_eig=15, k=8):
    """
    计算Laplacian特征向量并估算环绕数
    返回: (n_near_zero, winding, dim95)
    """
    N, D = X_pca.shape
    if N < 20 or D < 2:
        return 0, 0.0, 0
    
    # k-近邻图
    from sklearn.neighbors import NearestNeighbors
    k_actual = min(k, N - 1)
    nn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
    nn.fit(X_pca)
    dists, indices = nn.kneighbors(X_pca)
    
    # 构建权重矩阵
    W = np.zeros((N, N))
    sigma = float(np.median(dists[:, 1:])) + 1e-10
    for i in range(N):
        for j_idx in range(1, k_actual):
            j = indices[i, j_idx]
            W[i, j] = np.exp(-dists[i, j_idx]**2 / (2 * sigma**2))
            W[j, i] = W[i, j]
    
    # 度矩阵
    D = np.diag(W.sum(axis=1) + 1e-10)
    L = D - W
    
    # 特征分解
    n_eig_actual = min(n_eig, N - 1)
    try:
        eigvals, eigvecs = eigh(L, D, subset_by_index=[0, n_eig_actual - 1])
    except:
        eigvals, eigvecs = eigh(L)
        eigvals = eigvals[:n_eig_actual]
        eigvecs = eigvecs[:, :n_eig_actual]
    
    # near-zero特征值个数
    threshold = 0.01 * (eigvals[-1] - eigvals[0] + 1e-10) + 1e-10
    n_near_zero = int(np.sum(eigvals < threshold))
    
    # 环绕数: 用前几个非零特征向量的相位变化估算
    if n_near_zero > 0 and n_near_zero < n_eig_actual:
        phi = eigvecs[:, n_near_zero]  # 第一个非零特征向量
        # 计算相位变化
        phases = np.arctan2(phi[1:], phi[:-1])
        # 相位累积
        total_phase = float(np.sum(np.abs(phases)))
        winding = min(total_phase / (2 * np.pi), 1.0)
    else:
        winding = 0.0
    
    # dim95
    total_var = float(np.sum(eigvals))
    cumvar = np.cumsum(eigvals) / (total_var + 1e-10)
    dim95 = int(np.searchsorted(cumvar, 0.95)) + 1 if len(cumvar) > 0 else n_eig_actual
    
    return n_near_zero, winding, dim95


# ==================== P305: 信号子空间拓扑回归 ====================

def run_p305(G_dict, key_layers, model_name):
    """P305: 信号子空间拓扑回归 — 去噪声后环绕数是否趋近1.0"""
    L.log("=== P305: 信号子空间拓扑回归 ===")
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
            G_centered, _ = noun_centered_G(G_sub, labels)
            if np.any(np.isnan(G_centered)) or np.all(G_centered == 0):
                continue
            
            # PCA降维到30维
            pca_dim = min(30, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            # 检查NaN
            if np.any(np.isnan(G_pca)) or np.any(np.isinf(G_pca)):
                L.log(f"    L{layer} {attr_type}: NaN/Inf in G_pca, skipping")
                continue
            
            # 原始空间的拓扑
            n_near_zero_orig, winding_orig, dim95_orig = compute_winding_number(G_pca)
            
            # 构建indicator矩阵
            Y, attr_list = build_indicator_matrix(labels, attr_type)
            
            # CCA信号/噪声分解
            for n_cca_dim in [5, 8, 10]:
                if n_cca_dim >= min(pca_dim, Y.shape[1]):
                    continue
                G_signal, G_noise, cca, P_s, cca_corrs = compute_signal_noise_subspaces(
                    G_pca, Y, n_components=n_cca_dim
                )
                
                # 信号空间的拓扑
                n_near_zero_sig, winding_sig, dim95_sig = compute_winding_number(G_signal)
                
                # 噪声空间的拓扑
                n_near_zero_noise, winding_noise, dim95_noise = compute_winding_number(G_noise)
                
                # 信号/噪声的方差比
                var_signal = float(np.mean(np.sum(G_signal**2, axis=1)))
                var_noise = float(np.mean(np.sum(G_noise**2, axis=1)))
                snr = var_signal / (var_noise + 1e-10)
                
                # 信号空间重构余弦
                cos_sig = []
                for i in range(min(N, 200)):
                    n1 = np.linalg.norm(G_pca[i])
                    n2 = np.linalg.norm(G_signal[i])
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_sig.append(float(np.dot(G_pca[i], G_signal[i]) / (n1 * n2)))
                mean_cos_sig = float(np.mean(cos_sig)) if cos_sig else 0.0
                
                layer_result = {
                    "layer": layer,
                    "n_cca": n_cca_dim,
                    "original": {"n_near_zero": n_near_zero_orig, "winding": round(winding_orig, 4), "dim95": dim95_orig},
                    "signal": {"n_near_zero": n_near_zero_sig, "winding": round(winding_sig, 4), "dim95": dim95_sig},
                    "noise": {"n_near_zero": n_near_zero_noise, "winding": round(winding_noise, 4), "dim95": dim95_noise},
                    "snr": round(snr, 4),
                    "cos_signal": round(mean_cos_sig, 4),
                    "cca_corrs": cca_corrs[:5],
                }
                results[attr_type].append(layer_result)
                
                if n_cca_dim == 10:
                    L.log(f"    L{layer} {attr_type}(k={n_cca_dim}): "
                          f"wind_orig={winding_orig:.3f}→wind_sig={winding_sig:.3f}, "
                          f"SNR={snr:.2f}, cos_sig={mean_cos_sig:.3f}")
    
    return results


# ==================== P306: 噪声独立性测试 ====================

def run_p306(G_dict, key_layers, model_name):
    """P306: 噪声独立性测试 — 噪声是通用的还是属性特异的"""
    L.log("=== P306: 噪声独立性测试 ===")
    results = []
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_total = N_color + N_taste + N_size
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        # 对3个属性子空间分别做信号/噪声分解
        noise_spaces = {}
        signal_spaces = {}
        
        for attr_type, n_start, n_end, labels in [
            ("color", 0, N_color, COLOR_LABELS),
            ("taste", N_color, N_color + N_taste, TASTE_LABELS),
            ("size", N_color + N_taste, N_total, SIZE_LABELS),
        ]:
            G_sub = G_all[n_start:n_end]
            N, D = G_sub.shape
            if N < 60:
                continue
            
            G_centered, _ = noun_centered_G(G_sub, labels)
            pca_dim = min(30, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            Y, attr_list = build_indicator_matrix(labels, attr_type)
            n_cca = min(10, pca_dim - 1, Y.shape[1] - 1)
            
            G_signal, G_noise, cca, P_s, cca_corrs = compute_signal_noise_subspaces(
                G_pca, Y, n_components=n_cca
            )
            
            noise_spaces[attr_type] = G_noise
            signal_spaces[attr_type] = G_signal
        
        # 计算噪声空间之间的重叠度
        if len(noise_spaces) == 3:
            # 用CCA衡量两个噪声空间的重叠
            pairs = [("color", "taste"), ("color", "size"), ("taste", "size")]
            overlap_results = {}
            
            for a1, a2 in pairs:
                N1 = noise_spaces[a1].shape[0]
                N2 = noise_spaces[a2].shape[0]
                # 用子采样使两个集合大小相同
                N_min = min(N1, N2)
                idx1 = np.random.choice(N1, N_min, replace=False)
                idx2 = np.random.choice(N2, N_min, replace=False)
                
                noise1 = noise_spaces[a1][idx1]
                noise2 = noise_spaces[a2][idx2]
                
                # CCA重叠度
                n_overlap = min(5, noise1.shape[1] - 1, noise2.shape[1] - 1)
                if n_overlap > 0:
                    try:
                        cca_overlap = CCA(n_components=n_overlap, max_iter=300)
                        n1_cca, n2_cca = cca_overlap.fit_transform(noise1, noise2)
                        overlap_corrs = [round(abs(float(np.corrcoef(n1_cca[:, i], n2_cca[:, i])[0, 1])), 4) 
                                        for i in range(n_overlap)]
                    except:
                        overlap_corrs = [0.0] * n_overlap
                else:
                    overlap_corrs = []
                
                # 子空间重叠度: ||P1·P2||_F / ||P1||_F (P1,P2是投影矩阵)
                # 简化: 用PCA方向的重叠
                pca1 = PCA(n_components=10).fit(noise1)
                pca2 = PCA(n_components=10).fit(noise2)
                V1 = pca1.components_  # (10, D)
                V2 = pca2.components_  # (10, D)
                # 子空间重叠 = ||V1 @ V2^T||_F / sqrt(||V1||_F * ||V2||_F)
                overlap_matrix = V1 @ V2.T
                subspace_overlap = float(np.linalg.norm(overlap_matrix, 'fro') / 
                                        np.sqrt(np.linalg.norm(V1, 'fro')**2 * np.linalg.norm(V2, 'fro')**2))
                
                overlap_results[f"{a1}_vs_{a2}"] = {
                    "cca_corrs": overlap_corrs[:3],
                    "subspace_overlap": round(subspace_overlap, 4),
                }
                
                L.log(f"    L{layer} {a1} vs {a2} noise: cca_r={overlap_corrs[:3]}, "
                      f"subspace_overlap={subspace_overlap:.3f}")
            
            # 噪声的各向同性检测
            # 如果噪声是随机的, 特征值应该均匀衰减
            # 如果噪声有结构, 会有少量大特征值
            for attr_type in ["color", "taste", "size"]:
                noise_pca = PCA(n_components=15).fit(noise_spaces[attr_type])
                ev_ratio = float(noise_pca.explained_variance_ratio_[0] / 
                                (noise_pca.explained_variance_ratio_[-1] + 1e-10))
                ev_top3 = [round(float(v), 4) for v in noise_pca.explained_variance_ratio_[:3]]
                overlap_results[f"{attr_type}_noise_ev_ratio"] = round(ev_ratio, 2)
                overlap_results[f"{attr_type}_noise_ev_top3"] = ev_top3
            
            layer_result = {
                "layer": layer,
                "overlaps": overlap_results,
            }
            results.append(layer_result)
    
    return results


# ==================== P307: 算子干预实验 ====================

def run_p307(mdl, tok, device, G_dict, key_layers, model_name, word_hs):
    """P307: 算子干预实验 — 注入纯净算子, 观察logits"""
    L.log("=== P307: 算子干预实验 — 注入纯净算子 ===")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_total = N_color + N_taste + N_size
    
    # 获取lm_head
    lm_head = mdl.lm_head if hasattr(mdl, 'lm_head') else mdl.get_output_embeddings()
    
    for layer in key_layers[-3:]:  # 只在末3层做干预
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
            
            G_centered, noun_means = noun_centered_G(G_sub, labels)
            pca_dim = min(30, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_centered)
            
            Y, attr_list = build_indicator_matrix(labels, attr_type)
            n_cca = min(10, pca_dim - 1, Y.shape[1] - 1)
            
            G_signal, G_noise, cca, P_s, cca_corrs = compute_signal_noise_subspaces(
                G_pca, Y, n_components=n_cca
            )
            
            # 构造每个属性值的纯净算子(信号子空间中的均值)
            attr_operators_signal = {}  # 信号空间的属性算子
            attr_operators_full = {}    # 完整空间的属性算子(含f_base)
            
            attr_values = [l[2] for l in labels]
            for av in attr_list:
                mask = np.array([v == av for v in attr_values])
                if mask.sum() > 0:
                    # 信号空间的均值(在PCA空间)
                    mean_signal = G_signal[mask].mean(axis=0)
                    attr_operators_signal[av] = mean_signal
                    
                    # 完整G(含f_base): 先回到原始空间再加回noun_mean
                    mean_full_centered = pca.inverse_transform(mean_signal.reshape(1, -1))[0]
                    attr_operators_full[av] = mean_full_centered
            
            # 干预实验: 选3个测试名词(未见过的组合) + 3个属性
            test_nouns = ["cat", "car", "hammer"]  # 3个不同族的名词
            if attr_type == "color":
                test_attrs = ["red", "blue", "green"]
                # 扩展: 检查所有12个颜色
                eval_attrs = attr_list
            elif attr_type == "taste":
                test_attrs = ["sweet", "salty", "spicy"]
                eval_attrs = attr_list
            else:
                test_attrs = ["big", "tiny", "long"]
                eval_attrs = attr_list
            
            for test_noun in test_nouns:
                if test_noun not in word_hs or layer not in word_hs[test_noun]:
                    continue
                
                h_noun = word_hs[test_noun][layer]  # 名词的隐藏状态
                
                for test_attr in test_attrs:
                    if test_attr not in attr_operators_full:
                        continue
                    
                    # 纯净算子: f_base(noun) + signal_operator(attr)
                    operator = attr_operators_full[test_attr]
                    if test_noun in noun_means:
                        # 用训练集中该名词的f_base
                        f_base_noun = noun_means[test_noun]
                    else:
                        # 用平均f_base
                        f_base_noun = np.mean(list(noun_means.values()), axis=0)
                    
                    # 干预: h_new = h_noun + f_base + signal_operator
                    h_intervened = h_noun.numpy() + f_base_noun + operator
                    
                    # Baseline: h_noun (无干预)
                    h_baseline = h_noun.numpy()
                    
                    # 通过lm_head获取logits
                    h_intervened_t = torch.tensor(h_intervened, dtype=torch.float32).to(device)
                    h_baseline_t = torch.tensor(h_baseline, dtype=torch.float32).to(device)
                    
                    # lm_head需要的是模型dtype
                    h_intervened_t = h_intervened_t.to(lm_head.weight.dtype)
                    h_baseline_t = h_baseline_t.to(lm_head.weight.dtype)
                    
                    with torch.no_grad():
                        logits_intervened = lm_head(h_intervened_t.unsqueeze(0))  # (1, vocab)
                        logits_baseline = lm_head(h_baseline_t.unsqueeze(0))
                    
                    logits_diff = logits_intervened[0] - logits_baseline[0]
                    
                    # 检查目标属性词的logit提升
                    target_token_ids = []
                    for attr_word in eval_attrs:
                        tokens = tok.encode(attr_word, add_special_tokens=False)
                        target_token_ids.extend(tokens)
                    
                    # top-k logit变化
                    top_k = 20
                    top_vals, top_idx = torch.topk(logits_diff, top_k)
                    top_tokens = [tok.decode([idx.item()]) for idx in top_idx]
                    
                    # 目标属性词的logit变化
                    target_logit_changes = {}
                    for attr_word in eval_attrs:
                        tokens = tok.encode(attr_word, add_special_tokens=False)
                        if len(tokens) == 1:
                            tid = tokens[0]
                            if tid < logits_diff.shape[0]:
                                target_logit_changes[attr_word] = round(float(logits_diff[tid]), 4)
                    
                    # 目标属性词是否在top-k中
                    target_in_topk = sum(1 for t in test_attrs if t in top_tokens)
                    
                    result_entry = {
                        "layer": layer,
                        "noun": test_noun,
                        "attr": test_attr,
                        "top_k_tokens": top_tokens[:10],
                        "top_k_values": [round(float(v), 3) for v in top_vals[:10]],
                        "target_logit_changes": target_logit_changes,
                        "target_in_topk": target_in_topk,
                        "num_test_attrs": len(test_attrs),
                    }
                    results[attr_type].append(result_entry)
                    
                    L.log(f"    L{layer} {test_noun}+{test_attr}: top5={top_tokens[:5]}, "
                          f"target_changes={dict(list(target_logit_changes.items())[:3])}")
    
    return results


# ==================== P308: 算子线性组合实验 ====================

def run_p308(mdl, tok, device, G_dict, key_layers, model_name, word_hs):
    """P308: 算子线性组合 — 像调音台一样控制输出"""
    L.log("=== P308: 算子线性组合实验 ===")
    results = []
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    N_total = N_color + N_taste + N_size
    
    lm_head = mdl.lm_head if hasattr(mdl, 'lm_head') else mdl.get_output_embeddings()
    
    for layer in key_layers[-2:]:  # 只在末2层
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        # 颜色信号算子
        G_color = G_all[:N_color]
        G_centered_c, noun_means_c = noun_centered_G(G_color, COLOR_LABELS)
        if np.any(np.isnan(G_centered_c)) or np.all(G_centered_c == 0):
            continue
        pca_c = PCA(n_components=min(30, G_centered_c.shape[0]-1, G_centered_c.shape[1]))
        G_pca_c = pca_c.fit_transform(G_centered_c)
        if np.any(np.isnan(G_pca_c)) or np.any(np.isinf(G_pca_c)):
            continue
        Y_c, attr_list_c = build_indicator_matrix(COLOR_LABELS, "color")
        
        G_signal_c, _, cca_c, _, _ = compute_signal_noise_subspaces(G_pca_c, Y_c, n_components=10)
        
        # 大小信号算子
        G_size = G_all[N_color + N_taste:]
        G_centered_s, noun_means_s = noun_centered_G(G_size, SIZE_LABELS)
        if np.any(np.isnan(G_centered_s)) or np.all(G_centered_s == 0):
            continue
        pca_s = PCA(n_components=min(30, G_centered_s.shape[0]-1, G_centered_s.shape[1]))
        G_pca_s = pca_s.fit_transform(G_centered_s)
        if np.any(np.isnan(G_pca_s)) or np.any(np.isinf(G_pca_s)):
            continue
        Y_s, attr_list_s = build_indicator_matrix(SIZE_LABELS, "size")
        
        G_signal_s, _, cca_s, _, _ = compute_signal_noise_subspaces(G_pca_s, Y_s, n_components=10)
        
        # 构造组合算子: α·red + β·big
        color_operators = {}
        for av in attr_list_c:
            mask = np.array([l[2] == av for l in COLOR_LABELS])
            if mask.sum() > 0:
                mean_sig = G_signal_c[mask].mean(axis=0)
                color_operators[av] = pca_c.inverse_transform(mean_sig.reshape(1, -1))[0]
        
        size_operators = {}
        for av in attr_list_s:
            mask = np.array([l[2] == av for l in SIZE_LABELS])
            if mask.sum() > 0:
                mean_sig = G_signal_s[mask].mean(axis=0)
                size_operators[av] = pca_s.inverse_transform(mean_sig.reshape(1, -1))[0]
        
        # 测试: "red" + "big" = "big red object"
        test_noun = "cat"
        if test_noun not in word_hs or layer not in word_hs[test_noun]:
            continue
        
        h_noun = word_hs[test_noun][layer].numpy()
        
        # 不同α,β组合
        for alpha in [0.5, 1.0, 1.5]:
            for beta in [0.5, 1.0, 1.5]:
                if "red" not in color_operators or "big" not in size_operators:
                    continue
                
                combo_operator = alpha * color_operators["red"] + beta * size_operators["big"]
                
                # 平均f_base
                f_base_avg = np.mean(list(noun_means_c.values()), axis=0)
                h_combo = h_noun + f_base_avg + combo_operator
                
                h_baseline = h_noun.copy()
                
                h_combo_t = torch.tensor(h_combo, dtype=torch.float32).to(device).to(lm_head.weight.dtype)
                h_base_t = torch.tensor(h_baseline, dtype=torch.float32).to(device).to(lm_head.weight.dtype)
                
                with torch.no_grad():
                    logits_combo = lm_head(h_combo_t.unsqueeze(0))
                    logits_base = lm_head(h_base_t.unsqueeze(0))
                
                logits_diff = logits_combo[0] - logits_base[0]
                top_vals, top_idx = torch.topk(logits_diff, 15)
                top_tokens = [tok.decode([idx.item()]) for idx in top_idx]
                
                # 颜色词logit变化
                color_logit_changes = {}
                for cw in ["red", "blue", "green", "yellow", "black", "white"]:
                    tids = tok.encode(cw, add_special_tokens=False)
                    if len(tids) == 1 and tids[0] < logits_diff.shape[0]:
                        color_logit_changes[cw] = round(float(logits_diff[tids[0]]), 3)
                
                # 大小词logit变化
                size_logit_changes = {}
                for sw in ["big", "small", "huge", "tiny", "tall", "short"]:
                    tids = tok.encode(sw, add_special_tokens=False)
                    if len(tids) == 1 and tids[0] < logits_diff.shape[0]:
                        size_logit_changes[sw] = round(float(logits_diff[tids[0]]), 3)
                
                result_entry = {
                    "layer": layer,
                    "alpha_red": alpha,
                    "beta_big": beta,
                    "top_tokens": top_tokens[:10],
                    "top_values": [round(float(v), 3) for v in top_vals[:10]],
                    "color_logit_changes": color_logit_changes,
                    "size_logit_changes": size_logit_changes,
                }
                results.append(result_entry)
                
                L.log(f"    L{layer} α_red={alpha}, β_big={beta}: "
                      f"red_logit={color_logit_changes.get('red', 0):.3f}, "
                      f"big_logit={size_logit_changes.get('big', 0):.3f}")
    
    return results


# ==================== P309: 跨名词泛化实验 ====================

def run_p309(mdl, tok, device, G_dict, key_layers, model_name, word_hs):
    """P309: 跨名词泛化 — 用水果族训练的算子注入到动物/工具"""
    L.log("=== P309: 跨名词泛化实验 ===")
    results = []
    
    N_color = len(COLOR_TRIPLES)
    
    lm_head = mdl.lm_head if hasattr(mdl, 'lm_head') else mdl.get_output_embeddings()
    
    for layer in key_layers[-2:]:
        if layer not in G_dict or len(G_dict[layer]) < 2000:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        G_color = G_all[:N_color]
        
        # 只用水果族训练算子
        fruit_labels = [(n, "color", c, "fruit_family") for n in STIMULI["fruit_family"] 
                        for c in STIMULI["color_attrs"]]
        fruit_nouns = set(STIMULI["fruit_family"])
        fruit_mask = np.array([l[0] in fruit_nouns for l in COLOR_LABELS])
        
        G_fruit = G_color[fruit_mask]
        labels_fruit = [l for l, m in zip(COLOR_LABELS, fruit_mask) if m]
        
        if len(G_fruit) < 60:
            continue
        
        G_centered_f, noun_means_f = noun_centered_G(G_fruit, labels_fruit)
        pca_f = PCA(n_components=min(30, len(G_fruit) - 1, G_fruit.shape[1]))
        G_pca_f = pca_f.fit_transform(G_centered_f)
        if np.any(np.isnan(G_pca_f)) or np.any(np.isinf(G_pca_f)):
            continue
        Y_f, attr_list_f = build_indicator_matrix(labels_fruit, "color")
        
        G_signal_f, _, cca_f, _, cca_corrs_f = compute_signal_noise_subspaces(
            G_pca_f, Y_f, n_components=min(10, G_pca_f.shape[1] - 1, Y_f.shape[1] - 1)
        )
        
        # 构造水果族算子
        fruit_operators = {}
        for av in attr_list_f:
            mask = np.array([l[2] == av for l in labels_fruit])
            if mask.sum() > 0:
                mean_sig = G_signal_f[mask].mean(axis=0)
                fruit_operators[av] = pca_f.inverse_transform(mean_sig.reshape(1, -1))[0]
        
        # 注入到非水果族名词
        test_nouns = {
            "animal": ["cat", "dog", "lion"],
            "vehicle": ["car", "bus", "plane"],
            "tool": ["hammer", "knife", "drill"],
        }
        
        for family, nouns in test_nouns.items():
            for test_noun in nouns:
                if test_noun not in word_hs or layer not in word_hs[test_noun]:
                    continue
                
                h_noun = word_hs[test_noun][layer].numpy()
                
                for test_attr in ["red", "blue", "green"]:
                    if test_attr not in fruit_operators:
                        continue
                    
                    # 用平均f_base (因为非水果名词没有训练集f_base)
                    f_base_avg = np.mean(list(noun_means_f.values()), axis=0)
                    h_intervened = h_noun + f_base_avg + fruit_operators[test_attr]
                    
                    h_base_t = torch.tensor(h_noun, dtype=torch.float32).to(device).to(lm_head.weight.dtype)
                    h_int_t = torch.tensor(h_intervened, dtype=torch.float32).to(device).to(lm_head.weight.dtype)
                    
                    with torch.no_grad():
                        logits_int = lm_head(h_int_t.unsqueeze(0))
                        logits_base = lm_head(h_base_t.unsqueeze(0))
                    
                    logits_diff = logits_int[0] - logits_base[0]
                    top_vals, top_idx = torch.topk(logits_diff, 15)
                    top_tokens = [tok.decode([idx.item()]) for idx in top_idx]
                    
                    # 目标颜色词logit变化
                    target_changes = {}
                    for cw in STIMULI["color_attrs"]:
                        tids = tok.encode(cw, add_special_tokens=False)
                        if len(tids) == 1 and tids[0] < logits_diff.shape[0]:
                            target_changes[cw] = round(float(logits_diff[tids[0]]), 3)
                    
                    result_entry = {
                        "layer": layer,
                        "family": family,
                        "noun": test_noun,
                        "attr": test_attr,
                        "top_tokens": top_tokens[:8],
                        "target_color_changes": target_changes,
                        "target_attr_in_top10": test_attr in top_tokens[:10],
                    }
                    results.append(result_entry)
                    
                    L.log(f"    L{layer} {family}/{test_noun}+{test_attr}: "
                          f"in_top10={test_attr in top_tokens[:10]}, "
                          f"target_change={target_changes.get(test_attr, 0):.3f}")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"Phase LIII: 正交空间分解与算子提纯 — {model_name}")
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
    
    # 收集干预实验所需的隐藏状态
    L.log("收集干预实验所需的隐藏状态...")
    intervention_words = list(set(ALL_NOUNS + ["cat", "car", "hammer", "dog", "bus", "knife", 
                                                 "lion", "plane", "drill"]))
    word_hs = collect_hidden_states(mdl, tok, device, key_layers, intervention_words, PROMPT_TEMPLATES_30)
    L.log(f"  收集了{len(word_hs)}个词的隐藏状态")
    
    # P305: 信号子空间拓扑回归
    p305_results = run_p305(G_dict, key_layers, model_name)
    
    # P306: 噪声独立性测试
    p306_results = run_p306(G_dict, key_layers, model_name)
    
    # P307: 算子干预实验
    p307_results = run_p307(mdl, tok, device, G_dict, key_layers, model_name, word_hs)
    
    # P308: 算子线性组合
    p308_results = run_p308(mdl, tok, device, G_dict, key_layers, model_name, word_hs)
    
    # P309: 跨名词泛化
    p309_results = run_p309(mdl, tok, device, G_dict, key_layers, model_name, word_hs)
    
    # 保存结果
    all_results = {
        "model": model_name,
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M'),
        "p305_signal_topology": p305_results,
        "p306_noise_independence": p306_results,
        "p307_operator_intervention": p307_results,
        "p308_operator_combination": p308_results,
        "p309_cross_noun_generalization": p309_results,
    }
    
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_file = OUT_DIR / f"phase_liii_p305_309_{model_name}_{ts}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    L.log(f"结果已保存: {out_file}")
    
    # 最终总结
    L.log("\n===== Phase LIII 总结 =====")
    
    # P305总结
    L.log("--- P305: 信号子空间拓扑 ---")
    for attr_type in ["color", "taste", "size"]:
        for r in p305_results.get(attr_type, []):
            if r["n_cca"] == 10 and r["layer"] == key_layers[-1]:
                L.log(f"  {attr_type} L{r['layer']}: wind_orig={r['original']['winding']:.3f} "
                      f"→ wind_sig={r['signal']['winding']:.3f}, SNR={r['snr']:.2f}")
    
    # P307总结
    L.log("--- P307: 算子干预 ---")
    for attr_type in ["color", "taste", "size"]:
        entries = p307_results.get(attr_type, [])
        if entries:
            in_topk_count = sum(1 for e in entries if e.get("target_in_topk", 0) > 0)
            L.log(f"  {attr_type}: {in_topk_count}/{len(entries)} interventions had target in top-k")
    
    # P309总结
    L.log("--- P309: 跨名词泛化 ---")
    if p309_results:
        in_top10 = sum(1 for r in p309_results if r.get("target_attr_in_top10", False))
        L.log(f"  Cross-noun: {in_top10}/{len(p309_results)} had target attr in top-10")
    
    L.log(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.close()

if __name__ == "__main__":
    main()
