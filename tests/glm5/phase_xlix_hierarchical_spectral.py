"""
Phase XLIX-P285/286/287/288/289: 分层谱分析 — 揭示属性子空间内部几何
======================================================================

Phase XLVIII核心突破:
  1. Laplacian频谱 = 语义层级: φ1=属性类型, φ2-4=家族, φ5+=名词
  2. 颜色编码弱且分布式: max_color_corr<0.13
  3. 属性类型是最大尺度结构(φ1, r=0.51-0.82)
  4. Laplacian重构不如PCA: 0.83 vs 0.91

关键问题:
  1. 6维颜色子空间的内部几何是什么? (Phase XLVII发现dim_95=6)
  2. 在属性类型子空间内, Laplacian特征向量是否编码属性内部结构?
  3. 颜色子空间是否真的是环面? 还是球面? 还是更复杂?
  4. 味道和大小子空间的几何是什么?

核心思路:
  在全局Laplacian分析中, φ1编码属性类型, 这是跨类型的区分
  但在属性类型内部(如固定"颜色"), φ应该编码颜色内部的结构
  固定属性类型后, Laplacian特征向量应该揭示:
    - 颜色子空间: 色相环方向, 亮度, 饱和度
    - 味道子空间: 甜-苦轴, 酸-咸轴
    - 大小子空间: 大-小轴, 长-短轴, 重-轻轴

五大实验:
  P285: 属性子空间Laplacian谱
    - 对每种属性类型(color/taste/size), 单独做Laplacian分析
    - 颜色子空间: 120个样本(10名词×12色)
    - 味道子空间: 120个样本(10名词×12味)
    - 大小子空间: 120个样本(10名词×12大小)
    - 每个子空间求Laplacian特征值和特征向量

  P286: 子空间特征向量与属性值对应(核心!)
    - 颜色子空间: φ_i与12种颜色的相关 → 色相? 亮度? 饱和度?
    - 味道子空间: φ_i与12种味道的相关 → 甜-苦轴? 酸-咸轴?
    - 大小子空间: φ_i与12种大小的相关 → 大-小轴? 重-轻轴?
    - 回答"6维颜色子空间中每个维度编码什么?"

  P287: 子空间流形类型判断
    - 颜色子空间: 环面(T^k)? 球面(S^k)? 欧氏(R^k)?
    - 用环绕数、PCA维度、Laplacian特征值分布判断
    - 关键: 如果λ1≈0, λ2≈0 (两个接近零的特征值), 则环面假设成立

  P288: 子空间正交性与交互
    - 颜色/味道/大小子空间是否正交?
    - 子空间之间的夹角
    - 交互项: 颜色子空间的方向是否依赖名词?

  P289: 分层Laplacian重构公式
    - G ≈ Σ a_i·φ_i^(全局) + Σ b_j·φ_j^(颜色) + Σ c_k·φ_k^(味道) + Σ d_l·φ_l^(大小)
    - 计算分层重构的精度
    - 与单层PCA30对比

数据规模: 360三元组(与Phase XLVIII相同) × 30模板
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
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_xlix_log.txt"

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

SELECTED_NOUNS = ["apple","cherry", "cat","dog", "car","bus", "chair","table", "knife","hammer"]
NOUN_TO_FAMILY = {}
for fam, words in STIMULI.items():
    if "family" in fam:
        for w in words:
            NOUN_TO_FAMILY[w] = fam

# 三种属性类型各120个三元组
COLOR_TRIPLES = [(n, c, f"{c} {n}") for n in SELECTED_NOUNS for c in STIMULI["color_attrs"]]
TASTE_TRIPLES = [(n, t, f"{t} {n}") for n in SELECTED_NOUNS for t in STIMULI["taste_attrs"]]
SIZE_TRIPLES = [(n, s, f"{s} {n}") for n in SELECTED_NOUNS for s in STIMULI["size_attrs"]]

# 标签
COLOR_LABELS = [(n, "color", c, NOUN_TO_FAMILY.get(n,"unknown")) for n in SELECTED_NOUNS for c in STIMULI["color_attrs"]]
TASTE_LABELS = [(n, "taste", t, NOUN_TO_FAMILY.get(n,"unknown")) for n in SELECTED_NOUNS for t in STIMULI["taste_attrs"]]
SIZE_LABELS = [(n, "size", s, NOUN_TO_FAMILY.get(n,"unknown")) for n in SELECTED_NOUNS for s in STIMULI["size_attrs"]]

# 全部360个三元组
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


# ==================== P285: 属性子空间Laplacian谱 ====================

def run_p285(G_dict, key_layers):
    """P285: 三种属性子空间的Laplacian谱分析"""
    L.log("P285: 属性子空间Laplacian谱分析")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)  # 120
    N_taste = len(TASTE_TRIPLES)  # 120
    N_size = len(SIZE_TRIPLES)    # 120
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 300:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_attr, attr_list in [
            ("color", N_color, STIMULI["color_attrs"]),
            ("taste", N_taste, STIMULI["taste_attrs"]),
            ("size", N_size, STIMULI["size_attrs"]),
        ]:
            if attr_type == "color":
                G_sub = G_all[:N_color]
                labels = COLOR_LABELS
            elif attr_type == "taste":
                G_sub = G_all[N_color:N_color+N_taste]
                labels = TASTE_LABELS
            else:
                G_sub = G_all[N_color+N_taste:]
                labels = SIZE_LABELS
            
            N, D = G_sub.shape
            if N < 30:
                continue
            
            # PCA降维
            pca_dim = min(15, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_sub)
            
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            dim_95 = int(np.searchsorted(cumvar, 0.95) + 1) if len(cumvar) > 0 else 0
            dim_90 = int(np.searchsorted(cumvar, 0.90) + 1) if len(cumvar) > 0 else 0
            
            # Laplacian谱
            eigenvalues, eigenvectors = compute_graph_laplacian_spectra(G_pca, n_neighbors=10, n_eigenvectors=20)
            if eigenvalues is None:
                continue
            
            # 特征值间隙
            eig_gaps = []
            for i in range(1, len(eigenvalues)):
                if eigenvalues[i-1] > 1e-10:
                    eig_gaps.append(eigenvalues[i] / eigenvalues[i-1])
                else:
                    eig_gaps.append(0.0)
            
            # 接近零的特征值数量(暗示环面/连通分量)
            n_near_zero = int(np.sum(eigenvalues < 0.01))
            n_very_small = int(np.sum(eigenvalues < 0.05))
            
            # 环绕数(前2个Laplacian特征向量平面)
            winding = 0.0
            if eigenvectors.shape[1] >= 3:
                # 用φ1和φ2(跳过φ0常数向量)画环绕数
                phi1 = eigenvectors[:, 1]
                phi2 = eigenvectors[:, 2]
                angles = np.arctan2(phi2, phi1)
                sorted_idx = np.argsort(angles)
                angle_diffs = np.diff(angles[sorted_idx])
                angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
                winding = float(np.sum(angle_diffs) / (2 * np.pi))
            
            layer_result = {
                "layer": layer,
                "N_samples": N,
                "pca_dim": pca_dim,
                "dim_95": dim_95,
                "dim_90": dim_90,
                "eigenvalues_top10": [round(float(e), 6) for e in eigenvalues[:10]],
                "n_near_zero_eigenvalues": n_near_zero,
                "n_very_small_eigenvalues": n_very_small,
                "winding_number_phi1_phi2": round(winding, 3),
                "eigenvalue_gaps_top10": [round(g, 4) for g in eig_gaps[:10]],
            }
            
            results[attr_type].append(layer_result)
            L.log(f"    L{layer} {attr_type}: dim_95={dim_95}, λ1-3={[round(float(e),4) for e in eigenvalues[:3]]}, "
                  f"near_zero={n_near_zero}, winding={winding:.3f}")
    
    return results


# ==================== P286: 子空间特征向量与属性值对应 ====================

def run_p286(G_dict, key_layers):
    """P286: 子空间内Laplacian特征向量与属性值的对应"""
    L.log("P286: 子空间特征向量-属性值对应分析")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 300:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_attr, attr_list, labels in [
            ("color", N_color, STIMULI["color_attrs"], COLOR_LABELS),
            ("taste", N_taste, STIMULI["taste_attrs"], TASTE_LABELS),
            ("size", N_size, STIMULI["size_attrs"], SIZE_LABELS),
        ]:
            if attr_type == "color":
                G_sub = G_all[:N_color]
            elif attr_type == "taste":
                G_sub = G_all[N_color:N_color+N_taste]
            else:
                G_sub = G_all[N_color+N_taste:]
            
            N, D = G_sub.shape
            nouns = [l[0] for l in labels]
            attr_values = [l[2] for l in labels]
            families = [l[3] for l in labels]
            
            if N < 30:
                continue
            
            pca_dim = min(15, N - 1, D)
            pca = PCA(n_components=pca_dim)
            G_pca = pca.fit_transform(G_sub)
            
            eigenvalues, eigenvectors = compute_graph_laplacian_spectra(G_pca, n_neighbors=10, n_eigenvectors=15)
            if eigenvalues is None:
                continue
            
            n_eig = eigenvectors.shape[1]
            
            # 对每个特征向量, 分析其与属性值/名词/家族的相关
            eig_semantics = []
            for i in range(min(10, n_eig)):
                phi = eigenvectors[:, i]
                
                # 1) 与每个属性值的indicator相关
                attr_value_corrs = {}
                for av in attr_list:
                    indicator = np.array([1.0 if v == av else 0.0 for v in attr_values])
                    if np.std(phi) > 1e-10 and np.std(indicator) > 1e-10:
                        corr = float(np.corrcoef(phi, indicator)[0, 1])
                    else:
                        corr = 0.0
                    attr_value_corrs[av] = round(corr, 4)
                
                # 2) 与每个名词的indicator相关
                unique_nouns = sorted(set(nouns))
                noun_corrs = {}
                for n in unique_nouns:
                    indicator = np.array([1.0 if nn == n else 0.0 for nn in nouns])
                    if np.std(phi) > 1e-10 and np.std(indicator) > 1e-10:
                        corr = float(np.corrcoef(phi, indicator)[0, 1])
                    else:
                        corr = 0.0
                    noun_corrs[n] = round(corr, 4)
                
                # 3) 与家族的indicator相关
                unique_families = sorted(set(families))
                family_corrs = {}
                for f in unique_families:
                    indicator = np.array([1.0 if ff == f else 0.0 for ff in families])
                    if np.std(phi) > 1e-10 and np.std(indicator) > 1e-10:
                        corr = float(np.corrcoef(phi, indicator)[0, 1])
                    else:
                        corr = 0.0
                    family_corrs[f] = round(corr, 4)
                
                # 4) 最佳语义标签
                all_corrs = {}
                all_corrs.update({f"attr_{k}": v for k, v in attr_value_corrs.items()})
                all_corrs.update({f"noun_{k}": v for k, v in noun_corrs.items()})
                all_corrs.update({f"fam_{k}": v for k, v in family_corrs.items()})
                
                best_label = max(all_corrs, key=all_corrs.get)
                best_corr = all_corrs[best_label]
                
                max_attr_corr = max(abs(v) for v in attr_value_corrs.values())
                max_noun_corr = max(abs(v) for v in noun_corrs.values())
                max_fam_corr = max(abs(v) for v in family_corrs.values())
                
                # 5) 语义分类: 这个φ主要编码属性值? 名词? 还是家族?
                if max_attr_corr > max_noun_corr and max_attr_corr > max_fam_corr:
                    primary = "attr_value"
                elif max_noun_corr > max_fam_corr:
                    primary = "noun"
                else:
                    primary = "family"
                
                # 6) 属性值的排序(φ_i值从小到大的属性排列)
                attr_phi_means = {}
                for av in attr_list:
                    mask = np.array(attr_values) == av
                    if mask.sum() > 0:
                        attr_phi_means[av] = float(np.mean(phi[mask]))
                sorted_attrs = sorted(attr_phi_means.items(), key=lambda x: x[1])
                
                eig_semantics.append({
                    "eig_idx": i,
                    "eigenvalue": round(float(eigenvalues[i]), 6) if i < len(eigenvalues) else 0,
                    "best_semantic": best_label,
                    "best_corr": round(best_corr, 4),
                    "max_attr_corr": round(max_attr_corr, 4),
                    "max_noun_corr": round(max_noun_corr, 4),
                    "max_fam_corr": round(max_fam_corr, 4),
                    "primary_encoding": primary,
                    "attr_value_corrs": attr_value_corrs,
                    "noun_corrs": noun_corrs,
                    "family_corrs": family_corrs,
                    "sorted_attrs_by_phi": [(a, round(v, 4)) for a, v in sorted_attrs],
                })
            
            layer_result = {
                "layer": layer,
                "attr_type": attr_type,
                "N_samples": N,
                "eigenvector_semantics": eig_semantics,
            }
            
            results[attr_type].append(layer_result)
            
            if layer == max(key_layers):
                L.log(f"    L{layer} {attr_type}子空间特征向量:")
                for es in eig_semantics[:7]:
                    sorted_top3 = es["sorted_attrs_by_phi"][:3]
                    sorted_bot3 = es["sorted_attrs_by_phi"][-3:]
                    L.log(f"      φ{es['eig_idx']}: {es['primary_encoding']}(best={es['best_semantic']},r={es['best_corr']:.3f}), "
                          f"attr_r={es['max_attr_corr']:.3f}, noun_r={es['max_noun_corr']:.3f}, fam_r={es['max_fam_corr']:.3f}")
                    L.log(f"        负端: {sorted_top3}, 正端: {sorted_bot3}")
    
    return results


# ==================== P287: 子空间流形类型判断 ====================

def run_p287(G_dict, key_layers):
    """P287: 子空间流形类型判断 — 环面? 球面? 欧氏?"""
    L.log("P287: 子空间流形类型判断")
    results = {"color": [], "taste": [], "size": []}
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    N_size = len(SIZE_TRIPLES)
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 300:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        for attr_type, n_attr, attr_list, labels in [
            ("color", N_color, STIMULI["color_attrs"], COLOR_LABELS),
            ("taste", N_taste, STIMULI["taste_attrs"], TASTE_LABELS),
            ("size", N_size, STIMULI["size_attrs"], SIZE_LABELS),
        ]:
            if attr_type == "color":
                G_sub = G_all[:N_color]
            elif attr_type == "taste":
                G_sub = G_all[N_color:N_color+N_taste]
            else:
                G_sub = G_all[N_color+N_taste:]
            
            N, D = G_sub.shape
            if N < 30:
                continue
            
            # 固定名词后的子空间(单个名词×12属性值)
            per_noun_results = {}
            nouns = [l[0] for l in labels]
            unique_nouns = sorted(set(nouns))
            
            for noun in unique_nouns[:5]:  # 取前5个名词
                mask = np.array(nouns) == noun
                G_noun = G_sub[mask]  # (12, D)
                
                if G_noun.shape[0] < 12:
                    continue
                
                # PCA
                pca = PCA(n_components=min(12, G_noun.shape[1]))
                G_pca = pca.fit_transform(G_noun)
                
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                dim_95 = int(np.searchsorted(cumvar, 0.95) + 1) if len(cumvar) > 0 else 0
                
                # 环绕数(PC1-PC2平面)
                winding = 0.0
                if G_pca.shape[1] >= 2:
                    x, y = G_pca[:, 0], G_pca[:, 1]
                    angles = np.arctan2(y, x)
                    sorted_idx = np.argsort(angles)
                    angle_diffs = np.diff(angles[sorted_idx])
                    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
                    winding = float(np.sum(angle_diffs) / (2 * np.pi))
                
                # 闭合度(首尾距离 vs 平均步长)
                dists = pdist(G_pca[:, :min(3, G_pca.shape[1])])
                mean_dist = np.mean(dists) if len(dists) > 0 else 1.0
                if G_pca.shape[0] >= 2:
                    closure_dist = np.linalg.norm(G_pca[0, :min(3, G_pca.shape[1])] - G_pca[-1, :min(3, G_pca.shape[1])])
                    closure_ratio = closure_dist / (mean_dist + 1e-10)
                else:
                    closure_ratio = 0.0
                
                # 相邻属性余弦
                adj_cos = []
                for i in range(len(G_noun) - 1):
                    c = float(np.dot(G_noun[i], G_noun[i+1]) / 
                              (np.linalg.norm(G_noun[i]) * np.linalg.norm(G_noun[i+1]) + 1e-10))
                    adj_cos.append(c)
                mean_adj_cos = float(np.mean(adj_cos)) if adj_cos else 0.0
                
                # Laplacian(如果样本足够)
                lap_eigenvalues = None
                if G_pca.shape[0] >= 8:
                    ev, _ = compute_graph_laplacian_spectra(G_pca, n_neighbors=min(5, G_pca.shape[0]-1), n_eigenvectors=min(10, G_pca.shape[0]-1))
                    if ev is not None:
                        lap_eigenvalues = [round(float(e), 6) for e in ev[:5]]
                
                per_noun_results[noun] = {
                    "dim_95": dim_95,
                    "winding_number": round(winding, 3),
                    "closure_ratio": round(closure_ratio, 3),
                    "mean_adj_cos": round(mean_adj_cos, 4),
                    "lap_eigenvalues_top5": lap_eigenvalues,
                    "top3_var_ratio": pca.explained_variance_ratio_[:3].tolist(),
                }
            
            # 全局子空间分析
            pca_all = PCA(n_components=min(30, N-1, D))
            G_pca_all = pca_all.fit_transform(G_sub)
            cumvar_all = np.cumsum(pca_all.explained_variance_ratio_)
            dim_95_all = int(np.searchsorted(cumvar_all, 0.95) + 1) if len(cumvar_all) > 0 else 0
            
            # 判断流形类型
            manifold_type = "unknown"
            manifold_reason = ""
            
            # 环面判断: 有≥1个接近零的Laplacian特征值 + 环绕数≈整数
            eigenvalues_all, _ = compute_graph_laplacian_spectra(G_pca_all, n_neighbors=10, n_eigenvectors=20)
            if eigenvalues_all is not None:
                n_near_zero = int(np.sum(eigenvalues_all < 0.05))
                avg_winding = np.mean([v["winding_number"] for v in per_noun_results.values()])
                
                if n_near_zero >= 2 and abs(avg_winding) > 0.5:
                    manifold_type = "torus"
                    manifold_reason = f"n_near_zero={n_near_zero}, avg_winding={avg_winding:.3f}"
                elif n_near_zero >= 1 and abs(avg_winding) > 0.3:
                    manifold_type = "partial_torus"
                    manifold_reason = f"n_near_zero={n_near_zero}, avg_winding={avg_winding:.3f}"
                else:
                    manifold_type = "euclidean"
                    manifold_reason = f"n_near_zero={n_near_zero}, avg_winding={avg_winding:.3f}"
            
            layer_result = {
                "layer": layer,
                "attr_type": attr_type,
                "global_dim_95": dim_95_all,
                "manifold_type": manifold_type,
                "manifold_reason": manifold_reason,
                "per_noun": per_noun_results,
            }
            
            results[attr_type].append(layer_result)
            L.log(f"    L{layer} {attr_type}: dim_95={dim_95_all}, type={manifold_type}({manifold_reason})")
    
    return results


# ==================== P288: 子空间正交性与交互 ====================

def run_p288(G_dict, key_layers):
    """P288: 颜色/味道/大小子空间的正交性与交互"""
    L.log("P288: 子空间正交性与交互分析")
    results = []
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 300:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        
        G_color = G_all[:N_color]
        G_taste = G_all[N_color:N_color+N_taste]
        G_size = G_all[N_color+N_taste:]
        
        # 每个子空间做PCA
        pca_c = PCA(n_components=min(10, G_color.shape[1]))
        pca_t = PCA(n_components=min(10, G_taste.shape[1]))
        pca_s = PCA(n_components=min(10, G_size.shape[1]))
        
        pca_c.fit(G_color)
        pca_t.fit(G_taste)
        pca_s.fit(G_size)
        
        # 子空间主方向
        dirs_c = pca_c.components_[:6]  # (6, D)
        dirs_t = pca_t.components_[:6]
        dirs_s = pca_s.components_[:6]
        
        # 子空间间余弦矩阵
        cos_ct = np.abs(dirs_c @ dirs_t.T)  # (6, 6)
        cos_cs = np.abs(dirs_c @ dirs_s.T)
        cos_ts = np.abs(dirs_t @ dirs_s.T)
        
        # 子空间夹角(Principal angles)
        from scipy.linalg import subspace_angles
        try:
            angles_ct = subspace_angles(dirs_c.T, dirs_t.T)  # (min(6,6),)
            angles_cs = subspace_angles(dirs_c.T, dirs_s.T)
            angles_ts = subspace_angles(dirs_t.T, dirs_s.T)
            min_angle_ct = float(np.min(angles_ct)) * 180 / np.pi
            min_angle_cs = float(np.min(angles_cs)) * 180 / np.pi
            min_angle_ts = float(np.min(angles_ts)) * 180 / np.pi
            mean_angle_ct = float(np.mean(angles_ct)) * 180 / np.pi
            mean_angle_cs = float(np.mean(angles_cs)) * 180 / np.pi
            mean_angle_ts = float(np.mean(angles_ts)) * 180 / np.pi
        except Exception:
            min_angle_ct = 90.0
            min_angle_cs = 90.0
            min_angle_ts = 90.0
            mean_angle_ct = 90.0
            mean_angle_cs = 90.0
            mean_angle_ts = 90.0
        
        # 交互: 颜色方向是否依赖名词?
        nouns = [l[0] for l in COLOR_LABELS]
        unique_nouns = sorted(set(nouns))
        
        per_noun_color_dirs = {}
        for noun in unique_nouns:
            mask = np.array(nouns) == noun
            G_noun = G_color[mask]
            if G_noun.shape[0] >= 12:
                pca_n = PCA(n_components=min(6, G_noun.shape[1]))
                pca_n.fit(G_noun)
                per_noun_color_dirs[noun] = pca_n.components_[:6]
        
        # 跨名词颜色子空间的对齐度
        cross_noun_alignments = {}
        noun_list = list(per_noun_color_dirs.keys())
        for i in range(len(noun_list)):
            for j in range(i+1, len(noun_list)):
                n1, n2 = noun_list[i], noun_list[j]
                d1 = per_noun_color_dirs[n1]
                d2 = per_noun_color_dirs[n2]
                # 子空间余弦矩阵的平均
                cos_matrix = np.abs(d1 @ d2.T)
                mean_align = float(np.mean(np.max(cos_matrix, axis=1)))
                cross_noun_alignments[f"{n1}_vs_{n2}"] = round(mean_align, 4)
        
        layer_result = {
            "layer": layer,
            "subspace_max_cos_ct": round(float(np.max(cos_ct)), 4),
            "subspace_max_cos_cs": round(float(np.max(cos_cs)), 4),
            "subspace_max_cos_ts": round(float(np.max(cos_ts)), 4),
            "subspace_mean_cos_ct": round(float(np.mean(cos_ct)), 4),
            "subspace_mean_cos_cs": round(float(np.mean(cos_cs)), 4),
            "subspace_mean_cos_ts": round(float(np.mean(cos_ts)), 4),
            "min_angle_ct_deg": round(min_angle_ct, 2),
            "min_angle_cs_deg": round(min_angle_cs, 2),
            "min_angle_ts_deg": round(min_angle_ts, 2),
            "mean_angle_ct_deg": round(mean_angle_ct, 2),
            "mean_angle_cs_deg": round(mean_angle_cs, 2),
            "mean_angle_ts_deg": round(mean_angle_ts, 2),
            "cross_noun_color_alignment": cross_noun_alignments,
        }
        
        results.append(layer_result)
        L.log(f"    L{layer}: angle(C,T)={mean_angle_ct:.1f}°, angle(C,S)={mean_angle_cs:.1f}°, "
              f"angle(T,S)={mean_angle_ts:.1f}°, cross_noun_align={np.mean(list(cross_noun_alignments.values())):.3f}")
    
    return results


# ==================== P289: 分层Laplacian重构 ====================

def run_p289(G_dict, key_layers):
    """P289: 分层Laplacian重构 — 全局+子空间"""
    L.log("P289: 分层Laplacian重构")
    results = []
    
    N_color = len(COLOR_TRIPLES)
    N_taste = len(TASTE_TRIPLES)
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 300:
            continue
        
        G_all = torch.stack(G_dict[layer]).numpy()
        N, D = G_all.shape
        
        # 全局PCA30重构(基准)
        pca30 = PCA(n_components=min(30, N-1, D))
        G_pca30 = pca30.fit_transform(G_all)
        G_recon_pca30 = pca30.inverse_transform(G_pca30)
        
        cos_pca30_vals = []
        for i in range(N):
            no = np.linalg.norm(G_all[i])
            nr = np.linalg.norm(G_recon_pca30[i])
            if no > 1e-10 and nr > 1e-10:
                cos_pca30_vals.append(float(np.dot(G_all[i], G_recon_pca30[i]) / (no * nr)))
        cos_pca30 = float(np.mean(cos_pca30_vals))
        
        # 全局Laplacian重构
        pca_dim = min(30, N-1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_all)
        eigenvalues, eigenvectors = compute_graph_laplacian_spectra(G_pca, n_neighbors=15, n_eigenvectors=30)
        
        cos_lap_global = 0.0
        if eigenvalues is not None:
            k_eig = min(10, eigenvectors.shape[1])
            Phi_k = eigenvectors[:, 1:k_eig+1]
            nn_model = NearestNeighbors(n_neighbors=min(10, N-1), algorithm='ball_tree').fit(Phi_k)
            _, nn_idx = nn_model.kneighbors(Phi_k)
            
            G_recon_lap = np.zeros_like(G_pca)
            for j in range(N):
                G_recon_lap[j] = np.mean(G_pca[nn_idx[j]], axis=0)
            G_recon_lap_orig = pca.inverse_transform(G_recon_lap)
            
            cos_lap_vals = []
            for i in range(N):
                no = np.linalg.norm(G_all[i])
                nr = np.linalg.norm(G_recon_lap_orig[i])
                if no > 1e-10 and nr > 1e-10:
                    cos_lap_vals.append(float(np.dot(G_all[i], G_recon_lap_orig[i]) / (no * nr)))
            cos_lap_global = float(np.mean(cos_lap_vals))
        
        # 分层重构: 在每个子空间内单独做PCA+近邻重构
        sub_recon_cos = {}
        for attr_type, n_start, n_count in [("color", 0, N_color), ("taste", N_color, N_taste), 
                                              ("size", N_color+N_taste, len(SIZE_TRIPLES))]:
            G_sub = G_all[n_start:n_start+n_count]
            n_sub = G_sub.shape[0]
            if n_sub < 30:
                continue
            
            pca_sub = PCA(n_components=min(15, n_sub-1, D))
            G_sub_pca = pca_sub.fit_transform(G_sub)
            
            ev_sub, evec_sub = compute_graph_laplacian_spectra(G_sub_pca, n_neighbors=10, n_eigenvectors=15)
            if ev_sub is None:
                continue
            
            k_sub = min(5, evec_sub.shape[1])
            Phi_sub = evec_sub[:, 1:k_sub+1]
            nn_sub = NearestNeighbors(n_neighbors=min(5, n_sub-1), algorithm='ball_tree').fit(Phi_sub)
            _, nn_sub_idx = nn_sub.kneighbors(Phi_sub)
            
            G_sub_recon = np.zeros_like(G_sub_pca)
            for j in range(n_sub):
                G_sub_recon[j] = np.mean(G_sub_pca[nn_sub_idx[j]], axis=0)
            G_sub_recon_orig = pca_sub.inverse_transform(G_sub_recon)
            
            cos_sub_vals = []
            for i in range(n_sub):
                no = np.linalg.norm(G_sub[i])
                nr = np.linalg.norm(G_sub_recon_orig[i])
                if no > 1e-10 and nr > 1e-10:
                    cos_sub_vals.append(float(np.dot(G_sub[i], G_sub_recon_orig[i]) / (no * nr)))
            
            sub_recon_cos[attr_type] = round(float(np.mean(cos_sub_vals)), 4)
        
        layer_result = {
            "layer": layer,
            "cos_pca30": round(cos_pca30, 4),
            "cos_laplacian_global": round(cos_lap_global, 4),
            "cos_subspace_color": sub_recon_cos.get("color", 0),
            "cos_subspace_taste": sub_recon_cos.get("taste", 0),
            "cos_subspace_size": sub_recon_cos.get("size", 0),
        }
        
        results.append(layer_result)
        L.log(f"    L{layer}: PCA30={cos_pca30:.3f}, Laplacian={cos_lap_global:.3f}, "
              f"sub_color={sub_recon_cos.get('color',0):.3f}, sub_taste={sub_recon_cos.get('taste',0):.3f}, "
              f"sub_size={sub_recon_cos.get('size',0):.3f}")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    L.log(f"="*60)
    L.log(f"Phase XLIX: 分层谱分析 — {model_name}")
    L.log(f"="*60)
    
    L.log(f"加载模型 {model_name}...")
    mdl, tok, device = load_model(model_name)
    n_layers = mdl.config.num_hidden_layers
    d_model = mdl.config.hidden_size
    key_layers = get_key_layers(n_layers)
    L.log(f"  模型: {n_layers}层, d_model={d_model}, key_layers={key_layers}")
    
    # 数据收集(与Phase XLVIII相同)
    L.log(f"数据收集 — {len(ALL_TRIPLES)}三元组 × {len(PROMPT_TEMPLATES_30)}模板")
    G_dict = collect_G_terms_large_scale(mdl, tok, device, key_layers, ALL_TRIPLES, PROMPT_TEMPLATES_30)
    
    for layer in key_layers:
        if layer in G_dict:
            L.log(f"  L{layer}: {len(G_dict[layer])}个G向量")
    
    del mdl
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    L.log("模型已释放")
    
    # 运行5个实验
    p285_results = run_p285(G_dict, key_layers)
    p286_results = run_p286(G_dict, key_layers)
    p287_results = run_p287(G_dict, key_layers)
    p288_results = run_p288(G_dict, key_layers)
    p289_results = run_p289(G_dict, key_layers)
    
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "key_layers": key_layers,
        "n_triples": len(ALL_TRIPLES),
        "timestamp": timestamp,
        "p285_subspace_laplacian": p285_results,
        "p286_subspace_eigenvector_semantics": p286_results,
        "p287_subspace_manifold_type": p287_results,
        "p288_subspace_orthogonality": p288_results,
        "p289_hierarchical_reconstruction": p289_results,
    }
    
    out_file = OUT_DIR / f"phase_xlix_p285_289_{model_name}_{timestamp}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_file}")
    
    # 摘要
    L.log(f"\n{'='*60}")
    L.log(f"Phase XLIX 核心摘要 — {model_name}")
    L.log(f"{'='*60}")
    
    # P285摘要
    for attr_type in ["color", "taste", "size"]:
        if p285_results[attr_type]:
            last = p285_results[attr_type][-1]
            L.log(f"P285 {attr_type}: dim_95={last.get('dim_95',0)}, "
                  f"λ1-3={last.get('eigenvalues_top10',[])[:3]}, "
                  f"near_zero={last.get('n_near_zero_eigenvalues',0)}, "
                  f"winding={last.get('winding_number_phi1_phi2',0)}")
    
    # P286摘要
    for attr_type in ["color", "taste", "size"]:
        if p286_results[attr_type]:
            last = p286_results[attr_type][-1]
            eig_sems = last.get("eigenvector_semantics", [])
            L.log(f"P286 {attr_type}: 前5个φ编码:")
            for es in eig_sems[:5]:
                L.log(f"  φ{es['eig_idx']}: {es['primary_encoding']}(best={es['best_semantic']},r={es['best_corr']:.3f})")
    
    # P287摘要
    for attr_type in ["color", "taste", "size"]:
        if p287_results[attr_type]:
            last = p287_results[attr_type][-1]
            L.log(f"P287 {attr_type}: manifold_type={last.get('manifold_type','?')}, "
                  f"reason={last.get('manifold_reason','')}")
    
    # P288摘要
    if p288_results:
        last = p288_results[-1]
        L.log(f"P288: angle(C,T)={last.get('mean_angle_ct_deg',0):.1f}°, "
              f"angle(C,S)={last.get('mean_angle_cs_deg',0):.1f}°, "
              f"angle(T,S)={last.get('mean_angle_ts_deg',0):.1f}°")
    
    # P289摘要
    if p289_results:
        last = p289_results[-1]
        L.log(f"P289: PCA30={last.get('cos_pca30',0):.3f}, Laplacian={last.get('cos_laplacian_global',0):.3f}, "
              f"sub_C={last.get('cos_subspace_color',0):.3f}, sub_T={last.get('cos_subspace_taste',0):.3f}, "
              f"sub_S={last.get('cos_subspace_size',0):.3f}")
    
    L.close()


if __name__ == "__main__":
    main()
