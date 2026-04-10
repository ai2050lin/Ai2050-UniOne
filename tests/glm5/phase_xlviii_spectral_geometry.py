"""
Phase XLVIII-P280/281/282/283/284: 谱几何分析 — 从拓扑到几何
============================================================

Phase XLVII核心否定:
  1. 色相环假设被推翻: 纯颜色角度R2仅0.17
  2. β1不是真正的拓扑不变量: 无稳定平台
  3. 47%方差未解释: 解析参数化R2=0.53
  4. 但有突破: 颜色内在维度=6(跨名词一致)

关键瓶颈:
  1. 数据量不足: 214样本对30维谱分析过少 → 扩充到1200+
  2. 缺乏物理意义: Laplacian特征函数需要与语义建立对应

核心思路转换:
  放弃Betti数(拓扑不变量), 转向Laplacian谱(几何不变量)
  Laplacian特征值λ_i和特征函数φ_i:
    - λ_i是流形曲率的度量: 小λ=平坦, 大λ=弯曲
    - φ_i是流形上的"振动模式": 类比鼓面的振动
    - φ_i直接定义了G项的数学形式: G ≈ Σ_i c_i · φ_i

五大实验:
  P280: 大规模数据收集(1200+样本)
    - 10名词 × 12色 = 120颜色组合
    - 10名词 × 12味 = 120味道组合
    - 10名词 × 12大小 = 120大小组合
    - 30个prompt模板 → 30 × 360 = 10800个前向传播
    - G项取模板平均: 360个独立G向量

  P281: 图Laplacian谱分析
    - 在G项PCA30空间中构建k-近邻图
    - 计算图Laplacian: L = D - W
    - 求L的前30个最小特征值和特征向量
    - 特征值谱 = G项流形的"频率谱"

  P282: 特征向量-语义对应(核心!)
    - 对每个Laplacian特征向量φ_i:
      a) 计算φ_i与12个颜色的相关 → 颜色解释力
      b) 计算φ_i与12个味道的相关 → 味道解释力
      c) 计算φ_i与12个大小的相关 → 大小解释力
      d) 计算φ_i与5个家族的相关 → 家族解释力
    - 找出"为什么第3个特征向量对应亮度"的答案

  P283: 谱聚类 vs 语义类别
    - 用前k个Laplacian特征向量做谱聚类
    - 聚类结果与真实语义类别对比(ARI, NMI)
    - 如果谱聚类恢复语义类别 → 几何编码了语义

  P284: G项的谱重构公式
    - G ≈ Σ_i c_i · φ_i (Laplacian特征展开)
    - 计算截断误差: 用前k个特征重构G项
    - 与PCA30重构对比: Laplacian是否优于PCA?
    - 导出G项的显式公式

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免OOM)
"""

import torch
import torch.nn as nn
import numpy as np
import os, sys, gc, time, json, argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_xlviii_log.txt"

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
    tok = AutoTokenizer.from_pretrained(
        p_abs, trust_remote_code=True,
        local_files_only=True,
        use_fast=False
    )
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        p_abs, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager", device_map="cpu"
    )
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    mdl.eval()
    device = next(mdl.parameters()).device
    return mdl, tok, device


# ===================== 数据集定义(大规模) =====================
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

# 选10个有代表性的名词(每族2个)
SELECTED_NOUNS = ["apple","cherry", "cat","dog", "car","bus", "chair","table", "knife","hammer"]
NOUN_TO_FAMILY = {}
for fam, words in STIMULI.items():
    if "family" in fam:
        for w in words:
            NOUN_TO_FAMILY[w] = fam

# P280: 大规模三元组 — 10名词 × 3属性类型 × 12属性值 = 360组合
TRIPLES_P280 = []
TRIPLE_LABELS = []  # (noun, attr_type, attr_value, family)
for noun in SELECTED_NOUNS:
    for color in STIMULI["color_attrs"]:
        combo = f"{color} {noun}"
        TRIPLES_P280.append((noun, color, combo))
        TRIPLE_LABELS.append((noun, "color", color, NOUN_TO_FAMILY.get(noun, "unknown")))
    for taste in STIMULI["taste_attrs"]:
        combo = f"{taste} {noun}"
        TRIPLES_P280.append((noun, taste, combo))
        TRIPLE_LABELS.append((noun, "taste", taste, NOUN_TO_FAMILY.get(noun, "unknown")))
    for size in STIMULI["size_attrs"]:
        combo = f"{size} {noun}"
        TRIPLES_P280.append((noun, size, combo))
        TRIPLE_LABELS.append((noun, "size", size, NOUN_TO_FAMILY.get(noun, "unknown")))

# 30个prompt模板(增加多样性)
PROMPT_TEMPLATES_30 = [
    "The {word} is",
    "A {word} can be",
    "This {word} has",
    "I saw a {word}",
    "The {word} was",
    "My {word} is",
    "That {word} looks",
    "One {word} might",
    "Every {word} has",
    "Some {word} are",
    "Look at the {word}",
    "The {word} feels",
    "There is a {word}",
    "I like the {word}",
    "What a {word}",
    "The {word} seems",
    "A {word} always",
    "The {word} became",
    "Many {word} exist",
    "This {word} could",
    "The {word} appears",
    "A {word} usually",
    "The {word} remains",
    "I found a {word}",
    "The {word} shows",
    "A {word} makes",
    "The {word} gives",
    "Such {word} are",
    "The {word} holds",
    "A {word} provides",
]

def get_key_layers(n_layers):
    return sorted(set([0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))


# ==================== 数据收集 ====================

def collect_hidden_states_single(mdl, tok, device, word):
    """收集单个词在所有层的hidden state"""
    ids = tok.encode(word, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl(ids, output_hidden_states=True)
        hs = out.hidden_states
    layer_data = {}
    for l in range(len(hs)):
        layer_data[l] = hs[l][0, -1, :].detach().float().cpu()
    del out, hs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return layer_data


def collect_G_terms_large_scale(mdl, tok, device, key_layers, triples, templates):
    """大规模收集G项 — 对每个模板分别计算再平均"""
    # 收集所有需要的词
    all_words = set()
    for noun, attr, combo in triples:
        all_words.add(noun)
        all_words.add(attr)
        all_words.add(combo)
    
    L.log(f"  预计算{len(all_words)}个词 × {len(templates)}个模板的hidden states...")
    
    # 对每个模板分别收集，取平均
    word_hs_avg = {}
    total_combos = len(all_words) * len(templates)
    done = 0
    
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
                L.log(f"    {done}/{total_combos} 完成")
        
        # 模板平均
        word_hs_avg[word] = {l: layer_sums[l] / layer_counts[l] for l in layer_sums}
        
        if len(word_hs_avg) % 50 == 0:
            L.log(f"    {len(word_hs_avg)}/{len(all_words)} 词完成")
            gc.collect()
    
    L.log(f"  预计算完成, 计算G项...")
    
    # 计算G项
    G_dict = {l: [] for l in key_layers}
    for noun, attr, combo in triples:
        for layer in key_layers:
            if layer in word_hs_avg.get(combo, {}) and layer in word_hs_avg.get(noun, {}):
                G = word_hs_avg[combo][layer] - word_hs_avg[noun][layer]
                G_dict[layer].append(G)
    
    del word_hs_avg
    gc.collect()
    return G_dict


# ==================== P281: 图Laplacian谱分析 ====================

def compute_graph_laplacian_spectra(X_pca, n_neighbors=15, n_eigenvectors=30):
    """
    计算图Laplacian的特征谱
    X_pca: (N, d) PCA降维后的数据
    """
    N, d = X_pca.shape
    if N < 10:
        return None, None, None
    
    # 构建k-近邻图
    k = min(n_neighbors, N - 1)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_pca)
    distances, indices = nbrs.kneighbors(X_pca)
    
    # 构建权重矩阵(高斯核)
    sigma = np.median(distances[:, 1:]) + 1e-10
    W = np.zeros((N, N))
    for i in range(N):
        for j_idx in range(1, k):  # 跳过自身
            j = indices[i, j_idx]
            W[i, j] = np.exp(-distances[i, j_idx]**2 / (2 * sigma**2))
            W[j, i] = W[i, j]  # 对称
    
    # 归一化Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(W.sum(axis=1)) + 1e-10))
    L_norm = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    
    # 求特征值和特征向量(最小的n_eigenvectors个)
    n_eig = min(n_eigenvectors, N - 1)
    try:
        eigenvalues, eigenvectors = eigh(L_norm, subset_by_index=[0, n_eig - 1])
    except Exception:
        eigenvalues, eigenvectors = eigh(L_norm)
        eigenvalues = eigenvalues[:n_eig]
        eigenvectors = eigenvectors[:, :n_eig]
    
    return eigenvalues, eigenvectors, W


def run_p281(G_dict, key_layers, labels):
    """P281: 图Laplacian谱分析"""
    L.log("P281: 图Laplacian谱分析")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 30:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()  # (N, d_model)
        N, D = G_matrix.shape
        
        if N < 50:
            continue
        
        # PCA降到30维
        pca_dim = min(30, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_matrix)
        
        # 计算图Laplacian谱
        eigenvalues, eigenvectors, W = compute_graph_laplacian_spectra(G_pca, n_neighbors=15, n_eigenvectors=30)
        
        if eigenvalues is None:
            continue
        
        # 特征值间隙(谱间隙 = 两个连续特征值之比)
        eig_gaps = []
        for i in range(1, len(eigenvalues)):
            if eigenvalues[i-1] > 1e-10:
                eig_gaps.append(eigenvalues[i] / eigenvalues[i-1])
            else:
                eig_gaps.append(0.0)
        
        # 找最大谱间隙(暗示自然聚类数)
        max_gap_idx = np.argmax(eig_gaps[:15]) if len(eig_gaps) > 0 else 0
        natural_clusters = max_gap_idx + 2  # +2因为跳过了λ0=0
        
        # 前10个特征值的分布
        top_eigenvalues = eigenvalues[:min(15, len(eigenvalues))].tolist()
        
        # PCA累积方差(参考)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "pca_dim": pca_dim,
            "pca30_cumvar": round(float(cumvar[min(29, len(cumvar)-1)]), 4) if len(cumvar) > 0 else 0,
            "eigenvalues_top15": [round(e, 6) for e in top_eigenvalues],
            "eigenvalue_gaps_top15": [round(g, 4) for g in eig_gaps[:15]],
            "natural_clusters_from_gap": natural_clusters,
            "sigma_median": float(np.median(np.sqrt(np.sum(G_pca**2, axis=1)))),
        }
        
        results["per_layer"].append(layer_result)
        L.log(f"    L{layer}: λ1-5={[round(e,4) for e in eigenvalues[:5]]}, natural_clusters={natural_clusters}")
    
    return results


# ==================== P282: 特征向量-语义对应(核心!) ====================

def run_p282(G_dict, key_layers, labels):
    """P282: Laplacian特征向量与语义的对应关系"""
    L.log("P282: 特征向量-语义对应分析")
    results = {"per_layer": [], "eigenvector_semantic_map": {}}
    
    # 提取语义标签
    nouns = [l[0] for l in labels]
    attr_types = [l[1] for l in labels]
    attr_values = [l[2] for l in labels]
    families = [l[3] for l in labels]
    
    # 创建one-hot编码
    unique_nouns = sorted(set(nouns))
    unique_attrs = sorted(set(attr_values))
    unique_families = sorted(set(families))
    unique_attr_types = sorted(set(attr_types))
    
    # 颜色/味道/大小的one-hot
    colors_list = STIMULI["color_attrs"]
    tastes_list = STIMULI["taste_attrs"]
    sizes_list = STIMULI["size_attrs"]
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 30:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()
        N, D = G_matrix.shape
        
        if N < 50:
            continue
        
        # PCA降维
        pca_dim = min(30, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_matrix)
        
        # 计算图Laplacian
        eigenvalues, eigenvectors, W = compute_graph_laplacian_spectra(G_pca, n_neighbors=15, n_eigenvectors=30)
        if eigenvalues is None:
            continue
        
        n_eig = eigenvectors.shape[1]
        
        # 对每个特征向量φ_i, 分析其与语义的相关性
        eig_semantic = []
        
        for i in range(min(15, n_eig)):
            phi = eigenvectors[:, i]  # (N,)
            
            # 1) 属性类型相关: φ_i是否区分颜色/味道/大小?
            attr_type_corrs = {}
            for atype in unique_attr_types:
                indicator = np.array([1.0 if t == atype else 0.0 for t in attr_types])
                if np.std(phi) > 1e-10 and np.std(indicator) > 1e-10:
                    corr = float(np.corrcoef(phi, indicator)[0, 1])
                else:
                    corr = 0.0
                attr_type_corrs[atype] = round(corr, 4)
            
            # 2) 颜色属性相关: φ_i与每种颜色的相关
            color_corrs = {}
            for c in colors_list:
                indicator = np.array([1.0 if (av == c and at == "color") else 0.0 
                                     for av, at in zip(attr_values, attr_types)])
                if np.std(phi) > 1e-10 and np.std(indicator) > 1e-10:
                    corr = float(np.corrcoef(phi, indicator)[0, 1])
                else:
                    corr = 0.0
                color_corrs[c] = round(corr, 4)
            
            # 3) 味道属性相关
            taste_corrs = {}
            for t in tastes_list:
                indicator = np.array([1.0 if (av == t and at == "taste") else 0.0 
                                     for av, at in zip(attr_values, attr_types)])
                if np.std(phi) > 1e-10 and np.std(indicator) > 1e-10:
                    corr = float(np.corrcoef(phi, indicator)[0, 1])
                else:
                    corr = 0.0
                taste_corrs[t] = round(corr, 4)
            
            # 4) 大小属性相关
            size_corrs = {}
            for s in sizes_list:
                indicator = np.array([1.0 if (av == s and at == "size") else 0.0 
                                     for av, at in zip(attr_values, attr_types)])
                if np.std(phi) > 1e-10 and np.std(indicator) > 1e-10:
                    corr = float(np.corrcoef(phi, indicator)[0, 1])
                else:
                    corr = 0.0
                size_corrs[s] = round(corr, 4)
            
            # 5) 家族相关
            family_corrs = {}
            for f in unique_families:
                indicator = np.array([1.0 if fam == f else 0.0 for fam in families])
                if np.std(phi) > 1e-10 and np.std(indicator) > 1e-10:
                    corr = float(np.corrcoef(phi, indicator)[0, 1])
                else:
                    corr = 0.0
                family_corrs[f] = round(corr, 4)
            
            # 6) 名词相关
            noun_corrs = {}
            for n in unique_nouns:
                indicator = np.array([1.0 if nn == n else 0.0 for nn in nouns])
                if np.std(phi) > 1e-10 and np.std(indicator) > 1e-10:
                    corr = float(np.corrcoef(phi, indicator)[0, 1])
                else:
                    corr = 0.0
                noun_corrs[n] = round(corr, 4)
            
            # 7) 最佳语义解释
            all_corrs = {}
            all_corrs.update({f"attr_{k}": v for k, v in attr_type_corrs.items()})
            all_corrs.update({f"color_{k}": v for k, v in color_corrs.items()})
            all_corrs.update({f"taste_{k}": v for k, v in taste_corrs.items()})
            all_corrs.update({f"size_{k}": v for k, v in size_corrs.items()})
            all_corrs.update({f"fam_{k}": v for k, v in family_corrs.items()})
            all_corrs.update({f"noun_{k}": v for k, v in noun_corrs.items()})
            
            best_semantic = max(all_corrs, key=all_corrs.get)
            best_corr = all_corrs[best_semantic]
            
            # 计算总体解释力
            max_attr_type_corr = max(abs(v) for v in attr_type_corrs.values())
            max_color_corr = max(abs(v) for v in color_corrs.values())
            max_taste_corr = max(abs(v) for v in taste_corrs.values())
            max_size_corr = max(abs(v) for v in size_corrs.values())
            max_family_corr = max(abs(v) for v in family_corrs.values())
            max_noun_corr = max(abs(v) for v in noun_corrs.values())
            
            eig_semantic.append({
                "eig_idx": i,
                "eigenvalue": round(float(eigenvalues[i]), 6) if i < len(eigenvalues) else 0,
                "attr_type_corrs": attr_type_corrs,
                "color_corrs": color_corrs,
                "taste_corrs": taste_corrs,
                "size_corrs": size_corrs,
                "family_corrs": family_corrs,
                "noun_corrs": noun_corrs,
                "best_semantic_label": best_semantic,
                "best_corr": round(best_corr, 4),
                "max_abs_attr_type_corr": round(max_attr_type_corr, 4),
                "max_abs_color_corr": round(max_color_corr, 4),
                "max_abs_taste_corr": round(max_taste_corr, 4),
                "max_abs_size_corr": round(max_size_corr, 4),
                "max_abs_family_corr": round(max_family_corr, 4),
                "max_abs_noun_corr": round(max_noun_corr, 4),
            })
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "eigenvector_semantics": eig_semantic,
        }
        
        results["per_layer"].append(layer_result)
        
        # 末层详细输出
        if layer == max(key_layers):
            results["eigenvector_semantic_map"][f"L{layer}"] = eig_semantic
            L.log(f"    L{layer} 特征向量-语义对应:")
            for es in eig_semantic[:10]:
                L.log(f"      φ{es['eig_idx']}: best={es['best_semantic_label']}(r={es['best_corr']}), "
                      f"attr_type={es['max_abs_attr_type_corr']}, color={es['max_abs_color_corr']}, "
                      f"taste={es['max_abs_taste_corr']}, size={es['max_abs_size_corr']}, "
                      f"family={es['max_abs_family_corr']}, noun={es['max_abs_noun_corr']}")
    
    return results


# ==================== P283: 谱聚类 vs 语义类别 ====================

def run_p283(G_dict, key_layers, labels):
    """P283: 谱聚类与语义类别的对应"""
    L.log("P283: 谱聚类 vs 语义类别")
    results = {"per_layer": []}
    
    nouns = [l[0] for l in labels]
    attr_types = [l[1] for l in labels]
    attr_values = [l[2] for l in labels]
    families = [l[3] for l in labels]
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 30:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()
        N, D = G_matrix.shape
        
        if N < 50:
            continue
        
        # PCA降维
        pca_dim = min(30, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_matrix)
        
        # 计算图Laplacian
        eigenvalues, eigenvectors, W = compute_graph_laplacian_spectra(G_pca, n_neighbors=15, n_eigenvectors=30)
        if eigenvalues is None:
            continue
        
        # 用前k个特征向量做聚类
        n_eig = eigenvectors.shape[1]
        
        clustering_results = {}
        for n_clusters in [3, 5, 10, 12]:
            if n_clusters >= N:
                continue
            
            k_eig = min(n_clusters, n_eig)
            # 取前k_eig个特征向量(跳过第0个常数向量)
            phi_embed = eigenvectors[:, 1:k_eig+1]  # (N, k_eig)
            
            # K-means聚类在特征向量空间中
            km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
            cluster_labels = km.fit_predict(phi_embed)
            
            # 与属性类型的ARI/NMI
            ari_attr_type = adjusted_rand_score(attr_types, cluster_labels)
            nmi_attr_type = normalized_mutual_info_score(attr_types, cluster_labels)
            
            # 与家族的ARI/NMI
            ari_family = adjusted_rand_score(families, cluster_labels)
            nmi_family = normalized_mutual_info_score(families, cluster_labels)
            
            # 与名词的ARI/NMI
            ari_noun = adjusted_rand_score(nouns, cluster_labels)
            nmi_noun = normalized_mutual_info_score(nouns, cluster_labels)
            
            # 与具体属性值的ARI/NMI
            ari_attr_val = adjusted_rand_score(attr_values, cluster_labels)
            nmi_attr_val = normalized_mutual_info_score(attr_values, cluster_labels)
            
            # 聚类内的属性类型纯度
            purity_attr_type = 0
            for c in range(n_clusters):
                mask = cluster_labels == c
                if mask.sum() > 0:
                    most_common = max(set(np.array(attr_types)[mask]), 
                                    key=lambda x: (np.array(attr_types)[mask] == x).sum())
                    purity_attr_type += (np.array(attr_types)[mask] == most_common).sum()
            purity_attr_type /= N
            
            clustering_results[f"k={n_clusters}"] = {
                "ari_attr_type": round(ari_attr_type, 4),
                "nmi_attr_type": round(nmi_attr_type, 4),
                "ari_family": round(ari_family, 4),
                "nmi_family": round(nmi_family, 4),
                "ari_noun": round(ari_noun, 4),
                "nmi_noun": round(nmi_noun, 4),
                "ari_attr_val": round(ari_attr_val, 4),
                "nmi_attr_val": round(nmi_attr_val, 4),
                "purity_attr_type": round(purity_attr_type, 4),
            }
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "clustering_results": clustering_results,
        }
        
        results["per_layer"].append(layer_result)
        L.log(f"    L{layer}: ARI(attr_type,k=3)={clustering_results.get('k=3',{}).get('ari_attr_type',0):.3f}, "
              f"ARI(family,k=5)={clustering_results.get('k=5',{}).get('ari_family',0):.3f}")
    
    return results


# ==================== P284: G项的谱重构公式 ====================

def run_p284(G_dict, key_layers, labels):
    """P284: G项的Laplacian特征展开重构"""
    L.log("P284: G项的谱重构公式")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 30:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()
        N, D = G_matrix.shape
        
        if N < 50:
            continue
        
        # PCA降维
        pca_dim = min(30, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_matrix)
        
        # 图Laplacian
        eigenvalues, eigenvectors, W = compute_graph_laplacian_spectra(G_pca, n_neighbors=15, n_eigenvectors=30)
        if eigenvalues is None:
            continue
        
        n_eig = eigenvectors.shape[1]
        
        # Laplacian特征展开: G ≈ Σ_i <G, φ_i> φ_i
        # 但这是在PCA空间中, 需要投影回原空间
        # 先在PCA空间做重构, 再用PCA逆变换
        
        # 计算G_pca在每个特征向量上的投影系数
        # φ_i是归一化的, 所以 c_i = G_pca @ φ_i (逐样本)
        # G_pca ≈ Σ_i c_i φ_i (c_i是N维向量)
        
        recon_errors_laplacian = {}
        recon_cos_laplacian = {}
        
        for k in [3, 5, 10, 15, 20, 30]:
            if k >= n_eig or k >= N:
                continue
            
            # 取前k个特征向量(跳过φ0常数向量)
            Phi_k = eigenvectors[:, 1:k+1]  # (N, k)
            
            # Laplacian特征展开的重构:
            # G_pca的每个样本g_j在PCA空间中是d维向量
            # φ_i是N维向量(每个样本一个值)
            # 投影: c_{j,i} = g_j · (Phi_k @ Phi_k^T)[j, :] 不对
            # 正确: Laplacian特征向量是图的坐标, 不是PCA空间的基底
            # 重构公式: G_recon[j] = Σ_i (g_j · φ_i) φ_i? 不对
            # 
            # 实际意义: φ_i给每个样本一个坐标值, 
            # 用前k个φ_i的坐标做K-means聚类等任务
            # 对于重构: 用φ_i做低维嵌入, 然后在嵌入空间中做近邻重构
            # 
            # 简化: 直接用Φ_k^T @ G_pca 得到k维嵌入, 然后用Φ_k @ 嵌入 重构
            # 但Φ_k是(N,k), G_pca是(N,d), 这是样本空间不是特征空间
            #
            # 正确的谱重构:
            # 样本在φ空间的嵌入: Z = Φ_k (N, k) — 每行是一个样本的k维嵌入
            # 要重构G_pca: G_recon ≈ Z @ (Z^T Z)^{-1} Z^T G_pca = P_Z @ G_pca
            # 其中P_Z = Φ_k (Φ_k^T Φ_k)^{-1} Φ_k^T 是投影矩阵
            # 因为φ_i正交归一化, 所以P_Z = Φ_k Φ_k^T
            
            # 方法: 用Laplacian特征向量做近邻重构
            # 在φ空间中找近邻, 用近邻的G_pca平均值重构
            from sklearn.neighbors import NearestNeighbors as NN
            
            # 在Laplacian嵌入空间中的近邻
            Z_k = Phi_k  # (N, k)
            nn_model = NN(n_neighbors=min(10, N-1), algorithm='ball_tree').fit(Z_k)
            _, nn_indices = nn_model.kneighbors(Z_k)
            
            # 用近邻在PCA空间中的值加权重构
            G_recon_pca = np.zeros_like(G_pca)
            for j in range(N):
                neighbors = nn_indices[j]
                G_recon_pca[j] = np.mean(G_pca[neighbors], axis=0)
            
            # PCA逆变换回原空间
            G_recon_orig = pca.inverse_transform(G_recon_pca)
            
            # 计算重构误差
            mse = float(np.mean((G_matrix - G_recon_orig)**2))
            cos_vals = []
            for i in range(N):
                g_orig = G_matrix[i]
                g_recon = G_recon_orig[i]
                norm_orig = np.linalg.norm(g_orig)
                norm_recon = np.linalg.norm(g_recon)
                if norm_orig > 1e-10 and norm_recon > 1e-10:
                    cos_vals.append(float(np.dot(g_orig, g_recon) / (norm_orig * norm_recon)))
            
            mean_cos = float(np.mean(cos_vals)) if cos_vals else 0.0
            
            recon_errors_laplacian[f"k={k}"] = round(mse, 6)
            recon_cos_laplacian[f"k={k}"] = round(mean_cos, 4)
        
        # PCA30重构(参考)
        pca30 = PCA(n_components=min(30, pca_dim))
        G_pca30_proj = pca30.fit_transform(G_matrix)
        G_recon_pca30 = pca30.inverse_transform(G_pca30_proj)
        
        # 正确的PCA30重构
        pca30 = PCA(n_components=min(30, pca_dim))
        G_pca30_proj = pca30.fit_transform(G_matrix)
        G_recon_pca30 = pca30.inverse_transform(G_pca30_proj)
        
        mse_pca30 = float(np.mean((G_matrix - G_recon_pca30)**2))
        cos_pca30_vals = []
        for i in range(N):
            g_o = G_matrix[i]
            g_r = G_recon_pca30[i]
            no = np.linalg.norm(g_o)
            nr = np.linalg.norm(g_r)
            if no > 1e-10 and nr > 1e-10:
                cos_pca30_vals.append(float(np.dot(g_o, g_r) / (no * nr)))
        cos_pca30 = float(np.mean(cos_pca30_vals)) if cos_pca30_vals else 0.0
        
        # 特征值与解释方差的关系
        # Laplacian特征值小 = 低频 = 大尺度结构
        # Laplacian特征值大 = 高频 = 小尺度结构
        # φ_i是N维向量(样本空间), 它的方差直接就是流形上的信号强度
        eig_variance_explained = []
        total_var = float(np.var(G_pca)) * G_pca.shape[1]  # 总方差
        for i in range(min(15, n_eig)):
            if i > 0:
                # φ_i解释的方差: 每个样本的φ_i值加权其G_pca, 计算加权方差
                phi_i = eigenvectors[:, i]  # (N,)
                # 用φ_i值对G_pca做加权平均: 高φ值样本的平均 vs 低φ值样本的平均
                # 方差解释力 = φ_i区分样本的能力
                # 用φ_i值的方差直接作为信号强度
                var_explained = float(np.var(phi_i))
                eig_variance_explained.append(round(var_explained, 6))
            else:
                eig_variance_explained.append(0.0)
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "recon_errors_laplacian": recon_errors_laplacian,
            "recon_cos_laplacian": recon_cos_laplacian,
            "recon_error_pca30": round(mse_pca30, 6),
            "recon_cos_pca30": round(cos_pca30, 4),
            "eig_variance_explained": eig_variance_explained,
            "total_variance_pca": round(float(np.sum(pca.explained_variance_)), 2),
        }
        
        results["per_layer"].append(layer_result)
        L.log(f"    L{layer}: Laplacian(k=10)_cos={recon_cos_laplacian.get('k=10',0):.3f}, "
              f"PCA30_cos={cos_pca30:.3f}")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    L.log(f"="*60)
    L.log(f"Phase XLVIII: 谱几何分析 — {model_name}")
    L.log(f"="*60)
    
    # 加载模型
    L.log(f"加载模型 {model_name}...")
    mdl, tok, device = load_model(model_name)
    n_layers = mdl.config.num_hidden_layers
    d_model = mdl.config.hidden_size
    key_layers = get_key_layers(n_layers)
    L.log(f"  模型: {n_layers}层, d_model={d_model}, key_layers={key_layers}")
    
    # P280: 大规模数据收集
    L.log(f"P280: 大规模数据收集 — {len(TRIPLES_P280)}三元组 × {len(PROMPT_TEMPLATES_30)}模板")
    L.log(f"  唯一词数: {len(set(w for t in TRIPLES_P280 for w in t))}")
    
    G_dict = collect_G_terms_large_scale(mdl, tok, device, key_layers, TRIPLES_P280, PROMPT_TEMPLATES_30)
    
    for layer in key_layers:
        if layer in G_dict:
            L.log(f"  L{layer}: {len(G_dict[layer])}个G向量")
    
    # 释放模型内存
    del mdl
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    L.log("模型已释放")
    
    # P281: 图Laplacian谱分析
    p281_results = run_p281(G_dict, key_layers, TRIPLE_LABELS)
    
    # P282: 特征向量-语义对应
    p282_results = run_p282(G_dict, key_layers, TRIPLE_LABELS)
    
    # P283: 谱聚类
    p283_results = run_p283(G_dict, key_layers, TRIPLE_LABELS)
    
    # P284: 谱重构
    p284_results = run_p284(G_dict, key_layers, TRIPLE_LABELS)
    
    # 汇总结果
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "key_layers": key_layers,
        "n_triples": len(TRIPLES_P280),
        "n_templates": len(PROMPT_TEMPLATES_30),
        "timestamp": timestamp,
        "p281_laplacian_spectra": p281_results,
        "p282_eigenvector_semantics": p282_results,
        "p283_spectral_clustering": p283_results,
        "p284_spectral_reconstruction": p284_results,
    }
    
    # 保存
    out_file = OUT_DIR / f"phase_xlviii_p280_284_{model_name}_{timestamp}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_file}")
    
    # 打印核心摘要
    L.log(f"\n{'='*60}")
    L.log(f"Phase XLVIII 核心摘要 — {model_name}")
    L.log(f"{'='*60}")
    
    # P281摘要
    if p281_results["per_layer"]:
        last_p281 = p281_results["per_layer"][-1]
        L.log(f"P281 末层: λ1-5={last_p281.get('eigenvalues_top15',[])[:5]}, "
              f"natural_clusters={last_p281.get('natural_clusters_from_gap',0)}")
    
    # P282摘要
    if p282_results["per_layer"]:
        last_p282 = p282_results["per_layer"][-1]
        eig_sems = last_p282.get("eigenvector_semantics", [])
        L.log(f"P282 末层特征向量-语义对应(前10个):")
        for es in eig_sems[:10]:
            L.log(f"  φ{es['eig_idx']}: best={es['best_semantic_label']}(r={es['best_corr']:.3f}), "
                  f"attr_type={es['max_abs_attr_type_corr']:.3f}, color={es['max_abs_color_corr']:.3f}, "
                  f"taste={es['max_abs_taste_corr']:.3f}, size={es['max_abs_size_corr']:.3f}, "
                  f"fam={es['max_abs_family_corr']:.3f}, noun={es['max_abs_noun_corr']:.3f}")
    
    # P283摘要
    if p283_results["per_layer"]:
        last_p283 = p283_results["per_layer"][-1]
        cr = last_p283.get("clustering_results", {})
        L.log(f"P283 末层谱聚类:")
        for k_name, k_res in cr.items():
            L.log(f"  {k_name}: ARI(attr_type)={k_res['ari_attr_type']:.3f}, "
                  f"ARI(family)={k_res['ari_family']:.3f}, "
                  f"NMI(attr_type)={k_res['nmi_attr_type']:.3f}")
    
    # P284摘要
    if p284_results["per_layer"]:
        last_p284 = p284_results["per_layer"][-1]
        L.log(f"P284 末层谱重构: Laplacian_cos={last_p284.get('recon_cos_laplacian',{})}, "
              f"PCA30_cos={last_p284.get('recon_cos_pca30',0):.3f}")
    
    L.close()


if __name__ == "__main__":
    main()
