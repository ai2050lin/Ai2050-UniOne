"""
Phase XLV-P265/266/267/268/269: G项拓扑结构分析
====================================================

Phase XLIV核心结论:
  1. G项占组合编码70-94%, 符号回归全部失败(R2<0.4)
  2. G项不是线性/门控/分段函数 — 可能是拓扑/几何结构
  3. 两种FFN策略(冗余vs正交)实现相同语言能力
  4. DeepSeek7B的attn_ffn_cos≈-1(极端反aligned)

核心假设:
  G项不是函数(function)而是流形上的切向量场(Tangent Vector Field on Manifold)
  — 如果G项是拓扑结构, 则符号回归必然失败(因为函数无法描述拓扑)
  — 需要从"拟合函数"转向"描述流形/拓扑"

五大实验:
  P265: G项流形曲率分析
    - 对每层计算G项的局部曲率(local curvature)
    - 计算G项空间的intrinsic dimension (内在维度)
    - 如果intrinsic_dim << d_model, 说明G项在低维流形上
    - 方法: PCA特征值谱 + Grassberger-Procaccia维度估计 + 局部PCA

  P266: G项拓扑不变量
    - 计算G项空间的Betti数(β0, β1, β2) — 拓扑不变量
    - 使用持续同调(Persistent Homology)分析G项的拓扑结构
    - 如果G项有非平凡的Betti数, 说明存在"洞"(拓扑障碍), 函数无法穿越
    - 同时分析G项的连通分量(β0)和环结构(β1)

  P267: G项的切空间与法空间分解
    - 在G项流形上, 分解为切空间(可线性逼近部分)和法空间(曲率部分)
    - 如果法空间维度远小于切空间, G项可被局部线性化
    - 如果法空间维度≈切空间, G项是高度弯曲的流形
    - 对比不同层、不同家族的切-法空间维度比

  P268: G项的向量场结构分析
    - 将G项视为从(noun, attr)到组合编码空间的映射
    - 分析这个映射的雅可比矩阵(Jacobian)的秩
    - 如果Jacobian秩 << d_model, G项是低秩映射(流形上的向量场)
    - 如果Jacobian秩 ≈ d_model, G项是满秩映射(非流形)

  P269: G项的几何不变性测试
    - 测试G项在旋转、缩放、平移下的不变性
    - 如果G项对某些变换不变, 说明存在对称性(群结构)
    - 特别测试: G(apple, red)与G(apple, red)·R (旋转后)的关系
    - 这将揭示G项的几何本质(标量场 vs 向量场 vs 张量场)

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免OOM)
数据规模: 5家族(60词) + 3类属性(36词) + 30个组合三元组 + 7个prompt模板
          → 扩大到50个组合三元组 + 10个prompt模板
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, gc, time, json, argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_xlv_log.txt"

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

# ===================== 增强版数据集(更大规模) =====================
STIMULI = {
    # 5个语义家族, 每个12个成员
    "fruit_family": ["apple","banana","pear","orange","grape","mango","strawberry","watermelon","cherry","peach","lemon","lime"],
    "animal_family": ["cat","dog","rabbit","horse","lion","eagle","elephant","dolphin","tiger","bear","fox","wolf"],
    "vehicle_family": ["car","bus","train","plane","boat","bicycle","truck","helicopter","motorcycle","ship","subway","taxi"],
    "furniture_family": ["chair","table","desk","sofa","bed","cabinet","shelf","bench","stool","dresser","couch","armchair"],
    "tool_family": ["hammer","wrench","screwdriver","saw","drill","pliers","knife","scissors","shovel","rake","axe","chisel"],
    # 3类属性
    "color_attrs": ["red","green","yellow","orange","brown","white","blue","black","pink","purple","gray","gold"],
    "taste_attrs": ["sweet","sour","bitter","salty","crisp","soft","spicy","fresh","tart","savory","rich","mild"],
    "size_attrs": ["big","small","tall","short","long","wide","thin","thick","heavy","light","huge","tiny"],
}

# 扩大到50个组合三元组
TEST_TRIPLES = [
    # 水果+颜色 (20个)
    ("apple","red","red apple"), ("apple","green","green apple"), ("apple","yellow","yellow apple"),
    ("banana","yellow","yellow banana"), ("banana","green","green banana"), ("banana","brown","brown banana"),
    ("pear","green","green pear"), ("pear","yellow","yellow pear"), ("pear","brown","brown pear"),
    ("orange","orange","orange orange"), ("grape","purple","purple grape"), ("grape","red","red grape"),
    ("cherry","red","red cherry"), ("peach","pink","pink peach"), ("mango","yellow","yellow mango"),
    ("lemon","yellow","yellow lemon"), ("lemon","green","green lemon"),
    ("strawberry","red","red strawberry"), ("watermelon","green","green watermelon"),
    ("lime","green","green lime"),
    # 水果+味道 (12个)
    ("apple","sweet","sweet apple"), ("apple","sour","sour apple"), ("apple","crisp","crisp apple"),
    ("banana","sweet","sweet banana"), ("banana","soft","soft banana"),
    ("pear","sweet","sweet pear"), ("pear","fresh","fresh pear"),
    ("orange","sour","sour orange"), ("grape","sweet","sweet grape"),
    ("mango","sweet","sweet mango"), ("cherry","tart","tart cherry"),
    ("lemon","tart","tart lemon"),
    # 动物+颜色 (8个)
    ("cat","brown","brown cat"), ("cat","white","white cat"), ("cat","black","black cat"),
    ("dog","white","white dog"), ("dog","black","black dog"),
    ("horse","black","black horse"), ("tiger","orange","orange tiger"),
    ("bear","brown","brown bear"),
    # 动物+大小 (5个)
    ("elephant","big","big elephant"), ("cat","small","small cat"),
    ("horse","tall","tall horse"), ("fox","small","small fox"),
    ("bear","heavy","heavy bear"),
    # 工具+颜色 (3个)
    ("car","red","red car"), ("car","blue","blue car"),
    ("knife","sharp","sharp knife"),
]

# 10个prompt模板(比之前7个更多)
PROMPT_TEMPLATES = [
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
]

# 关键层采样
def get_key_layers(n_layers):
    return sorted(set([0,1,2,3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))


# ==================== 数据收集 ====================

def collect_hidden_states(mdl, tok, device, word, templates=None):
    """收集一个词在所有层的hidden states, 使用多个prompt模板取平均"""
    if templates is None:
        templates = PROMPT_TEMPLATES
    
    n_layers = mdl.config.num_hidden_layers
    d_model = mdl.config.hidden_size
    all_hs = []
    
    for tpl in templates:
        text = tpl.format(word=word)
        ids = tok.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(ids, output_hidden_states=True)
            hs = out.hidden_states  # tuple of (1, seq_len, d_model)
        # 取最后一个token的hidden state
        last_hs = hs[-1][0, -1, :].detach().float().cpu()
        all_hs.append(last_hs)
        del out, hs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 多模板平均
    avg_hs_per_layer = {}
    # 重新计算每层的平均
    all_layer_hs = []
    for tpl in templates:
        text = tpl.format(word=word)
        ids = tok.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(ids, output_hidden_states=True)
            hs = out.hidden_states
        layer_data = []
        for l in range(len(hs)):
            layer_data.append(hs[l][0, -1, :].detach().float().cpu())
        all_layer_hs.append(layer_data)
        del out, hs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 按层平均
    n_actual_layers = len(all_layer_hs[0])
    result = {}
    for l in range(n_actual_layers):
        tensors = [ald[l] for ald in all_layer_hs]
        result[l] = torch.stack(tensors).mean(dim=0)
    
    return result


def compute_G_term(h_noun, h_attr, h_combo, layer):
    """计算G项: G = h(combo, l+1) - h(noun, l+1) - (h(attr, l+1) - h(noun, l+1))的非线性部分
    简化定义: G = h(combo) - h(noun) - h(attr) + h_global_ref
    但更直接的: G = h(combo) - h(noun)  (组合增量)
    或者: G_nonlinear = h(combo) - h(noun) - proj(h(attr)-h(noun)) onto (h(combo)-h(noun))
    使用最简单定义: G = h(combo, l) - h(noun, l)
    """
    G = h_combo - h_noun
    return G


# ==================== P265: 流形曲率与内在维度 ====================

def estimate_intrinsic_dim_pca(X, explained_var_threshold=0.95):
    """用PCA估计内在维度"""
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim = np.searchsorted(cumvar, explained_var_threshold) + 1
    return dim, pca.explained_variance_ratio_[:20], cumvar[:20]


def estimate_intrinsic_dim_grassberger(X, k=10):
    """Grassberger-Procaccia维度估计"""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # 使用k近邻距离的平均值
    r_k = distances[:, 1:].mean(axis=1)  # 排除自身
    r_k = r_k[r_k > 0]
    if len(r_k) < 5:
        return None
    
    # 在log-log空间拟合斜率
    r_sorted = np.sort(r_k)
    n_points = len(r_sorted)
    log_r = np.log(r_sorted[1:] + 1e-10)
    log_C = np.log(np.arange(1, n_points) / n_points + 1e-10)
    
    # 线性区域拟合
    valid = (log_C > -10) & (log_r > -10)
    if valid.sum() < 3:
        return None
    
    try:
        slope, _ = np.polyfit(log_r[valid], log_C[valid], 1)
        return max(0, slope)
    except:
        return None


def compute_local_curvature(X, k=10):
    """计算局部曲率: 每个点到其k近邻切空间的距离"""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    curvatures = []
    for i in range(len(X)):
        neighbors = X[indices[i, 1:]]  # k近邻
        center = neighbors.mean(axis=0)
        centered = neighbors - center
        
        if len(centered) < 2:
            curvatures.append(0.0)
            continue
        
        # 局部PCA
        local_pca = PCA(n_components=min(3, len(centered)-1))
        local_pca.fit(centered)
        
        # 曲率 = 残差方差 / 总方差
        total_var = centered.var()
        residual_var = max(0, total_var - local_pca.explained_variance_.sum())
        curvatures.append(np.sqrt(residual_var / (total_var + 1e-10)))
    
    return np.array(curvatures)


def run_p265(G_dict, key_layers):
    """P265: G项流形曲率与内在维度分析"""
    L.log("P265: 流形曲率与内在维度分析")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 5:
            continue
        
        # 收集该层所有G项
        G_matrix = torch.stack(G_dict[layer]).numpy()  # (N, d_model)
        N, D = G_matrix.shape
        
        if N < 5:
            continue
        
        # 1. PCA内在维度
        dim_95, sv_top20, cumvar_top20 = estimate_intrinsic_dim_pca(G_matrix, 0.95)
        dim_90, _, _ = estimate_intrinsic_dim_pca(G_matrix, 0.90)
        dim_99, _, _ = estimate_intrinsic_dim_pca(G_matrix, 0.99)
        
        # 2. Grassberger-Procaccia维度
        gp_dim = estimate_intrinsic_dim_grassberger(G_matrix, k=min(10, N-1))
        
        # 3. 局部曲率
        curvatures = compute_local_curvature(G_matrix, k=min(10, N-1))
        mean_curvature = float(curvatures.mean())
        std_curvature = float(curvatures.std())
        max_curvature = float(curvatures.max())
        
        # 4. 全局PCA特征值谱 (判断流形结构)
        pca_full = PCA()
        pca_full.fit(G_matrix)
        ev_ratio_top5 = pca_full.explained_variance_ratio_[:5].tolist()
        ev_top5 = pca_full.explained_variance_[:5].tolist()
        
        # 5. 判断: 是否是低维流形
        is_low_dim = dim_95 < D * 0.3  # 内在维度 < 30% of d_model
        compression_ratio = dim_95 / D
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "d_model": D,
            "pca_dim_90": int(dim_90),
            "pca_dim_95": int(dim_95),
            "pca_dim_99": int(dim_99),
            "gp_dim": float(gp_dim) if gp_dim is not None else None,
            "mean_curvature": round(mean_curvature, 4),
            "std_curvature": round(std_curvature, 4),
            "max_curvature": round(max_curvature, 4),
            "ev_ratio_top5": [round(x, 4) for x in ev_ratio_top5],
            "ev_top5": [round(x, 2) for x in ev_top5],
            "is_low_dim_manifold": is_low_dim,
            "compression_ratio": round(compression_ratio, 4),
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: pca_dim95={dim_95}/{D}({compression_ratio:.2%}), curvature={mean_curvature:.4f}, gp_dim={gp_dim}")
    
    return results


# ==================== P266: 拓扑不变量(持续同调) ====================

def compute_persistent_homology_simple(X, max_dim=2):
    """简化版持续同调计算 - 使用距离矩阵方法"""
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    
    N = len(X)
    if N < 3:
        return {"beta0": 1, "beta1": 0, "beta2": 0, "n_components_vs_epsilon": []}
    
    # 计算距离矩阵
    dist_matrix = squareform(pdist(X))
    
    # 过滤零距离
    nonzero_dists = dist_matrix[dist_matrix > 0]
    if len(nonzero_dists) == 0:
        # 所有点重合
        return {"beta0_final": 1, "beta1_at_median_dist": 0, "persistence_diagram": [],
                "n_components_vs_epsilon": [], "median_pairwise_dist": 0.0}
    
    # β0 (连通分量数) 随epsilon变化
    epsilons = np.percentile(nonzero_dists, np.arange(5, 96, 5))
    components_vs_eps = []
    for eps in epsilons:
        adj = (dist_matrix <= eps).astype(float)
        np.fill_diagonal(adj, 0)
        n_comp, _ = connected_components(csr_matrix(adj), directed=False)
        components_vs_eps.append({"epsilon": round(float(eps), 2), "n_components": int(n_comp)})
    
    # β1 (环结构) - 简化估计: 使用Euler特征
    median_dist = float(np.median(nonzero_dists))
    adj_median = (dist_matrix <= median_dist).astype(float)
    np.fill_diagonal(adj_median, 0)
    n_comp_median, _ = connected_components(csr_matrix(adj_median), directed=False)
    n_edges = int(adj_median.sum() / 2)
    beta1_est = max(0, n_edges - N + n_comp_median)
    
    # 持续性图 (simplified): 在不同epsilon下的β0, β1
    persistence_diagram = []
    for i, item in enumerate(components_vs_eps):
        eps = item["epsilon"]
        adj = (dist_matrix <= eps).astype(float)
        np.fill_diagonal(adj, 0)
        n_c, _ = connected_components(csr_matrix(adj), directed=False)
        n_e = int(adj.sum() / 2)
        b1 = max(0, n_e - N + n_c)
        persistence_diagram.append({
            "epsilon": eps,
            "beta0": n_c,
            "beta1": b1,
        })
    
    # 最终β0 (在足够大epsilon下)
    final_beta0 = components_vs_eps[-1]["n_components"] if components_vs_eps else 1
    
    return {
        "beta0_final": int(final_beta0),
        "beta1_at_median_dist": int(beta1_est),
        "persistence_diagram": persistence_diagram[:10],  # 前10个点
        "n_components_vs_epsilon": components_vs_eps,
        "median_pairwise_dist": round(median_dist, 2),
    }


def run_p266(G_dict, key_layers):
    """P266: G项拓扑不变量分析"""
    L.log("P266: 拓扑不变量(持续同调)分析")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 5:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()
        N, D = G_matrix.shape
        
        if N < 5:
            continue
        
        topo = compute_persistent_homology_simple(G_matrix, max_dim=2)
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "d_model": D,
            "beta0_final": topo["beta0_final"],
            "beta1_at_median": topo["beta1_at_median_dist"],
            "median_pairwise_dist": topo["median_pairwise_dist"],
            "persistence_summary": topo["persistence_diagram"][:5],
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: β0={topo['beta0_final']}, β1≈{topo['beta1_at_median_dist']}, median_dist={topo['median_pairwise_dist']:.2f}")
    
    return results


# ==================== P267: 切空间与法空间分解 ====================

def compute_tangent_normal_decomposition(X, tangent_dim=None):
    """切空间-法空间分解"""
    N, D = X.shape
    
    # 中心化
    center = X.mean(axis=0)
    X_centered = X - center
    
    # 全局PCA
    pca = PCA()
    pca.fit(X_centered)
    
    if tangent_dim is None:
        # 自动确定切空间维度: 解释95%方差的维度
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        tangent_dim = int(np.searchsorted(cumvar, 0.95) + 1)
    
    tangent_dim = min(tangent_dim, D - 1)
    
    # 切空间: 前tangent_dim个主成分
    tangent_basis = pca.components_[:tangent_dim]  # (tangent_dim, D)
    normal_basis = pca.components_[tangent_dim:]    # (D-tangent_dim, D)
    
    # 投影
    X_tangent = X_centered @ tangent_basis.T  # (N, tangent_dim)
    X_normal = X_centered @ normal_basis.T     # (N, D-tangent_dim)
    
    # 能量比
    tangent_energy = float(pca.explained_variance_ratio_[:tangent_dim].sum())
    normal_energy = float(pca.explained_variance_ratio_[tangent_dim:].sum())
    
    # 切空间范数 vs 法空间范数
    tangent_norms = np.linalg.norm(X_tangent, axis=1)
    normal_norms = np.linalg.norm(X_normal, axis=1)
    
    return {
        "tangent_dim": tangent_dim,
        "normal_dim": D - tangent_dim,
        "tangent_energy_ratio": round(tangent_energy, 4),
        "normal_energy_ratio": round(normal_energy, 4),
        "mean_tangent_norm": round(float(tangent_norms.mean()), 4),
        "mean_normal_norm": round(float(normal_norms.mean()), 4),
        "std_tangent_norm": round(float(tangent_norms.std()), 4),
        "std_normal_norm": round(float(normal_norms.std()), 4),
        "tangent_normal_ratio": round(float(tangent_norms.mean() / (normal_norms.mean() + 1e-10)), 4),
        "is_locally_flat": normal_energy < 0.05,  # 法空间能量<5% → 近似平坦
    }


def run_p267(G_dict, key_layers):
    """P267: 切空间与法空间分解"""
    L.log("P267: 切空间-法空间分解")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 5:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()
        N, D = G_matrix.shape
        
        if N < 5:
            continue
        
        tn = compute_tangent_normal_decomposition(G_matrix)
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "d_model": D,
            **tn,
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: tangent_dim={tn['tangent_dim']}/{D}, tangent_energy={tn['tangent_energy_ratio']:.2%}, "
              f"normal_energy={tn['normal_energy_ratio']:.2%}, flat={tn['is_locally_flat']}")
    
    return results


# ==================== P268: Jacobian秩分析 ====================

def run_p268(G_dict, key_layers, triples_info):
    """P268: Jacobian秩分析"""
    L.log("P268: Jacobian秩分析")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 3:
            continue
        
        G_vecs = G_dict[layer]  # list of (d_model,) tensors
        
        # 方法: 将所有G向量堆叠, 矩阵的秩≈Jacobian的秩
        # 如果G = f(noun, attr), 且f是光滑映射, 
        # 则G的样本矩阵的秩与f的Jacobian秩相关
        G_matrix = torch.stack(G_vecs).numpy()  # (N, d_model)
        N, D = G_matrix.shape
        
        if N < 3:
            continue
        
        # SVD确定有效秩
        U, S, Vt = np.linalg.svd(G_matrix, full_matrices=False)
        
        # 有效秩定义: 能解释95%方差的奇异值数量
        total_var = (S ** 2).sum()
        cumvar = np.cumsum(S ** 2) / total_var
        rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        rank_90 = int(np.searchsorted(cumvar, 0.90) + 1)
        rank_99 = int(np.searchsorted(cumvar, 0.99) + 1)
        
        # 数值秩: S > max(S) * eps 的数量
        eps = max(S) * N * np.finfo(float).eps
        numerical_rank = int((S > eps).sum())
        
        # 核维度(null space dimension)
        null_dim = D - numerical_rank
        
        # 条件数
        cond = float(S[0] / (S[min(N-1, D-1)] + 1e-10))
        
        # 奇异值衰减率
        if len(S) > 2:
            sv_decay_rate = float(S[1] / (S[0] + 1e-10))
        else:
            sv_decay_rate = 0.0
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "d_model": D,
            "effective_rank_90": rank_90,
            "effective_rank_95": rank_95,
            "effective_rank_99": rank_99,
            "numerical_rank": numerical_rank,
            "null_space_dim": null_dim,
            "condition_number": round(cond, 2),
            "sv_top5": [round(float(s), 4) for s in S[:5]],
            "sv_decay_rate": round(sv_decay_rate, 4),
            "is_low_rank": rank_95 < D * 0.3,
            "rank_to_dim_ratio": round(rank_95 / D, 4),
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: eff_rank95={rank_95}/{D}({rank_95/D:.2%}), num_rank={numerical_rank}, "
              f"cond={cond:.1f}, sv_decay={sv_decay_rate:.4f}")
    
    return results


# ==================== P269: 几何不变性测试 ====================

def compute_rotation_invariance(G_dict, key_layers):
    """测试G项在旋转下的不变性"""
    L.log("P269: 几何不变性测试")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 5:
            continue
        
        G_vecs = G_dict[layer]
        G_matrix = torch.stack(G_vecs)  # (N, d_model)
        N, D = G_matrix.shape
        
        # 1. 旋转不变性: G·R 与其他G的关系
        # 生成随机旋转矩阵
        torch.manual_seed(42)
        R = torch.linalg.qr(torch.randn(D, D))[0]  # 正交矩阵
        
        # 对每个G向量旋转
        G_rotated = G_matrix @ R.T  # (N, D)
        
        # 计算旋转后的G与原始G集合的余弦相似度
        cos_between = F.cosine_similarity(
            G_rotated.unsqueeze(1),  # (N, 1, D)
            G_matrix.unsqueeze(0),   # (1, N, D)
            dim=2
        )  # (N, N)
        
        # 自身旋转后与原始自身的余弦
        cos_self = F.cosine_similarity(G_rotated, G_matrix, dim=1)
        
        # 2. 缩放不变性: ||αG|| / ||G|| 的比例
        G_norms = G_matrix.norm(dim=1)
        cv_norm = float((G_norms.std() / (G_norms.mean() + 1e-10)))  # 变异系数
        
        # 3. 平移不变性: G - G_mean 与 G 的关系
        G_mean = G_matrix.mean(dim=0)
        G_centered = G_matrix - G_mean
        cos_centered = F.cosine_similarity(G_centered, G_matrix, dim=1)
        
        # 4. 方向聚集度: 所有G与均值的余弦
        G_mean_norm = F.normalize(G_mean.unsqueeze(0), dim=1)
        G_normed = F.normalize(G_matrix, dim=1)
        cos_to_mean = F.cosine_similarity(G_normed, G_mean_norm.expand(N, -1), dim=1)
        
        # 5. 对称性: G是否关于某个超平面对称
        # 检查G和-G的分布
        cos_neg = F.cosine_similarity(-G_matrix, G_matrix, dim=1)  # 应该≈-1
        anti_sym_ratio = float((cos_neg < -0.9).float().mean())
        
        # 6. 内积矩阵结构: G_i · G_j 的模式
        gram = (G_matrix @ G_matrix.T).numpy()
        gram_diag = np.diag(gram)
        gram_offdiag = gram[np.triu_indices(N, k=1)]
        diag_to_offdiag_ratio = float(gram_diag.mean() / (abs(gram_offdiag).mean() + 1e-10))
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "d_model": D,
            # 旋转不变性
            "mean_cos_self_after_rotation": round(float(cos_self.abs().mean()), 4),
            "std_cos_self_after_rotation": round(float(cos_self.abs().std()), 4),
            # 缩放不变性
            "norm_cv": round(cv_norm, 4),  # 范数变异系数
            "mean_norm": round(float(G_norms.mean()), 2),
            "std_norm": round(float(G_norms.std()), 2),
            # 中心化后
            "mean_cos_after_centering": round(float(cos_centered.mean()), 4),
            # 方向聚集度
            "mean_cos_to_mean": round(float(cos_to_mean.mean()), 4),
            "std_cos_to_mean": round(float(cos_to_mean.std()), 4),
            "max_cos_to_mean": round(float(cos_to_mean.max()), 4),
            "min_cos_to_mean": round(float(cos_to_mean.min()), 4),
            # 反对称性
            "anti_symmetric_ratio": round(anti_sym_ratio, 4),
            # Gram矩阵
            "gram_diag_to_offdiag_ratio": round(diag_to_offdiag_ratio, 4),
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: cos_to_mean={float(cos_to_mean.mean()):.4f}, norm_cv={cv_norm:.4f}, "
              f"anti_sym={anti_sym_ratio:.4f}, gram_ratio={diag_to_offdiag_ratio:.2f}")
    
    return results


# ==================== 主流程 ====================

def run_phase_xlv(model_name):
    """Phase XLV主流程"""
    L.log(f"======== Phase XLV: G项拓扑结构分析 - {model_name} ========")
    
    # 加载模型
    L.log(f"加载模型 {model_name}...")
    mdl, tok, device = load_model(model_name)
    n_layers = mdl.config.num_hidden_layers
    d_model = mdl.config.hidden_size
    key_layers = get_key_layers(n_layers)
    L.log(f"模型: {model_name}, {n_layers}层, d={d_model}, key_layers={key_layers}")
    
    # 收集所有G项
    L.log("收集hidden states...")
    G_dict = {l: [] for l in key_layers}
    triple_info = []
    
    for idx, (noun, attr, combo) in enumerate(TEST_TRIPLES):
        L.log(f"  [{idx+1}/{len(TEST_TRIPLES)}] {noun} + {attr} = {combo}")
        
        # 收集三个词的hidden states
        hs_noun = collect_hidden_states(mdl, tok, device, noun, templates=PROMPT_TEMPLATES[:3])  # 用3个模板加速
        hs_attr = collect_hidden_states(mdl, tok, device, attr, templates=PROMPT_TEMPLATES[:3])
        hs_combo = collect_hidden_states(mdl, tok, device, combo, templates=PROMPT_TEMPLATES[:3])
        
        triple_info.append({"noun": noun, "attr": attr, "combo": combo})
        
        for layer in key_layers:
            if layer in hs_noun and layer in hs_combo:
                G = compute_G_term(hs_noun[layer], hs_attr[layer], hs_combo[layer], layer)
                G_dict[layer].append(G)
        
        # 清理
        del hs_noun, hs_attr, hs_combo
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    L.log(f"收集完成: {sum(len(v) for v in G_dict.values())} 个G向量")
    
    # 运行五大实验
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_triples": len(TEST_TRIPLES),
        "key_layers": key_layers,
        "timestamp": datetime.now().isoformat(),
        "triples_used": triple_info,
    }
    
    # P265: 流形曲率与内在维度
    L.log("--- P265: 流形曲率与内在维度 ---")
    results["p265_manifold_curvature"] = run_p265(G_dict, key_layers)
    
    # P266: 拓扑不变量
    L.log("--- P266: 拓扑不变量 ---")
    results["p266_topology"] = run_p266(G_dict, key_layers)
    
    # P267: 切空间-法空间分解
    L.log("--- P267: 切空间-法空间 ---")
    results["p267_tangent_normal"] = run_p267(G_dict, key_layers)
    
    # P268: Jacobian秩分析
    L.log("--- P268: Jacobian秩 ---")
    results["p268_jacobian_rank"] = run_p268(G_dict, key_layers, triple_info)
    
    # P269: 几何不变性
    L.log("--- P269: 几何不变性 ---")
    results["p269_geometric_invariance"] = compute_rotation_invariance(G_dict, key_layers)
    
    # 释放模型
    del mdl, tok, G_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # 保存结果
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUT_DIR / f"phase_xlv_p265_269_{model_name}_{ts}.json"
    
    # 转换numpy为python类型
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    L.log(f"结果已保存: {out_path}")
    L.log(f"======== Phase XLV {model_name} 完成 ========")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"],
                        help="模型名称")
    args = parser.parse_args()
    
    run_phase_xlv(args.model)
