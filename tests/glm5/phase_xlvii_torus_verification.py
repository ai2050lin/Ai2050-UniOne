"""
Phase XLVII-P275/276/277/278/279: 非线性参数化与拓扑验证
====================================================

Phase XLVI核心结论:
  1. G项 = T^7 × R^28: 7维环面 × 28维欧氏空间
  2. 30维 = 3类属性 × 10维: 颜色/味道/大小各占~10维正交子空间
  3. 颜色产生最多环(β1=734): 色相环是环面结构来源
  4. 三模型结构完全一致: k≈7, pca30_cos≈0.927
  5. 留一误差60%: 30维线性参数化不足

核心假设:
  7维环面来自颜色空间的色相环结构
  需要验证: 固定名词遍历颜色时, G项是否形成环面

五大实验:
  P275: 纯颜色实验(固定名词, 遍历12色)
    - 固定名词(如apple), 遍历12种颜色
    - 计算G(red apple), G(green apple), ..., G(gold apple)
    - 验证这12个G向量是否在低维子空间中形成环(色相环)
    - 如果形成环, 则色相假设成立

  P276: VAE非线性参数化
    - 训练VAE在G项30维切空间上
    - VAE潜在空间维度=10(足够覆盖环面+欧氏)
    - 检查VAE潜在空间是否有周期性结构
    - 重构误差与PCA30对比

  P277: 环面维度精细估计
    - 使用Cech复形/Vietoris-Rips复形的精细计算
    - 对颜色子空间单独计算Betti数
    - 验证β1(颜色子空间)是否≈7

  P278: 跨名词颜色环一致性
    - apple的颜色环 vs banana的颜色环 vs pear的颜色环
    - 这些环是否在同一个7维子空间中?
    - 如果是, 则环面是属性的内在结构, 不是名词特定的

  P279: 拓扑不变量的解析验证
    - 如果G项确实是T^7×R^28, 则存在解析公式
    - 尝试拟合: G ≈ Σ_i a_i·cos(θ_i) + Σ_j b_j·φ_j (7个周期+28个线性)
    - 验证这个参数化是否能重构G项

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免OOM)
数据规模: 5个固定名词 × 12种颜色 = 60个颜色组合
          + 3个固定名词 × 12种味道 = 36个味道组合
          + 3个固定名词 × 12种大小 = 36个大小组合
          + 80个通用组合 = 总计212个样本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, gc, time, json, argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA, FastICA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_xlvii_log.txt"

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

# P275: 固定名词遍历颜色 — 5个名词 × 12色 = 60组合
COLOR_FIXED_TRIPLES = []
for noun in ["apple", "banana", "cat", "car", "knife"]:
    for color in STIMULI["color_attrs"]:
        COLOR_FIXED_TRIPLES.append((noun, color, f"{color} {noun}"))

# P278: 跨名词颜色环 — 额外3个名词 × 12色 = 36组合
CROSS_NOUN_COLOR_TRIPLES = []
for noun in ["pear", "dog", "chair"]:
    for color in STIMULI["color_attrs"]:
        CROSS_NOUN_COLOR_TRIPLES.append((noun, color, f"{color} {noun}"))

# P275b: 固定名词遍历味道/大小
TASTE_FIXED_TRIPLES = []
for noun in ["apple", "banana", "pear"]:
    for taste in STIMULI["taste_attrs"]:
        TASTE_FIXED_TRIPLES.append((noun, taste, f"{taste} {noun}"))

SIZE_FIXED_TRIPLES = []
for noun in ["apple", "cat", "car"]:
    for size in STIMULI["size_attrs"]:
        SIZE_FIXED_TRIPLES.append((noun, size, f"{size} {noun}"))

# 通用80个组合(与Phase XLVI一致)
GENERAL_TRIPLES = [
    ("apple","red","red apple"), ("apple","green","green apple"), ("apple","yellow","yellow apple"),
    ("banana","yellow","yellow banana"), ("banana","green","green banana"), ("banana","brown","brown banana"),
    ("pear","green","green pear"), ("pear","yellow","yellow pear"), ("orange","orange","orange orange"),
    ("grape","purple","purple grape"), ("cherry","red","red cherry"), ("peach","pink","pink peach"),
    ("mango","yellow","yellow mango"), ("lemon","yellow","yellow lemon"),
    ("strawberry","red","red strawberry"), ("watermelon","green","green watermelon"),
    ("apple","sweet","sweet apple"), ("apple","sour","sour apple"), ("apple","crisp","crisp apple"),
    ("banana","sweet","sweet banana"), ("banana","soft","soft banana"),
    ("pear","sweet","sweet pear"), ("orange","sour","sour orange"), ("grape","sweet","sweet grape"),
    ("apple","big","big apple"), ("apple","small","small apple"),
    ("banana","long","long banana"), ("grape","small","small grape"),
    ("cat","brown","brown cat"), ("cat","white","white cat"), ("cat","black","black cat"),
    ("dog","white","white dog"), ("dog","black","black dog"),
    ("horse","black","black horse"), ("tiger","orange","orange tiger"),
    ("bear","brown","brown bear"), ("fox","red","red fox"),
    ("elephant","big","big elephant"), ("cat","small","small cat"),
    ("car","red","red car"), ("car","blue","blue car"), ("car","white","white car"),
    ("knife","sharp","sharp knife"), ("hammer","heavy","heavy hammer"),
    ("screwdriver","small","small screwdriver"), ("axe","heavy","heavy axe"),
]

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

def get_key_layers(n_layers):
    return sorted(set([0,1,2,3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))


# ==================== 数据收集 ====================

def collect_hidden_states(mdl, tok, device, word, templates=None):
    if templates is None:
        templates = PROMPT_TEMPLATES
    
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
    
    n_actual_layers = len(all_layer_hs[0])
    result = {}
    for l in range(n_actual_layers):
        tensors = [ald[l] for ald in all_layer_hs]
        result[l] = torch.stack(tensors).mean(dim=0)
    
    return result


def collect_G_terms_for_triples(mdl, tok, device, key_layers, triples):
    """为指定三元组收集G项"""
    all_words = set()
    for noun, attr, combo in triples:
        all_words.add(noun)
        all_words.add(attr)
        all_words.add(combo)
    
    L.log(f"  预计算{len(all_words)}个词的hidden states...")
    word_hs = {}
    for i, word in enumerate(sorted(all_words)):
        hs = collect_hidden_states(mdl, tok, device, word)
        word_hs[word] = hs
        if (i+1) % 10 == 0:
            L.log(f"    {i+1}/{len(all_words)} 词完成")
    
    G_dict = {l: [] for l in key_layers}
    for noun, attr, combo in triples:
        for layer in key_layers:
            G = word_hs[combo][layer] - word_hs[noun][layer]
            G_dict[layer].append(G)
    
    del word_hs
    gc.collect()
    return G_dict


# ==================== P275: 纯颜色实验 ====================

def run_p275(G_color_dict, G_taste_dict, G_size_dict, key_layers, color_triples, taste_triples, size_triples):
    """P275: 固定名词遍历属性 — 验证颜色环"""
    L.log("P275: 固定名词遍历属性(颜色/味道/大小)")
    results = {"color_per_noun": {}, "taste_per_noun": {}, "size_per_noun": {}}
    
    for layer in key_layers:
        # === 颜色分析 ===
        color_data = G_color_dict.get(layer, [])
        if len(color_data) < 12:
            continue
        
        G_color = torch.stack(color_data).numpy()  # (60, d_model)
        N, D = G_color.shape
        
        # 按名词分组(5个名词, 每个12色)
        nouns = ["apple", "banana", "cat", "car", "knife"]
        per_noun_color = {}
        
        for ni, noun in enumerate(nouns):
            start_idx = ni * 12
            end_idx = start_idx + 12
            G_noun_color = G_color[start_idx:end_idx]  # (12, d_model)
            
            # PCA分析这个名词的颜色子空间
            pca = PCA(n_components=min(12, D))
            G_pca = pca.fit_transform(G_noun_color)
            
            # 检测12个颜色是否形成环
            # 方法1: 前2个PCA维度的环绕数
            if G_pca.shape[1] >= 2:
                x = G_pca[:, 0]
                y = G_pca[:, 1]
                angles = np.arctan2(y, x)
                angle_diffs = np.diff(np.sort(angles))
                angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
                winding = float(np.sum(angle_diffs) / (2 * np.pi))
            else:
                winding = 0.0
            
            # 方法2: 环的闭合度(首尾距离 vs 平均步长)
            dists = pdist(G_pca[:, :min(3, G_pca.shape[1])])
            mean_step = np.mean(dists)
            if G_pca.shape[1] >= 2:
                closure_dist = np.linalg.norm(G_pca[0, :2] - G_pca[-1, :2])
                closure_ratio = closure_dist / (mean_step + 1e-10)
            else:
                closure_ratio = 0.0
            
            # 方法3: PCA内在维度
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            dim_95 = int(np.searchsorted(cumvar, 0.95) + 1) if len(cumvar) > 0 else 0
            dim_90 = int(np.searchsorted(cumvar, 0.90) + 1) if len(cumvar) > 0 else 0
            
            # 方法4: 相邻颜色的余弦(色相环应该相邻颜色高相关)
            cos_adj = []
            for i in range(11):
                c = float(np.dot(G_noun_color[i], G_noun_color[i+1]) / 
                          (np.linalg.norm(G_noun_color[i]) * np.linalg.norm(G_noun_color[i+1]) + 1e-10))
                cos_adj.append(c)
            mean_adj_cos = float(np.mean(cos_adj))
            
            per_noun_color[noun] = {
                "layer": layer,
                "winding_number_pc12": round(winding, 3),
                "closure_ratio": round(closure_ratio, 3),
                "pca_dim_90": dim_90,
                "pca_dim_95": dim_95,
                "mean_adjacent_cos": round(mean_adj_cos, 4),
                "top3_var_ratio": pca.explained_variance_ratio_[:3].tolist(),
            }
        
        results["color_per_noun"][f"L{layer}"] = per_noun_color
        
        # === 味道分析 ===
        taste_data = G_taste_dict.get(layer, [])
        if len(taste_data) >= 12:
            G_taste = torch.stack(taste_data).numpy()
            taste_nouns = ["apple", "banana", "pear"]
            per_noun_taste = {}
            
            for ni, noun in enumerate(taste_nouns):
                start_idx = ni * 12
                end_idx = start_idx + 12
                G_noun_taste = G_taste[start_idx:end_idx]
                
                pca = PCA(n_components=min(12, G_noun_taste.shape[1]))
                G_pca = pca.fit_transform(G_noun_taste)
                
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                dim_95 = int(np.searchsorted(cumvar, 0.95) + 1) if len(cumvar) > 0 else 0
                
                # 味道是否形成环?
                if G_pca.shape[1] >= 2:
                    x, y = G_pca[:, 0], G_pca[:, 1]
                    angles = np.arctan2(y, x)
                    angle_diffs = np.diff(np.sort(angles))
                    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
                    winding = float(np.sum(angle_diffs) / (2 * np.pi))
                else:
                    winding = 0.0
                
                per_noun_taste[noun] = {
                    "pca_dim_95": dim_95,
                    "winding_number": round(winding, 3),
                    "top3_var_ratio": pca.explained_variance_ratio_[:3].tolist(),
                }
            
            results["taste_per_noun"][f"L{layer}"] = per_noun_taste
        
        # === 大小分析 ===
        size_data = G_size_dict.get(layer, [])
        if len(size_data) >= 12:
            G_size = torch.stack(size_data).numpy()
            size_nouns = ["apple", "cat", "car"]
            per_noun_size = {}
            
            for ni, noun in enumerate(size_nouns):
                start_idx = ni * 12
                end_idx = start_idx + 12
                G_noun_size = G_size[start_idx:end_idx]
                
                pca = PCA(n_components=min(12, G_noun_size.shape[1]))
                G_pca = pca.fit_transform(G_noun_size)
                
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                dim_95 = int(np.searchsorted(cumvar, 0.95) + 1) if len(cumvar) > 0 else 0
                
                if G_pca.shape[1] >= 2:
                    x, y = G_pca[:, 0], G_pca[:, 1]
                    angles = np.arctan2(y, x)
                    angle_diffs = np.diff(np.sort(angles))
                    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
                    winding = float(np.sum(angle_diffs) / (2 * np.pi))
                else:
                    winding = 0.0
                
                per_noun_size[noun] = {
                    "pca_dim_95": dim_95,
                    "winding_number": round(winding, 3),
                    "top3_var_ratio": pca.explained_variance_ratio_[:3].tolist(),
                }
            
            results["size_per_noun"][f"L{layer}"] = per_noun_size
    
    return results


# ==================== P276: VAE非线性参数化 ====================

class SimpleVAE(nn.Module):
    """简单的VAE用于G项非线性参数化"""
    def __init__(self, input_dim, latent_dim=10, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def run_p276(G_dict, key_layers):
    """P276: VAE非线性参数化"""
    L.log("P276: VAE非线性参数化")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 20:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()  # (N, d_model)
        N, D = G_matrix.shape
        
        if N < 20:
            continue
        
        # 先PCA降到35维(加速VAE训练)
        pca_dim = min(35, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_matrix)  # (N, pca_dim)
        
        # VAE训练
        G_tensor = torch.FloatTensor(G_pca)
        
        best_recon_cos = 0
        best_latent_dim = 0
        best_mse = 0
        best_mu = None
        
        for latent_dim in [5, 10, 15, 20]:
            vae = SimpleVAE(pca_dim, latent_dim=latent_dim, hidden_dim=64)
            optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
            
            # 训练200步
            vae.train()
            for epoch in range(200):
                recon, mu, logvar = vae(G_tensor)
                recon_loss = F.mse_loss(recon, G_tensor)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.01 * kl_loss  # 低KL权重, 侧重重构
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 评估重构质量
            vae.eval()
            with torch.no_grad():
                recon, mu, logvar = vae(G_tensor)
                recon_np = recon.numpy()
                
                # 余弦相似度
                cos_vals = []
                for i in range(N):
                    c = float(np.dot(G_pca[i], recon_np[i]) / 
                             (np.linalg.norm(G_pca[i]) * np.linalg.norm(recon_np[i]) + 1e-10))
                    cos_vals.append(c)
                mean_cos = float(np.mean(cos_vals))
                
                # MSE
                mse = float(np.mean((G_pca - recon_np)**2))
            
            if mean_cos > best_recon_cos:
                best_recon_cos = mean_cos
                best_latent_dim = latent_dim
                best_mu = mu.numpy()
                best_mse = mse
        
        # 分析最佳VAE的潜在空间
        # 检测潜在空间的周期性
        periodic_count = 0
        for dim_idx in range(best_latent_dim):
            dim_vals = best_mu[:, dim_idx]
            # 简单周期性检测: 值域是否环绕
            val_range = dim_vals.max() - dim_vals.min()
            val_std = dim_vals.std()
            if val_range > 0 and val_std > 0:
                # 如果标准差相对于范围较小, 说明值集中在某些区域(非均匀=周期性?)
                uniformity = val_std / (val_range + 1e-10)
                # 暂时用简单阈值
                if uniformity > 0.3:  # 非常粗糙的估计
                    pass  # 不确定
        
        # PCA30重构作为基线
        pca30 = PCA(n_components=min(30, pca_dim))
        G_pca30 = pca30.fit_transform(G_pca)
        G_recon30 = pca30.inverse_transform(G_pca30)
        pca30_mse = float(np.mean((G_pca - G_recon30)**2))
        pca30_cos_vals = []
        for i in range(N):
            c = float(np.dot(G_pca[i], G_recon30[i]) / 
                     (np.linalg.norm(G_pca[i]) * np.linalg.norm(G_recon30[i]) + 1e-10))
            pca30_cos_vals.append(c)
        pca30_mean_cos = float(np.mean(pca30_cos_vals))
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "pca_dim": pca_dim,
            "best_vae_latent_dim": best_latent_dim,
            "vae_recon_cos": round(best_recon_cos, 4),
            "vae_recon_mse": round(best_mse, 4),
            "pca30_recon_cos": round(pca30_mean_cos, 4),
            "pca30_recon_mse": round(pca30_mse, 4),
            "vae_vs_pca_cos_diff": round(best_recon_cos - pca30_mean_cos, 4),
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: vae_cos={best_recon_cos:.4f}(dim={best_latent_dim}), pca30_cos={pca30_mean_cos:.4f}, diff={best_recon_cos-pca30_mean_cos:.4f}")
    
    return results


# ==================== P277: 环面维度精细估计 ====================

def compute_beta1_precise(X, max_epsilon_percentile=95):
    """精细β1估计: 对不同epsilon计算β1, 找"平台"区域"""
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    
    N = len(X)
    if N < 4:
        return {"beta1_max": 0, "beta1_plateau": 0, "plateau_start": None, "plateau_end": None}
    
    dist_matrix = squareform(pdist(X))
    nonzero_dists = dist_matrix[dist_matrix > 0]
    
    if len(nonzero_dists) == 0:
        return {"beta1_max": 0, "beta1_plateau": 0, "plateau_start": None, "plateau_end": None}
    
    epsilons = np.percentile(nonzero_dists, np.arange(5, max_epsilon_percentile, 2))
    beta1_list = []
    
    for eps in epsilons:
        adj = (dist_matrix <= eps).astype(float)
        np.fill_diagonal(adj, 0)
        n_comp, _ = connected_components(csr_matrix(adj), directed=False)
        n_edges = int(adj.sum() / 2)
        beta1 = max(0, n_edges - N + n_comp)
        beta1_list.append(beta1)
    
    # 找"平台": β1连续5个点变化<5%的区间
    beta1_arr = np.array(beta1_list, dtype=float)
    plateau_beta1 = 0
    plateau_start = None
    plateau_end = None
    
    for i in range(len(beta1_arr) - 5):
        window = beta1_arr[i:i+5]
        if window.max() > 0:
            variation = (window.max() - window.min()) / (window.mean() + 1e-10)
            if variation < 0.05:  # 变化<5%视为平台
                plateau_beta1 = int(window.mean())
                plateau_start = float(epsilons[i])
                plateau_end = float(epsilons[i+4])
                break
    
    return {
        "beta1_max": int(max(beta1_list)) if beta1_list else 0,
        "beta1_plateau": plateau_beta1,
        "plateau_start": round(plateau_start, 2) if plateau_start else None,
        "plateau_end": round(plateau_end, 2) if plateau_end else None,
        "beta1_vs_eps": [{"epsilon": round(float(epsilons[i]), 2), "beta1": int(beta1_list[i])} 
                         for i in range(0, len(epsilons), 3)],
    }


def run_p277(G_color_dict, G_general_dict, key_layers, color_triples):
    """P277: 环面维度精细估计"""
    L.log("P277: 环面维度精细估计")
    results = {"color_global": {}, "per_noun": {}, "general": {}}
    
    for layer in key_layers:
        # 颜色子空间全局β1
        color_data = G_color_dict.get(layer, [])
        if len(color_data) >= 12:
            G_color = torch.stack(color_data).numpy()
            
            # 全局颜色β1
            ph_color = compute_beta1_precise(G_color)
            results["color_global"][f"L{layer}"] = ph_color
            
            # 按名词分组
            nouns = ["apple", "banana", "cat", "car", "knife"]
            per_noun = {}
            for ni, noun in enumerate(nouns):
                start_idx = ni * 12
                end_idx = start_idx + 12
                G_noun = G_color[start_idx:end_idx]
                
                # PCA降维后计算β1
                pca = PCA(n_components=min(10, G_noun.shape[1]-1))
                G_noun_pca = pca.fit_transform(G_noun)
                
                ph_noun = compute_beta1_precise(G_noun_pca)
                
                # 该名词颜色子空间的内在维度
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
                
                per_noun[noun] = {
                    "dim_95": dim_95,
                    "beta1_max": ph_noun["beta1_max"],
                    "beta1_plateau": ph_noun["beta1_plateau"],
                }
            
            results["per_noun"][f"L{layer}"] = per_noun
        
        # 通用组合的β1
        general_data = G_general_dict.get(layer, [])
        if len(general_data) >= 20:
            G_general = torch.stack(general_data).numpy()
            ph_general = compute_beta1_precise(G_general)
            results["general"][f"L{layer}"] = ph_general
    
    return results


# ==================== P278: 跨名词颜色环一致性 ====================

def run_p278(G_color_dict, G_cross_noun_dict, key_layers):
    """P278: 跨名词颜色环一致性"""
    L.log("P278: 跨名词颜色环一致性")
    results = {"per_layer": []}
    
    for layer in key_layers:
        color_data = G_color_dict.get(layer, [])
        cross_data = G_cross_noun_dict.get(layer, [])
        
        if len(color_data) < 60 or len(cross_data) < 36:
            continue
        
        G_color = torch.stack(color_data).numpy()  # (60, d_model)
        G_cross = torch.stack(cross_data).numpy()  # (36, d_model)
        
        # PCA: 用颜色数据拟合
        pca = PCA(n_components=min(30, G_color.shape[0]-1, G_color.shape[1]))
        pca.fit(G_color)
        
        # 5个主名词的颜色子空间基底
        G_color_pca = pca.transform(G_color)  # (60, 30)
        
        # 3个额外名词投影到相同子空间
        G_cross_pca = pca.transform(G_cross)  # (36, 30)
        
        # 分析: 不同名词的颜色轨迹是否在相同子空间?
        nouns_main = ["apple", "banana", "cat", "car", "knife"]
        nouns_extra = ["pear", "dog", "chair"]
        
        # 计算每个名词颜色轨迹的主成分(在30维PCA空间中)
        per_noun_subspace = {}
        all_noun_centroids = []
        
        for ni, noun in enumerate(nouns_main):
            G_noun = G_color_pca[ni*12:(ni+1)*12]
            centroid = G_noun.mean(axis=0)
            all_noun_centroids.append(centroid)
            
            # 该名词的颜色子空间维度
            pca_noun = PCA(n_components=min(5, 11))
            pca_noun.fit(G_noun)
            per_noun_subspace[noun] = {
                "centroid_norm": round(float(np.linalg.norm(centroid)), 2),
                "color_dim_95": int(np.searchsorted(np.cumsum(pca_noun.explained_variance_ratio_), 0.95) + 1),
                "top3_var_ratio": [round(x, 4) for x in pca_noun.explained_variance_ratio_[:3]],
            }
        
        for ni, noun in enumerate(nouns_extra):
            G_noun = G_cross_pca[ni*12:(ni+1)*12]
            centroid = G_noun.mean(axis=0)
            all_noun_centroids.append(centroid)
            
            pca_noun = PCA(n_components=min(5, 11))
            pca_noun.fit(G_noun)
            per_noun_subspace[noun] = {
                "centroid_norm": round(float(np.linalg.norm(centroid)), 2),
                "color_dim_95": int(np.searchsorted(np.cumsum(pca_noun.explained_variance_ratio_), 0.95) + 1),
                "top3_var_ratio": [round(x, 4) for x in pca_noun.explained_variance_ratio_[:3]],
            }
        
        # 不同名词间的颜色子空间对齐度
        # 方法: 两个名词的颜色PCA方向余弦
        noun_pairs_alignment = {}
        for i, n1 in enumerate(nouns_main[:3]):
            for j, n2 in enumerate(nouns_main[:3]):
                if i < j:
                    G1 = G_color_pca[i*12:(i+1)*12]
                    G2 = G_color_pca[j*12:(j+1)*12]
                    
                    # CCA-like: 两个子空间的最大余弦
                    pca1 = PCA(n_components=3).fit(G1)
                    pca2 = PCA(n_components=3).fit(G2)
                    
                    # 前3个PC方向的平均余弦
                    cos_vals = []
                    for k in range(3):
                        c = float(np.abs(np.dot(pca1.components_[k], pca2.components_[k])))
                        cos_vals.append(c)
                    
                    noun_pairs_alignment[f"{n1}-{n2}"] = {
                        "pc_cos_mean": round(float(np.mean(cos_vals)), 4),
                        "pc_cos_top3": [round(x, 4) for x in cos_vals],
                    }
        
        layer_result = {
            "layer": layer,
            "per_noun_subspace": per_noun_subspace,
            "noun_pairs_alignment": noun_pairs_alignment,
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: color_dim_95 across nouns: {[per_noun_subspace[n]['color_dim_95'] for n in nouns_main]}")
    
    return results


# ==================== P279: 拓扑不变量的解析验证 ====================

def run_p279(G_color_dict, key_layers, color_triples):
    """P279: 解析参数化尝试 — G ≈ Σ a_i·cos(θ_i) + Σ b_j·φ_j"""
    L.log("P279: 解析参数化尝试")
    results = {"per_layer": []}
    
    for layer in key_layers:
        color_data = G_color_dict.get(layer, [])
        if len(color_data) < 12:
            continue
        
        G_color = torch.stack(color_data).numpy()  # (60, d_model)
        N, D = G_color.shape
        
        # 先PCA降到低维
        pca_dim = min(20, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_color)  # (60, pca_dim)
        
        # 尝试不同的参数化:
        # 1. 纯线性: G = A·φ (φ是属性参数)
        # 2. 纯余弦: G = A·cos(θ) (θ是角度参数)
        # 3. 混合: G = A·cos(θ) + B·φ

        # 构建属性参数矩阵
        # 5个名词(用one-hot), 12种颜色(用角度编码)
        nouns = ["apple", "banana", "cat", "car", "knife"]
        colors = STIMULI["color_attrs"]
        
        # 颜色的HSV角度编码(假设12色在色相环上均匀分布)
        color_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)  # [0, π/6, ..., 11π/6]
        color_cos = np.cos(color_angles)
        color_sin = np.sin(color_angles)
        
        # 名词one-hot
        noun_onehot = np.eye(5)
        
        # 构建设计矩阵
        # 方法1: 纯颜色角度
        X_cos_only = np.column_stack([color_cos.repeat(5), color_sin.repeat(5)])  # (60, 2)
        
        # 方法2: 颜色角度 + 名词one-hot
        X_cos_noun = []
        for ni in range(5):
            for ci in range(12):
                row = list(noun_onehot[ni]) + [color_cos[ci], color_sin[ci]]
                X_cos_noun.append(row)
        X_cos_noun = np.array(X_cos_noun)  # (60, 7)
        
        # 方法3: 颜色角度 × 名词交互
        X_interaction = []
        for ni in range(5):
            for ci in range(12):
                row = list(noun_onehot[ni]) + [color_cos[ci], color_sin[ci]]
                # 交互项: noun_i × cos, noun_i × sin
                row += [noun_onehot[ni][k] * color_cos[ci] for k in range(5)]
                row += [noun_onehot[ni][k] * color_sin[ci] for k in range(5)]
                X_interaction.append(row)
        X_interaction = np.array(X_interaction)  # (60, 17)
        
        # 对每个PCA维度拟合
        models_comparison = {"cos_only": [], "cos_noun": [], "interaction": []}
        
        for dim_idx in range(min(10, pca_dim)):
            y = G_pca[:, dim_idx]
            
            # 方法1: cos+sin
            try:
                coef1 = np.linalg.lstsq(X_cos_only, y, rcond=None)[0]
                y_pred1 = X_cos_only @ coef1
                r2_1 = max(0, 1 - np.mean((y - y_pred1)**2) / (np.var(y) + 1e-10))
            except:
                r2_1 = 0
            
            # 方法2: noun+cos+sin
            try:
                coef2 = np.linalg.lstsq(X_cos_noun, y, rcond=None)[0]
                y_pred2 = X_cos_noun @ coef2
                r2_2 = max(0, 1 - np.mean((y - y_pred2)**2) / (np.var(y) + 1e-10))
            except:
                r2_2 = 0
            
            # 方法3: 交互
            try:
                coef3 = np.linalg.lstsq(X_interaction, y, rcond=None)[0]
                y_pred3 = X_interaction @ coef3
                r2_3 = max(0, 1 - np.mean((y - y_pred3)**2) / (np.var(y) + 1e-10))
            except:
                r2_3 = 0
            
            models_comparison["cos_only"].append(round(r2_1, 4))
            models_comparison["cos_noun"].append(round(r2_2, 4))
            models_comparison["interaction"].append(round(r2_3, 4))
        
        # 汇总
        mean_r2_cos = float(np.mean(models_comparison["cos_only"]))
        mean_r2_cos_noun = float(np.mean(models_comparison["cos_noun"]))
        mean_r2_interaction = float(np.mean(models_comparison["interaction"]))
        
        layer_result = {
            "layer": layer,
            "pca_dim": pca_dim,
            "mean_r2_cos_only": round(mean_r2_cos, 4),
            "mean_r2_cos_noun": round(mean_r2_cos_noun, 4),
            "mean_r2_interaction": round(mean_r2_interaction, 4),
            "per_dim_r2": models_comparison,
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: R2 cos_only={mean_r2_cos:.3f}, cos_noun={mean_r2_cos_noun:.3f}, interaction={mean_r2_interaction:.3f}")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase XLVII: 非线性参数化与拓扑验证 - {model_name} =====")
    start_time = time.time()
    
    # 加载模型
    L.log("加载模型...")
    mdl, tok, device = load_model(model_name)
    n_layers = mdl.config.num_hidden_layers
    d_model = mdl.config.hidden_size
    key_layers = get_key_layers(n_layers)
    L.log(f"模型: {model_name}, n_layers={n_layers}, d_model={d_model}, key_layers={key_layers}")
    
    # 收集G项数据(4组)
    L.log("收集G项数据...")
    
    L.log(f"  [1/4] 颜色固定名词: {len(COLOR_FIXED_TRIPLES)}个三元组")
    G_color_dict = collect_G_terms_for_triples(mdl, tok, device, key_layers, COLOR_FIXED_TRIPLES)
    
    L.log(f"  [2/4] 味道固定名词: {len(TASTE_FIXED_TRIPLES)}个三元组")
    G_taste_dict = collect_G_terms_for_triples(mdl, tok, device, key_layers, TASTE_FIXED_TRIPLES)
    
    L.log(f"  [3/4] 大小固定名词: {len(SIZE_FIXED_TRIPLES)}个三元组")
    G_size_dict = collect_G_terms_for_triples(mdl, tok, device, key_layers, SIZE_FIXED_TRIPLES)
    
    L.log(f"  [4/4] 跨名词颜色: {len(CROSS_NOUN_COLOR_TRIPLES)}个三元组")
    G_cross_noun_dict = collect_G_terms_for_triples(mdl, tok, device, key_layers, CROSS_NOUN_COLOR_TRIPLES)
    
    # 通用组合(复用颜色数据的一部分)
    L.log(f"  [5/5] 通用组合: {len(GENERAL_TRIPLES)}个三元组")
    G_general_dict = collect_G_terms_for_triples(mdl, tok, device, key_layers, GENERAL_TRIPLES)
    
    # 释放模型
    del mdl, tok
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    L.log("模型已释放, 开始分析...")
    
    # 运行5个实验
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "key_layers": key_layers,
        "n_color_triples": len(COLOR_FIXED_TRIPLES),
        "n_taste_triples": len(TASTE_FIXED_TRIPLES),
        "n_size_triples": len(SIZE_FIXED_TRIPLES),
        "n_cross_noun_triples": len(CROSS_NOUN_COLOR_TRIPLES),
        "n_general_triples": len(GENERAL_TRIPLES),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    
    L.log("运行P275: 固定名词遍历属性...")
    all_results["p275_color_taste_size"] = run_p275(
        G_color_dict, G_taste_dict, G_size_dict, key_layers,
        COLOR_FIXED_TRIPLES, TASTE_FIXED_TRIPLES, SIZE_FIXED_TRIPLES
    )
    
    L.log("运行P276: VAE非线性参数化...")
    all_results["p276_vae_parameterization"] = run_p276(G_general_dict, key_layers)
    
    L.log("运行P277: 环面维度精细估计...")
    all_results["p277_torus_dim_precise"] = run_p277(
        G_color_dict, G_general_dict, key_layers, COLOR_FIXED_TRIPLES
    )
    
    L.log("运行P278: 跨名词颜色环一致性...")
    all_results["p278_cross_noun_consistency"] = run_p278(
        G_color_dict, G_cross_noun_dict, key_layers
    )
    
    L.log("运行P279: 解析参数化尝试...")
    all_results["p279_analytical_parameterization"] = run_p279(
        G_color_dict, key_layers, COLOR_FIXED_TRIPLES
    )
    
    # 保存结果
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
        elif isinstance(obj, tuple):
            return list(convert(v) for v in obj)
        return obj
    
    all_results = convert(all_results)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"phase_xlvii_p275_279_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - start_time
    L.log(f"===== 完成! 耗时{elapsed:.1f}秒, 结果: {out_path} =====")
    L.close()
    
    print(f"\n{'='*60}")
    print(f"Phase XLVII 结果摘要 - {model_name}")
    print(f"{'='*60}")
    
    # P275摘要
    p275 = all_results.get("p275_color_taste_size", {})
    color_per_noun = p275.get("color_per_noun", {})
    last_layer_key = f"L{key_layers[-1]}"
    if last_layer_key in color_per_noun:
        noun_data = color_per_noun[last_layer_key]
        print(f"P275 颜色环 (L{key_layers[-1]}):")
        for noun, data in noun_data.items():
            print(f"  {noun}: winding={data.get('winding_number_pc12', 'N/A')}, "
                  f"dim_95={data.get('pca_dim_95', 'N/A')}, "
                  f"adj_cos={data.get('mean_adjacent_cos', 'N/A')}")
    
    # P276摘要
    p276 = all_results.get("p276_vae_parameterization", {}).get("per_layer", [])
    if p276:
        last = p276[-1]
        print(f"P276 VAE (L{last['layer']}):")
        print(f"  vae_cos={last.get('vae_recon_cos', 'N/A')}, "
              f"pca30_cos={last.get('pca30_recon_cos', 'N/A')}, "
              f"diff={last.get('vae_vs_pca_cos_diff', 'N/A')}")
    
    # P277摘要
    p277 = all_results.get("p277_torus_dim_precise", {})
    color_global = p277.get("color_global", {})
    if last_layer_key in color_global:
        ph = color_global[last_layer_key]
        print(f"P277 颜色β1 (L{key_layers[-1]}):")
        print(f"  beta1_max={ph.get('beta1_max', 'N/A')}, "
              f"beta1_plateau={ph.get('beta1_plateau', 'N/A')}")
    
    # P278摘要
    p278 = all_results.get("p278_cross_noun_consistency", {}).get("per_layer", [])
    if p278:
        last = p278[-1]
        print(f"P278 跨名词一致性 (L{last['layer']}):")
        subspace = last.get("per_noun_subspace", {})
        for noun, data in list(subspace.items())[:3]:
            print(f"  {noun}: dim_95={data.get('color_dim_95', 'N/A')}")
    
    # P279摘要
    p279 = all_results.get("p279_analytical_parameterization", {}).get("per_layer", [])
    if p279:
        last = p279[-1]
        print(f"P279 解析参数化 (L{last['layer']}):")
        print(f"  R2 cos_only={last.get('mean_r2_cos_only', 'N/A')}, "
              f"cos_noun={last.get('mean_r2_cos_noun', 'N/A')}, "
              f"interaction={last.get('mean_r2_interaction', 'N/A')}")


if __name__ == "__main__":
    main()
