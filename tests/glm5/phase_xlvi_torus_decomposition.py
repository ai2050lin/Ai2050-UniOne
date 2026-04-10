"""
Phase XLVI-P270/271/272/273/274: 环面结构验证与30维分解
====================================================

Phase XLV核心结论:
  1. G项是极低维流形: pca_dim95=25-32, 仅占d_model的0.7-1.2%
  2. G项有大量环结构: β1≈518-520(拓扑障碍)
  3. G项局部平坦: 法空间能量<5%
  4. G项映射极低秩: eff_rank95≈30-31
  5. G项完全反对称, 对旋转不不变
  6. 三模型内在维度高度一致: ≈30, 独立于架构

核心假设:
  G项是环面(Torus) T^k × R^(30-k) 上的向量场
  - 需要验证: k是多少? 哪些维度是周期性的(环面)、哪些是非周期性的(欧氏)?
  - 30维中分别编码了什么语义信息?

五大实验:
  P270: ICA切空间分解
    - 对G项的30维切空间做独立成分分析(ICA)
    - 每个IC编码了什么语义? (颜色/味道/大小/家族?)
    - IC之间的独立性如何? (是否真正独立还是仍有残余相关)

  P271: 周期性维度检测
    - 检测G项30维中哪些维度具有周期性(环面维度)
    - 方法: 自相关函数的周期性 + FFT频谱 + 环绕数(winding number)
    - 如果某个维度是周期性的, 它对应环面T^1的一个因子

  P272: 精细持续同调
    - 使用Vietoris-Rips复形精确计算β0和β1
    - 使用Ripser或giotto-tda库
    - 分析persistence barcode: 哪些环是"长命"的(拓扑不变量)?
    - 与P266的粗估对比

  P273: 语义-拓扑对应
    - 不同语义类别(颜色/味道/大小)的G项子集是否有不同拓扑?
    - 水果G项 vs 动物G项 vs 工具G项的Betti数差异
    - 这揭示拓扑结构是否编码语义类别

  P274: 流形参数化重构
    - 用30维切空间坐标重构G项
    - 重构误差是多少? (验证30维是否充分)
    - 加入周期性修正后, 重构是否改善?

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免OOM)
数据规模: 5家族(60词) × 3类属性(36词) × 80个组合三元组 × 10个prompt模板
          → 扩大到80个组合三元组 + 10个prompt模板
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
from sklearn.manifold import Isomap, TSNE
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

LOG_FILE = OUT_DIR / "phase_xlvi_log.txt"

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

# ===================== 增强版数据集(80个三元组) =====================
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

# 扩大到80个组合三元组(比Phase XLV的50个更多)
TEST_TRIPLES = [
    # 水果+颜色 (24个)
    ("apple","red","red apple"), ("apple","green","green apple"), ("apple","yellow","yellow apple"),
    ("banana","yellow","yellow banana"), ("banana","green","green banana"), ("banana","brown","brown banana"),
    ("pear","green","green pear"), ("pear","yellow","yellow pear"), ("pear","brown","brown pear"),
    ("orange","orange","orange orange"), ("grape","purple","purple grape"), ("grape","red","red grape"),
    ("cherry","red","red cherry"), ("peach","pink","pink peach"), ("mango","yellow","yellow mango"),
    ("lemon","yellow","yellow lemon"), ("lemon","green","green lemon"),
    ("strawberry","red","red strawberry"), ("watermelon","green","green watermelon"),
    ("lime","green","green lime"), ("apple","pink","pink apple"),
    ("grape","green","green grape"), ("peach","orange","orange peach"),
    ("mango","red","red mango"),
    # 水果+味道 (16个)
    ("apple","sweet","sweet apple"), ("apple","sour","sour apple"), ("apple","crisp","crisp apple"),
    ("banana","sweet","sweet banana"), ("banana","soft","soft banana"),
    ("pear","sweet","sweet pear"), ("pear","fresh","fresh pear"),
    ("orange","sour","sour orange"), ("grape","sweet","sweet grape"),
    ("mango","sweet","sweet mango"), ("cherry","tart","tart cherry"),
    ("lemon","tart","tart lemon"), ("apple","bitter","bitter apple"),
    ("peach","sweet","sweet peach"), ("strawberry","sweet","sweet strawberry"),
    ("watermelon","fresh","fresh watermelon"),
    # 水果+大小 (8个)
    ("apple","big","big apple"), ("apple","small","small apple"),
    ("banana","long","long banana"), ("grape","small","small grape"),
    ("watermelon","big","big watermelon"), ("cherry","small","small cherry"),
    ("mango","small","small mango"), ("lemon","small","small lemon"),
    # 动物+颜色 (12个)
    ("cat","brown","brown cat"), ("cat","white","white cat"), ("cat","black","black cat"),
    ("dog","white","white dog"), ("dog","black","black dog"), ("dog","brown","brown dog"),
    ("horse","black","black horse"), ("tiger","orange","orange tiger"),
    ("bear","brown","brown bear"), ("fox","red","red fox"),
    ("rabbit","white","white rabbit"), ("eagle","white","white eagle"),
    # 动物+大小 (8个)
    ("elephant","big","big elephant"), ("cat","small","small cat"),
    ("horse","tall","tall horse"), ("fox","small","small fox"),
    ("bear","heavy","heavy bear"), ("dog","big","big dog"),
    ("rabbit","small","small rabbit"), ("dolphin","big","big dolphin"),
    # 车辆+颜色 (6个)
    ("car","red","red car"), ("car","blue","blue car"), ("car","white","white car"),
    ("bus","yellow","yellow bus"), ("truck","black","black truck"),
    ("bicycle","green","green bicycle"),
    # 工具+属性 (6个)
    ("knife","sharp","sharp knife"), ("hammer","heavy","heavy hammer"),
    ("drill","loud","loud drill"), ("scissors","sharp","sharp scissors"),
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
    """收集一个词在所有层的hidden states, 使用多个prompt模板取平均"""
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


def collect_G_terms(mdl, tok, device, key_layers):
    """收集所有三元组的G项"""
    L.log(f"收集G项数据: {len(TEST_TRIPLES)}个三元组 × {len(PROMPT_TEMPLATES)}个模板")
    
    # 预计算所有词的hidden states
    all_words = set()
    for noun, attr, combo in TEST_TRIPLES:
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
    
    # 计算G项
    G_dict = {l: [] for l in key_layers}
    triple_meta = []  # 元数据: (noun, attr, combo, attr_type, noun_family)
    
    for noun, attr, combo in TEST_TRIPLES:
        # 判断属性类型
        attr_type = "unknown"
        for atype, attrs in [("color", STIMULI["color_attrs"]), 
                             ("taste", STIMULI["taste_attrs"]),
                             ("size", STIMULI["size_attrs"])]:
            if attr in attrs:
                attr_type = atype
                break
        
        # 判断名词家族
        noun_family = "unknown"
        for fname, members in STIMULI.items():
            if noun in members:
                noun_family = fname
                break
        
        triple_meta.append({
            "noun": noun, "attr": attr, "combo": combo,
            "attr_type": attr_type, "noun_family": noun_family
        })
        
        for layer in key_layers:
            G = word_hs[combo][layer] - word_hs[noun][layer]
            G_dict[layer].append(G)
    
    del word_hs
    gc.collect()
    
    return G_dict, triple_meta


# ==================== P270: ICA切空间分解 ====================

def run_p270(G_dict, key_layers, triple_meta):
    """P270: ICA切空间分解"""
    L.log("P270: ICA切空间分解")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 10:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()  # (N, d_model)
        N, D = G_matrix.shape
        
        if N < 10:
            continue
        
        # Step 1: PCA降到30维(切空间)
        pca_dim = min(35, N - 1, D)  # 稍大于30, 确保覆盖
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_matrix)  # (N, pca_dim)
        
        # Step 2: ICA分解
        n_ics = min(30, pca_dim, N - 1)
        ica = FastICA(n_components=n_ics, max_iter=1000, tol=1e-4, random_state=42)
        try:
            G_ica = ica.fit_transform(G_pca)  # (N, n_ics)
        except Exception as e:
            L.log(f"  L{layer}: ICA失败 - {e}")
            continue
        
        # Step 3: 分析每个IC的语义相关性
        ic_semantics = []
        for ic_idx in range(n_ics):
            ic_values = G_ica[:, ic_idx]  # (N,)
            
            # 检查IC是否与属性类型相关
            color_mask = np.array([m["attr_type"] == "color" for m in triple_meta[:N]])
            taste_mask = np.array([m["attr_type"] == "taste" for m in triple_meta[:N]])
            size_mask = np.array([m["attr_type"] == "size" for m in triple_meta[:N]])
            
            # ANOVA-like: 各组的均值差异
            color_mean = np.abs(ic_values[color_mask]).mean() if color_mask.any() else 0
            taste_mean = np.abs(ic_values[taste_mask]).mean() if taste_mask.any() else 0
            size_mean = np.abs(ic_values[size_mask]).mean() if size_mask.any() else 0
            
            # 检查IC是否与名词家族相关
            fruit_mask = np.array([m["noun_family"] == "fruit_family" for m in triple_meta[:N]])
            animal_mask = np.array([m["noun_family"] == "animal_family" for m in triple_meta[:N]])
            vehicle_mask = np.array([m["noun_family"] == "vehicle_family" for m in triple_meta[:N]])
            tool_mask = np.array([m["noun_family"] == "tool_family" for m in triple_meta[:N]])
            
            fruit_mean = np.abs(ic_values[fruit_mask]).mean() if fruit_mask.any() else 0
            animal_mean = np.abs(ic_values[animal_mask]).mean() if animal_mask.any() else 0
            vehicle_mean = np.abs(ic_values[vehicle_mask]).mean() if vehicle_mask.any() else 0
            tool_mean = np.abs(ic_values[tool_mask]).mean() if tool_mask.any() else 0
            
            # 判别比: 最大组均值 / 最小组均值
            attr_values = [color_mean, taste_mean, size_mean]
            family_values = [fruit_mean, animal_mean, vehicle_mean, tool_mean]
            
            attr_discrim = max(attr_values) / (min(attr_values) + 1e-10)
            family_discrim = max(family_values) / (min(family_values) + 1e-10)
            
            # 最强关联类别
            attr_assoc = ["color", "taste", "size"][np.argmax(attr_values)]
            family_assoc = ["fruit", "animal", "vehicle", "tool"][np.argmax(family_values)]
            
            ic_semantics.append({
                "ic_idx": ic_idx,
                "kurtosis": round(float(np.mean((ic_values - ic_values.mean())**4) / (ic_values.std()**4 + 1e-10)), 2),
                "attr_discrim_ratio": round(float(attr_discrim), 3),
                "family_discrim_ratio": round(float(family_discrim), 3),
                "strongest_attr_assoc": attr_assoc,
                "strongest_family_assoc": family_assoc,
                "color_mean": round(float(color_mean), 4),
                "taste_mean": round(float(taste_mean), 4),
                "size_mean": round(float(size_mean), 4),
                "fruit_mean": round(float(fruit_mean), 4),
                "animal_mean": round(float(animal_mean), 4),
            })
        
        # Step 4: IC间独立性
        ic_corr_matrix = np.corrcoef(G_ica.T)  # (n_ics, n_ics)
        off_diag = ic_corr_matrix[np.triu_indices(n_ics, k=1)]
        mean_abs_corr = float(np.abs(off_diag).mean())
        max_abs_corr = float(np.abs(off_diag).max())
        
        # Step 5: 各IC的方差贡献
        ic_variances = np.var(G_ica, axis=0)
        total_var = ic_variances.sum()
        ic_var_ratios = (ic_variances / total_var).tolist()
        
        # Step 6: 重构误差
        G_reconstructed_pca = pca.inverse_transform(G_pca)
        recon_error_pca = float(np.mean((G_matrix - G_reconstructed_pca)**2) / np.mean(G_matrix**2))
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "d_model": D,
            "pca_dim": pca_dim,
            "n_ics": n_ics,
            "pca_recon_error": round(recon_error_pca, 4),
            "ic_mean_abs_corr": round(mean_abs_corr, 4),
            "ic_max_abs_corr": round(max_abs_corr, 4),
            "ic_var_ratios_top10": [round(x, 4) for x in ic_var_ratios[:10]],
            "ic_semantics": ic_semantics,
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: pca_recon_err={recon_error_pca:.4f}, ic_mean_corr={mean_abs_corr:.4f}, ic_max_corr={max_abs_corr:.4f}")
    
    return results


# ==================== P271: 周期性维度检测 ====================

def detect_periodicity(series, min_period=2, max_period=None):
    """检测一维序列的周期性"""
    N = len(series)
    if N < 8:
        return {"is_periodic": False, "period": None, "strength": 0.0}
    
    # 标准化
    series = (series - series.mean()) / (series.std() + 1e-10)
    
    # 方法1: FFT频谱分析
    fft_vals = np.abs(fft(series))
    freqs = fftfreq(N)
    
    # 排除DC分量
    fft_vals[0] = 0
    if N > 1:
        fft_vals[N//2] = 0
    
    # 找主频
    peak_idx = np.argmax(fft_vals[1:N//2]) + 1
    peak_freq = abs(freqs[peak_idx])
    
    if peak_freq > 0:
        period_fft = 1.0 / peak_freq
    else:
        period_fft = None
    
    # 频谱强度: 峰值/均值比
    spectral_strength = float(fft_vals[peak_idx] / (fft_vals[1:N//2].mean() + 1e-10))
    
    # 方法2: 自相关函数
    autocorr = np.correlate(series, series, mode='full')
    autocorr = autocorr[N-1:]  # 取正半部分
    autocorr = autocorr / (autocorr[0] + 1e-10)
    
    # 找自相关的第一个正峰值(排除0延迟)
    if max_period is None:
        max_period = N // 2
    
    ac_peaks, _ = find_peaks(autocorr[1:max_period+1], height=0.1)
    
    if len(ac_peaks) > 0:
        period_ac = int(ac_peaks[0] + 1)
        ac_strength = float(autocorr[period_ac])
    else:
        period_ac = None
        ac_strength = 0.0
    
    # 综合判断
    is_periodic = (spectral_strength > 3.0) and (ac_strength > 0.2)
    
    best_period = period_ac if period_ac is not None else period_fft
    
    return {
        "is_periodic": is_periodic,
        "period_ac": period_ac,
        "period_fft": round(float(period_fft), 2) if period_fft is not None else None,
        "spectral_strength": round(spectral_strength, 3),
        "ac_strength": round(ac_strength, 3),
        "best_period": round(float(best_period), 2) if best_period is not None else None,
    }


def run_p271(G_dict, key_layers, triple_meta):
    """P271: 周期性维度检测"""
    L.log("P271: 周期性维度检测")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 10:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()  # (N, d_model)
        N, D = G_matrix.shape
        
        if N < 10:
            continue
        
        # PCA降维到切空间
        pca_dim = min(35, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_matrix)  # (N, pca_dim)
        
        # 对每个PCA维度检测周期性
        periodic_dims = []
        non_periodic_dims = []
        
        for dim_idx in range(pca_dim):
            dim_series = G_pca[:, dim_idx]
            
            # 按属性类型排序, 看是否有周期性
            # 方法1: 原始顺序的周期性
            orig_period = detect_periodicity(dim_series)
            
            # 方法2: 按noun字母排序后的周期性
            sort_idx = np.argsort([m["noun"] for m in triple_meta[:N]])
            sorted_series = dim_series[sort_idx]
            sorted_period = detect_periodicity(sorted_series)
            
            # 方法3: 按attr_type分组后的周期性
            # 先color, 再taste, 再size
            attr_order = {"color": 0, "taste": 1, "size": 2, "unknown": 3}
            attr_sort_idx = np.argsort([attr_order.get(m["attr_type"], 3) for m in triple_meta[:N]])
            attr_sorted_series = dim_series[attr_sort_idx]
            attr_period = detect_periodicity(attr_sorted_series)
            
            # 方法4: 按noun_family分组后的周期性
            family_order = {"fruit_family": 0, "animal_family": 1, "vehicle_family": 2, "tool_family": 3, "unknown": 4}
            family_sort_idx = np.argsort([family_order.get(m["noun_family"], 4) for m in triple_meta[:N]])
            family_sorted_series = dim_series[family_sort_idx]
            family_period = detect_periodicity(family_sorted_series)
            
            is_periodic = (orig_period["is_periodic"] or sorted_period["is_periodic"] or
                          attr_period["is_periodic"] or family_period["is_periodic"])
            
            dim_info = {
                "dim_idx": dim_idx,
                "var_ratio": round(float(pca.explained_variance_ratio_[dim_idx]), 4),
                "original_periodicity": orig_period,
                "sorted_periodicity": sorted_period,
                "attr_grouped_periodicity": attr_period,
                "family_grouped_periodicity": family_period,
                "is_periodic_any": is_periodic,
            }
            
            if is_periodic:
                periodic_dims.append(dim_info)
            else:
                non_periodic_dims.append(dim_info)
        
        # 环绕数(winding number)估计: 对G向量在切空间中的角度变化
        # 选取前2个PCA维度构成平面, 计算环绕数
        winding_numbers = []
        for plane_idx in range(min(5, pca_dim - 1)):
            x = G_pca[:, plane_idx]
            y = G_pca[:, plane_idx + 1]
            angles = np.arctan2(y, x)
            angle_diffs = np.diff(angles)
            # 修正角度跳变
            angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
            winding = float(np.sum(angle_diffs) / (2 * np.pi))
            winding_numbers.append({
                "plane": f"PC{plane_idx+1}-PC{plane_idx+2}",
                "winding_number": round(winding, 3),
            })
        
        n_periodic = len(periodic_dims)
        n_nonperiodic = len(non_periodic_dims)
        torus_dim_estimate = n_periodic  # 环面维度 ≈ 周期性维度数
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "pca_dim": pca_dim,
            "n_periodic_dims": n_periodic,
            "n_nonperiodic_dims": n_nonperiodic,
            "torus_dim_estimate": torus_dim_estimate,
            "euclidean_dim_estimate": n_nonperiodic,
            "winding_numbers": winding_numbers,
            "periodic_dims_summary": [{
                "dim_idx": d["dim_idx"],
                "var_ratio": d["var_ratio"],
                "best_period": d.get("original_periodicity", {}).get("best_period"),
                "spectral_strength": d.get("original_periodicity", {}).get("spectral_strength"),
            } for d in periodic_dims[:10]],
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: periodic={n_periodic}, nonperiodic={n_nonperiodic}, torus_dim≈{torus_dim_estimate}")
    
    return results


# ==================== P272: 精细持续同调 ====================

def compute_persistent_homology_vietoris_rips(X, max_dim=1, max_edge_length=None):
    """使用Vietoris-Rips复形计算持续同调
    
    不依赖giotto-tda/ripser, 使用scipy手工实现
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
    from scipy.sparse import csr_matrix, lil_matrix
    import itertools
    
    N = len(X)
    if N < 4:
        return {"beta0_persistence": [], "beta1_persistence": [], "n_long_bars_h0": 0, "n_long_bars_h1": 0,
                "max_beta1": 0, "median_pairwise_dist": 0.0, "h0_persistence": [], "h1_persistence": [],
                "beta1_vs_epsilon": []}
    
    # 距离矩阵
    dist_matrix = squareform(pdist(X))
    nonzero_dists = dist_matrix[dist_matrix > 0]
    
    if len(nonzero_dists) == 0:
        return {"beta0_persistence": [], "beta1_persistence": [], "n_long_bars_h0": 0, "n_long_bars_h1": 0,
                "max_beta1": 0, "median_pairwise_dist": 0.0, "h0_persistence": [], "h1_persistence": [],
                "beta1_vs_epsilon": []}
    
    if max_edge_length is None:
        max_edge_length = float(np.percentile(nonzero_dists, 95))
    
    # H0: 连通分量的持续性
    # 对每个epsilon, 计算连通分量数
    epsilons = np.percentile(nonzero_dists, np.arange(1, 96, 2))
    
    h0_persistence = []  # (birth, death) pairs
    prev_n_components = N  # 初始每个点一个分量
    
    # 用并查集跟踪分量合并
    parent = list(range(N))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    # 按距离排序所有边
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if dist_matrix[i,j] <= max_edge_length:
                edges.append((dist_matrix[i,j], i, j))
    edges.sort()
    
    # 逐步添加边, 跟踪H0
    n_components = N
    births = {i: 0.0 for i in range(N)}  # 每个分量出生在0
    
    for dist, i, j in edges:
        pi, pj = find(i), find(j)
        if pi != pj:
            # 合并: 较新的分量死亡
            h0_persistence.append((0.0, float(dist)))
            union(i, j)
            n_components -= 1
            if n_components == 1:
                # 所有点连通, 添加无穷持续条
                h0_persistence.append((0.0, float('inf')))
                break
    
    # H1: 环的持续性
    # 简化方法: 用Euler特征在不同epsilon下估计
    # β1 = n_edges - n_vertices + n_components (在某个epsilon下)
    h1_persistence = []
    
    # 对不同epsilon计算β1
    epsilons_h1 = np.percentile(nonzero_dists, np.arange(10, 91, 5))
    beta1_vs_eps = []
    
    for eps in epsilons_h1:
        adj = (dist_matrix <= eps).astype(float)
        np.fill_diagonal(adj, 0)
        n_comp, _ = connected_components(csr_matrix(adj), directed=False)
        n_edges = int(adj.sum() / 2)
        # β1 = edges - vertices + components (Euler公式)
        beta1 = max(0, n_edges - N + n_comp)
        beta1_vs_eps.append({"epsilon": round(float(eps), 2), "beta1": int(beta1)})
    
    # 找到β1的峰值和持续区间
    if beta1_vs_eps:
        max_beta1 = max(b["beta1"] for b in beta1_vs_eps)
        # H1的"长条"近似为β1保持高值的epsilon区间
        h1_birth = None
        h1_death = None
        for b in beta1_vs_eps:
            if b["beta1"] > max_beta1 * 0.5:
                if h1_birth is None:
                    h1_birth = b["epsilon"]
                h1_death = b["epsilon"]
        
        if h1_birth is not None and h1_death is not None:
            h1_persistence.append({"birth": round(h1_birth, 2), "death": round(h1_death, 2), 
                                   "max_beta1": int(max_beta1)})
    
    # 判断"长条": 持续区间 > 中位距离
    median_dist = float(np.median(nonzero_dists))
    n_long_h0 = sum(1 for b, d in h0_persistence if d != float('inf') and (d - b) > median_dist)
    n_long_h0_inf = sum(1 for b, d in h0_persistence if d == float('inf'))
    
    max_b1 = int(max(b["beta1"] for b in beta1_vs_eps)) if beta1_vs_eps else 0
    
    return {
        "h0_persistence": [(round(b, 2), round(d, 2) if d != float('inf') else "inf") for b, d in h0_persistence[:20]],
        "h1_persistence": h1_persistence,
        "beta1_vs_epsilon": beta1_vs_eps,
        "n_long_bars_h0": n_long_h0 + n_long_h0_inf,
        "n_long_bars_h1": len(h1_persistence),
        "max_beta1": max_b1,
        "median_pairwise_dist": round(median_dist, 2),
    }


def run_p272(G_dict, key_layers, triple_meta):
    """P272: 精细持续同调"""
    L.log("P272: 精细持续同调")
    results = {"per_layer": [], "per_family": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 10:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()  # (N, d_model)
        N, D = G_matrix.shape
        
        if N < 10:
            continue
        
        # 全局持续同调
        ph_global = compute_persistent_homology_vietoris_rips(G_matrix)
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "max_beta1": ph_global["max_beta1"],
            "n_long_bars_h0": ph_global["n_long_bars_h0"],
            "n_long_bars_h1": ph_global["n_long_bars_h1"],
            "h1_persistence": ph_global["h1_persistence"],
            "h0_persistence_sample": ph_global["h0_persistence"][:10],
            "median_pairwise_dist": ph_global["median_pairwise_dist"],
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: max_beta1={ph_global['max_beta1']}, n_long_h0={ph_global['n_long_bars_h0']}, n_long_h1={ph_global['n_long_bars_h1']}")
    
    # 按家族分组的拓扑分析
    family_names = ["fruit_family", "animal_family", "vehicle_family", "tool_family"]
    for fname in family_names:
        family_indices = [i for i, m in enumerate(triple_meta) if m["noun_family"] == fname]
        
        if len(family_indices) < 5:
            continue
        
        # 取最后一个关键层的数据
        last_layer = key_layers[-1]
        if last_layer not in G_dict:
            continue
        
        G_all = torch.stack(G_dict[last_layer]).numpy()
        G_family = G_all[family_indices]
        
        ph_family = compute_persistent_homology_vietoris_rips(G_family)
        
        results["per_family"].append({
            "family": fname,
            "layer": last_layer,
            "N_samples": len(family_indices),
            "max_beta1": ph_family["max_beta1"],
            "n_long_bars_h0": ph_family["n_long_bars_h0"],
            "median_pairwise_dist": ph_family["median_pairwise_dist"],
            "beta1_vs_epsilon": ph_family["beta1_vs_epsilon"][:5],
        })
        L.log(f"  {fname} L{last_layer}: max_beta1={ph_family['max_beta1']}, N={len(family_indices)}")
    
    # 按属性类型分组的拓扑分析
    attr_types = ["color", "taste", "size"]
    for atype in attr_types:
        attr_indices = [i for i, m in enumerate(triple_meta) if m["attr_type"] == atype]
        
        if len(attr_indices) < 5:
            continue
        
        last_layer = key_layers[-1]
        if last_layer not in G_dict:
            continue
        
        G_all = torch.stack(G_dict[last_layer]).numpy()
        G_attr = G_all[attr_indices]
        
        ph_attr = compute_persistent_homology_vietoris_rips(G_attr)
        
        results["per_family"].append({
            "attr_type": atype,
            "layer": last_layer,
            "N_samples": len(attr_indices),
            "max_beta1": ph_attr["max_beta1"],
            "n_long_bars_h0": ph_attr["n_long_bars_h0"],
            "median_pairwise_dist": ph_attr["median_pairwise_dist"],
        })
        L.log(f"  {atype} L{last_layer}: max_beta1={ph_attr['max_beta1']}, N={len(attr_indices)}")
    
    return results


# ==================== P273: 语义-拓扑对应 ====================

def run_p273(G_dict, key_layers, triple_meta):
    """P273: 语义-拓扑对应"""
    L.log("P273: 语义-拓扑对应")
    results = {"per_layer": [], "cross_family_topology": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 10:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()  # (N, d_model)
        N, D = G_matrix.shape
        
        if N < 10:
            continue
        
        # PCA降维
        pca_dim = min(35, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_matrix)
        
        # 1. 族内距离 vs 跨族距离
        family_masks = {}
        for fname in ["fruit_family", "animal_family", "vehicle_family", "tool_family"]:
            mask = np.array([m["noun_family"] == fname for m in triple_meta[:N]])
            if mask.any():
                family_masks[fname] = mask
        
        intra_family_dists = []
        inter_family_dists = []
        
        for fname, mask in family_masks.items():
            G_fam = G_pca[mask]
            if len(G_fam) < 2:
                continue
            dists = pdist(G_fam)
            intra_family_dists.extend(dists.tolist())
        
        # 跨族距离
        fam_names = list(family_masks.keys())
        for i in range(len(fam_names)):
            for j in range(i+1, len(fam_names)):
                G_i = G_pca[family_masks[fam_names[i]]]
                G_j = G_pca[family_masks[fam_names[j]]]
                for gi in G_i:
                    for gj in G_j:
                        inter_family_dists.append(float(np.linalg.norm(gi - gj)))
        
        intra_mean = float(np.mean(intra_family_dists)) if intra_family_dists else 0
        inter_mean = float(np.mean(inter_family_dists)) if inter_family_dists else 0
        separation_ratio = inter_mean / (intra_mean + 1e-10)
        
        # 2. 属性类型在PCA空间中的分离度
        attr_masks = {}
        for atype in ["color", "taste", "size"]:
            mask = np.array([m["attr_type"] == atype for m in triple_meta[:N]])
            if mask.any():
                attr_masks[atype] = mask
        
        # 属性类型的中心间距离
        attr_centroids = {}
        for atype, mask in attr_masks.items():
            attr_centroids[atype] = G_pca[mask].mean(axis=0)
        
        attr_separations = {}
        for a1 in attr_centroids:
            for a2 in attr_centroids:
                if a1 < a2:
                    d = float(np.linalg.norm(attr_centroids[a1] - attr_centroids[a2]))
                    attr_separations[f"{a1}-{a2}"] = round(d, 2)
        
        # 3. 语义维度与PCA维度的对齐
        # 对每个PCA维度, 计算其与语义标签的相关性
        semantic_alignment = []
        for dim_idx in range(min(10, pca_dim)):
            dim_vals = G_pca[:, dim_idx]
            
            # 与属性类型的相关(ANOVA F-statistic)
            groups = []
            for atype in ["color", "taste", "size"]:
                mask = np.array([m["attr_type"] == atype for m in triple_meta[:N]])
                if mask.any():
                    groups.append(dim_vals[mask])
            
            if len(groups) >= 2:
                # 简单F统计
                grand_mean = dim_vals.mean()
                ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
                ss_within = sum(((g - g.mean())**2).sum() for g in groups)
                f_stat = ss_between / (ss_within + 1e-10) * (N - len(groups)) / (len(groups) - 1)
            else:
                f_stat = 0
            
            semantic_alignment.append({
                "dim": dim_idx,
                "var_ratio": round(float(pca.explained_variance_ratio_[dim_idx]), 4),
                "attr_f_stat": round(float(f_stat), 2),
            })
        
        layer_result = {
            "layer": layer,
            "intra_family_dist_mean": round(intra_mean, 2),
            "inter_family_dist_mean": round(inter_mean, 2),
            "family_separation_ratio": round(separation_ratio, 3),
            "attr_centroid_separations": attr_separations,
            "semantic_alignment": semantic_alignment,
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: family_sep={separation_ratio:.3f}, intra={intra_mean:.1f}, inter={inter_mean:.1f}")
    
    return results


# ==================== P274: 流形参数化重构 ====================

def run_p274(G_dict, key_layers, triple_meta):
    """P274: 流形参数化重构"""
    L.log("P274: 流形参数化重构")
    results = {"per_layer": []}
    
    for layer in key_layers:
        if layer not in G_dict or len(G_dict[layer]) < 10:
            continue
        
        G_matrix = torch.stack(G_dict[layer]).numpy()  # (N, d_model)
        N, D = G_matrix.shape
        
        if N < 10:
            continue
        
        # 1. PCA重构(线性)
        for n_comp in [10, 20, 30, 35]:
            if n_comp >= N or n_comp >= D:
                continue
            pca = PCA(n_components=n_comp)
            G_pca = pca.fit_transform(G_matrix)
            G_recon = pca.inverse_transform(G_pca)
            recon_mse = float(np.mean((G_matrix - G_recon)**2))
            recon_cos = float(np.mean([
                np.dot(G_matrix[i], G_recon[i]) / (np.linalg.norm(G_matrix[i]) * np.linalg.norm(G_recon[i]) + 1e-10)
                for i in range(N)
            ]))
            
            if n_comp == 30:
                recon_30_mse = recon_mse
                recon_30_cos = recon_cos
        
        # 2. 语义条件PCA: 对每个属性类型分别做PCA再合并
        attr_pca_results = {}
        for atype in ["color", "taste", "size"]:
            mask = np.array([m["attr_type"] == atype for m in triple_meta[:N]])
            if mask.sum() < 5:
                continue
            G_attr = G_matrix[mask]
            n_comp_attr = min(10, len(G_attr) - 1, D)
            pca_attr = PCA(n_components=n_comp_attr)
            pca_attr.fit(G_attr)
            attr_pca_results[atype] = {
                "dim_95": int(np.searchsorted(np.cumsum(pca_attr.explained_variance_ratio_), 0.95) + 1),
                "dim_90": int(np.searchsorted(np.cumsum(pca_attr.explained_variance_ratio_), 0.90) + 1),
                "top3_var_ratio": pca_attr.explained_variance_ratio_[:3].tolist(),
            }
        
        # 3. Isomap重构(非线性)
        try:
            n_neighbors_iso = min(10, N - 1)
            isomap = Isomap(n_components=min(30, N-1, D), n_neighbors=n_neighbors_iso)
            G_iso = isomap.fit_transform(G_matrix)
            G_iso_recon = isomap.inverse_transform(G_iso) if hasattr(isomap, 'inverse_transform') else None
            
            if G_iso_recon is not None:
                iso_recon_mse = float(np.mean((G_matrix - G_iso_recon)**2))
                iso_recon_cos = float(np.mean([
                    np.dot(G_matrix[i], G_iso_recon[i]) / (np.linalg.norm(G_matrix[i]) * np.linalg.norm(G_iso_recon[i]) + 1e-10)
                    for i in range(N)
                ]))
            else:
                iso_recon_mse = None
                iso_recon_cos = None
                # 用重建误差代替: isomap的reconstruction_error
                iso_recon_mse = float(isomap.reconstruction_error()) if hasattr(isomap, 'reconstruction_error') else None
        except Exception as e:
            iso_recon_mse = None
            iso_recon_cos = None
            L.log(f"  L{layer}: Isomap失败 - {e}")
        
        # 4. 留一重构误差
        if N >= 20:
            loo_errors = []
            n_loo = min(20, N)  # 限制计算量
            indices = np.random.choice(N, n_loo, replace=False)
            
            for idx in indices:
                train_mask = np.ones(N, dtype=bool)
                train_mask[idx] = False
                G_train = G_matrix[train_mask]
                G_test = G_matrix[idx]
                
                pca_loo = PCA(n_components=min(30, N-2, D))
                pca_loo.fit(G_train)
                
                # 用训练集的PCA投影重构测试点
                G_test_proj = pca_loo.transform(G_test.reshape(1, -1))
                G_test_recon = pca_loo.inverse_transform(G_test_proj)
                
                error = float(np.linalg.norm(G_test - G_test_recon.flatten()) / (np.linalg.norm(G_test) + 1e-10))
                loo_errors.append(error)
            
            mean_loo_error = float(np.mean(loo_errors))
        else:
            mean_loo_error = None
        
        layer_result = {
            "layer": layer,
            "N_samples": N,
            "pca30_recon_mse": round(recon_30_mse, 4) if 'recon_30_mse' in dir() else None,
            "pca30_recon_cos": round(recon_30_cos, 4) if 'recon_30_cos' in dir() else None,
            "isomap_recon_mse": round(iso_recon_mse, 4) if iso_recon_mse is not None else None,
            "isomap_recon_cos": round(iso_recon_cos, 4) if iso_recon_cos is not None else None,
            "mean_loo_error": round(mean_loo_error, 4) if mean_loo_error is not None else None,
            "attr_conditional_pca": attr_pca_results,
        }
        results["per_layer"].append(layer_result)
        L.log(f"  L{layer}: pca30_cos={recon_30_cos:.4f}, iso_cos={iso_recon_cos}, loo_err={mean_loo_error}")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase XLVI: 环面结构验证与30维分解 - {model_name} =====")
    start_time = time.time()
    
    # 加载模型
    L.log("加载模型...")
    mdl, tok, device = load_model(model_name)
    n_layers = mdl.config.num_hidden_layers
    d_model = mdl.config.hidden_size
    key_layers = get_key_layers(n_layers)
    L.log(f"模型: {model_name}, n_layers={n_layers}, d_model={d_model}, key_layers={key_layers}")
    
    # 收集G项
    G_dict, triple_meta = collect_G_terms(mdl, tok, device, key_layers)
    L.log(f"收集完成: {len(triple_meta)}个三元组, {len(key_layers)}个层")
    
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
        "n_triples": len(triple_meta),
        "triple_meta_sample": triple_meta[:5],
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    
    L.log("运行P270: ICA切空间分解...")
    all_results["p270_ica_decomposition"] = run_p270(G_dict, key_layers, triple_meta)
    
    L.log("运行P271: 周期性维度检测...")
    all_results["p271_periodicity_detection"] = run_p271(G_dict, key_layers, triple_meta)
    
    L.log("运行P272: 精细持续同调...")
    all_results["p272_persistent_homology"] = run_p272(G_dict, key_layers, triple_meta)
    
    L.log("运行P273: 语义-拓扑对应...")
    all_results["p273_semantic_topology"] = run_p273(G_dict, key_layers, triple_meta)
    
    L.log("运行P274: 流形参数化重构...")
    all_results["p274_manifold_reconstruction"] = run_p274(G_dict, key_layers, triple_meta)
    
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
    out_path = OUT_DIR / f"phase_xlvi_p270_274_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - start_time
    L.log(f"===== 完成! 耗时{elapsed:.1f}秒, 结果: {out_path} =====")
    L.close()
    
    print(f"\n{'='*60}")
    print(f"Phase XLVI 结果摘要 - {model_name}")
    print(f"{'='*60}")
    
    # P270摘要
    p270 = all_results.get("p270_ica_decomposition", {}).get("per_layer", [])
    if p270:
        last = p270[-1]
        print(f"P270 ICA分解 (L{last['layer']}):")
        print(f"  pca_recon_error={last.get('pca_recon_error', 'N/A')}")
        print(f"  ic_mean_corr={last.get('ic_mean_abs_corr', 'N/A')}")
    
    # P271摘要
    p271 = all_results.get("p271_periodicity_detection", {}).get("per_layer", [])
    if p271:
        last = p271[-1]
        print(f"P271 周期性 (L{last['layer']}):")
        print(f"  n_periodic={last.get('n_periodic_dims', 'N/A')}")
        print(f"  torus_dim≈{last.get('torus_dim_estimate', 'N/A')}")
        print(f"  euclidean_dim≈{last.get('euclidean_dim_estimate', 'N/A')}")
    
    # P272摘要
    p272 = all_results.get("p272_persistent_homology", {}).get("per_layer", [])
    if p272:
        last = p272[-1]
        print(f"P272 持续同调 (L{last['layer']}):")
        print(f"  max_beta1={last.get('max_beta1', 'N/A')}")
        print(f"  n_long_bars_h1={last.get('n_long_bars_h1', 'N/A')}")
    
    # P273摘要
    p273 = all_results.get("p273_semantic_topology", {}).get("per_layer", [])
    if p273:
        last = p273[-1]
        print(f"P273 语义-拓扑 (L{last['layer']}):")
        print(f"  family_separation={last.get('family_separation_ratio', 'N/A')}")
    
    # P274摘要
    p274 = all_results.get("p274_manifold_reconstruction", {}).get("per_layer", [])
    if p274:
        last = p274[-1]
        print(f"P274 流形重构 (L{last['layer']}):")
        print(f"  pca30_cos={last.get('pca30_recon_cos', 'N/A')}")
        print(f"  isomap_cos={last.get('isomap_recon_cos', 'N/A')}")
        print(f"  loo_error={last.get('mean_loo_error', 'N/A')}")


if __name__ == "__main__":
    main()
