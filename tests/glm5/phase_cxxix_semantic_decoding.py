"""
Phase CXXIX-CXXX: 语义编码的深层机制
P563: W_U方向0,1,2,3编码了什么? — 用PCA分析这些方向的语义内容
P564: 功能词vs内容词的频谱双峰假设 — W_U空间是否存在两个独立的子空间?
P565: 频谱力学如何实现token预测 — 从频谱到logits的完整数学路径
P566: 统一语言理论: 频谱力学+语义子空间 = 语言能力的数学基础
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, get_model_info, get_W_U, get_sample_layers, get_layers, get_layer_weights

import torch


def compute_wu_svd(model, k=200):
    """计算W_U的SVD"""
    W_U = get_W_U(model)
    d_model, n_vocab = W_U.shape
    k = min(k, min(d_model, n_vocab) - 1)
    U_wu, S_wu, Vt_wu = svds(W_U.T, k=k)
    sort_idx = np.argsort(S_wu)[::-1]
    U_wu = U_wu[:, sort_idx]
    S_wu = S_wu[sort_idx]
    return U_wu, S_wu, W_U


def project_and_spectrum(h, U_wu, k):
    """将h投影到W_U空间并计算频谱"""
    coeffs = U_wu[:, :k].T @ h
    spectrum = np.sort(np.abs(coeffs))[::-1]
    return spectrum


def compute_ratio_k_from_h(h, U_wu, k):
    """从hidden state计算ratio(k)"""
    proj = U_wu[:, :k].T @ h
    ratio = np.sum(proj**2) / (np.sum(h**2) + 1e-10)
    return ratio


def safe_decode(tokenizer, token_id):
    """安全解码token"""
    try:
        s = tokenizer.decode([token_id])
        return s.encode('ascii', 'replace').decode('ascii')
    except:
        return f"id={token_id}"


# ============== P563: W_U方向0,1,2,3的语义解码 ==============
def experiment_p563(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """W_U方向0,1,2,3编码了什么? — 解码这些方向的语义内容"""
    print(f"\n{'='*60}")
    print(f"P563: W_U方向0-9的语义解码 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(200, U_wu.shape[1])
    n_vocab = W_U.shape[0]
    
    # 1. 每个W_U方向对应的Top词汇
    print("\n--- W_U前10个SVD方向的Top词汇 ---")
    direction_top_tokens = {}
    for d in range(10):
        # W_U的第d个SVD方向 = U_wu[:, d]
        direction = U_wu[:, d]
        # 每个token在这个方向上的投影 = W_U @ direction
        projections = W_U @ direction  # [n_vocab]
        
        # Top-20 最正和最负的token
        top_pos_ids = np.argsort(projections)[::-1][:20]
        top_neg_ids = np.argsort(projections)[:20]
        
        top_pos = [(safe_decode(tokenizer, tid), float(projections[tid])) for tid in top_pos_ids]
        top_neg = [(safe_decode(tokenizer, tid), float(projections[tid])) for tid in top_neg_ids]
        
        direction_top_tokens[d] = {"pos": top_pos, "neg": top_neg}
        
        pos_str = ", ".join([f"{t}({v:.2f})" for t, v in top_pos[:5]])
        neg_str = ", ".join([f"{t}({v:.2f})" for t, v in top_neg[:5]])
        print(f"  方向{d}: Top+=[{pos_str}]")
        print(f"         Top-=[{neg_str}]")
    
    # 2. 每个方向的词性/语法角色分析
    print("\n--- 方向的词性/语法角色分析 ---")
    # 定义简单的词性分类
    function_words = set()
    content_words = set()
    
    # 构建词性映射(内置词表)
    # 用内置词表
    func_list = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
                 "may", "might", "can", "could", "must", "to", "of", "in", "for", "on",
                 "with", "at", "by", "from", "as", "into", "through", "during", "before", "after",
                 "and", "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
                 "that", "which", "who", "whom", "this", "these", "those", "it", "its", "he",
                 "she", "they", "we", "you", "i", "me", "him", "her", "us", "them"]
    
    for d in range(10):
        direction = U_wu[:, d]
        projections = W_U @ direction
        top100_ids = np.argsort(np.abs(projections))[::-1][:100]
        
        func_count = 0
        content_count = 0
        for tid in top100_ids:
            try:
                tok_str = tokenizer.decode([tid]).strip().lower()
                if tok_str in func_list:
                    func_count += 1
                elif tok_str.isalpha() and len(tok_str) > 1:
                    content_count += 1
            except:
                pass
        
        total = func_count + content_count + 1e-10
        print(f"  方向{d}: Top-100中功能词={func_count}({func_count/total:.1%}), "
              f"内容词={content_count}({content_count/total:.1%})")
    
    # 3. 每个方向的token频率分析
    print("\n--- 方向与token频率的关系 ---")
    # 计算每个token在训练语料中的近似频率(用W_U的行范数近似)
    token_norms = np.linalg.norm(W_U, axis=1)
    
    for d in range(10):
        direction = U_wu[:, d]
        projections = W_U @ direction
        abs_proj = np.abs(projections)
        
        # 按投影大小分组
        top100_mask = np.zeros(n_vocab, dtype=bool)
        top100_ids = np.argsort(abs_proj)[::-1][:100]
        top100_mask[top100_ids] = True
        
        top100_norms = token_norms[top100_mask]
        other_norms = token_norms[~top100_mask]
        
        if len(other_norms) > 0:
            norm_ratio = np.mean(top100_norms) / (np.mean(other_norms) + 1e-10)
        else:
            norm_ratio = 1.0
        
        print(f"  方向{d}: Top-100平均范数/其他平均范数 = {norm_ratio:.3f}")
    
    # 4. 方向间的正交性和独立性
    print("\n--- 方向间的正交性和独立性 ---")
    for i in range(5):
        for j in range(i+1, 5):
            cos = np.dot(U_wu[:, i], U_wu[:, j])
            print(f"  方向{i} vs 方向{j}: cos={cos:.6f}")
    
    # 5. h在各方向上的投影随层演化
    print("\n--- h在各方向上的投影随层演化 ---")
    text = "The development of artificial intelligence has transformed many aspects of modern life"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    sample_layers = get_sample_layers(n_layers, 6)
    for d in range(5):
        print(f"  方向{d}:", end="")
        for li in sample_layers:
            h = outputs.hidden_states[li][0, -1].detach().cpu().float().numpy()
            proj = np.dot(U_wu[:, d], h)
            print(f" L{li}={proj:.3f}", end="")
        print()
    
    result = {
        "model": model_name,
        "direction_top_tokens": {
            str(d): {
                "pos": [(t, float(v)) for t, v in direction_top_tokens[d]["pos"][:10]],
                "neg": [(t, float(v)) for t, v in direction_top_tokens[d]["neg"][:10]],
            }
            for d in range(10)
        },
    }
    
    return result


# ============== P564: 功能词vs内容词的频谱双峰假设 ==============
def experiment_p564(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """功能词vs内容词的频谱双峰假设 — W_U空间是否存在两个独立的子空间?"""
    print(f"\n{'='*60}")
    print(f"P564: 功能词vs内容词双峰假设 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(200, U_wu.shape[1])
    
    # 1. 定义功能词和内容词集合
    func_words = ["the", "a", "an", "is", "are", "was", "were", "be", "have", "has",
                  "do", "does", "will", "would", "can", "could", "to", "of", "in", "for",
                  "on", "with", "at", "by", "and", "but", "or", "not", "that", "this"]
    
    content_words = ["cat", "dog", "house", "car", "tree", "water", "book", "run", "eat",
                     "big", "small", "red", "blue", "happy", "think", "speak", "write",
                     "love", "time", "world", "power", "nature", "science", "music", "art"]
    
    # 2. 获取功能词和内容词在W_U空间中的表示
    func_projs = []  # [n_func, k_wu]
    content_projs = []  # [n_content, k_wu]
    
    for word in func_words:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if not tok_ids:
            continue
        tok_id = tok_ids[0]
        coord = W_U[tok_id]
        proj = U_wu[:, :k_wu].T @ coord
        func_projs.append(proj)
    
    for word in content_words:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if not tok_ids:
            continue
        tok_id = tok_ids[0]
        coord = W_U[tok_id]
        proj = U_wu[:, :k_wu].T @ coord
        content_projs.append(proj)
    
    func_projs = np.array(func_projs)  # [n_func, k_wu]
    content_projs = np.array(content_projs)  # [n_content, k_wu]
    
    print(f"功能词数: {len(func_projs)}, 内容词数: {len(content_projs)}")
    
    # 3. 分析两个子空间的重叠
    print("\n--- 子空间重叠分析 ---")
    
    # 功能词的主方向
    func_cov = func_projs.T @ func_projs / len(func_projs)
    func_eigvals, func_eigvecs = np.linalg.eigh(func_cov)
    func_eigvals = func_eigvals[::-1]
    func_eigvecs = func_eigvecs[:, ::-1]
    
    # 内容词的主方向
    content_cov = content_projs.T @ content_projs / len(content_projs)
    content_eigvals, content_eigvecs = np.linalg.eigh(content_cov)
    content_eigvals = content_eigvals[::-1]
    content_eigvecs = content_eigvecs[:, ::-1]
    
    # 子空间重叠: 功能词前5个主方向 vs 内容词前5个主方向
    for k in [5, 10, 20]:
        # 用子空间投影计算重叠
        func_subspace = func_eigvecs[:, :k]
        content_subspace = content_eigvecs[:, :k]
        
        # 子空间重叠 = ||P_func @ P_content||_F / k
        overlap_matrix = func_subspace.T @ content_subspace
        overlap = np.linalg.norm(overlap_matrix, 'fro') / np.sqrt(k)
        
        print(f"  前{k}个主方向子空间重叠: {overlap:.4f} (1.0=完全重叠, 0=正交)")
    
    # 4. 功能词和内容词的频谱分布差异
    print("\n--- 频谱分布差异 ---")
    func_spectra = np.abs(func_projs)  # [n_func, k_wu]
    content_spectra = np.abs(content_projs)  # [n_content, k_wu]
    
    # 各方向的平均能量
    func_energy = np.mean(func_spectra**2, axis=0)  # [k_wu]
    content_energy = np.mean(content_spectra**2, axis=0)  # [k_wu]
    
    # 前10个方向的能量占比
    for k in [5, 10, 20, 50]:
        func_ratio = np.sum(func_energy[:k]) / (np.sum(func_energy) + 1e-10)
        content_ratio = np.sum(content_energy[:k]) / (np.sum(content_energy) + 1e-10)
        print(f"  前{k}方向: 功能词能量比={func_ratio:.3f}, 内容词能量比={content_ratio:.3f}, "
              f"差距={func_ratio - content_ratio:+.3f}")
    
    # 5. 双峰检验: 混合分布vs单一分布
    print("\n--- 双峰检验 ---")
    # 计算每个token的"功能度" = 在前5个W_U方向的能量比
    all_projs = np.vstack([func_projs, content_projs])  # [n_total, k_wu]
    n_func = len(func_projs)
    labels = ["func"] * n_func + ["content"] * len(content_projs)
    
    func_degrees = []
    content_degrees = []
    
    for i, proj in enumerate(all_projs):
        abs_proj = np.abs(proj)
        total = np.sum(abs_proj**2) + 1e-10
        func_degree = np.sum(abs_proj[:5]**2) / total  # 前5方向的能量比
        if labels[i] == "func":
            func_degrees.append(func_degree)
        else:
            content_degrees.append(func_degree)
    
    print(f"  功能词'功能度'(前5方向能量比): {np.mean(func_degrees):.3f} ± {np.std(func_degrees):.3f}")
    print(f"  内容词'功能度'(前5方向能量比): {np.mean(content_degrees):.3f} ± {np.std(content_degrees):.3f}")
    
    # t-test
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(func_degrees, content_degrees)
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.6f} {'**显著**' if p_val < 0.01 else '不显著'}")
    
    # 6. 混合模型的似然比检验
    print("\n--- 高斯混合模型检验 ---")
    all_func_degrees = np.array(func_degrees + content_degrees)
    
    # 单高斯
    mu_single = np.mean(all_func_degrees)
    sigma_single = np.std(all_func_degrees)
    log_lik_single = np.sum(-0.5 * ((all_func_degrees - mu_single) / (sigma_single + 1e-10))**2 
                            - np.log(sigma_single + 1e-10))
    
    # 双高斯
    mu_func = np.mean(func_degrees)
    sigma_func = np.std(func_degrees) + 1e-10
    mu_content = np.mean(content_degrees)
    sigma_content = np.std(content_degrees) + 1e-10
    
    log_lik_func = np.sum(-0.5 * ((np.array(func_degrees) - mu_func) / sigma_func)**2 - np.log(sigma_func))
    log_lik_content = np.sum(-0.5 * ((np.array(content_degrees) - mu_content) / sigma_content)**2 - np.log(sigma_content))
    log_lik_dual = log_lik_func + log_lik_content
    
    bic_single = -2 * log_lik_single + 2 * np.log(len(all_func_degrees))
    bic_dual = -2 * log_lik_dual + 4 * np.log(len(all_func_degrees))  # 4参数: mu1,sigma1,mu2,sigma2
    
    print(f"  单高斯 BIC={bic_single:.1f}")
    print(f"  双高斯 BIC={bic_dual:.1f}")
    print(f"  双高斯更优? {'是' if bic_dual < bic_single else '否'} (差距={bic_single - bic_dual:.1f})")
    
    # 7. 子空间维度估计
    print("\n--- 子空间维度估计 ---")
    # 功能词子空间的参与率
    func_var_explained = func_eigvals / (np.sum(func_eigvals) + 1e-10)
    content_var_explained = content_eigvals / (np.sum(content_eigvals) + 1e-10)
    
    for threshold in [0.80, 0.90, 0.95]:
        func_dim = np.searchsorted(np.cumsum(func_var_explained), threshold) + 1
        content_dim = np.searchsorted(np.cumsum(content_var_explained), threshold) + 1
        print(f"  {threshold:.0%}方差: 功能词维度={func_dim}, 内容词维度={content_dim}")
    
    result = {
        "model": model_name,
        "func_func_degree_mean": float(np.mean(func_degrees)),
        "content_func_degree_mean": float(np.mean(content_degrees)),
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "bic_single": float(bic_single),
        "bic_dual": float(bic_dual),
        "dual_better": bool(bic_dual < bic_single),
    }
    
    return result


# ============== P565: 频谱力学到token预测的完整路径 ==============
def experiment_p565(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """频谱力学如何实现token预测 — 从频谱到logits的完整数学路径"""
    print(f"\n{'='*60}")
    print(f"P565: 频谱→logits完整路径 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(200, U_wu.shape[1])
    
    # 1. 从h到logits的分解
    print("\n--- h→logits分解 ---")
    text = "The development of artificial intelligence has transformed many aspects of modern life"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    # 末层h
    h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    
    # 完整logits
    full_logits = W_U @ h_last  # [n_vocab]
    
    # 按W_U方向分解: logits = Σ_k (U_wu[:, k] · h) * (V_wu[k, :])
    # 其中 V_wu[k, :] = W_U @ U_wu[:, k] / S_wu[k]
    h_coeffs = U_wu[:, :k_wu].T @ h_last  # [k_wu]
    
    # 每个W_U方向对logits的贡献
    direction_logit_contributions = []
    for k in range(k_wu):
        # 第k个方向对logits的贡献 = h_coeffs[k] * S_wu[k] * Vt_wu[k, :]
        # 但直接用: (U_wu[:, k] · h) * (W_U @ U_wu[:, k])
        direction_in_logit_space = W_U @ U_wu[:, k]  # [n_vocab]
        contribution = h_coeffs[k] * direction_in_logit_space  # [n_vocab]
        direction_logit_contributions.append(contribution)
    
    direction_logit_contributions = np.array(direction_logit_contributions)  # [k_wu, n_vocab]
    
    # 验证: 累积贡献应接近full_logits
    reconstructed_logits = np.sum(direction_logit_contributions, axis=0)
    cos_sim = np.dot(full_logits, reconstructed_logits) / (np.linalg.norm(full_logits) * np.linalg.norm(reconstructed_logits) + 1e-10)
    print(f"  完整重建logits余弦相似: {cos_sim:.6f}")
    
    # 2. 各方向对top-5预测的贡献
    print("\n--- 各方向对top-5预测的贡献 ---")
    top5_ids = np.argsort(full_logits)[::-1][:5]
    top5_tokens = [safe_decode(tokenizer, tid) for tid in top5_ids]
    print(f"  Top-5 tokens: {top5_tokens}")
    print(f"  Top-5 logits: {[f'{full_logits[tid]:.2f}' for tid in top5_ids]}")
    
    # 每个方向的累积贡献
    cumsum_logits = np.cumsum(direction_logit_contributions, axis=0)  # [k_wu, n_vocab]
    
    for k in [1, 5, 10, 20, 50, 100, 200]:
        partial_logits = cumsum_logits[k-1]
        cos_partial = np.dot(full_logits, partial_logits) / (np.linalg.norm(full_logits) * np.linalg.norm(partial_logits) + 1e-10)
        
        # Top-5预测的重叠
        partial_top5 = set(np.argsort(partial_logits)[::-1][:5])
        overlap = len(partial_top5 & set(top5_ids))
        
        print(f"  前{k}方向: logits余弦={cos_partial:.4f}, Top-5重叠={overlap}/5")
    
    # 3. 频谱系数的统计特性
    print("\n--- 频谱系数统计 ---")
    abs_coeffs = np.abs(h_coeffs)
    total_energy = np.sum(abs_coeffs**2)
    
    # 幂律拟合
    sorted_coeffs = np.sort(abs_coeffs)[::-1]
    x = np.arange(1, len(sorted_coeffs)+1)
    log_x = np.log10(x[5:50])
    log_y = np.log10(sorted_coeffs[5:50] + 1e-10)
    valid = np.isfinite(log_y)
    if np.sum(valid) > 3:
        slope, intercept = np.polyfit(log_x[valid], log_y[valid], 1)
        print(f"  频谱幂律斜率: {slope:.3f} (log-log)")
    else:
        slope = 0
        print(f"  频谱幂律斜率: N/A")
    
    # 能量集中度
    for k in [5, 10, 20, 50]:
        energy_ratio = np.sum(abs_coeffs[:k]**2) / (total_energy + 1e-10)
        print(f"  前{k}方向能量比: {energy_ratio:.4f}")
    
    # 4. 方向贡献的独立性
    print("\n--- 方向贡献独立性 ---")
    # 各方向对logits的贡献是否独立?
    # 如果独立, 则logits方差 = Σ 方向贡献方差
    logit_var = np.var(full_logits)
    dir_vars = np.var(direction_logit_contributions, axis=1)  # [k_wu]
    sum_dir_vars = np.sum(dir_vars)
    
    print(f"  logits方差: {logit_var:.2f}")
    print(f"  Σ方向贡献方差: {sum_dir_vars:.2f}")
    print(f"  比值(=1则独立): {sum_dir_vars / (logit_var + 1e-10):.4f}")
    
    # 方向间的logit贡献相关
    if k_wu > 10:
        cors = []
        for i in range(10):
            for j in range(i+1, 10):
                r, _ = pearsonr(direction_logit_contributions[i], direction_logit_contributions[j])
                cors.append(r)
        print(f"  前10方向间logit贡献平均相关: {np.mean(cors):.4f}")
    
    # 5. 预测信息在频谱中的分布
    print("\n--- 预测信息的频谱分布 ---")
    # 用信息论方法: 每个方向消除多少不确定性
    
    with torch.no_grad():
        full_probs = torch.softmax(torch.tensor(full_logits, dtype=torch.float32), dim=0).numpy()
    full_entropy = -np.sum(full_probs * np.log(full_probs + 1e-10))
    
    # 截断到前K个方向后的entropy
    for K in [5, 10, 20, 50, 100]:
        partial_logits_k = cumsum_logits[K-1]
        partial_probs = torch.softmax(torch.tensor(partial_logits_k, dtype=torch.float32), dim=0).numpy()
        partial_entropy = -np.sum(partial_probs * np.log(partial_probs + 1e-10))
        info_gain = full_entropy - partial_entropy
        print(f"  前{K}方向: entropy={partial_entropy:.3f}, 信息增益={info_gain:.3f} bits, "
              f"消除{(info_gain/full_entropy*100) if full_entropy > 0 else 0:.1f}%不确定性")
    
    print(f"  完整entropy: {full_entropy:.3f}")
    
    result = {
        "model": model_name,
        "spectral_slope": float(slope),
        "logit_reconstruction_cos": float(cos_sim),
        "full_entropy": float(full_entropy),
        "var_ratio": float(sum_dir_vars / (logit_var + 1e-10)),
    }
    
    return result


# ============== P566: 统一语言理论验证 ==============
def experiment_p566(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """统一语言理论: 频谱力学+语义子空间 = 语言能力的数学基础"""
    print(f"\n{'='*60}")
    print(f"P566: 统一语言理论验证 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(200, U_wu.shape[1])
    
    # 1. 完整因果链验证: W_down频谱 → h频谱 → logits → 预测
    print("\n--- 完整因果链验证 ---")
    text = "The development of artificial intelligence has transformed many aspects of modern life"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    
    # 链1: h频谱 → logits
    h_spec = project_and_spectrum(h_last, U_wu, k_wu)
    full_logits = W_U @ h_last
    
    # 用频谱重建h
    h_coeffs = U_wu[:, :k_wu].T @ h_last
    
    # 链2: W_down频谱 → h频谱 (已在P558验证)
    layers = get_layers(model)
    last_layer = layers[n_layers - 1]
    lw = get_layer_weights(last_layer, info.d_model, info.mlp_type)
    W_down_last = lw.W_down
    
    W_down_spec = np.linalg.svd(W_down_last, compute_uv=False)
    
    # 2. 频谱力学参数汇总
    print("\n--- 频谱力学参数汇总 ---")
    
    # alpha (残差保持)
    h_prev = outputs.hidden_states[-2][0, -1].detach().cpu().float().numpy()
    alpha = np.dot(h_last, h_prev) / (np.dot(h_prev, h_prev) + 1e-10)
    print(f"  alpha(残差保持): {alpha:.4f}")
    
    # beta (微调幅度)
    delta = h_last - alpha * h_prev
    beta = np.linalg.norm(delta) / (np.linalg.norm(h_prev) + 1e-10)
    print(f"  beta(微调幅度): {beta:.4f}")
    
    # ratio(50)
    ratio_50 = compute_ratio_k_from_h(h_last, U_wu, 50)
    print(f"  ratio(50): {ratio_50:.4f}")
    
    # 频谱幂律斜率
    sorted_spec = np.sort(np.abs(h_coeffs))[::-1]
    x = np.arange(1, len(sorted_spec)+1)
    log_x = np.log10(x[3:30])
    log_y = np.log10(sorted_spec[3:30] + 1e-10)
    valid = np.isfinite(log_y)
    if np.sum(valid) > 3:
        spectral_slope, _ = np.polyfit(log_x[valid], log_y[valid], 1)
    else:
        spectral_slope = 0
    print(f"  频谱幂律斜率: {spectral_slope:.3f}")
    
    # 功能方向能量比 (前5个W_U方向)
    abs_coeffs = np.abs(h_coeffs)
    total_energy = np.sum(abs_coeffs**2) + 1e-10
    func_energy_ratio = np.sum(abs_coeffs[:5]**2) / total_energy
    print(f"  功能方向(前5)能量比: {func_energy_ratio:.4f}")
    
    # 3. 统一预测模型: 用频谱参数预测logits
    print("\n--- 统一预测模型 ---")
    
    # 模型A: 完整h → logits (上界)
    full_logits = W_U @ h_last
    
    # 模型B: 频谱截断K=50 → logits
    h_recon_50 = U_wu[:, :50] @ h_coeffs[:50]
    logits_50 = W_U @ h_recon_50
    cos_50 = np.dot(full_logits, logits_50) / (np.linalg.norm(full_logits) * np.linalg.norm(logits_50) + 1e-10)
    
    # 模型C: 频谱截断K=10 → logits
    h_recon_10 = U_wu[:, :10] @ h_coeffs[:10]
    logits_10 = W_U @ h_recon_10
    cos_10 = np.dot(full_logits, logits_10) / (np.linalg.norm(full_logits) * np.linalg.norm(logits_10) + 1e-10)
    
    # 模型D: 仅功能方向(K=5) → logits
    h_recon_5 = U_wu[:, :5] @ h_coeffs[:5]
    logits_5 = W_U @ h_recon_5
    cos_5 = np.dot(full_logits, logits_5) / (np.linalg.norm(full_logits) * np.linalg.norm(logits_5) + 1e-10)
    
    print(f"  K=50 → logits余弦: {cos_50:.4f}")
    print(f"  K=10 → logits余弦: {cos_10:.4f}")
    print(f"  K=5  → logits余弦: {cos_5:.4f}")
    
    # 4. 预测质量的频谱分解
    print("\n--- 预测质量的频谱分解 ---")
    
    with torch.no_grad():
        all_logits_t = outputs.logits[0]  # [seq_len, n_vocab]
    
    seq_len = all_logits_t.shape[0]
    position_data = []
    
    for pos in range(1, min(seq_len, 15)):
        h_pos = outputs.hidden_states[-1][0, pos].detach().cpu().float().numpy()
        h_pos_coeffs = U_wu[:, :k_wu].T @ h_pos
        abs_c = np.abs(h_pos_coeffs)
        total_e = np.sum(abs_c**2) + 1e-10
        
        r5 = np.sum(abs_c[:5]**2) / total_e
        r10 = np.sum(abs_c[:10]**2) / total_e
        r50 = np.sum(abs_c[:50]**2) / total_e
        
        probs = torch.softmax(all_logits_t[pos].float(), dim=0)
        top1_prob = probs.max().item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        position_data.append({
            "r5": r5, "r10": r10, "r50": r50,
            "top1_prob": top1_prob, "entropy": entropy
        })
    
    if len(position_data) > 2:
        r10s = [d["r10"] for d in position_data]
        r50s = [d["r50"] for d in position_data]
        top1s = [d["top1_prob"] for d in position_data]
        entropies = [d["entropy"] for d in position_data]
        
        corr_r10_prob, _ = spearmanr(r10s, top1s)
        corr_r50_prob, _ = spearmanr(r50s, top1s)
        corr_r10_ent, _ = spearmanr(r10s, entropies)
        corr_r50_ent, _ = spearmanr(r50s, entropies)
        
        print(f"  r(10) vs top1_prob: Spearman r={corr_r10_prob:.3f}")
        print(f"  r(50) vs top1_prob: Spearman r={corr_r50_prob:.3f}")
        print(f"  r(10) vs entropy: Spearman r={corr_r10_ent:.3f}")
        print(f"  r(50) vs entropy: Spearman r={corr_r50_ent:.3f}")
    else:
        corr_r10_prob = 0
        corr_r50_prob = 0
        corr_r10_ent = 0
        corr_r50_ent = 0
    
    # 5. 统一理论总结
    print("\n--- 统一理论总结 ---")
    print(f"  核心参数: alpha={alpha:.3f}, beta={beta:.3f}, slope={spectral_slope:.3f}")
    print(f"  频谱集中度: r(5)={func_energy_ratio:.3f}, r(10)={np.sum(abs_coeffs[:10]**2)/total_energy:.3f}, r(50)={ratio_50:.3f}")
    print(f"  预测效率: K=5余弦={cos_5:.3f}, K=10余弦={cos_10:.3f}, K=50余弦={cos_50:.3f}")
    print(f"  频谱→预测: r(10)~top1_prob r={corr_r10_prob:.3f}, r(50)~top1_prob r={corr_r50_prob:.3f}")
    
    result = {
        "model": model_name,
        "alpha": float(alpha),
        "beta": float(beta),
        "spectral_slope": float(spectral_slope),
        "ratio_50": float(ratio_50),
        "func_energy_ratio": float(func_energy_ratio),
        "cos_k5": float(cos_5),
        "cos_k10": float(cos_10),
        "cos_k50": float(cos_50),
        "corr_r10_prob": float(corr_r10_prob),
        "corr_r50_prob": float(corr_r50_prob),
    }
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["p563", "p564", "p565", "p566"])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    
    print(f"\n模型: {args.model}, 层数: {info.n_layers}, d_model: {info.d_model}")
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    
    if args.experiment == "p563":
        result = experiment_p563(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p564":
        result = experiment_p564(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p565":
        result = experiment_p565(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p566":
        result = experiment_p566(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    
    # 保存结果
    result_dir = f"results/phase_cxxix"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{args.model}_{args.experiment}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {result_file}")
    
    # 释放GPU
    del model
    torch.cuda.empty_cache()
    print("GPU内存已释放")


if __name__ == "__main__":
    main()
