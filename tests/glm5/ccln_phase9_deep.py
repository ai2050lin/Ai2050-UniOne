"""
CCL-N(255): Phase 9 深入验证 — W_U⊥精细结构 + Final Norm + 跨层因果链
======================================================================
Phase 8核心发现: 语法角色75-94%编码在W_U⊥子空间中!
Phase 9目标: 理解W_U⊥的内部结构和因果机制

实验:
  Exp1: ★★★★★ W_U⊥子空间中的语法角色几何
    → 在W_U⊥投影后的hidden states中分析角色几何
    → W_U⊥中的余弦矩阵 vs 完整空间中的余弦矩阵
    → W_U⊥中是否仍有正四面体结构?

  Exp2: ★★★★★ Final Norm的耦合效应
    → logits = W_U @ LayerNorm(h)  (非线性!)
    → 测量: 纯W_U⊥扰动 → logits变化有多大?
    → 对比: 绕过LayerNorm直接计算 W_U @ delta_h (线性预测)
    → 如果LayerNorm是主要耦合源, 线性预测应远小于实际logits变化

  Exp3: ★★★★★ 跨层因果链 — W_U⊥操控的传播
    → 在中间层(1/3, 1/2, 2/3深度)做W_U⊥操控
    → 测量: (a)该层分类翻转, (b)最后层分类翻转, (c)最后层logits变化
    → 建立因果链: 操控层→最后层→logits
    → 预测: 如果W_U⊥操控是"局部的", 中间层操控不应传播到logits

  Exp4: ★★★★ W_U⊥的维度容量
    → W_U⊥有多少维? 语法角色编码需要多少维?
    → 在W_U⊥的PCA子空间中逐步降维, 测量分类准确率
    → 语法角色的"维度需求" vs W_U⊥的"容量"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode, get_W_U


# ===== 数据集 =====
EXTENDED_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom",
            "The doctor treated the patient",
            "The artist painted the portrait",
            "The soldier defended the castle",
            "The cat sat on the mat",
            "The dog ran through the park",
            "The bird sang a beautiful song",
            "The child played with the toys",
            "The student read the textbook",
            "The teacher explained the lesson",
            "The woman drove the car",
            "The man fixed the roof",
            "The girl sang a song",
            "The boy kicked the ball",
            "The president signed the bill",
            "The chef cooked the meal",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier",
            "cat", "dog", "bird", "child",
            "student", "teacher", "woman", "man",
            "girl", "boy", "president", "chef",
        ],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday",
            "She visited the doctor recently",
            "He admired the artist greatly",
            "We honored the soldier today",
            "She chased the cat away",
            "He found the dog outside",
            "They watched the bird closely",
            "We helped the child today",
            "I praised the student loudly",
            "You thanked the teacher warmly",
            "The police arrested the man quickly",
            "The company hired the woman recently",
            "The coach trained the girl daily",
            "The teacher praised the boy warmly",
            "The nation elected the president fairly",
            "The customer tipped the chef generously",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier",
            "cat", "dog", "bird", "child",
            "student", "teacher", "man", "woman",
            "girl", "boy", "president", "chef",
        ],
    },
    "amod": {
        "sentences": [
            "The brave king fought hard",
            "The kind doctor helped many",
            "The creative artist worked well",
            "The strong soldier marched far",
            "The beautiful cat sat quietly",
            "The large dog ran swiftly",
            "The small bird sang softly",
            "The young child played happily",
            "The bright student read carefully",
            "The wise teacher explained clearly",
            "The old woman walked slowly",
            "The tall man stood quietly",
            "The little girl smiled sweetly",
            "The smart boy answered quickly",
            "The powerful president decided firmly",
            "The skilled chef cooked perfectly",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong",
            "beautiful", "large", "small", "young",
            "bright", "wise", "old", "tall",
            "little", "smart", "powerful", "skilled",
        ],
    },
    "advmod": {
        "sentences": [
            "The king ruled wisely forever",
            "The doctor worked carefully always",
            "The artist painted beautifully daily",
            "The soldier fought bravely there",
            "The cat ran quickly home",
            "The dog barked loudly today",
            "The bird sang softly outside",
            "The child played happily inside",
            "The student read carefully alone",
            "The teacher spoke clearly again",
            "The woman drove slowly home",
            "The man spoke quietly now",
            "The girl laughed happily then",
            "The boy ran fast away",
            "The president spoke firmly today",
            "The chef worked quickly then",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely",
            "quickly", "loudly", "softly", "happily",
            "carefully", "clearly", "slowly", "quietly",
            "happily", "fast", "firmly", "quickly",
        ],
    },
}

ROLE_NAMES = ["nsubj", "dobj", "amod", "advmod"]

FUNCTION_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
    'or', 'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these',
    'those', 'it', 'its', 'he', 'she', 'they', 'them', 'we', 'us', 'me',
    'my', 'your', 'his', 'her', 'their', 'our', 'what', 'which', 'who',
    'whom', 'whose', ',', '.', '!', '?', ';', ':', "'", '"', '-',
    "'s", "'t", "'re", "'ve", "'ll", "'d", "'m",
}


def find_token_index(tokens, word):
    word_lower = word.lower()
    for i, tok in enumerate(tokens):
        if word_lower in tok.lower() or tok.lower().startswith(word_lower[:3]):
            return i
    return None


def is_content_word(token_str):
    t = token_str.strip().lower().lstrip('▁').lstrip(' ')
    if not t:
        return False
    if t in FUNCTION_WORDS:
        return False
    if len(t) <= 1:
        return False
    return True


def get_layer_hidden_with_logits(model, tokenizer, device, sentence, layer_idx):
    """获取指定层的hidden states + 最终logits"""
    layers = get_layers(model)
    target_layer = layers[layer_idx]
    
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float().clone()
        else:
            captured['h'] = output.detach().float().clone()
    
    h_handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output = model(**toks)
    
    h_handle.remove()
    
    logits = output.logits.detach().float().clone() if hasattr(output, 'logits') else None
    
    if 'h' not in captured:
        return None, logits, toks
    
    return captured['h'], logits, toks


def collect_hidden_states(model, tokenizer, device, layer_idx=-1):
    """收集所有语法角色的hidden states"""
    all_h = []
    all_labels = []
    
    layers = get_layers(model)
    if layer_idx == -1:
        target_layer = layers[-1]
    else:
        target_layer = layers[layer_idx]
    
    for role_idx, role in enumerate(ROLE_NAMES):
        data = EXTENDED_DATA[role]
        for sent, target_word in zip(data["sentences"], data["target_words"]):
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens_list, target_word)
            if dep_idx is None:
                continue
            
            captured = {}
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured['h'] = output[0].detach().float().cpu().numpy()
                else:
                    captured['h'] = output.detach().float().cpu().numpy()
            
            h_handle = target_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = model(**toks)
            h_handle.remove()
            
            if 'h' not in captured:
                continue
            
            h_vec = captured['h'][0, dep_idx, :]
            all_h.append(h_vec)
            all_labels.append(role_idx)
    
    return np.array(all_h), np.array(all_labels)


def compute_W_U_subspaces(W_U, n_components=None):
    """计算W_U的行空间和正交子空间的投影矩阵
    
    使用W_U^T @ W_U的完整特征分解, 
    根据特征值分布选择有效阈值
    """
    d_model = W_U.shape[1]
    W_U_f64 = W_U.astype(np.float64)  # [vocab_size, d_model]
    
    # W_U^T @ W_U: [d_model, d_model]
    print(f"  计算W_U^T @ W_U ({d_model}x{d_model})...")
    WtW = W_U_f64.T @ W_U_f64
    
    # 完整特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(WtW)
    eigenvalues = eigenvalues[::-1]  # 降序
    eigenvectors = eigenvectors[:, ::-1]
    
    # 显示特征值谱
    print(f"  W_U^T@W_U 特征值谱:")
    print(f"    Top-5: {eigenvalues[:5].tolist()}")
    print(f"    Median: {np.median(eigenvalues):.2f}")
    print(f"    Bottom-5: {eigenvalues[-5:].tolist()}")
    print(f"    Condition number: {eigenvalues[0]/max(eigenvalues[-1],1e-20):.1f}")
    
    # 计算累积能量
    total_energy = np.sum(eigenvalues)
    cum_energy = np.cumsum(eigenvalues) / total_energy
    for threshold in [0.5, 0.9, 0.95, 0.99, 0.999]:
        dim = np.searchsorted(cum_energy, threshold) + 1
        print(f"    {threshold:.1%}能量: {dim}维")
    
    # 使用能量阈值定义W_U行空间
    # 取99.9%能量的维度作为"有效行空间"
    energy_threshold = 0.999
    par_dim = int(np.searchsorted(cum_energy, energy_threshold) + 1)
    print(f"  W_U有效行空间维度(99.9%能量): {par_dim}")
    
    U_par = eigenvectors[:, :par_dim]  # [d_model, par_dim]
    
    # 行空间投影
    P_parallel = U_par @ U_par.T
    P_perp = np.eye(d_model) - P_parallel
    
    # 验证: P_perp的范数
    perp_norm = np.linalg.norm(P_perp)
    print(f"  P_perp Frobenius范数: {perp_norm:.4f}")
    print(f"  W_U⊥维度: {d_model - par_dim}")
    
    return U_par, P_parallel, P_perp, eigenvalues[:200]


# ===== Exp1: W_U⊥子空间中的语法角色几何 =====
def exp1_W_U_perp_geometry(model, tokenizer, device):
    """分析W_U⊥投影后的语法角色几何"""
    print("\n" + "="*70)
    print("Exp1: W_U⊥子空间中的语法角色几何 ★★★★★")
    print("="*70)
    
    # 收集hidden states
    print("  收集hidden states...")
    H, labels = collect_hidden_states(model, tokenizer, device, layer_idx=-1)
    print(f"  收集到 {len(H)} 个样本")
    
    # 计算W_U子空间
    W_U = get_W_U(model)
    print(f"  W_U shape: {W_U.shape}")
    U, P_par, P_perp, svals = compute_W_U_subspaces(W_U, n_components=200)
    print(f"  W_U行空间维度: {U.shape[1]}, top-50奇异值能量比: {np.sum(svals[:50]**2)/np.sum(svals**2):.4f}")
    
    results = {}
    
    # 1. 完整空间的余弦矩阵
    centers_full = {}
    for ri, role in enumerate(ROLE_NAMES):
        mask = labels == ri
        centers_full[role] = H[mask].mean(axis=0)
    
    cos_mat_full = np.zeros((4, 4))
    for i, r1 in enumerate(ROLE_NAMES):
        for j, r2 in enumerate(ROLE_NAMES):
            v1, v2 = centers_full[r1], centers_full[r2]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_mat_full[i, j] = np.dot(v1, v2) / (n1 * n2)
    
    print("\n  完整空间 聚类中心余弦矩阵:")
    print("        ", "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r in enumerate(ROLE_NAMES):
        row = "  ".join(f"{cos_mat_full[i,j]:8.4f}" for j in range(4))
        print(f"  {r:>8s} {row}")
    mean_off_full = np.mean(cos_mat_full[np.triu_indices(4, k=1)])
    print(f"  mean_off_diag = {mean_off_full:.4f}")
    
    results['cos_mat_full'] = cos_mat_full.tolist()
    results['mean_off_diag_full'] = float(mean_off_full)
    
    # 2. W_U⊥投影后的余弦矩阵
    # 先投影所有hidden states到W_U⊥
    H_perp = (P_perp @ H.T).T  # [N, d_model]
    
    centers_perp = {}
    for ri, role in enumerate(ROLE_NAMES):
        mask = labels == ri
        centers_perp[role] = H_perp[mask].mean(axis=0)
    
    cos_mat_perp = np.zeros((4, 4))
    for i, r1 in enumerate(ROLE_NAMES):
        for j, r2 in enumerate(ROLE_NAMES):
            v1, v2 = centers_perp[r1], centers_perp[r2]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_mat_perp[i, j] = np.dot(v1, v2) / (n1 * n2)
    
    print("\n  W_U⊥投影 聚类中心余弦矩阵:")
    print("        ", "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r in enumerate(ROLE_NAMES):
        row = "  ".join(f"{cos_mat_perp[i,j]:8.4f}" for j in range(4))
        print(f"  {r:>8s} {row}")
    mean_off_perp = np.mean(cos_mat_perp[np.triu_indices(4, k=1)])
    print(f"  mean_off_diag = {mean_off_perp:.4f}")
    
    # 检查是否接近-1/3(正四面体)
    tetra_error_perp = abs(mean_off_perp - (-1/3))
    print(f"  正四面体误差 = |mean_off_diag + 1/3| = {tetra_error_perp:.4f}")
    
    results['cos_mat_perp'] = cos_mat_perp.tolist()
    results['mean_off_diag_perp'] = float(mean_off_perp)
    results['tetra_error_perp'] = float(tetra_error_perp)
    
    # 3. W_U∥投影后的余弦矩阵
    H_par = (P_par @ H.T).T
    centers_par = {}
    for ri, role in enumerate(ROLE_NAMES):
        mask = labels == ri
        centers_par[role] = H_par[mask].mean(axis=0)
    
    cos_mat_par = np.zeros((4, 4))
    for i, r1 in enumerate(ROLE_NAMES):
        for j, r2 in enumerate(ROLE_NAMES):
            v1, v2 = centers_par[r1], centers_par[r2]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_mat_par[i, j] = np.dot(v1, v2) / (n1 * n2)
    
    print("\n  W_U∥投影 聚类中心余弦矩阵:")
    print("        ", "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r in enumerate(ROLE_NAMES):
        row = "  ".join(f"{cos_mat_par[i,j]:8.4f}" for j in range(4))
        print(f"  {r:>8s} {row}")
    mean_off_par = np.mean(cos_mat_par[np.triu_indices(4, k=1)])
    print(f"  mean_off_diag = {mean_off_par:.4f}")
    
    results['cos_mat_par'] = cos_mat_par.tolist()
    results['mean_off_diag_par'] = float(mean_off_par)
    
    # 4. 在W_U⊥空间中训练分类器
    scaler_perp = StandardScaler()
    H_perp_scaled = scaler_perp.fit_transform(H_perp)
    
    probe_perp = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    cv_perp = cross_val_score(probe_perp, H_perp_scaled, labels, cv=5, scoring='accuracy')
    print(f"\n  W_U⊥空间 CV准确率: {cv_perp.mean():.4f} ± {cv_perp.std():.4f}")
    
    probe_perp.fit(H_perp_scaled, labels)
    # 探针方向
    probe_dirs = {}
    for ri, role in enumerate(ROLE_NAMES):
        w = probe_perp.coef_[ri]
        probe_dirs[role] = w / np.linalg.norm(w)
    
    # W_U⊥中探针方向的余弦矩阵
    cos_probe_perp = np.zeros((4, 4))
    for i, r1 in enumerate(ROLE_NAMES):
        for j, r2 in enumerate(ROLE_NAMES):
            cos_probe_perp[i, j] = np.dot(probe_dirs[r1], probe_dirs[r2])
    
    print("\n  W_U⊥空间 探针方向余弦矩阵:")
    print("        ", "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r in enumerate(ROLE_NAMES):
        row = "  ".join(f"{cos_probe_perp[i,j]:8.4f}" for j in range(4))
        print(f"  {r:>8s} {row}")
    mean_off_probe_perp = np.mean(cos_probe_perp[np.triu_indices(4, k=1)])
    print(f"  mean_off_diag = {mean_off_probe_perp:.4f}")
    print(f"  ★ 注意: 这也应该是-1/3 (LR artifact)")
    
    results['cv_perp'] = cv_perp.tolist()
    results['cos_probe_perp'] = cos_probe_perp.tolist()
    results['mean_off_diag_probe_perp'] = float(mean_off_probe_perp)
    
    # 5. W_U⊥中的PCA分析
    # 先去掉全局均值
    H_perp_centered = H_perp - H_perp.mean(axis=0)
    pca_perp = PCA(n_components=20)
    pca_perp.fit(H_perp_centered)
    
    print("\n  W_U⊥空间 PCA方差解释比:")
    cumvar = 0
    for i in range(min(10, len(pca_perp.explained_variance_ratio_))):
        vr = pca_perp.explained_variance_ratio_[i]
        cumvar += vr
        print(f"    PC{i+1}: {vr:.4f} (累计: {cumvar:.4f})")
    
    # 在PCA子空间中分类
    for n_comp in [3, 5, 10, 20]:
        pca_sub = PCA(n_components=n_comp)
        H_pca = pca_sub.fit_transform(H_perp_centered)
        probe_pca = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
        cv_pca = cross_val_score(probe_pca, H_pca, labels, cv=5, scoring='accuracy')
        print(f"  W_U⊥ PCA({n_comp}D) CV: {cv_pca.mean():.4f}")
    
    results['pca_variance_perp'] = pca_perp.explained_variance_ratio_[:20].tolist()
    
    return results


# ===== Exp2: Final Norm的耦合效应 =====
def exp2_final_norm_coupling(model, tokenizer, device):
    """分析LayerNorm如何将W_U⊥信息耦合到logits"""
    print("\n" + "="*70)
    print("Exp2: Final Norm的耦合效应 ★★★★★")
    print("="*70)
    
    # 收集hidden states
    print("  收集hidden states...")
    H, labels = collect_hidden_states(model, tokenizer, device, layer_idx=-1)
    print(f"  收集到 {len(H)} 个样本")
    
    # 计算W_U子空间
    W_U = get_W_U(model)
    U_wu, P_par, P_perp, svals = compute_W_U_subspaces(W_U, n_components=200)
    
    # 获取Final LayerNorm权重
    layers = get_layers(model)
    last_layer = layers[-1]
    
    # 查找final layernorm
    ln_weight = None
    for ln_name in ["input_layernorm", "ln_1"]:
        if hasattr(last_layer, ln_name):
            ln = getattr(last_layer, ln_name)
            if hasattr(ln, "weight"):
                ln_weight = ln.weight.detach().cpu().float().numpy()
                break
    
    # 模型的final norm可能在model.model.norm或model.model.final_layernorm
    model_norm = None
    if hasattr(model, "model"):
        for attr in ["norm", "final_layernorm"]:
            if hasattr(model.model, attr):
                norm_obj = getattr(model.model, attr)
                if hasattr(norm_obj, "weight"):
                    model_norm = norm_obj.weight.detach().cpu().float().numpy()
                    print(f"  找到model-level final norm: {attr}")
                    break
    
    results = {}
    
    # 训练探针
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)
    probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    probe.fit(H_scaled, labels)
    
    # 获取探针方向
    probe_dirs = {}
    for ri, role in enumerate(ROLE_NAMES):
        w = probe.coef_[ri]
        probe_dirs[role] = w / np.linalg.norm(w)
    
    # 选择操控测试对
    manipulation_tests = [
        ("The king ruled the kingdom", "king", "nsubj", "dobj"),
        ("The cat sat on the mat", "cat", "nsubj", "advmod"),
        ("The brave king fought hard", "brave", "amod", "nsubj"),
        ("The king ruled wisely forever", "wisely", "advmod", "amod"),
    ]
    
    alphas = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    print("\n  核心对比: 线性预测 vs 实际logits变化")
    print("  线性预测: delta_logits_linear = W_U @ delta_h (绕过LayerNorm)")
    print("  实际变化: delta_logits_real = logits(h+delta) - logits(h)")
    
    stats = {
        'linear_vs_real': [],
        'perp_leak_ratio': [],
        'par_leak_ratio': [],
    }
    
    for sent, target, src_role, tgt_role in manipulation_tests:
        print(f"\n  [{src_role}→{tgt_role}] {sent} / '{target}'")
        
        h, logits_base, toks = get_layer_hidden_with_logits(model, tokenizer, device, sent, -1)
        if h is None:
            continue
        
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens_list, target)
        if dep_idx is None:
            continue
        
        h_vec = h[0, dep_idx, :].float().cpu().numpy()
        h_norm = float(np.linalg.norm(h_vec))
        logits_base_np = logits_base[0, dep_idx].float().cpu().numpy()
        
        # 探针方向差
        delta_dir = probe_dirs[tgt_role] - probe_dirs[src_role]
        delta_dir = delta_dir / np.linalg.norm(delta_dir)
        
        # 分解到W_U∥和W_U⊥
        delta_par = P_par @ delta_dir
        delta_perp = P_perp @ delta_dir
        
        par_norm = np.linalg.norm(delta_par)
        perp_norm = np.linalg.norm(delta_perp)
        print(f"    h_norm={h_norm:.1f}, par_norm={par_norm:.4f}, perp_norm={perp_norm:.4f}")
        
        for alpha in alphas:
            # === W_U⊥扰动 ===
            delta_h_perp = alpha * h_norm * delta_perp
            h_new_perp = h_vec + delta_h_perp
            
            # 线性预测: W_U @ delta_h_perp (应该≈0, 因为delta_h_perp在W_U⊥中)
            delta_logits_linear_perp = W_U @ delta_h_perp
            linear_change_perp = float(np.linalg.norm(delta_logits_linear_perp))
            
            # 实际计算logits变化
            h_tensor_new = torch.tensor(h_new_perp.reshape(1, 1, -1), dtype=torch.float32)
            # 手动应用LayerNorm + W_U
            # LayerNorm: (h - mean) / std * gamma + beta
            h_mean = h_new_perp.mean()
            h_std = np.sqrt(np.var(h_new_perp) + 1e-5)
            h_normed = (h_new_perp - h_mean) / h_std
            if model_norm is not None:
                h_normed = h_normed * model_norm
            delta_logits_layernorm_perp = W_U @ h_normed - W_U @ ((h_vec - h_vec.mean()) / np.sqrt(np.var(h_vec) + 1e-5) * (model_norm if model_norm is not None else 1.0))
            layernorm_change_perp = float(np.linalg.norm(delta_logits_layernorm_perp))
            
            # 实际模型forward获取logits
            # 由于直接forward太慢, 这里用近似方法
            # 通过hook获取实际logits
            h_base_normed = (h_vec - h_vec.mean()) / np.sqrt(np.var(h_vec) + 1e-5)
            if model_norm is not None:
                h_base_normed = h_base_normed * model_norm
            logits_from_normed = W_U @ h_base_normed
            
            stats['linear_vs_real'].append({
                'alpha': alpha,
                'src_role': src_role,
                'tgt_role': tgt_role,
                'linear_change_perp': linear_change_perp,
                'layernorm_change_perp': layernorm_change_perp,
                'perp_leak_ratio': layernorm_change_perp / max(linear_change_perp, 1e-10),
            })
            
            if alpha == 0.2:
                print(f"    alpha={alpha:.2f}: linear_change={linear_change_perp:.4f}, "
                      f"layernorm_change={layernorm_change_perp:.4f}, "
                      f"leak_ratio={layernorm_change_perp/max(linear_change_perp,1e-10):.2f}x")
    
    # 汇总统计
    perp_leak_ratios = [s['perp_leak_ratio'] for s in stats['linear_vs_real'] 
                       if np.isfinite(s['perp_leak_ratio']) and s['perp_leak_ratio'] < 1000]
    
    if perp_leak_ratios:
        print(f"\n  ★ W_U⊥扰动 → LayerNorm泄露放大比:")
        print(f"    Mean leak ratio: {np.mean(perp_leak_ratios):.2f}x")
        print(f"    Median leak ratio: {np.median(perp_leak_ratios):.2f}x")
        print(f"    如果leak_ratio >> 1: LayerNorm是语法-语义耦合的主要来源!")
    
    # 补充: 测量W_U⊥扰动对logits top-k的实际影响
    print("\n  --- 实际模型forward验证 ---")
    for sent, target, src_role, tgt_role in manipulation_tests[:2]:
        h, logits_base, toks = get_layer_hidden_with_logits(model, tokenizer, device, sent, -1)
        if h is None:
            continue
        
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens_list, target)
        if dep_idx is None:
            continue
        
        h_vec = h[0, dep_idx, :].float().cpu().numpy()
        h_norm = float(np.linalg.norm(h_vec))
        logits_base_np = logits_base[0, dep_idx].float().cpu().numpy()
        
        delta_dir = probe_dirs[tgt_role] - probe_dirs[src_role]
        delta_dir = delta_dir / np.linalg.norm(delta_dir)
        delta_perp = P_perp @ delta_dir
        
        # 在W_U⊥方向扰动, 用hook修改最后层输出
        for alpha in [0.2, 0.5]:
            delta_h_perp = alpha * h_norm * delta_perp
            
            # 注册hook修改最后层输出
            layers_list = get_layers(model)
            last_layer = layers_list[-1]
            
            modified_logits = {}
            def make_modify_hook(delta, pos_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h_out = output[0].detach().clone()
                        h_out[0, pos_idx, :] += torch.tensor(delta, dtype=h_out.dtype, device=h_out.device)
                        return (h_out,) + output[1:]
                    return output
                return hook_fn
            
            hook_handle = last_layer.register_forward_hook(make_modify_hook(delta_h_perp, dep_idx))
            
            with torch.no_grad():
                output = model(**toks)
            
            hook_handle.remove()
            
            logits_modified = output.logits[0, dep_idx].float().cpu().numpy()
            
            # logits变化
            delta_logits = logits_modified - logits_base_np
            logits_change_norm = float(np.linalg.norm(delta_logits))
            
            # top-5 token变化
            base_top5 = np.argsort(logits_base_np)[-5:][::-1]
            mod_top5 = np.argsort(logits_modified)[-5:][::-1]
            
            base_top1 = safe_decode(tokenizer, int(base_top5[0]))
            mod_top1 = safe_decode(tokenizer, int(mod_top5[0]))
            
            print(f"    [{src_role}→{tgt_role}] alpha={alpha:.2f}: "
                  f"top1: {base_top1}→{mod_top1}, "
                  f"logits_change_norm={logits_change_norm:.2f}")
    
    results['leak_stats'] = stats
    if perp_leak_ratios:
        results['mean_leak_ratio'] = float(np.mean(perp_leak_ratios))
        results['median_leak_ratio'] = float(np.median(perp_leak_ratios))
    
    return results


# ===== Exp3: 跨层因果链 =====
def exp3_cross_layer_causal(model, tokenizer, device, model_info):
    """在中间层做W_U⊥操控, 测量传播到最后一层的效果"""
    print("\n" + "="*70)
    print("Exp3: 跨层因果链 — W_U⊥操控的传播 ★★★★★")
    print("="*70)
    
    n_layers = model_info.n_layers
    # 测试层: 1/3, 1/2, 2/3, 最后一层
    test_layer_indices = [
        n_layers // 3,
        n_layers // 2,
        2 * n_layers // 3,
        n_layers - 1,
    ]
    test_layer_names = [f"L{li}" for li in test_layer_indices]
    print(f"  测试层: {test_layer_names}")
    
    # 收集每层的hidden states
    print("  收集各层hidden states...")
    layer_data = {}
    for li in test_layer_indices:
        H, labels = collect_hidden_states(model, tokenizer, device, layer_idx=li)
        layer_data[li] = (H, labels)
        print(f"    L{li}: {len(H)} samples")
    
    # 计算W_U子空间
    W_U = get_W_U(model)
    U_wu, P_par, P_perp, svals = compute_W_U_subspaces(W_U, n_components=200)
    
    # 在每层训练探针
    layer_probes = {}
    for li in test_layer_indices:
        H, labels = layer_data[li]
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H)
        probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
        cv = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
        probe.fit(H_scaled, labels)
        
        probe_dirs = {}
        for ri, role in enumerate(ROLE_NAMES):
            w = probe.coef_[ri]
            probe_dirs[role] = w / np.linalg.norm(w)
        
        layer_probes[li] = {
            'scaler': scaler,
            'probe': probe,
            'dirs': probe_dirs,
            'cv': cv.mean(),
        }
        print(f"    L{li} CV: {cv.mean():.4f}")
    
    # 在各层做W_U⊥操控, 测量最后层的效果
    manipulation_tests = [
        ("The king ruled the kingdom", "king", "nsubj", "dobj"),
        ("The cat sat on the mat", "cat", "nsubj", "advmod"),
        ("The brave king fought hard", "brave", "amod", "nsubj"),
    ]
    
    alphas = [0.2, 0.5]
    
    results = {}
    
    for li_idx, src_layer_idx in enumerate(test_layer_indices):
        layer_name = f"L{src_layer_idx}"
        print(f"\n  === 在{layer_name}层操控 ===")
        
        probe_info = layer_probes[src_layer_idx]
        scaler = probe_info['scaler']
        probe = probe_info['probe']
        probe_dirs = probe_info['dirs']
        
        # 获取该层的W_U⊥分量(需要用该层的H计算)
        H_src, _ = layer_data[src_layer_idx]
        
        layer_results = {
            'src_layer': src_layer_idx,
            'src_cv': probe_info['cv'],
            'cases': [],
        }
        
        for sent, target, src_role, tgt_role in manipulation_tests:
            # 获取该层的hidden state
            layers_list = get_layers(model)
            target_layer = layers_list[src_layer_idx]
            
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens_list, target)
            if dep_idx is None:
                continue
            
            # 获取base的该层hidden state和最终logits
            captured_src = {}
            def make_capture_hook(key):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        captured_src[key] = output[0].detach().float().cpu().numpy()
                    else:
                        captured_src[key] = output.detach().float().cpu().numpy()
                return hook_fn
            
            h_handle = target_layer.register_forward_hook(make_capture_hook('src_h'))
            with torch.no_grad():
                output_base = model(**toks)
            h_handle.remove()
            
            if 'src_h' not in captured_src:
                continue
            
            h_src = captured_src['src_h'][0, dep_idx, :]
            logits_base = output_base.logits[0, dep_idx].float().cpu().numpy()
            h_src_norm = float(np.linalg.norm(h_src))
            
            # 探针方向
            delta_dir = probe_dirs[tgt_role] - probe_dirs[src_role]
            delta_dir = delta_dir / max(np.linalg.norm(delta_dir), 1e-10)
            delta_perp = P_perp @ delta_dir
            
            for alpha in alphas:
                delta_h = alpha * h_src_norm * delta_perp
                
                # 修改该层输出并forward到最后
                def make_modify_hook(delta, pos_idx):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h_out = output[0].detach().clone()
                            h_out[0, pos_idx, :] += torch.tensor(delta, dtype=h_out.dtype, device=h_out.device)
                            return (h_out,) + output[1:]
                        return output
                    return hook_fn
                
                hook_handle = target_layer.register_forward_hook(make_modify_hook(delta_h, dep_idx))
                
                with torch.no_grad():
                    output_mod = model(**toks)
                
                hook_handle.remove()
                
                logits_mod = output_mod.logits[0, dep_idx].float().cpu().numpy()
                
                # 计算效果
                delta_logits = logits_mod - logits_base
                logits_change = float(np.linalg.norm(delta_logits))
                
                # top token变化
                base_top = safe_decode(tokenizer, int(np.argmax(logits_base)))
                mod_top = safe_decode(tokenizer, int(np.argmax(logits_mod)))
                
                # 用最后层探针预测分类
                # 需要获取操控后的最后层hidden state
                # 简化: 用logits变化和top token变化作为代理
                base_top5 = np.argsort(logits_base)[-5:][::-1]
                mod_top5 = np.argsort(logits_mod)[-5:][::-1]
                top5_overlap = len(set(base_top5) & set(mod_top5)) / 5
                
                case_result = {
                    'alpha': alpha,
                    'src_role': src_role,
                    'tgt_role': tgt_role,
                    'base_top': base_top,
                    'mod_top': mod_top,
                    'logits_change': logits_change,
                    'top5_overlap': top5_overlap,
                }
                layer_results['cases'].append(case_result)
                
                print(f"    [{src_role}→{tgt_role}] alpha={alpha:.1f}: "
                      f"top1: {base_top}→{mod_top}, "
                      f"Δlogits={logits_change:.2f}, top5_overlap={top5_overlap:.0%}")
        
        # 汇总该层
        if layer_results['cases']:
            mean_logits_change = np.mean([c['logits_change'] for c in layer_results['cases']])
            mean_top5_overlap = np.mean([c['top5_overlap'] for c in layer_results['cases']])
            print(f"  {layer_name} 汇总: mean_Δlogits={mean_logits_change:.2f}, "
                  f"mean_top5_overlap={mean_top5_overlap:.0%}")
            layer_results['mean_logits_change'] = float(mean_logits_change)
            layer_results['mean_top5_overlap'] = float(mean_top5_overlap)
        
        results[layer_name] = layer_results
    
    # 关键对比
    print("\n  === 跨层传播对比 ===")
    print(f"  {'Layer':>8s} {'CV':>6s} {'mean_Δlogits':>14s} {'top5_overlap':>13s}")
    for li in test_layer_indices:
        name = f"L{li}"
        if name in results and 'mean_logits_change' in results[name]:
            r = results[name]
            print(f"  {name:>8s} {r['src_cv']:6.3f} {r['mean_logits_change']:14.2f} {r['mean_top5_overlap']:13.0%}")
    
    return results


# ===== Exp4: W_U⊥的维度容量 =====
def exp4_perp_dimension_capacity(model, tokenizer, device):
    """分析W_U⊥子空间的维度容量"""
    print("\n" + "="*70)
    print("Exp4: W_U⊥的维度容量 ★★★★")
    print("="*70)
    
    # 收集hidden states
    print("  收集hidden states...")
    H, labels = collect_hidden_states(model, tokenizer, device, layer_idx=-1)
    print(f"  收集到 {len(H)} 个样本, d_model={H.shape[1]}")
    
    # 计算W_U子空间
    W_U = get_W_U(model)
    U_wu, P_par, P_perp, svals = compute_W_U_subspaces(W_U, n_components=200)
    
    # W_U行空间的有效维度
    total_energy = np.sum(svals**2)
    cum_energy = np.cumsum(svals**2) / total_energy
    
    print(f"\n  W_U行空间(即W_U∥)的有效维度:")
    for threshold in [0.5, 0.9, 0.95, 0.99]:
        dim = np.searchsorted(cum_energy, threshold) + 1
        print(f"    {threshold:.0%}能量需要 {dim} 维")
    
    # W_U⊥的维度
    d_model = H.shape[1]
    perp_dim = d_model - U_wu.shape[1]
    print(f"\n  d_model={d_model}, W_U∥维度={U_wu.shape[1]}, W_U⊥维度={perp_dim}")
    print(f"  W_U⊥占比: {perp_dim/d_model:.1%}")
    
    # 投影到W_U⊥, 分析语法角色编码需要的维度
    H_perp = (P_perp @ H.T).T  # [N, d_model]
    H_perp_centered = H_perp - H_perp.mean(axis=0)
    
    # PCA on W_U⊥
    pca = PCA(n_components=min(50, H_perp_centered.shape[1], H_perp_centered.shape[0] - 1))
    pca.fit(H_perp_centered)
    
    print(f"\n  W_U⊥空间的PCA方差解释比:")
    cumvar = 0
    for i in range(min(15, len(pca.explained_variance_ratio_))):
        vr = pca.explained_variance_ratio_[i]
        cumvar += vr
        print(f"    PC{i+1}: {vr:.4f} (累计: {cumvar:.4f})")
    
    # 逐步降维, 测量分类准确率
    print(f"\n  W_U⊥空间 逐步降维 分类准确率:")
    dims_to_test = [2, 3, 5, 10, 15, 20, 30, 50]
    
    results = {}
    dim_accuracies = []
    
    for n_dim in dims_to_test:
        if n_dim > pca.n_components_:
            continue
        H_pca = pca.transform(H_perp_centered)[:, :n_dim]
        probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
        cv = cross_val_score(probe, H_pca, labels, cv=5, scoring='accuracy')
        print(f"    W_U⊥ PCA({n_dim}D): CV={cv.mean():.4f} ± {cv.std():.4f}")
        dim_accuracies.append({'dim': n_dim, 'cv_mean': float(cv.mean()), 'cv_std': float(cv.std())})
    
    # 找到"足够维度" - 达到95%最大准确率的最小维度
    max_acc = max(d['cv_mean'] for d in dim_accuracies) if dim_accuracies else 0
    sufficient_dim = None
    for d in dim_accuracies:
        if d['cv_mean'] >= 0.95 * max_acc:
            sufficient_dim = d['dim']
            break
    
    print(f"\n  ★ 4个语法角色在W_U⊥中需要的最小维度: {sufficient_dim}")
    print(f"  ★ W_U⊥总维度: {perp_dim}")
    print(f"  ★ 维度利用率: {sufficient_dim}/{perp_dim} = {sufficient_dim/perp_dim:.4%}")
    
    # 对比: 在完整空间中的维度需求
    H_centered = H - H.mean(axis=0)
    pca_full = PCA(n_components=min(50, H_centered.shape[1], H_centered.shape[0] - 1))
    pca_full.fit(H_centered)
    
    print(f"\n  完整空间 逐步降维 分类准确率:")
    dim_accuracies_full = []
    for n_dim in dims_to_test:
        if n_dim > pca_full.n_components_:
            continue
        H_pca = pca_full.transform(H_centered)[:, :n_dim]
        probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
        cv = cross_val_score(probe, H_pca, labels, cv=5, scoring='accuracy')
        print(f"    Full PCA({n_dim}D): CV={cv.mean():.4f} ± {cv.std():.4f}")
        dim_accuracies_full.append({'dim': n_dim, 'cv_mean': float(cv.mean()), 'cv_std': float(cv.std())})
    
    results['w_u_parallel_effective_dim'] = int(np.searchsorted(cum_energy, 0.95) + 1)
    results['perp_dim'] = int(perp_dim)
    results['d_model'] = int(d_model)
    results['dim_accuracies_perp'] = dim_accuracies
    results['dim_accuracies_full'] = dim_accuracies_full
    results['sufficient_dim_perp'] = sufficient_dim
    results['max_acc_perp'] = float(max_acc)
    
    return results


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3, 4])
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"CCL-N Phase 9 深入验证 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")
    
    try:
        if args.exp == 1:
            results = exp1_W_U_perp_geometry(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_final_norm_coupling(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_cross_layer_causal(model, tokenizer, device, model_info)
        elif args.exp == 4:
            results = exp4_perp_dimension_capacity(model, tokenizer, device)
        
        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 
                               f"ccln_exp{args.exp}_{args.model}_results.json")
        
        # JSON序列化
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(convert(v) for v in obj)
            return obj
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {out_path}")
    
    finally:
        release_model(model)
        print(f"  模型已释放")


if __name__ == "__main__":
    main()
