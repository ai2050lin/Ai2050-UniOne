"""
CCL-L(253): 语法角色的真实几何结构分析
==================================================
核心洞察: CCL-K证明正四面体是LR的数学性质, 不是hidden state的结构!

关键否定性发现:
  - 随机标签也产生正四面体 (10/10)
  - 随机向量也产生正四面体 (10/10)
  - 原因: LogisticRegression的K个归一化权重自然满足sum≈0, 
    对K=4这恰好是正四面体条件

正确方法: 不分析探针权重的几何, 而分析**聚类中心**的几何!

实验:
  Exp1: ★★★★★ 聚类中心几何分析
    → 计算各语法角色的聚类中心
    → 分析聚类中心之间的余弦矩阵
    → 分析聚类中心是否构成正四面体
    → 对比: 聚类中心几何 vs 探针权重几何

  Exp2: ★★★★★ 类间距离vs类内方差的信噪比分析
    → 对每对角色: 类间距离 / 类内标准差
    → 哪些角色对最容易/最难区分?
    → 信噪比 = 真正衡量角色区分度的指标

  Exp3: ★★★★ PCA维度分析
    → 4个聚类中心张成多少维?
    → 各主成分解释多少方差?
    → 这告诉我们语法角色编码的内在维度

  Exp4: ★★★★ 聚类中心的W_U分解
    → 聚类中心在W_U行空间中的投影比例
    → 哪些维度是"语法专属"的(与W_U正交)?
    → 语法信息vs语义信息的维度分配
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
from scipy.spatial.distance import cosine as cosine_distance

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


def find_token_index(tokens, word):
    word_lower = word.lower()
    for i, tok in enumerate(tokens):
        if word_lower in tok.lower() or tok.lower().startswith(word_lower[:3]):
            return i
    return None


def get_layer_hidden(model, tokenizer, device, sentence, layer_idx):
    """获取指定层的hidden states"""
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
    
    if 'h' not in captured:
        return None, toks
    
    return captured['h'], toks


def collect_hidden_states(model, tokenizer, device, layer_idx, data_dict, role_names):
    """收集指定层所有hidden states"""
    all_hidden = []
    all_labels = []
    
    for role_idx, role in enumerate(role_names):
        data = data_dict[role]
        for sent, target in zip(data["sentences"], data["target_words"]):
            h, toks = get_layer_hidden(model, tokenizer, device, sent, layer_idx)
            if h is None:
                continue
            
            input_ids = toks.input_ids
            tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens, target)
            if dep_idx is None:
                continue
            
            h_vec = h[0, dep_idx, :].float().cpu().numpy()
            all_hidden.append(h_vec)
            all_labels.append(role_idx)
    
    return np.array(all_hidden), np.array(all_labels)


# ================================================================
# Exp1: 聚类中心几何分析
# ================================================================

def exp1_cluster_center_geometry(model, tokenizer, device, model_info):
    """
    ★★★★★ 聚类中心几何分析
    
    不分析探针权重(已知是LR artifact), 而分析聚类中心!
    """
    print("\n" + "="*70)
    print("Exp1: 聚类中心几何分析 — 真实几何结构")
    print("="*70)
    
    # 1. 收集hidden states
    print("\n[1] Collecting hidden states from last layer...")
    X, y = collect_hidden_states(
        model, tokenizer, device, model_info.n_layers - 1, EXTENDED_DATA, ROLE_NAMES
    )
    print(f"  {len(X)} samples, {X.shape[1]} dims")
    print(f"  Label distribution: {np.bincount(y)}")
    
    # 2. 计算聚类中心
    print("\n[2] Computing cluster centers...")
    centers = {}
    class_data = {}
    for role_idx, role in enumerate(ROLE_NAMES):
        mask = y == role_idx
        X_class = X[mask]
        center = X_class.mean(axis=0)
        centers[role] = center
        class_data[role] = X_class
        print(f"  {role}: n={mask.sum()}, ||center||={np.linalg.norm(center):.2f}")
    
    # 3. 聚类中心余弦矩阵
    print("\n[3] Cluster center cosine matrix...")
    center_matrix = np.array([centers[r] for r in ROLE_NAMES])
    
    cos_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            ni = np.linalg.norm(center_matrix[i])
            nj = np.linalg.norm(center_matrix[j])
            if ni > 1e-10 and nj > 1e-10:
                cos_matrix[i, j] = np.dot(center_matrix[i], center_matrix[j]) / (ni * nj)
            else:
                cos_matrix[i, j] = 0
    
    print("  Cosine matrix of cluster centers:")
    print("         " + "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r1 in enumerate(ROLE_NAMES):
        row = f"  {r1:>6s}"
        for j in range(4):
            row += f"  {cos_matrix[i,j]:8.4f}"
        print(row)
    
    # 非对角元素
    off_diag = []
    for i in range(4):
        for j in range(i+1, 4):
            off_diag.append(cos_matrix[i, j])
    
    mean_off_diag = np.mean(off_diag)
    print(f"\n  Mean off-diagonal cosine: {mean_off_diag:.4f}")
    print(f"  (正四面体预期: -0.3333, 随机预期: ~0)")
    
    # 4. 聚类中心距离矩阵
    print("\n[4] Cluster center distance matrix...")
    dist_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            dist_matrix[i, j] = np.linalg.norm(center_matrix[i] - center_matrix[j])
    
    print("  Distance matrix:")
    print("         " + "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r1 in enumerate(ROLE_NAMES):
        row = f"  {r1:>6s}"
        for j in range(4):
            row += f"  {dist_matrix[i,j]:8.2f}"
        print(row)
    
    # 5. 聚类中心是否构成正四面体?
    print("\n[5] Tetrahedron check for CLUSTER CENTERS...")
    
    # 归一化后的聚类中心
    center_norms = [np.linalg.norm(c) for c in center_matrix]
    center_hat = center_matrix / np.array(center_norms)[:, np.newaxis]
    
    cos_hat_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            cos_hat_matrix[i, j] = np.dot(center_hat[i], center_hat[j])
    
    off_diag_hat = []
    for i in range(4):
        for j in range(i+1, 4):
            off_diag_hat.append(cos_hat_matrix[i, j])
    
    mean_off_diag_hat = np.mean(off_diag_hat)
    error_tetra = abs(mean_off_diag_hat - (-1/3))
    is_tetra = error_tetra < 0.05
    
    print(f"  Normalized centers mean off-diag cosine: {mean_off_diag_hat:.4f}")
    print(f"  Error from tetrahedron: {error_tetra:.4f}")
    print(f"  Is tetrahedron: {is_tetra}")
    
    # 6. PCA分析聚类中心
    print("\n[6] PCA of cluster centers...")
    pca = PCA(n_components=4)
    pca.fit(center_matrix)
    print(f"  Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"  Intrinsic dimensionality: ", end="")
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for i, cv in enumerate(cumvar):
        print(f"PC{i+1}={cv:.4f}", end=" ")
    print()
    
    # 7. 对比: 聚类中心几何 vs 探针权重几何
    print("\n[7] Comparison: Cluster centers vs Probe weights...")
    
    # 训练探针获取权重
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    probe.fit(X_scaled, y)
    
    probe_weights = probe.coef_ / scaler.scale_[np.newaxis, :]
    probe_hat = probe_weights / np.linalg.norm(probe_weights, axis=1, keepdims=True)
    
    cos_probe = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            cos_probe[i, j] = np.dot(probe_hat[i], probe_hat[j])
    
    off_diag_probe = []
    for i in range(4):
        for j in range(i+1, 4):
            off_diag_probe.append(cos_probe[i, j])
    
    mean_off_diag_probe = np.mean(off_diag_probe)
    
    print(f"  Cluster centers: mean_off_diag={mean_off_diag_hat:.4f}, tetra_error={error_tetra:.4f}")
    print(f"  Probe weights:   mean_off_diag={mean_off_diag_probe:.4f}, "
          f"tetra_error={abs(mean_off_diag_probe - (-1/3)):.4f}")
    print(f"  → Probe weights form tetrahedron (LR artifact)")
    print(f"  → Cluster centers {'DO' if is_tetra else 'DO NOT'} form tetrahedron")
    
    # 8. 关键分析: 聚类中心的"质心位移"
    print("\n[8] Centroid displacement analysis...")
    overall_center = X.mean(axis=0)
    displacements = {}
    for role in ROLE_NAMES:
        disp = centers[role] - overall_center
        displacements[role] = disp
        print(f"  {role}: ||displacement||={np.linalg.norm(disp):.2f}")
    
    # 位移方向的余弦
    disp_matrix = np.array([displacements[r] for r in ROLE_NAMES])
    cos_disp = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            ni = np.linalg.norm(disp_matrix[i])
            nj = np.linalg.norm(disp_matrix[j])
            if ni > 1e-10 and nj > 1e-10:
                cos_disp[i, j] = np.dot(disp_matrix[i], disp_matrix[j]) / (ni * nj)
    
    print("\n  Displacement cosine matrix:")
    print("         " + "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r1 in enumerate(ROLE_NAMES):
        row = f"  {r1:>6s}"
        for j in range(4):
            row += f"  {cos_disp[i,j]:8.4f}"
        print(row)
    
    off_diag_disp = []
    for i in range(4):
        for j in range(i+1, 4):
            off_diag_disp.append(cos_disp[i, j])
    
    mean_off_diag_disp = np.mean(off_diag_disp)
    error_disp_tetra = abs(mean_off_diag_disp - (-1/3))
    
    print(f"\n  Displacement mean off-diag: {mean_off_diag_disp:.4f}")
    print(f"  Error from tetrahedron: {error_disp_tetra:.4f}")
    print(f"  Is tetrahedron: {error_disp_tetra < 0.05}")
    
    return {
        'center_cos_matrix': cos_matrix.tolist(),
        'center_dist_matrix': dist_matrix.tolist(),
        'center_mean_off_diag': float(mean_off_diag_hat),
        'center_tetra_error': float(error_tetra),
        'center_is_tetrahedron': bool(is_tetra),
        'displacement_mean_off_diag': float(mean_off_diag_disp),
        'displacement_tetra_error': float(error_disp_tetra),
        'probe_mean_off_diag': float(mean_off_diag_probe),
        'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
    }


# ================================================================
# Exp2: 信噪比分析
# ================================================================

def exp2_snr_analysis(model, tokenizer, device, model_info):
    """
    ★★★★★ 类间距离vs类内方差的信噪比分析
    """
    print("\n" + "="*70)
    print("Exp2: 信噪比分析 — 类间距离 / 类内标准差")
    print("="*70)
    
    X, y = collect_hidden_states(
        model, tokenizer, device, model_info.n_layers - 1, EXTENDED_DATA, ROLE_NAMES
    )
    print(f"  {len(X)} samples")
    
    # 各类统计
    class_data = {}
    class_center = {}
    class_std = {}
    
    for role_idx, role in enumerate(ROLE_NAMES):
        mask = y == role_idx
        X_class = X[mask]
        class_data[role] = X_class
        class_center[role] = X_class.mean(axis=0)
        # 类内标准差(到中心的平均距离)
        dists = np.linalg.norm(X_class - class_center[role], axis=1)
        class_std[role] = np.mean(dists)
        print(f"  {role}: n={mask.sum()}, intra_std={class_std[role]:.2f}")
    
    # 类间距离
    print("\n  Inter-class SNR (distance / avg_intra_std):")
    snr_matrix = np.zeros((4, 4))
    dist_matrix = np.zeros((4, 4))
    
    for i, r1 in enumerate(ROLE_NAMES):
        for j, r2 in enumerate(ROLE_NAMES):
            if i == j:
                continue
            dist = np.linalg.norm(class_center[r1] - class_center[r2])
            avg_std = (class_std[r1] + class_std[r2]) / 2
            snr = dist / max(avg_std, 1e-10)
            snr_matrix[i, j] = snr
            dist_matrix[i, j] = dist
    
    print("         " + "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r1 in enumerate(ROLE_NAMES):
        row = f"  {r1:>6s}"
        for j in range(4):
            if i == j:
                row += f"  {'---':>8s}"
            else:
                row += f"  {snr_matrix[i,j]:8.2f}"
        print(row)
    
    # 最容易/最难区分的角色对
    pairs = []
    for i in range(4):
        for j in range(i+1, 4):
            pairs.append((ROLE_NAMES[i], ROLE_NAMES[j], snr_matrix[i, j], dist_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2])
    print(f"\n  Easiest to separate: {pairs[-1][0]}-{pairs[-1][1]}, SNR={pairs[-1][2]:.2f}")
    print(f"  Hardest to separate: {pairs[0][0]}-{pairs[0][1]}, SNR={pairs[0][2]:.2f}")
    print(f"  Mean SNR: {np.mean([p[2] for p in pairs]):.2f}")
    print(f"  SNR range: {min(p[2] for p in pairs):.2f} - {max(p[2] for p in pairs):.2f}")
    
    # Fisher判据
    print("\n  Fisher discriminant ratio (between/within variance):")
    overall_center = X.mean(axis=0)
    
    S_B = np.zeros((X.shape[1], X.shape[1]))  # between-class scatter
    S_W = np.zeros((X.shape[1], X.shape[1]))  # within-class scatter
    
    for role_idx, role in enumerate(ROLE_NAMES):
        mask = y == role_idx
        X_class = X[mask]
        n_k = X_class.shape[0]
        
        diff = (class_center[role] - overall_center).reshape(-1, 1)
        S_B += n_k * (diff @ diff.T)
        
        for x in X_class:
            d = (x - class_center[role]).reshape(-1, 1)
            S_W += d @ d.T
    
    fisher_ratio = np.trace(S_B) / max(np.trace(S_W), 1e-10)
    print(f"  Fisher ratio (trace): {fisher_ratio:.4f}")
    
    return {
        'snr_matrix': snr_matrix.tolist(),
        'pairs_sorted': [(p[0], p[1], float(p[2]), float(p[3])) for p in pairs],
        'mean_snr': float(np.mean([p[2] for p in pairs])),
        'fisher_ratio': float(fisher_ratio),
        'class_intra_std': {r: float(class_std[r]) for r in ROLE_NAMES},
    }


# ================================================================
# Exp3: PCA维度分析
# ================================================================

def exp3_pca_dimensionality(model, tokenizer, device, model_info):
    """
    ★★★★ PCA维度分析 — 语法角色编码的内在维度
    """
    print("\n" + "="*70)
    print("Exp3: PCA维度分析")
    print("="*70)
    
    X, y = collect_hidden_states(
        model, tokenizer, device, model_info.n_layers - 1, EXTENDED_DATA, ROLE_NAMES
    )
    print(f"  {len(X)} samples, {X.shape[1]} dims")
    
    # 1. 全部数据的PCA
    print("\n[1] PCA of all hidden states...")
    pca_full = PCA(n_components=20)
    pca_full.fit(X)
    print(f"  Top 10 variance ratios: {pca_full.explained_variance_ratio_[:10]}")
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    print(f"  Cumulative: {cumvar[:10]}")
    
    # 2. 聚类中心的PCA
    print("\n[2] PCA of cluster centers...")
    centers = []
    for role_idx, role in enumerate(ROLE_NAMES):
        mask = y == role_idx
        centers.append(X[mask].mean(axis=0))
    centers = np.array(centers)
    
    pca_centers = PCA(n_components=4)
    pca_centers.fit(centers)
    print(f"  Variance ratios: {pca_centers.explained_variance_ratio_}")
    print(f"  → 4个聚类中心的有效维度: ", end="")
    n_sig = sum(v > 0.01 for v in pca_centers.explained_variance_ratio_)
    print(f"{n_sig}")
    
    # 3. 质心位移的PCA
    print("\n[3] PCA of centroid displacements...")
    overall_center = X.mean(axis=0)
    displacements = []
    for role_idx, role in enumerate(ROLE_NAMES):
        mask = y == role_idx
        class_center = X[mask].mean(axis=0)
        displacements.append(class_center - overall_center)
    displacements = np.array(displacements)
    
    pca_disp = PCA(n_components=4)
    pca_disp.fit(displacements)
    print(f"  Variance ratios: {pca_disp.explained_variance_ratio_}")
    n_sig_disp = sum(v > 0.01 for v in pca_disp.explained_variance_ratio_)
    print(f"  → 质心位移的有效维度: {n_sig_disp}")
    
    # 4. 语法角色分类投影
    print("\n[4] Projection of grammar roles onto top PCs...")
    # 计算每个样本在前3个质心位移PC上的投影
    W_pca = pca_disp.components_[:3]  # [3, d_model]
    
    projections = {}
    for role_idx, role in enumerate(ROLE_NAMES):
        mask = y == role_idx
        X_class = X[mask]
        proj = X_class @ W_pca.T  # [n, 3]
        projections[role] = {
            'mean': proj.mean(axis=0).tolist(),
            'std': proj.std(axis=0).tolist(),
        }
        print(f"  {role}: PC1={proj.mean(axis=0)[0]:.2f}±{proj.std(axis=0)[0]:.2f}, "
              f"PC2={proj.mean(axis=0)[1]:.2f}±{proj.std(axis=0)[1]:.2f}, "
              f"PC3={proj.mean(axis=0)[2]:.2f}±{proj.std(axis=0)[2]:.2f}")
    
    return {
        'full_pca_variance': pca_full.explained_variance_ratio_[:10].tolist(),
        'centers_pca_variance': pca_centers.explained_variance_ratio_.tolist(),
        'displacement_pca_variance': pca_disp.explained_variance_ratio_.tolist(),
        'n_sig_dimensions': int(n_sig_disp),
        'projections': projections,
    }


# ================================================================
# Exp4: 聚类中心的W_U分解
# ================================================================

def exp4_wu_decomposition(model, tokenizer, device, model_info):
    """
    ★★★★ 聚类中心的W_U分解 — 语法vs语义的维度分配
    """
    print("\n" + "="*70)
    print("Exp4: 聚类中心的W_U分解")
    print("="*70)
    
    X, y = collect_hidden_states(
        model, tokenizer, device, model_info.n_layers - 1, EXTENDED_DATA, ROLE_NAMES
    )
    print(f"  {len(X)} samples")
    
    # 1. 计算质心位移
    print("\n[1] Computing centroid displacements...")
    overall_center = X.mean(axis=0)
    displacements = {}
    for role_idx, role in enumerate(ROLE_NAMES):
        mask = y == role_idx
        class_center = X[mask].mean(axis=0)
        displacements[role] = class_center - overall_center
        print(f"  {role}: ||disp||={np.linalg.norm(displacements[role]):.2f}")
    
    # 2. 获取W_U行空间基
    print("\n[2] Computing W_U row space basis...")
    W_U = get_W_U(model)
    print(f"  W_U shape: {W_U.shape}")
    
    W_U_T = W_U.T.astype(np.float32)
    k = min(300, min(W_U_T.shape[0], W_U_T.shape[1]) - 2)
    print(f"  SVD with k={k}...")
    U_wut, s_wut, _ = svds(W_U_T, k=k)
    U_wut = np.asarray(U_wut, dtype=np.float64)
    print(f"  Done. Top 5 singular values: {s_wut[:5]}")
    
    total_var = float(np.sum(s_wut**2))
    captured_var = float(np.sum(s_wut**2))
    print(f"  Captured variance ratio: {captured_var/total_var:.4f}")
    
    # 3. 分解每个质心位移
    print("\n[3] Decomposing centroid displacements...")
    decomp = {}
    for role in ROLE_NAMES:
        disp = displacements[role]
        disp_norm = np.linalg.norm(disp)
        
        # 投影到W_U行空间
        proj_coeffs = U_wut.T @ disp
        parallel = U_wut @ proj_coeffs
        perp = disp - parallel
        
        par_energy = np.dot(parallel, parallel)
        perp_energy = np.dot(perp, perp)
        
        decomp[role] = {
            'total_norm': float(disp_norm),
            'parallel_norm': float(np.linalg.norm(parallel)),
            'perp_norm': float(np.linalg.norm(perp)),
            'parallel_ratio': float(par_energy / max(disp_norm**2, 1e-20)),
            'perp_ratio': float(perp_energy / max(disp_norm**2, 1e-20)),
        }
        
        print(f"  {role}: parallel={decomp[role]['parallel_ratio']:.4f}, "
              f"perp={decomp[role]['perp_ratio']:.4f}")
    
    # 4. 分类器在W_U正交补空间中的表现
    print("\n[4] Classification in W_U orthogonal complement...")
    
    # 将所有数据投影到W_U正交补空间
    X_centered = X - overall_center
    proj_to_wu = X_centered @ U_wut @ U_wut.T  # 在W_U行空间中的投影
    X_perp = X_centered - proj_to_wu  # 正交补中的分量
    
    # 在正交补空间中训练分类器
    scaler_perp = StandardScaler()
    X_perp_scaled = scaler_perp.fit_transform(X_perp)
    
    probe_perp = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    cv_folds = min(5, min(np.bincount(y)))
    if cv_folds >= 2:
        cv_perp = cross_val_score(probe_perp, X_perp_scaled, y, cv=cv_folds, scoring='accuracy')
        print(f"  Classification in W_U⊥: CV={cv_perp.mean():.4f}±{cv_perp.std():.4f}")
    else:
        cv_perp = np.array([-1])
        print(f"  Not enough samples for CV")
    
    # 对比: 在W_U行空间中的分类
    X_par = proj_to_wu
    scaler_par = StandardScaler()
    X_par_scaled = scaler_par.fit_transform(X_par)
    
    probe_par = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    if cv_folds >= 2:
        cv_par = cross_val_score(probe_par, X_par_scaled, y, cv=cv_folds, scoring='accuracy')
        print(f"  Classification in W_U:  CV={cv_par.mean():.4f}±{cv_par.std():.4f}")
    
    # 完整空间
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_centered)
    probe_full = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    if cv_folds >= 2:
        cv_full = cross_val_score(probe_full, X_full_scaled, y, cv=cv_folds, scoring='accuracy')
        print(f"  Classification in full: CV={cv_full.mean():.4f}±{cv_full.std():.4f}")
    
    # 5. 各维度对分类的贡献
    print("\n[5] Dimension contribution to classification...")
    
    # 训练完整探针
    scaler_all = StandardScaler()
    X_all_scaled = scaler_all.fit_transform(X)
    probe_all = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    probe_all.fit(X_all_scaled, y)
    
    # 探针权重在W_U行空间中的投影比例
    probe_w = probe_all.coef_ / scaler_all.scale_[np.newaxis, :]
    for i, role in enumerate(ROLE_NAMES):
        w = probe_w[i]
        proj_c = U_wut.T @ w
        par = U_wut @ proj_c
        perp = w - par
        par_ratio = np.dot(par, par) / max(np.dot(w, w), 1e-20)
        perp_ratio = np.dot(perp, perp) / max(np.dot(w, w), 1e-20)
        print(f"  {role} probe: parallel={par_ratio:.4f}, perp={perp_ratio:.4f}")
    
    return {
        'displacement_decomposition': decomp,
        'classification_in_WU_perp': float(cv_perp.mean()),
        'classification_in_WU': float(cv_par.mean()) if cv_folds >= 2 else None,
        'classification_in_full': float(cv_full.mean()) if cv_folds >= 2 else None,
    }


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CCL-L: Real Geometry Analysis")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, 
                       choices=[1, 2, 3, 4],
                       help="1=cluster_centers, 2=snr, 3=pca, 4=wu_decomp")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"Model: {model_info.name}, layers={model_info.n_layers}, d_model={model_info.d_model}")
    
    exp_funcs = {
        1: exp1_cluster_center_geometry,
        2: exp2_snr_analysis,
        3: exp3_pca_dimensionality,
        4: exp4_wu_decomposition,
    }
    
    result = exp_funcs[args.exp](model, tokenizer, device, model_info)
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
    os.makedirs(results_dir, exist_ok=True)
    
    exp_names = {1: "cluster_centers", 2: "snr", 3: "pca", 4: "wu_decomp"}
    result_path = os.path.join(results_dir, 
                              f"ccll_exp{args.exp}_{exp_names[args.exp]}_{model_info.name}_results.json")
    
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
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(convert(result), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {result_path}")
    
    release_model(model)


if __name__ == "__main__":
    main()
