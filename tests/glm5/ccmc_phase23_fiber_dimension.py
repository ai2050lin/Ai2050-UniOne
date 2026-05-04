"""
CCMC(Phase 23): 纤维维度的真正验证——从3个角色到8个角色
=============================================================================
用户批评的5个硬伤:
  硬伤1: δ_role=2D是线性代数恒等式(3点rank≤2), 不是发现!
  硬伤2: δ_role不应是常量, 应分解为δ̄_r + ε_r(x)
  硬伤3: 合并LID不是"共享流形"的直接证据
  硬伤4: 切空间距离π的解释过度(小样本+高维噪声)
  硬伤5: CV<0.2是经验阈值, 需要统计检验

用户提出的严格数学框架:
  h_ℓ(x,r) = M_ℓ(x) + Δ_ℓ(r,x)
  Δ_ℓ(r,x) = A_ℓ(r) + B_ℓ(r,x)
  其中:
    M_ℓ(x): 共享语义流形坐标
    A_ℓ(r): 角色平均纤维
    B_ℓ(r,x): 词依赖修正项

核心命题(用户提出):
  命题1: 平均语法纤维是否低维? → rank{δ̄_r} with 8 roles
  命题2: 纤维是否跨层平行运输? → δ̄_{ℓ+1}(r) ≈ T_ℓ δ̄_ℓ(r)
  命题3: A_ℓ(r) vs B_ℓ(r,x)的相对贡献

实验:
  Exp1: ★★★★★★★★★★★ 8个语法角色的纤维维度
    → 收集8个角色(nsubj, dobj, amod, advmod, det, prep, iobj, poss)
    → 计算rank{δ̄_r}(8个角色平均偏移的秩)
    → PCA谱分析: 前k个主成分解释方差的累计比
    → 与随机baseline比较

  Exp2: ★★★★★★★★★ Δ = A + B分解
    → A_ℓ(r) = mean_x[Δ_ℓ(r,x)] (角色平均纤维)
    → B_ℓ(r,x) = Δ_ℓ(r,x) - A_ℓ(r) (词依赖修正)
    → ||B|| / ||Δ|| 的比值: 词特异偏移占多少?
    → B的有效维度: 词特异偏移需要多少维?

  Exp3: ★★★★★★★★★ 平行运输验证(带bootstrap)
    → δ̄_{ℓ+1}(r) ≈ T_ℓ δ̄_ℓ(r) (Procrustes)
    → bootstrap置信区间(不用CV<0.2)
    → permutation test: 打乱层序后T的拟合优度

  Exp4: ★★★★★★★ M_ℓ(x)的定义与验证
    → 共同PCA: 所有角色数据做PCA, 前k维作为M_ℓ(x)
    → CCA: 找跨角色共享的典型方向
    → 残差分析: h - M = Δ 的方差分解
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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import procrustes
from scipy.stats import permutation_test

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode, MODEL_CONFIGS


# ===== 8个英文语法角色数据 =====
# 关键设计: 同一组名词出现在不同角色中, 控制词汇效应
EN_ROLES_DATA_8 = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom wisely",
            "The doctor treated the patient carefully",
            "The artist painted the portrait beautifully",
            "The soldier defended the castle bravely",
            "The teacher explained the lesson clearly",
            "The chef cooked the meal perfectly",
            "The student read the book quietly",
            "The writer wrote the novel slowly",
            "The pilot flew the plane smoothly",
            "The farmer grew the crops diligently",
            "The judge decided the case fairly",
            "The nurse helped the patient kindly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "student", "writer", "pilot", "farmer",
            "judge", "nurse",
        ],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday",
            "She visited the doctor recently",
            "He admired the artist greatly",
            "We honored the soldier today",
            "You thanked the teacher warmly",
            "The customer tipped the chef generously",
            "I praised the student loudly",
            "The editor praised the writer highly",
            "He admired the pilot greatly",
            "We thanked the farmer sincerely",
            "They appointed the judge recently",
            "The hospital thanked the nurse warmly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "student", "writer", "pilot", "farmer",
            "judge", "nurse",
        ],
    },
    "amod": {
        "sentences": [
            "The brave king fought hard",
            "The kind doctor helped many",
            "The creative artist worked well",
            "The strong soldier marched far",
            "The wise teacher explained clearly",
            "The skilled chef cooked perfectly",
            "The bright student read carefully",
            "The thoughtful writer reflected deeply",
            "The careful pilot landed smoothly",
            "The hardworking farmer harvested early",
            "The fair judge decided wisely",
            "The gentle nurse cared tenderly",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "wise",
            "skilled", "bright", "thoughtful", "careful", "hardworking",
            "fair", "gentle",
        ],
    },
    "advmod": {
        "sentences": [
            "The king ruled extremely wisely",
            "The doctor treated very carefully",
            "The artist painted quite beautifully",
            "The soldier fought incredibly bravely",
            "The teacher explained remarkably clearly",
            "The chef cooked absolutely perfectly",
            "The student read very quietly",
            "The writer wrote extremely slowly",
            "The pilot flew incredibly smoothly",
            "The farmer worked quite diligently",
            "The judge decided very fairly",
            "The nurse helped remarkably kindly",
        ],
        "target_words": [
            "extremely", "very", "quite", "incredibly", "remarkably",
            "absolutely", "very", "extremely", "incredibly", "quite",
            "very", "remarkably",
        ],
    },
    "det": {
        "sentences": [
            "This king ruled the kingdom wisely",
            "That doctor treated the patient carefully",
            "Every artist painted the portrait beautifully",
            "Each soldier defended the castle bravely",
            "This teacher explained the lesson clearly",
            "That chef cooked the meal perfectly",
            "Every student read the book quietly",
            "Each writer wrote the novel slowly",
            "This pilot flew the plane smoothly",
            "That farmer grew the crops diligently",
            "Every judge decided the case fairly",
            "Each nurse helped the patient kindly",
        ],
        "target_words": [
            "This", "That", "Every", "Each", "This",
            "That", "Every", "Each", "This", "That",
            "Every", "Each",
        ],
    },
    "prep": {
        "sentences": [
            "The crown of the king glittered",
            "The office of the doctor opened",
            "The studio of the artist looked beautiful",
            "The uniform of the soldier was clean",
            "The book of the teacher sold quickly",
            "The restaurant of the chef opened today",
            "The essay of the student read well",
            "The pen of the writer wrote smoothly",
            "The license of the pilot was renewed",
            "The land of the farmer was fertile",
            "The decision of the judge was fair",
            "The care of the nurse was gentle",
        ],
        "target_words": [
            "of", "of", "of", "of", "of",
            "of", "of", "of", "of", "of",
            "of", "of",
        ],
    },
    "iobj": {
        "sentences": [
            "They gave the king a crown",
            "She handed the doctor a report",
            "He offered the artist a commission",
            "We sent the soldier a letter",
            "You told the teacher a story",
            "The customer gave the chef a tip",
            "I lent the student a book",
            "The publisher offered the writer a contract",
            "They assigned the pilot a route",
            "We gave the farmer a loan",
            "They handed the judge a verdict",
            "She gave the nurse a flower",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "student", "writer", "pilot", "farmer",
            "judge", "nurse",
        ],
    },
    "poss": {
        "sentences": [
            "The king's crown glittered brightly",
            "The doctor's office opened early",
            "The artist's studio looked beautiful",
            "The soldier's uniform was clean",
            "The teacher's book sold quickly",
            "The chef's restaurant opened today",
            "The student's essay read well",
            "The writer's pen wrote smoothly",
            "The pilot's license was renewed",
            "The farmer's land was fertile",
            "The judge's decision was fair",
            "The nurse's care was gentle",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "student", "writer", "pilot", "farmer",
            "judge", "nurse",
        ],
    },
}


def find_token_index(tokens, word):
    word_lower = word.lower()
    word_start = word_lower[:3]
    for i, tok in enumerate(tokens):
        tok_lower = tok.lower().strip()
        if word_lower in tok_lower or tok_lower.startswith(word_start):
            return i
    for i, tok in enumerate(tokens):
        if word_lower[:2] in tok.lower():
            return i
    if len(word) > 0:
        first_char = word[0]
        for i, tok in enumerate(tokens):
            if first_char in tok:
                return i
    return None


def collect_hs_at_layer(model, tokenizer, device, sentences, target_words, layer_idx):
    """在指定层收集target token的hidden states"""
    layers = get_layers(model)
    if layer_idx >= len(layers):
        layer_idx = len(layers) - 1
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    target_layer = layers[layer_idx]

    all_h = []
    valid_words = []

    for sent, target_word in zip(sentences, target_words):
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
        valid_words.append(target_word)

    return np.array(all_h) if all_h else None


def collect_all_data(model, tokenizer, device, layers_to_test):
    """收集所有角色在所有层的hidden states"""
    all_data = {}
    role_names = list(EN_ROLES_DATA_8.keys())
    print(f"  收集8个角色的数据: {role_names}")
    
    for layer_idx in layers_to_test:
        layer_data = {}
        for role in role_names:
            role_info = EN_ROLES_DATA_8[role]
            hs = collect_hs_at_layer(
                model, tokenizer, device,
                role_info["sentences"],
                role_info["target_words"],
                layer_idx
            )
            if hs is not None and len(hs) >= 3:
                layer_data[role] = hs
        all_data[layer_idx] = layer_data
        
        n_roles = len(layer_data)
        n_samples = sum(len(v) for v in layer_data.values())
        print(f"    Layer {layer_idx}: {n_roles} roles, {n_samples} samples")
    
    return all_data, role_names


# ==================== Exp1: 8个角色的纤维维度 ====================

def exp1_fiber_dimension(model_name, model, tokenizer, device, layers_to_test):
    """Exp1: 计算8个角色平均偏移δ̄_r的秩"""
    print(f"\n{'='*60}")
    print(f"Exp1: 8个角色的纤维维度 (rank{{δ̄_r}})")
    print(f"{'='*60}")
    
    all_data, role_names = collect_all_data(model, tokenizer, device, layers_to_test)
    
    results = {"model": model_name, "exp": 1, "layers": {}}
    
    for layer_idx in sorted(all_data.keys()):
        layer_data = all_data[layer_idx]
        if len(layer_data) < 4:
            print(f"  Layer {layer_idx}: too few roles ({len(layer_data)}), skipping")
            continue
        
        # 计算每个角色的中心
        role_centers = {}
        for role, hs in layer_data.items():
            role_centers[role] = np.mean(hs, axis=0)
        
        # 计算grand mean
        all_centers = np.array([role_centers[r] for r in layer_data.keys()])
        grand_mean = np.mean(all_centers, axis=0)
        
        # δ̄_r = center_r - grand_mean
        deltas = {}
        for role in layer_data.keys():
            deltas[role] = role_centers[role] - grand_mean
        
        delta_matrix = np.array([deltas[r] for r in layer_data.keys()])
        n_roles = delta_matrix.shape[0]
        
        # PCA分析
        pca = PCA()
        pca.fit(delta_matrix)
        
        # 累计方差解释比
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        
        # 找有效维度(解释95%方差的维度数)
        eff_dim_95 = np.searchsorted(cumvar, 0.95) + 1
        eff_dim_90 = np.searchsorted(cumvar, 0.90) + 1
        eff_dim_80 = np.searchsorted(cumvar, 0.80) + 1
        
        # Participation ratio (更稳健的维度估计)
        eigenvalues = pca.explained_variance_ratio_
        pr = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)
        
        # ★★★ 随机baseline: 随机方向偏移的秩
        n_random = 100
        random_ranks = []
        for _ in range(n_random):
            random_deltas = np.random.randn(n_roles, delta_matrix.shape[1])
            random_deltas -= random_deltas.mean(axis=0)
            rpca = PCA()
            rpca.fit(random_deltas)
            rc = np.cumsum(rpca.explained_variance_ratio_)
            random_ranks.append(np.searchsorted(rc, 0.95) + 1)
        
        # ★★★ Two-NN intrinsic dimension估计
        # 合并所有角色数据
        all_hs = np.vstack([hs for hs in layer_data.values()])
        two_nn_dim = compute_two_nn_id(all_hs)
        
        layer_result = {
            "n_roles": n_roles,
            "delta_shape": list(delta_matrix.shape),
            "pca_eigenvalues_pct": [round(float(v*100), 2) for v in pca.explained_variance_ratio_[:min(8, n_roles)]],
            "cumvar_pct": [round(float(v*100), 1) for v in cumvar[:min(8, n_roles)]],
            "eff_dim_80": int(eff_dim_80),
            "eff_dim_90": int(eff_dim_90),
            "eff_dim_95": int(eff_dim_95),
            "participation_ratio": round(float(pr), 2),
            "two_nn_dim": round(float(two_nn_dim), 2) if two_nn_dim else None,
            "random_baseline_eff_dim_95_mean": round(float(np.mean(random_ranks)), 2),
            "random_baseline_eff_dim_95_std": round(float(np.std(random_ranks)), 2),
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"  Layer {layer_idx}: n_roles={n_roles}")
        print(f"    PCA eigenvalues: {[f'{v:.1f}%' for v in pca.explained_variance_ratio_[:6]*100]}")
        print(f"    Cumvar: {[f'{v:.1f}%' for v in cumvar[:6]*100]}")
        print(f"    Eff dim(80/90/95%): {eff_dim_80}/{eff_dim_90}/{eff_dim_95}")
        print(f"    Participation ratio: {pr:.2f}")
        print(f"    Two-NN dim: {two_nn_dim:.2f}" if two_nn_dim else "    Two-NN dim: N/A")
        print(f"    Random baseline dim(95%): {np.mean(random_ranks):.1f} ± {np.std(random_ranks):.1f}")
    
    # ★★★ 关键分析: 纤维维度是否随角色数变化?
    # 对比3角色 vs 8角色的维度
    roles_3 = ["nsubj", "dobj", "amod"]
    roles_subset = ["nsubj", "dobj", "amod", "advmod", "det", "prep"]
    
    results["comparison_3v8"] = {}
    for layer_idx in sorted(all_data.keys()):
        layer_data = all_data[layer_idx]
        
        # 3角色维度
        if all(r in layer_data for r in roles_3):
            centers_3 = np.array([np.mean(layer_data[r], axis=0) for r in roles_3])
            gm3 = centers_3.mean(axis=0)
            deltas_3 = centers_3 - gm3
            pca3 = PCA()
            pca3.fit(deltas_3)
            dim_3 = np.searchsorted(np.cumsum(pca3.explained_variance_ratio_), 0.95) + 1
            pr_3 = (np.sum(pca3.explained_variance_ratio_))**2 / np.sum(pca3.explained_variance_ratio_**2)
        else:
            dim_3 = None
            pr_3 = None
        
        # 6角色维度
        avail_6 = [r for r in roles_subset if r in layer_data]
        if len(avail_6) >= 5:
            centers_6 = np.array([np.mean(layer_data[r], axis=0) for r in avail_6])
            gm6 = centers_6.mean(axis=0)
            deltas_6 = centers_6 - gm6
            pca6 = PCA()
            pca6.fit(deltas_6)
            dim_6 = np.searchsorted(np.cumsum(pca6.explained_variance_ratio_), 0.95) + 1
            pr_6 = (np.sum(pca6.explained_variance_ratio_))**2 / np.sum(pca6.explained_variance_ratio_**2)
        else:
            dim_6 = None
            pr_6 = None
        
        # 8角色维度
        n8 = len(layer_data)
        if n8 >= 6:
            centers_8 = np.array([np.mean(layer_data[r], axis=0) for r in layer_data.keys()])
            gm8 = centers_8.mean(axis=0)
            deltas_8 = centers_8 - gm8
            pca8 = PCA()
            pca8.fit(deltas_8)
            dim_8 = np.searchsorted(np.cumsum(pca8.explained_variance_ratio_), 0.95) + 1
            pr_8 = (np.sum(pca8.explained_variance_ratio_))**2 / np.sum(pca8.explained_variance_ratio_**2)
        else:
            dim_8 = None
            pr_8 = None
        
        results["comparison_3v8"][str(layer_idx)] = {
            "dim_3roles_95": dim_3,
            "pr_3roles": round(float(pr_3), 2) if pr_3 else None,
            "dim_6roles_95": dim_6,
            "pr_6roles": round(float(pr_6), 2) if pr_6 else None,
            "dim_8roles_95": dim_8,
            "pr_8roles": round(float(pr_8), 2) if pr_8 else None,
        }
        
        print(f"  ★ Layer {layer_idx} 纤维维度对比: 3角色={dim_3}, 6角色={dim_6}, 8角色={dim_8}")
    
    return results


def compute_two_nn_id(X, frac=0.9):
    """Two-NN intrinsic dimension估计
    Facco et al. (2017) "Estimating the intrinsic dimension of datasets by a minimal neighborhood information"
    """
    if len(X) < 10:
        return None
    
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # 排除零距离
    valid = (r1 > 1e-10) & (r2 > 1e-10) & (r2 > r1)
    if valid.sum() < 5:
        return None
    
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0]
    
    if len(mu) < 5:
        return None
    
    # MLE: d = 1 / <log(mu)>
    d_hat = len(mu) / np.sum(np.log(mu))
    
    return max(1.0, d_hat)


# ==================== Exp2: Δ = A + B 分解 ====================

def exp2_delta_decomposition(model_name, model, tokenizer, device, layers_to_test):
    """Exp2: Δ_ℓ(r,x) = A_ℓ(r) + B_ℓ(r,x) 的分解"""
    print(f"\n{'='*60}")
    print(f"Exp2: Δ = A + B 分解 (角色平均纤维 vs 词依赖修正)")
    print(f"{'='*60}")
    
    all_data, role_names = collect_all_data(model, tokenizer, device, layers_to_test)
    
    results = {"model": model_name, "exp": 2, "layers": {}}
    
    for layer_idx in sorted(all_data.keys()):
        layer_data = all_data[layer_idx]
        if len(layer_data) < 3:
            continue
        
        # 计算每个角色的中心
        role_centers = {}
        for role, hs in layer_data.items():
            role_centers[role] = np.mean(hs, axis=0)
        
        grand_mean = np.mean([role_centers[r] for r in layer_data.keys()], axis=0)
        
        # A_ℓ(r) = δ̄_r = center_r - grand_mean
        A = {}
        for role in layer_data.keys():
            A[role] = role_centers[role] - grand_mean
        
        # Δ_ℓ(r,x) = h(x,r) - grand_mean
        # B_ℓ(r,x) = Δ - A = h(x,r) - center_r (即角色内残差)
        A_norms = []
        B_norms = []
        Delta_norms = []
        
        B_all = []  # 所有B向量
        A_all = []  # 所有A向量(重复)
        
        for role in layer_data.keys():
            for h_vec in layer_data[role]:
                delta = h_vec - grand_mean
                b = h_vec - role_centers[role]  # B = Δ - A
                
                A_norms.append(np.linalg.norm(A[role]))
                B_norms.append(np.linalg.norm(b))
                Delta_norms.append(np.linalg.norm(delta))
                
                B_all.append(b)
                A_all.append(A[role])
        
        A_norms = np.array(A_norms)
        B_norms = np.array(B_norms)
        Delta_norms = np.array(Delta_norms)
        
        # ★★★ ||B|| / ||Δ|| 的比值
        ratio_B_Delta = np.mean(B_norms) / np.mean(Delta_norms) if np.mean(Delta_norms) > 0 else 0
        ratio_B_A = np.mean(B_norms) / np.mean(A_norms) if np.mean(A_norms) > 0 else float('inf')
        
        # B的有效维度
        B_matrix = np.array(B_all)
        if len(B_matrix) > 5:
            bpca = PCA()
            bpca.fit(B_matrix)
            B_cumvar = np.cumsum(bpca.explained_variance_ratio_)
            B_eff_dim_80 = int(np.searchsorted(B_cumvar, 0.80) + 1)
            B_eff_dim_90 = int(np.searchsorted(B_cumvar, 0.90) + 1)
            B_eff_dim_95 = int(np.searchsorted(B_cumvar, 0.95) + 1)
        else:
            B_eff_dim_80 = B_eff_dim_90 = B_eff_dim_95 = None
            B_cumvar = None
        
        # A的有效维度(同Exp1, 但这里重算)
        A_matrix = np.array([A[r] for r in layer_data.keys()])
        apca = PCA()
        apca.fit(A_matrix)
        A_cumvar = np.cumsum(apca.explained_variance_ratio_)
        A_eff_dim_95 = int(np.searchsorted(A_cumvar, 0.95) + 1)
        
        # ★★★ ANOVA风格分解: 总方差 = A方差 + B方差
        # 总方差(相对grand_mean)
        total_var = np.mean(Delta_norms**2)
        A_var = np.mean(A_norms**2)
        B_var = np.mean(B_norms**2)
        
        # 方差解释比
        A_var_pct = A_var / total_var * 100 if total_var > 0 else 0
        B_var_pct = B_var / total_var * 100 if total_var > 0 else 0
        
        # ★★★ 逐角色的||B||/||A||
        role_BA_ratios = {}
        for role in layer_data.keys():
            role_B_norms = [np.linalg.norm(h - role_centers[role]) for h in layer_data[role]]
            A_norm = np.linalg.norm(A[role])
            role_BA_ratios[role] = round(float(np.mean(role_B_norms) / A_norm), 3) if A_norm > 0 else float('inf')
        
        layer_result = {
            "n_roles": len(layer_data),
            "mean_A_norm": round(float(np.mean(A_norms)), 4),
            "mean_B_norm": round(float(np.mean(B_norms)), 4),
            "mean_Delta_norm": round(float(np.mean(Delta_norms)), 4),
            "ratio_B_Delta": round(float(ratio_B_Delta), 4),
            "ratio_B_A": round(float(ratio_B_A), 4),
            "total_var": round(float(total_var), 4),
            "A_var": round(float(A_var), 4),
            "B_var": round(float(B_var), 4),
            "A_var_pct": round(float(A_var_pct), 2),
            "B_var_pct": round(float(B_var_pct), 2),
            "A_eff_dim_95": A_eff_dim_95,
            "B_eff_dim_80": B_eff_dim_80,
            "B_eff_dim_90": B_eff_dim_90,
            "B_eff_dim_95": B_eff_dim_95,
            "B_cumvar_top5": [round(float(v*100), 1) for v in B_cumvar[:5]] if B_cumvar is not None else None,
            "role_BA_ratios": role_BA_ratios,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"  Layer {layer_idx}: n_roles={len(layer_data)}")
        print(f"    ||A||={np.mean(A_norms):.4f}, ||B||={np.mean(B_norms):.4f}, ||Δ||={np.mean(Delta_norms):.4f}")
        print(f"    ||B||/||Δ||={ratio_B_Delta:.4f}, ||B||/||A||={ratio_B_A:.4f}")
        print(f"    A方差={A_var_pct:.1f}%, B方差={B_var_pct:.1f}%")
        print(f"    A维度(95%)={A_eff_dim_95}, B维度(95%)={B_eff_dim_95}")
        print(f"    逐角色||B||/||A||: {role_BA_ratios}")
    
    return results


# ==================== Exp3: 平行运输验证(带bootstrap) ====================

def exp3_parallel_transport(model_name, model, tokenizer, device, layers_to_test):
    """Exp3: δ̄_{ℓ+1}(r) ≈ T_ℓ δ̄_ℓ(r) 的验证"""
    print(f"\n{'='*60}")
    print(f"Exp3: 平行运输验证 (带bootstrap)")
    print(f"{'='*60}")
    
    all_data, role_names = collect_all_data(model, tokenizer, device, layers_to_test)
    
    results = {"model": model_name, "exp": 3, "layers": {}}
    
    sorted_layers = sorted(all_data.keys())
    
    for i in range(len(sorted_layers) - 1):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i + 1]
        
        data1 = all_data[l1]
        data2 = all_data[l2]
        
        # 只保留两层都有的角色
        common_roles = [r for r in data1.keys() if r in data2]
        if len(common_roles) < 3:
            continue
        
        # δ̄_r for each layer
        gm1 = np.mean([np.mean(data1[r], axis=0) for r in common_roles], axis=0)
        gm2 = np.mean([np.mean(data2[r], axis=0) for r in common_roles], axis=0)
        
        deltas1 = np.array([np.mean(data1[r], axis=0) - gm1 for r in common_roles])
        deltas2 = np.array([np.mean(data2[r], axis=0) - gm2 for r in common_roles])
        
        # Procrustes alignment: 找T使T @ deltas1 ≈ deltas2
        # 标准Procrustes需要n_features一致
        if deltas1.shape != deltas2.shape:
            continue
        
        # scipy Procrustes (返回旋转后的矩阵)
        try:
            _, deltas2_aligned, _ = procrustes(deltas1, deltas2)
        except Exception:
            continue
        
        # 拟合优度: 旋转后的误差
        residual = deltas2 - deltas2_aligned
        error_norm = np.linalg.norm(residual)
        original_norm = np.linalg.norm(deltas2)
        fit_quality = 1.0 - error_norm / original_norm if original_norm > 0 else 0
        
        # ★★★ Bootstrap置信区间
        n_bootstrap = 200
        bootstrap_fits = []
        for _ in range(n_bootstrap):
            # 重采样角色(有放回)
            idx = np.random.choice(len(common_roles), size=len(common_roles), replace=True)
            boot_d1 = deltas1[idx]
            boot_d2 = deltas2[idx]
            
            try:
                _, boot_d2_aligned, _ = procrustes(boot_d1, boot_d2)
                boot_residual = boot_d2 - boot_d2_aligned
                boot_error = np.linalg.norm(boot_residual)
                boot_original = np.linalg.norm(boot_d2)
                boot_fit = 1.0 - boot_error / boot_original if boot_original > 0 else 0
                bootstrap_fits.append(boot_fit)
            except Exception:
                continue
        
        if len(bootstrap_fits) > 10:
            ci_low = np.percentile(bootstrap_fits, 2.5)
            ci_high = np.percentile(bootstrap_fits, 97.5)
            boot_mean = np.mean(bootstrap_fits)
            boot_std = np.std(bootstrap_fits)
        else:
            ci_low = ci_high = boot_mean = boot_std = None
        
        # ★★★ Permutation test
        n_perm = 200
        perm_fits = []
        for _ in range(n_perm):
            # 打乱角色标签
            perm_idx = np.random.permutation(len(common_roles))
            perm_d2 = deltas2[perm_idx]
            
            try:
                _, perm_d2_aligned, _ = procrustes(deltas1, perm_d2)
                perm_residual = perm_d2 - perm_d2_aligned
                perm_error = np.linalg.norm(perm_residual)
                perm_original = np.linalg.norm(perm_d2)
                perm_fit = 1.0 - perm_error / perm_original if perm_original > 0 else 0
                perm_fits.append(perm_fit)
            except Exception:
                continue
        
        if len(perm_fits) > 10:
            p_value = np.mean([f >= fit_quality for f in perm_fits])
            perm_mean = np.mean(perm_fits)
        else:
            p_value = None
            perm_mean = None
        
        # ★★★ 逐角色分析: 每个角色的方向变化角
        angle_changes = {}
        for j, role in enumerate(common_roles):
            d1 = deltas1[j]
            d2 = deltas2[j]
            cos_a = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10)
            cos_a = np.clip(cos_a, -1, 1)
            angle_changes[role] = round(float(np.degrees(np.arccos(cos_a))), 1)
        
        layer_key = f"L{l1}_L{l2}"
        layer_result = {
            "layer_from": l1,
            "layer_to": l2,
            "n_common_roles": len(common_roles),
            "fit_quality": round(float(fit_quality), 4),
            "bootstrap_mean": round(float(boot_mean), 4) if boot_mean is not None else None,
            "bootstrap_std": round(float(boot_std), 4) if boot_std is not None else None,
            "bootstrap_ci_low_95": round(float(ci_low), 4) if ci_low is not None else None,
            "bootstrap_ci_high_95": round(float(ci_high), 4) if ci_high is not None else None,
            "permutation_mean": round(float(perm_mean), 4) if perm_mean is not None else None,
            "permutation_p_value": round(float(p_value), 4) if p_value is not None else None,
            "angle_changes": angle_changes,
        }
        
        results["layers"][layer_key] = layer_result
        
        print(f"  {layer_key}: n_roles={len(common_roles)}")
        print(f"    Procrustes fit: {fit_quality:.4f}")
        print(f"    Bootstrap CI(95%): [{ci_low:.4f}, {ci_high:.4f}]" if ci_low is not None else "    Bootstrap: failed")
        print(f"    Permutation p-value: {p_value:.4f}" if p_value is not None else "    Permutation: failed")
        print(f"    Direction changes: {angle_changes}")
    
    # ★★★ 全局纤维不变性指标
    # 角色间夹角比例的跨层稳定性
    if len(sorted_layers) >= 3:
        angle_ratio_series = []
        for layer_idx in sorted_layers:
            layer_data = all_data[layer_idx]
            common_r = [r for r in layer_data.keys() if r in ["nsubj", "dobj", "amod"]]
            if len(common_r) < 3:
                continue
            
            gm = np.mean([np.mean(layer_data[r], axis=0) for r in common_r], axis=0)
            deltas = {r: np.mean(layer_data[r], axis=0) - gm for r in common_r}
            
            # nsubj-dobj夹角
            cos_nd = np.dot(deltas["nsubj"], deltas["dobj"]) / (np.linalg.norm(deltas["nsubj"]) * np.linalg.norm(deltas["dobj"]) + 1e-10)
            angle_nd = np.arccos(np.clip(cos_nd, -1, 1))
            
            # nsubj-amod夹角
            cos_na = np.dot(deltas["nsubj"], deltas["amod"]) / (np.linalg.norm(deltas["nsubj"]) * np.linalg.norm(deltas["amod"]) + 1e-10)
            angle_na = np.arccos(np.clip(cos_na, -1, 1))
            
            if angle_na > 0.01:
                angle_ratio_series.append(angle_nd / angle_na)
        
        if len(angle_ratio_series) >= 3:
            ar_mean = np.mean(angle_ratio_series)
            ar_std = np.std(angle_ratio_series)
            ar_cv = ar_std / ar_mean if ar_mean > 0 else 0
            
            # Bootstrap CI for CV
            n_boot = 500
            boot_cvs = []
            for _ in range(n_boot):
                boot_sample = np.random.choice(angle_ratio_series, size=len(angle_ratio_series), replace=True)
                bm = np.mean(boot_sample)
                bs = np.std(boot_sample)
                if bm > 0:
                    boot_cvs.append(bs / bm)
            
            if boot_cvs:
                cv_ci_low = np.percentile(boot_cvs, 2.5)
                cv_ci_high = np.percentile(boot_cvs, 97.5)
            else:
                cv_ci_low = cv_ci_high = None
            
            results["global_invariance"] = {
                "angle_ratio_mean": round(float(ar_mean), 4),
                "angle_ratio_std": round(float(ar_std), 4),
                "angle_ratio_cv": round(float(ar_cv), 4),
                "cv_bootstrap_ci_low_95": round(float(cv_ci_low), 4) if cv_ci_low is not None else None,
                "cv_bootstrap_ci_high_95": round(float(cv_ci_high), 4) if cv_ci_high is not None else None,
                "interpretation": "存在跨层稳定迹象" if ar_cv < 0.2 else "不稳定",
                "note": "CV<0.2是经验阈值, 需要更多统计检验",
            }
            
            print(f"\n  ★ 全局纤维不变性:")
            print(f"    角度比 mean={ar_mean:.4f}, std={ar_std:.4f}, CV={ar_cv:.4f}")
            print(f"    CV bootstrap CI(95%): [{cv_ci_low:.4f}, {cv_ci_high:.4f}]" if cv_ci_low else "    CV bootstrap: failed")
    
    return results


# ==================== Exp4: M_ℓ(x) 定义与验证 ====================

def exp4_manifold_definition(model_name, model, tokenizer, device, layers_to_test):
    """Exp4: M_ℓ(x)的定义——共享语义流形坐标"""
    print(f"\n{'='*60}")
    print(f"Exp4: M_ℓ(x)的定义 (共享语义流形坐标)")
    print(f"{'='*60}")
    
    all_data, role_names = collect_all_data(model, tokenizer, device, layers_to_test)
    
    results = {"model": model_name, "exp": 4, "layers": {}}
    
    for layer_idx in sorted(all_data.keys()):
        layer_data = all_data[layer_idx]
        if len(layer_data) < 3:
            continue
        
        # 合并所有角色数据
        all_hs = []
        all_labels = []  # (role, word_idx)
        for role, hs in layer_data.items():
            for j, h in enumerate(hs):
                all_hs.append(h)
                all_labels.append((role, j))
        
        all_hs = np.array(all_hs)
        
        # 方法1: 共同PCA, 前k维作为M_ℓ(x)
        pca = PCA()
        pca.fit(all_hs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        
        # 不同k值解释的方差
        k_values = [1, 2, 3, 5, 10, 20]
        k_var = {}
        for k in k_values:
            if k <= len(cumvar):
                k_var[str(k)] = round(float(cumvar[k-1] * 100), 2)
        
        # 方法2: 残差 = h - M 的方差分解
        # M = PCA前k维投影
        for k in [3, 5, 10]:
            if k > pca.n_components_:
                continue
            
            # 投影到前k维
            pca_k = PCA(n_components=k)
            M = pca_k.fit_transform(all_hs)
            M_reconstructed = pca_k.inverse_transform(M)
            residuals = all_hs - M_reconstructed
            
            # 残差的方差分解: 角色间 vs 角色内
            role_centers_residual = {}
            for j, (role, _) in enumerate(all_labels):
                if role not in role_centers_residual:
                    role_centers_residual[role] = []
                role_centers_residual[role].append(residuals[j])
            
            # 角色间方差
            all_role_center_residuals = np.array([np.mean(v, axis=0) for v in role_centers_residual.values()])
            between_var = np.mean(np.sum(all_role_center_residuals**2, axis=1))
            
            # 角色内方差
            within_var = 0
            n_within = 0
            for role, res_list in role_centers_residual.items():
                center = np.mean(res_list, axis=0)
                for r in res_list:
                    within_var += np.sum((r - center)**2)
                    n_within += 1
            within_var /= n_within if n_within > 0 else 1
            
            total_residual_var = np.mean(np.sum(residuals**2, axis=1))
            between_pct = between_var / total_residual_var * 100 if total_residual_var > 0 else 0
            within_pct = within_var / total_residual_var * 100 if total_residual_var > 0 else 0
            
            layer_result_key = f"k{k}"
            if str(layer_idx) not in results["layers"]:
                results["layers"][str(layer_idx)] = {}
            results["layers"][str(layer_idx)][layer_result_key] = {
                "total_residual_var": round(float(total_residual_var), 6),
                "between_role_var": round(float(between_var), 6),
                "within_role_var": round(float(within_var), 6),
                "between_pct": round(float(between_pct), 2),
                "within_pct": round(float(within_pct), 2),
            }
        
        results["layers"][str(layer_idx)]["k_variance_pct"] = k_var
        
        print(f"  Layer {layer_idx}:")
        print(f"    共同PCA k方差: {k_var}")
        for k in [3, 5, 10]:
            kk = f"k{k}"
            if kk in results["layers"][str(layer_idx)]:
                rr = results["layers"][str(layer_idx)][kk]
                print(f"    残差(k={k}): 角色间={rr['between_pct']:.1f}%, 角色内={rr['within_pct']:.1f}%")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Phase 23: 纤维维度的真正验证")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4],
                       help="实验编号(1-4)")
    args = parser.parse_args()
    
    model_name = args.model
    exp_num = args.exp
    
    # 加载模型 (DS7B需要8bit量化以适应12GB显存)
    if model_name == "deepseek7b":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cfg = MODEL_CONFIGS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
        model.eval()
        device = next(model.parameters()).device
        print(f"[ccmc] {model_name} loaded with 8bit, device={device}")
    else:
        model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    # 选择测试层(均匀采样7层)
    layers_to_test = [0]
    if n_layers > 6:
        step = (n_layers - 1) // 6
        layers_to_test = list(range(0, n_layers, step))[:7]
    if n_layers - 1 not in layers_to_test:
        layers_to_test.append(n_layers - 1)
    layers_to_test = sorted(set(layers_to_test))
    
    print(f"\nModel: {model_name}, Layers: {n_layers}, Test layers: {layers_to_test}")
    
    # 运行实验
    if exp_num == 1:
        results = exp1_fiber_dimension(model_name, model, tokenizer, device, layers_to_test)
    elif exp_num == 2:
        results = exp2_delta_decomposition(model_name, model, tokenizer, device, layers_to_test)
    elif exp_num == 3:
        results = exp3_parallel_transport(model_name, model, tokenizer, device, layers_to_test)
    elif exp_num == 4:
        results = exp4_manifold_definition(model_name, model, tokenizer, device, layers_to_test)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5_temp")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"ccmc_exp{exp_num}_{model_name}_results.json")
    
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        return obj
    
    results = convert_numpy(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 释放模型
    release_model(model)
    print("模型已释放")


if __name__ == "__main__":
    main()
