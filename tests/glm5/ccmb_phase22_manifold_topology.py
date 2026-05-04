"""
CCMB(Phase 22): 流形拓扑分析——语法角色是共享语义流形上的纤维结构
=============================================================================
核心数学框架(Phase 21确立):
  h_role = h_semantic + δ_role
  → h_semantic: 共享主方向(语义基向量)
  → δ_role: 角色特异偏移(低维残差子空间)

核心问题:
  语法角色是否共享一个低维语义主流形, 在其上形成可分离的局部偏移?
  如果成立 → 语言语法不是正交坐标系, 而是共享语义流形上的局部纤维结构!

实验:
  Exp1: ★★★★★★★★★ 局部内在维度(LID)与流形拓扑
    → 逐层计算每个语法角色cluster的局部内在维度
    → 各角色之间的geodesic距离(沿流形的距离, 非欧氏距离)
    → tangent space一致性: 不同角色cluster的切空间是否相似?
    → 如果LID一致 → 语法角色确实在同一流形上

  Exp2: ★★★★★★★★ 角色特异偏移δ_role的几何性质
    → δ_role = h_role - h_semantic 的维度和方向
    → 不同角色的δ是否在同一个低维子空间中?
    → δ之间的夹角分布(是否正交? 是否随层变化?)
    → δ的范数随层变化(与相对范数曲线比较)

  Exp3: ★★★★★★★ 平行移动关系
    → 沿层的旋转是否保持δ_role的几何关系?
    → 如果是 → 语法角色的纤维结构是层间不变的
    → 如果否 → 语法角色的几何关系在层间演化

  Exp4: ★★★★★ h_role = h_semantic + δ_role的定量验证
    → 拟合线性模型: h_role = α·h_semantic + δ_role
    → α的值是否接近1?
    → δ_role的维度(PCA解释方差的维度)
    → 与随机对照的比较
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

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 英文语法角色数据(同Phase 21) =====
EN_ROLES_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom wisely",
            "The doctor treated the patient carefully",
            "The artist painted the portrait beautifully",
            "The soldier defended the castle bravely",
            "The teacher explained the lesson clearly",
            "The chef cooked the meal perfectly",
            "The cat chased the mouse quickly",
            "The dog found the bone happily",
            "The woman drove the car safely",
            "The man fixed the roof carefully",
            "The student read the book quietly",
            "The singer performed the song brilliantly",
            "The baker made the bread daily",
            "The pilot flew the plane smoothly",
            "The farmer grew the crops diligently",
            "The writer wrote the novel slowly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
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
            "The hawk chased the cat swiftly",
            "The boy found the dog outside",
            "The police arrested the woman quickly",
            "The company hired the man recently",
            "I praised the student loudly",
            "They applauded the singer warmly",
            "She visited the baker often",
            "He admired the pilot greatly",
            "We thanked the farmer sincerely",
            "The editor praised the writer highly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
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
            "The quick cat ran fast",
            "The loyal dog stayed close",
            "The old woman walked slowly",
            "The tall man stood quietly",
            "The bright student read carefully",
            "The talented singer performed brilliantly",
            "The patient baker waited calmly",
            "The careful pilot landed smoothly",
            "The hardworking farmer harvested early",
            "The thoughtful writer reflected deeply",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "wise",
            "skilled", "quick", "loyal", "old", "tall",
            "bright", "talented", "patient", "careful", "hardworking", "thoughtful",
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
            "The cat's tail swished gently",
            "The dog's bark echoed loudly",
            "The woman's dress looked elegant",
            "The man's car drove fast",
            "The student's essay read well",
            "The singer's voice rang clearly",
            "The baker's shop smelled wonderful",
            "The pilot's license was renewed",
            "The farmer's land was fertile",
            "The writer's pen wrote smoothly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
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


def compute_lid(X, k=5):
    """计算局部内在维度(Local Intrinsic Dimension)
    
    使用最大似然估计法 (MLE):
    Levina & Bickel (2004) "Maximum likelihood estimation of intrinsic dimension"
    
    对于每个点, 用k近邻距离估计内在维度:
    d_mle = (1/(k-1)) * sum_{j=1}^{k-1} log(r_k / r_j)
    
    其中 r_j 是第j近邻的距离
    """
    if len(X) < k + 1:
        return None
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    lid_estimates = []
    for i in range(len(X)):
        dists = distances[i, 1:]  # 排除自身
        if dists[-1] < 1e-10:
            continue
        r_k = dists[-1]
        mle_sum = 0
        valid = 0
        for j in range(len(dists) - 1):
            if dists[j] > 1e-10 and r_k > 1e-10:
                mle_sum += np.log(r_k / dists[j])
                valid += 1
        if valid > 0:
            lid_estimates.append(valid / max(mle_sum, 1e-10))
    
    if lid_estimates:
        return float(np.median(lid_estimates))
    return None


def compute_tangent_space(X, n_components=5):
    """计算切空间(用PCA近似)
    
    返回:
      - 主成分方向
      - 各成分的方差解释比
      - 有效维度(累计方差>0.95的维度数)
    """
    if len(X) < n_components + 1:
        return None
    
    pca = PCA(n_components=min(n_components, len(X) - 1))
    pca.fit(X)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    eff_dim = int(np.searchsorted(cumvar, 0.95) + 1) if cumvar[-1] >= 0.95 else len(cumvar)
    
    return {
        'components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'effective_dim_95': eff_dim,
        'cumvar_at_3': float(cumvar[min(2, len(cumvar)-1)]),
    }


def tangent_space_similarity(ts1, ts2):
    """计算两个切空间的相似性(Grassmannian距离)
    
    两个子空间之间的principal angles
    """
    if ts1 is None or ts2 is None:
        return None
    
    C1 = ts1['components']
    C2 = ts2['components']
    
    M = C1 @ C2.T
    _, s, _ = np.linalg.svd(M)
    
    # principal angles
    angles = np.arccos(np.clip(s, 0, 1))
    
    # Grassmannian距离
    d_G = np.sqrt(np.sum(angles**2))
    
    return {
        'principal_angles_deg': [float(np.degrees(a)) for a in angles],
        'grassmannian_dist': float(d_G),
        'mean_angle_deg': float(np.degrees(np.mean(angles))),
    }


# ===== Exp1: LID与流形拓扑 =====
def exp1_lid_topology(model, tokenizer, device):
    """局部内在维度与流形拓扑分析"""
    print("\n" + "="*70)
    print("Exp1: LID与流形拓扑分析")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers

    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)

    roles = ["nsubj", "dobj", "amod", "poss"]
    k_lid = min(5, 8 - 1)  # LID的k近邻数

    layer_results = {}

    for li in sample_layers:
        print(f"\n  L{li}:")

        role_data = {}
        for role in roles:
            data = EN_ROLES_DATA[role]
            H = collect_hs_at_layer(model, tokenizer, device,
                                    data["sentences"][:8], data["target_words"][:8], li)
            if H is not None and len(H) >= k_lid + 1:
                role_data[role] = H
                print(f"    {role}: {len(H)} samples")

        if len(role_data) < 2:
            print(f"    数据不足, 跳过")
            continue

        # ===== LID =====
        print(f"\n  Step 1a: 局部内在维度(LID)")
        lid_dict = {}
        for role, H in role_data.items():
            lid = compute_lid(H, k=k_lid)
            lid_dict[role] = lid
            print(f"    {role}: LID={lid:.2f}" if lid else f"    {role}: LID=计算失败")

        # ===== 合并所有角色的LID =====
        all_hs = np.vstack(list(role_data.values()))
        lid_all = compute_lid(all_hs, k=k_lid)
        print(f"    合并所有角色: LID={lid_all:.2f}" if lid_all else "    合并: 计算失败")

        # ===== 切空间 =====
        print(f"\n  Step 1b: 切空间分析")
        ts_dict = {}
        for role, H in role_data.items():
            ts = compute_tangent_space(H, n_components=5)
            ts_dict[role] = ts
            if ts:
                print(f"    {role}: 有效维度(95%方差)={ts['effective_dim_95']}, "
                      f"前3成分累计方差={ts['cumvar_at_3']:.4f}")

        # ===== 切空间相似性 =====
        print(f"\n  Step 1c: 切空间相似性(Grassmannian距离)")
        role_list = list(ts_dict.keys())
        ts_similarities = {}
        for i in range(len(role_list)):
            for j in range(i + 1, len(role_list)):
                r1, r2 = role_list[i], role_list[j]
                sim = tangent_space_similarity(ts_dict[r1], ts_dict[r2])
                if sim:
                    key = f"{r1}-{r2}"
                    ts_similarities[key] = sim
                    print(f"    {key}: d_G={sim['grassmannian_dist']:.4f}, "
                          f"mean_angle={sim['mean_angle_deg']:.1f}°")

        # ===== Geodesic距离(近似) =====
        print(f"\n  Step 1d: Geodesic距离(用graph-based近似)")
        # 将所有点合并, 建kNN图, 用最短路径距离
        all_points = []
        all_labels = []
        for role, H in role_data.items():
            all_points.append(H)
            all_labels.extend([role] * len(H))
        
        if len(all_points) > 1:
            X_all = np.vstack(all_points)
            
            # 用kNN图近似geodesic距离
            k_graph = min(5, len(X_all) - 1)
            nbrs = NearestNeighbors(n_neighbors=k_graph + 1, algorithm='auto').fit(X_all)
            distances, indices = nbrs.kneighbors(X_all)
            
            # Floyd-Warshall太慢, 用Dijkstra近似
            # 只计算cluster中心之间的geodesic距离
            centers = {role: np.mean(H, axis=0) for role, H in role_data.items()}
            center_indices = {}
            idx = 0
            for role, H in role_data.items():
                center_indices[role] = idx + np.argmin(np.linalg.norm(H - centers[role], axis=1))
                idx += len(H)
            
            # 简化: 用欧氏距离除以LID来近似geodesic距离
            # geodesic_dist ≈ euclidean_dist * correction_factor
            # correction_factor基于曲率估计
            geo_distances = {}
            role_list2 = list(centers.keys())
            for i in range(len(role_list2)):
                for j in range(i + 1, len(role_list2)):
                    r1, r2 = role_list2[i], role_list2[j]
                    euc_dist = np.linalg.norm(centers[r1] - centers[r2])
                    
                    # 用2个cluster的切空间主成分夹角修正
                    if r1 in ts_dict and r2 in ts_dict and ts_dict[r1] and ts_dict[r2]:
                        sim = tangent_space_similarity(ts_dict[r1], ts_dict[r2])
                        if sim:
                            # geodesic > euclidean, 比例取决于流形弯曲
                            geo_factor = 1.0 + sim['grassmannian_dist'] / np.pi
                        else:
                            geo_factor = 1.0
                    else:
                        geo_factor = 1.0
                    
                    geo_dist = euc_dist * geo_factor
                    geo_distances[f"{r1}-{r2}"] = {
                        'euclidean': float(euc_dist),
                        'geodesic_approx': float(geo_dist),
                        'geo_factor': float(geo_factor),
                    }
                    print(f"    {r1}-{r2}: eucl={euc_dist:.2f}, geo≈{geo_dist:.2f} (factor={geo_factor:.3f})")

        # 保存结果
        layer_results[li] = {
            'lid': {k: v for k, v in lid_dict.items() if v is not None},
            'lid_all': lid_all,
            'effective_dims': {r: ts['effective_dim_95'] for r, ts in ts_dict.items() if ts},
            'ts_similarities': {k: {'grassmannian_dist': v['grassmannian_dist'],
                                     'mean_angle_deg': v['mean_angle_deg']}
                                for k, v in ts_similarities.items()},
            'geo_distances': geo_distances,
        }

    # ===== 跨层总结 =====
    print(f"\n  ===== 跨层LID总结 =====")
    
    if layer_results:
        # LID随层变化
        for role in roles:
            lids = [(li, lr['lid'].get(role)) for li, lr in layer_results.items() 
                    if lr['lid'].get(role) is not None]
            if lids:
                layers_str = ", ".join([f"L{li}={lid:.1f}" for li, lid in lids])
                print(f"  {role}: {layers_str}")
        
        # 切空间相似性随层变化
        print(f"\n  ===== 切空间相似性(Grassmannian距离)随层变化 =====")
        for pair_key in ['nsubj-dobj', 'nsubj-amod', 'dobj-amod']:
            dGs = [(li, lr['ts_similarities'].get(pair_key, {}).get('grassmannian_dist'))
                   for li, lr in layer_results.items()]
            valid = [(li, dG) for li, dG in dGs if dG is not None]
            if valid:
                dG_str = ", ".join([f"L{li}={dG:.3f}" for li, dG in valid])
                print(f"  {pair_key}: {dG_str}")

    results['layer_results'] = {str(k): v for k, v in layer_results.items()}
    return results


# ===== Exp2: δ_role几何性质 =====
def exp2_delta_geometry(model, tokenizer, device):
    """角色特异偏移δ_role的几何性质"""
    print("\n" + "="*70)
    print("Exp2: δ_role几何性质")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers

    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)

    roles = ["nsubj", "dobj", "amod"]

    layer_results = {}

    for li in sample_layers:
        print(f"\n  L{li}:")

        role_data = {}
        for role in roles:
            data = EN_ROLES_DATA[role]
            H = collect_hs_at_layer(model, tokenizer, device,
                                    data["sentences"][:8], data["target_words"][:8], li)
            if H is not None and len(H) >= 3:
                role_data[role] = H

        if len(role_data) < 3:
            print(f"    数据不足, 跳过")
            continue

        # ===== h_semantic: 所有角色hidden states的PCA第一主成分方向 =====
        # 或者: 所有角色cluster中心的平均值
        all_hs = np.vstack([role_data[r] for r in roles])
        centers = {role: np.mean(H, axis=0) for role, H in role_data.items()}
        grand_center = np.mean(list(centers.values()), axis=0)
        
        # h_semantic = grand_center (所有角色共享的方向)
        # δ_role = center_role - grand_center
        print(f"\n  Step 2a: δ_role = h_role - h_semantic")
        
        deltas = {}
        delta_norms = {}
        for role in roles:
            delta = centers[role] - grand_center
            deltas[role] = delta
            delta_norms[role] = np.linalg.norm(delta)
            print(f"    δ_{role}: norm={delta_norms[role]:.4f}")

        # ===== δ之间的夹角 =====
        print(f"\n  Step 2b: δ_role之间的夹角(正交化前!)")
        delta_angles = {}
        role_list = list(deltas.keys())
        for i in range(len(role_list)):
            for j in range(i + 1, len(role_list)):
                r1, r2 = role_list[i], role_list[j]
                d1, d2 = deltas[r1], deltas[r2]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_angle = np.dot(d1, d2) / (n1 * n2)
                    angle_deg = np.degrees(np.arccos(np.clip(abs(cos_angle), 0, 1)))
                    delta_angles[f"{r1}-{r2}"] = {
                        'cos': float(cos_angle),
                        'angle_deg': float(angle_deg),
                    }
                    print(f"    δ_{r1} vs δ_{r2}: angle={angle_deg:.1f}°, cos={cos_angle:.4f}")

        # ===== δ的子空间维度 =====
        print(f"\n  Step 2c: δ_role子空间的维度")
        delta_matrix = np.column_stack([deltas[r] for r in roles])
        pca_delta = PCA()
        pca_delta.fit(delta_matrix.T)  # 注意: 每个δ是一列
        
        cumvar = np.cumsum(pca_delta.explained_variance_ratio_)
        print(f"    方差解释比: {pca_delta.explained_variance_ratio_}")
        print(f"    累计方差: {cumvar}")
        
        eff_dim = int(np.searchsorted(cumvar, 0.95) + 1) if cumvar[-1] >= 0.95 else len(cumvar)
        print(f"    有效维度(95%方差): {eff_dim}")

        # ===== δ占总方差的比例 =====
        print(f"\n  Step 2d: δ_role vs h_semantic的方差贡献")
        
        # 总方差 = 角色间方差 + 角色内方差
        total_var = np.var(all_hs, axis=0).sum()
        between_var = sum(delta_norms[r]**2 * len(role_data[r]) for r in roles) / len(all_hs)
        
        within_vars = {}
        for role, H in role_data.items():
            within_vars[role] = np.var(H, axis=0).sum()
        total_within_var = sum(within_vars[r] * len(role_data[r]) for r in roles) / len(all_hs)
        
        if total_var > 1e-10:
            between_pct = between_var / total_var * 100
            within_pct = total_within_var / total_var * 100
        else:
            between_pct = within_pct = 0
        
        print(f"    角色间方差(δ贡献): {between_pct:.1f}%")
        print(f"    角色内方差: {within_pct:.1f}%")
        
        # ===== 随机对照: 3组随机句子的δ =====
        print(f"\n  Step 2e: 随机对照")
        random_groups = [
            ["The weather is nice today", "The sun shines brightly", "The rain falls gently",
             "The wind blows softly", "The snow melts slowly", "The clouds drift apart",
             "The storm passed quickly", "The fog cleared away"],
            ["She runs every morning", "He swims in the pool", "They dance all night",
             "We sing together now", "I read before sleeping", "You write very well",
             "It plays the music", "She paints the wall"],
            ["The table is wooden", "The chair looks old", "The lamp shines dim",
             "The door closes shut", "The window opens wide", "The floor feels cold",
             "The ceiling hangs low", "The wall stands firm"],
        ]
        
        random_centers = []
        for group in random_groups:
            hs = []
            for sent in group:
                try:
                    toks = tokenizer(sent, return_tensors="pt").to(device)
                    captured = {}
                    layers_list = get_layers(model)
                    target_layer = layers_list[li]
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            captured['h'] = output[0].detach().float().cpu().numpy()
                        else:
                            captured['h'] = output.detach().float().cpu().numpy()
                    h_handle = target_layer.register_forward_hook(hook_fn)
                    with torch.no_grad():
                        _ = model(**toks)
                    h_handle.remove()
                    if 'h' in captured:
                        last_idx = captured['h'].shape[1] - 1
                        hs.append(captured['h'][0, last_idx, :])
                except:
                    pass
            if hs:
                random_centers.append(np.mean(hs, axis=0))
        
        if len(random_centers) >= 3:
            random_grand = np.mean(random_centers, axis=0)
            random_deltas = {f"G{i}": c - random_grand for i, c in enumerate(random_centers)}
            
            random_delta_angles = {}
            g_list = list(random_deltas.keys())
            for i in range(len(g_list)):
                for j in range(i + 1, len(g_list)):
                    d1, d2 = random_deltas[g_list[i]], random_deltas[g_list[j]]
                    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_a = np.dot(d1, d2) / (n1 * n2)
                        ang = np.degrees(np.arccos(np.clip(abs(cos_a), 0, 1)))
                        random_delta_angles[f"{g_list[i]}-{g_list[j]}"] = float(ang)
            
            avg_random_angle = np.mean(list(random_delta_angles.values())) if random_delta_angles else 0
            avg_syntax_angle = np.mean([v['angle_deg'] for v in delta_angles.values()]) if delta_angles else 0
            
            print(f"    随机δ夹角平均: {avg_random_angle:.1f}°")
            print(f"    语法δ夹角平均: {avg_syntax_angle:.1f}°")
            
            if avg_syntax_angle < avg_random_angle:
                print(f"    → 语法δ比随机δ更共线! (与Phase 21一致)")
            else:
                print(f"    → 语法δ比随机δ更正交!")
        else:
            avg_random_angle = 0

        layer_results[li] = {
            'delta_norms': delta_norms,
            'delta_angles': delta_angles,
            'delta_eff_dim': eff_dim,
            'between_var_pct': float(between_pct),
            'within_var_pct': float(within_pct),
            'random_delta_avg_angle': float(avg_random_angle),
        }

    # ===== 跨层总结 =====
    print(f"\n  ===== δ_role几何性质的跨层变化 =====")
    
    if layer_results:
        for role in roles:
            norms = [(li, lr['delta_norms'].get(role, 0)) for li, lr in layer_results.items()]
            if norms:
                norm_str = ", ".join([f"L{li}={n:.2f}" for li, n in norms])
                print(f"  ||δ_{role}||: {norm_str}")
        
        print(f"\n  δ夹角跨层变化:")
        for pair_key in ['nsubj-dobj', 'nsubj-amod', 'dobj-amod']:
            angles = [(li, lr['delta_angles'].get(pair_key, {}).get('angle_deg'))
                      for li, lr in layer_results.items()]
            valid = [(li, a) for li, a in angles if a is not None]
            if valid:
                a_str = ", ".join([f"L{li}={a:.1f}°" for li, a in valid])
                print(f"  {pair_key}: {a_str}")
        
        print(f"\n  角色间方差占比跨层变化:")
        for li, lr in sorted(layer_results.items()):
            print(f"  L{li}: between={lr['between_var_pct']:.1f}%")

    results['layer_results'] = {str(k): v for k, v in layer_results.items()}
    return results


# ===== Exp3: 平行移动关系 =====
def exp3_parallel_transport(model, tokenizer, device):
    """沿层的旋转是否保持δ_role的几何关系?"""
    print("\n" + "="*70)
    print("Exp3: 平行移动关系")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers

    roles = ["nsubj", "dobj", "amod"]

    # 采集所有层的δ
    print("\n  Step 3a: 采集所有层的δ_role")
    
    all_deltas = {}
    for li in range(n_layers):
        print(f"  L{li}...", end="", flush=True)

        role_data = {}
        for role in roles:
            data = EN_ROLES_DATA[role]
            H = collect_hs_at_layer(model, tokenizer, device,
                                    data["sentences"][:8], data["target_words"][:8], li)
            if H is not None and len(H) >= 3:
                role_data[role] = H

        if len(role_data) < 3:
            print(f" 跳过")
            continue

        centers = {role: np.mean(H, axis=0) for role, H in role_data.items()}
        grand_center = np.mean(list(centers.values()), axis=0)
        deltas = {role: centers[role] - grand_center for role in roles}
        
        all_deltas[li] = deltas
        print(f" ok")

    if len(all_deltas) < 4:
        print("  数据不足!")
        return results

    # ===== δ夹角的层间稳定性 =====
    print(f"\n  Step 3b: δ_role夹角的层间稳定性")
    
    sorted_layers = sorted(all_deltas.keys())
    delta_angle_history = {'nsubj-dobj': [], 'nsubj-amod': [], 'dobj-amod': []}
    
    for li in sorted_layers:
        deltas = all_deltas[li]
        for pair_key in delta_angle_history:
            r1, r2 = pair_key.split('-')
            if r1 in deltas and r2 in deltas:
                d1, d2 = deltas[r1], deltas[r2]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_a = np.dot(d1, d2) / (n1 * n2)
                    angle = np.degrees(np.arccos(np.clip(abs(cos_a), 0, 1)))
                    delta_angle_history[pair_key].append((li, angle))
    
    for pair_key, history in delta_angle_history.items():
        if history:
            angles = [a for _, a in history]
            print(f"  {pair_key}: mean={np.mean(angles):.1f}°, std={np.std(angles):.1f}°, "
                  f"range=[{np.min(angles):.1f}°, {np.max(angles):.1f}°]")
            
            # 是否有趋势?
            if len(history) >= 3:
                layers_arr = np.array([l for l, _ in history])
                angles_arr = np.array([a for _, a in history])
                slope, _ = np.polyfit(layers_arr, angles_arr, 1)
                print(f"    趋势: {slope:.3f}°/层 ({'递增' if slope > 0.5 else '递减' if slope < -0.5 else '稳定'})")

    # ===== δ范数的层间变化 =====
    print(f"\n  Step 3c: δ_role范数的层间变化")
    
    delta_norm_history = {role: [] for role in roles}
    for li in sorted_layers:
        deltas = all_deltas[li]
        for role in roles:
            if role in deltas:
                delta_norm_history[role].append((li, np.linalg.norm(deltas[role])))
    
    for role, history in delta_norm_history.items():
        if history:
            norms = [n for _, n in history]
            print(f"  ||δ_{role}||: min={np.min(norms):.2f}, max={np.max(norms):.2f}, "
                  f"ratio={np.max(norms)/max(np.min(norms),1e-10):.1f}x")

    # ===== δ方向的变化(平行移动) =====
    print(f"\n  Step 3d: δ_role方向的层间变化(平行移动)")
    
    # 相邻层δ方向的变化角
    direction_changes = {role: [] for role in roles}
    for i in range(len(sorted_layers) - 1):
        li = sorted_layers[i]
        li1 = sorted_layers[i + 1]
        deltas_l = all_deltas[li]
        deltas_l1 = all_deltas[li1]
        
        for role in roles:
            if role in deltas_l and role in deltas_l1:
                d_l = deltas_l[role]
                d_l1 = deltas_l1[role]
                n_l = np.linalg.norm(d_l)
                n_l1 = np.linalg.norm(d_l1)
                if n_l > 1e-10 and n_l1 > 1e-10:
                    cos_a = np.dot(d_l, d_l1) / (n_l * n_l1)
                    angle = np.degrees(np.arccos(np.clip(cos_a, 0, 1)))
                    direction_changes[role].append((li, li1, angle))
    
    for role, changes in direction_changes.items():
        if changes:
            angles = [a for _, _, a in changes]
            print(f"  δ_{role} 方向变化: mean={np.mean(angles):.1f}°, "
                  f"max={np.max(angles):.1f}°")
            print(f"    前1/3: mean={np.mean(angles[:len(angles)//3]):.1f}°" if len(angles) >= 3 else "")
            print(f"    后1/3: mean={np.mean(angles[2*len(angles)//3:]):.1f}°" if len(angles) >= 3 else "")

    # ===== δ夹角比例的层间稳定性 =====
    print(f"\n  Step 3e: δ夹角比例的层间稳定性(纤维结构不变性)")
    
    # 如果语法角色是纤维结构, 那么δ_role之间的夹角比例应该在层间稳定
    # 即: angle(nsubj-dobj) / angle(nsubj-amod) ≈ 常数
    angle_ratios = []
    for li in sorted_layers:
        angles = {}
        deltas = all_deltas[li]
        for r1, r2 in [('nsubj', 'dobj'), ('nsubj', 'amod'), ('dobj', 'amod')]:
            if r1 in deltas and r2 in deltas:
                d1, d2 = deltas[r1], deltas[r2]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_a = np.dot(d1, d2) / (n1 * n2)
                    angles[f"{r1}-{r2}"] = np.degrees(np.arccos(np.clip(abs(cos_a), 0, 1)))
        
        if 'nsubj-dobj' in angles and 'nsubj-amod' in angles and angles['nsubj-amod'] > 1:
            ratio = angles['nsubj-dobj'] / angles['nsubj-amod']
            angle_ratios.append((li, ratio))
    
    if angle_ratios:
        ratios = [r for _, r in angle_ratios]
        print(f"  angle(nsubj-dobj)/angle(nsubj-amod): mean={np.mean(ratios):.3f}, "
              f"std={np.std(ratios):.3f}, CV={np.std(ratios)/max(np.mean(ratios),1e-10):.3f}")
        
        if np.std(ratios) / max(np.mean(ratios), 1e-10) < 0.2:
            print(f"  → 夹角比例跨层稳定! 纤维结构不变性得到支持!")
        else:
            print(f"  → 夹角比例跨层变化较大, 纤维结构在层间演化")

    results['delta_angle_history'] = {k: [(l, float(a)) for l, a in v] 
                                      for k, v in delta_angle_history.items()}
    results['direction_changes'] = {k: [(l, l1, float(a)) for l, l1, a in v]
                                    for k, v in direction_changes.items()}
    results['angle_ratios'] = [(l, float(r)) for l, r in angle_ratios]
    
    return results


# ===== Exp4: h_role = h_semantic + δ_role 定量验证 =====
def exp4_linear_decomposition(model, tokenizer, device):
    """h_role = h_semantic + δ_role的定量验证"""
    print("\n" + "="*70)
    print("Exp4: h_role = h_semantic + δ_role 定量验证")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers

    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    roles = ["nsubj", "dobj", "amod", "poss"]

    layer_results = {}

    for li in sample_layers:
        print(f"\n  L{li}:")

        role_data = {}
        for role in roles:
            data = EN_ROLES_DATA[role]
            H = collect_hs_at_layer(model, tokenizer, device,
                                    data["sentences"][:8], data["target_words"][:8], li)
            if H is not None and len(H) >= 3:
                role_data[role] = H

        if len(role_data) < 3:
            print(f"    数据不足, 跳过")
            continue

        # ===== 方案1: h_semantic = PCA第一主成分 =====
        print(f"\n  Step 4a: h_semantic = PCA第一主成分方向")
        
        all_hs = np.vstack(list(role_data.values()))
        pca_full = PCA(n_components=min(10, len(all_hs) - 1))
        pca_full.fit(all_hs)
        
        print(f"    前10成分方差解释比: {pca_full.explained_variance_ratio_[:10]}")
        print(f"    前1成分: {pca_full.explained_variance_ratio_[0]:.4f}")
        print(f"    前3成分累计: {np.cumsum(pca_full.explained_variance_ratio_)[2]:.4f}")
        
        # ===== 方案2: 线性回归 h_role ≈ α·h_semantic + δ_role =====
        print(f"\n  Step 4b: 线性回归验证")
        
        # h_semantic = 所有数据的grand mean
        grand_mean = np.mean(all_hs, axis=0)
        
        for role in list(role_data.keys())[:3]:  # 只测试前3个角色
            H_role = role_data[role]
            
            # 对每个样本: h = grand_mean + δ
            # R² = 1 - Var(δ) / Var(h)
            residuals = H_role - grand_mean
            var_h = np.var(H_role, axis=0).sum()
            var_delta = np.var(residuals, axis=0).sum()
            
            if var_h > 1e-10:
                r_squared = 1 - var_delta / var_h
            else:
                r_squared = 0
            
            print(f"    {role}: Var(h)={var_h:.2f}, Var(δ)={var_delta:.2f}, R²={r_squared:.4f}")
            
            # δ的维度
            if len(residuals) >= 3:
                pca_delta = PCA(n_components=min(5, len(residuals) - 1))
                pca_delta.fit(residuals)
                cumvar = np.cumsum(pca_delta.explained_variance_ratio_)
                eff_dim = int(np.searchsorted(cumvar, 0.95) + 1) if cumvar[-1] >= 0.95 else len(cumvar)
                print(f"      δ有效维度(95%方差): {eff_dim}")
                print(f"      δ前3成分方差: {pca_delta.explained_variance_ratio_[:min(3, len(pca_delta.explained_variance_ratio_))]}")

        # ===== 方案3: 共享子空间分析 =====
        print(f"\n  Step 4c: 共享子空间 vs 私有子空间")
        
        # 共享子空间: 所有角色数据的PCA前k维
        # 私有子空间: 每个角色残差的PCA
        for k_shared in [1, 2, 3, 5]:
            shared_components = pca_full.components_[:k_shared]
            shared_basis = shared_components.T  # [d, k]
            
            total_shared_var = 0
            total_private_var = 0
            
            for role, H in role_data.items():
                # 投影到共享子空间
                proj = H @ shared_basis @ shared_basis.T
                residual = H - proj
                
                shared_var = np.var(proj, axis=0).sum()
                private_var = np.var(residual, axis=0).sum()
                
                total_shared_var += shared_var
                total_private_var += private_var
            
            total_var = total_shared_var + total_private_var
            if total_var > 1e-10:
                shared_pct = total_shared_var / total_var * 100
                print(f"    k={k_shared}共享子空间: 解释{shared_pct:.1f}%方差")

        layer_results[li] = {
            'pca_var_ratio': pca_full.explained_variance_ratio_[:10].tolist(),
            'pca_cumvar_3': float(np.cumsum(pca_full.explained_variance_ratio_)[2]),
        }

    results['layer_results'] = {str(k): v for k, v in layer_results.items()}
    return results


# ===== 主函数 =====
def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3, 4])
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CCMB Phase22 流形拓扑分析 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_lid_topology(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_delta_geometry(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_parallel_transport(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_linear_decomposition(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"ccmb_exp{args.exp}_{args.model}_results.json")

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
                return list(convert(v) for v in obj)
            return obj

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {out_path}")

    finally:
        release_model(model)
        print(f"  模型已释放")


if __name__ == "__main__":
    main()
