"""
CCME(Phase 25): 在单位球面上研究语法几何
=============================================================================
Phase 24核心发现:
  ★★★ 中间层的"1D共线"是norm伪影, 不是语法结构!
  → PC1 = norm方向 (corr=1.0000)
  → det角色norm比其他大100-200倍
  → L2归一化后PC1从100%降到12-21% → 高维
  → 语法信息在方向残余ε上, 不在原始h上

Phase 25实验:
  Exp1: ★★★★★★★★★★★ L2归一化后的语法几何
    → ĥ = h/||h|| (L2归一化)
    → 在单位球面上计算δ̄_r维度、Δ=A+B分解
    → 去掉norm伪影后的真实语法几何

  Exp2: ★★★★★★★★★ 去掉det角色后重分析
    → det是主要norm异常值
    → 7个角色(无det)的δ̄_r维度
    → 是否仍然1D?

  Exp3: ★★★★★★★ 位置控制实验
    → 所有target token放在句子中相同位置
    → 分离位置效应和语法效应

  Exp4: ★★★★★ ε_ℓ残差的精细分析
    → ε = ĥ - <ĥ> (归一化后的残差)
    → ε的维度、PCA谱、角色可分性
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import defaultdict

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode, MODEL_CONFIGS

# 复用Phase 24的控制词汇数据
from ccmd_phase24_controlled_vocab import CONTROLLED_ROLES_DATA, find_token_index, collect_hs_at_layer


def collect_all_data_controlled(model, tokenizer, device, layers_to_test):
    """收集所有角色在所有层的hidden states"""
    all_data = {}
    role_names = list(CONTROLLED_ROLES_DATA.keys())
    print(f"  收集8个角色的数据(控制词汇版): {role_names}")
    
    for layer_idx in layers_to_test:
        layer_data = {}
        for role in role_names:
            role_info = CONTROLLED_ROLES_DATA[role]
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


def l2_normalize(X):
    """L2归一化: 每行除以其L2范数"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return X / norms


# ==================== Exp1: L2归一化后的语法几何 ====================

def exp1_normalized_geometry(model_name, model, tokenizer, device, layers_to_test):
    """Exp1: L2归一化后的语法几何——去掉norm伪影"""
    print(f"\n{'='*60}")
    print(f"Exp1: L2归一化后的语法几何")
    print(f"{'='*60}")
    
    all_data, role_names = collect_all_data_controlled(model, tokenizer, device, layers_to_test)
    
    results = {"model": model_name, "exp": 1, "experiment": "normalized_geometry", "layers": {}}
    
    for layer_idx in sorted(all_data.keys()):
        layer_data = all_data[layer_idx]
        if len(layer_data) < 4:
            continue
        
        # ===== 原始空间分析(对照) =====
        all_hs_raw = np.vstack([hs for hs in layer_data.values()])
        all_labels = []
        for role, hs in layer_data.items():
            all_labels.extend([role] * len(hs))
        
        raw_pca = PCA()
        raw_pca.fit(all_hs_raw)
        raw_cumvar = np.cumsum(raw_pca.explained_variance_ratio_)
        
        # 原始δ̄_r
        role_centers_raw = {r: np.mean(hs, axis=0) for r, hs in layer_data.items()}
        gm_raw = np.mean(list(role_centers_raw.values()), axis=0)
        deltas_raw = np.array([role_centers_raw[r] - gm_raw for r in layer_data.keys()])
        raw_delta_pca = PCA()
        raw_delta_pca.fit(deltas_raw)
        
        # ===== L2归一化空间分析 =====
        # 对每个hidden state做L2归一化
        layer_data_normed = {}
        for role, hs in layer_data.items():
            layer_data_normed[role] = l2_normalize(hs)
        
        all_hs_normed = np.vstack([hs for hs in layer_data_normed.values()])
        
        normed_pca = PCA()
        normed_pca.fit(all_hs_normed)
        normed_cumvar = np.cumsum(normed_pca.explained_variance_ratio_)
        
        # 归一化后的δ̄_r
        role_centers_normed = {r: np.mean(hs, axis=0) for r, hs in layer_data_normed.items()}
        gm_normed = np.mean(list(role_centers_normed.values()), axis=0)
        deltas_normed = np.array([role_centers_normed[r] - gm_normed for r in layer_data_normed.keys()])
        
        normed_delta_pca = PCA()
        normed_delta_pca.fit(deltas_normed)
        normed_delta_cumvar = np.cumsum(normed_delta_pca.explained_variance_ratio_)
        
        n_roles = len(deltas_normed)
        
        # 归一化后的δ̄_r有效维度
        normed_delta_eff_dim_95 = np.searchsorted(normed_delta_cumvar, 0.95) + 1
        normed_delta_eff_dim_90 = np.searchsorted(normed_delta_cumvar, 0.90) + 1
        normed_delta_pr = (np.sum(normed_delta_pca.explained_variance_ratio_))**2 / np.sum(normed_delta_pca.explained_variance_ratio_**2)
        
        # ===== 归一化后的Δ=A+B分解 =====
        A_norms = []
        B_norms = []
        Delta_norms = []
        
        for role in layer_data_normed.keys():
            for h_vec in layer_data_normed[role]:
                delta = h_vec - gm_normed
                b = h_vec - role_centers_normed[role]
                
                A_norms.append(np.linalg.norm(role_centers_normed[role] - gm_normed))
                B_norms.append(np.linalg.norm(b))
                Delta_norms.append(np.linalg.norm(delta))
        
        A_var = np.mean(np.array(A_norms)**2)
        B_var = np.mean(np.array(B_norms)**2)
        total_var = np.mean(np.array(Delta_norms)**2)
        A_var_pct = A_var / total_var * 100 if total_var > 0 else 0
        B_var_pct = B_var / total_var * 100 if total_var > 0 else 0
        
        # ===== 归一化后的角色可分性(LDA) =====
        normed_proj = normed_pca.transform(all_hs_normed)[:, :min(10, normed_pca.n_components_)]
        labels_idx = [list(layer_data.keys()).index(l) for l in all_labels]
        
        if len(set(labels_idx)) >= 2:
            lda = LinearDiscriminantAnalysis()
            try:
                lda.fit(normed_proj, labels_idx)
                normed_lda_acc = round(float(lda.score(normed_proj, labels_idx)), 3)
            except Exception:
                normed_lda_acc = None
        else:
            normed_lda_acc = None
        
        # ===== 归一化后的角色间角度 =====
        role_angles = {}
        role_list = list(layer_data_normed.keys())
        for i in range(len(role_list)):
            for j in range(i+1, len(role_list)):
                r1, r2 = role_list[i], role_list[j]
                d1 = deltas_normed[i]
                d2 = deltas_normed[j]
                cos_a = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10)
                cos_a = np.clip(cos_a, -1, 1)
                angle = np.degrees(np.arccos(cos_a))
                role_angles[f"{r1}-{r2}"] = round(float(angle), 1)
        
        layer_result = {
            # 原始空间(对照)
            "raw_joint_pc1_pct": round(float(raw_pca.explained_variance_ratio_[0]*100), 2),
            "raw_delta_pc1_pct": round(float(raw_delta_pca.explained_variance_ratio_[0]*100), 2),
            "raw_delta_eff_dim_95": int(np.searchsorted(np.cumsum(raw_delta_pca.explained_variance_ratio_), 0.95) + 1),
            
            # 归一化空间
            "normed_joint_pc1_pct": round(float(normed_pca.explained_variance_ratio_[0]*100), 2),
            "normed_joint_top5_pct": [round(float(v*100), 2) for v in normed_pca.explained_variance_ratio_[:5]],
            "normed_joint_k3_var": round(float(normed_cumvar[min(2, len(normed_cumvar)-1)]*100), 2),
            "normed_joint_k10_var": round(float(normed_cumvar[min(9, len(normed_cumvar)-1)]*100), 2),
            
            "normed_delta_pc1_pct": round(float(normed_delta_pca.explained_variance_ratio_[0]*100), 2),
            "normed_delta_top5_pct": [round(float(v*100), 2) for v in normed_delta_pca.explained_variance_ratio_[:5]],
            "normed_delta_cumvar_top5_pct": [round(float(v*100), 1) for v in normed_delta_cumvar[:5]],
            "normed_delta_eff_dim_90": int(normed_delta_eff_dim_90),
            "normed_delta_eff_dim_95": int(normed_delta_eff_dim_95),
            "normed_delta_participation_ratio": round(float(normed_delta_pr), 2),
            
            # Δ=A+B分解(归一化后)
            "normed_A_var_pct": round(float(A_var_pct), 2),
            "normed_B_var_pct": round(float(B_var_pct), 2),
            
            # 可分性
            "normed_lda_accuracy": normed_lda_acc,
            
            # 角色间角度(归一化后δ̄_r之间)
            "role_angles_normed_delta": role_angles,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"  Layer {layer_idx}:")
        print(f"    原始: 联合PC1={raw_pca.explained_variance_ratio_[0]*100:.1f}%, δ̄_r PC1={raw_delta_pca.explained_variance_ratio_[0]*100:.1f}%, eff_dim_95={int(np.searchsorted(np.cumsum(raw_delta_pca.explained_variance_ratio_), 0.95)+1)}")
        print(f"    归一化: 联合PC1={normed_pca.explained_variance_ratio_[0]*100:.1f}%, δ̄_r PC1={normed_delta_pca.explained_variance_ratio_[0]*100:.1f}%, eff_dim_95={normed_delta_eff_dim_95}")
        print(f"    归一化Δ=A+B: A={A_var_pct:.1f}%, B={B_var_pct:.1f}%")
        print(f"    归一化LDA: {normed_lda_acc}")
    
    return results


# ==================== Exp2: 去掉det角色后重分析 ====================

def exp2_without_det(model_name, model, tokenizer, device, layers_to_test):
    """Exp2: 去掉det角色(最大norm异常值)后的分析"""
    print(f"\n{'='*60}")
    print(f"Exp2: 去掉det角色后重分析")
    print(f"{'='*60}")
    
    all_data, role_names = collect_all_data_controlled(model, tokenizer, device, layers_to_test)
    
    results = {"model": model_name, "exp": 2, "experiment": "without_det", "layers": {}}
    
    for layer_idx in sorted(all_data.keys()):
        layer_data = all_data[layer_idx]
        
        # ===== 8个角色(含det) =====
        if len(layer_data) < 4:
            continue
        
        all_hs_8 = np.vstack([hs for hs in layer_data.values()])
        pca_8 = PCA()
        pca_8.fit(all_hs_8)
        
        role_centers_8 = {r: np.mean(hs, axis=0) for r, hs in layer_data.items()}
        gm_8 = np.mean(list(role_centers_8.values()), axis=0)
        deltas_8 = np.array([role_centers_8[r] - gm_8 for r in layer_data.keys()])
        delta_pca_8 = PCA()
        delta_pca_8.fit(deltas_8)
        
        # ===== 7个角色(去掉det) =====
        layer_data_no_det = {r: hs for r, hs in layer_data.items() if r != "det"}
        if len(layer_data_no_det) < 3:
            continue
        
        all_hs_7 = np.vstack([hs for hs in layer_data_no_det.values()])
        pca_7 = PCA()
        pca_7.fit(all_hs_7)
        
        role_centers_7 = {r: np.mean(hs, axis=0) for r, hs in layer_data_no_det.items()}
        gm_7 = np.mean(list(role_centers_7.values()), axis=0)
        deltas_7 = np.array([role_centers_7[r] - gm_7 for r in layer_data_no_det.keys()])
        delta_pca_7 = PCA()
        delta_pca_7.fit(deltas_7)
        
        # ===== 7个角色 + L2归一化 =====
        layer_data_7_normed = {r: l2_normalize(hs) for r, hs in layer_data_no_det.items()}
        all_hs_7_normed = np.vstack([hs for hs in layer_data_7_normed.values()])
        pca_7_normed = PCA()
        pca_7_normed.fit(all_hs_7_normed)
        
        role_centers_7_normed = {r: np.mean(hs, axis=0) for r, hs in layer_data_7_normed.items()}
        gm_7_normed = np.mean(list(role_centers_7_normed.values()), axis=0)
        deltas_7_normed = np.array([role_centers_7_normed[r] - gm_7_normed for r in layer_data_7_normed.keys()])
        delta_pca_7_normed = PCA()
        delta_pca_7_normed.fit(deltas_7_normed)
        
        delta_7_normed_cumvar = np.cumsum(delta_pca_7_normed.explained_variance_ratio_)
        delta_7_normed_eff_95 = np.searchsorted(delta_7_normed_cumvar, 0.95) + 1
        delta_7_normed_pr = (np.sum(delta_pca_7_normed.explained_variance_ratio_))**2 / np.sum(delta_pca_7_normed.explained_variance_ratio_**2)
        
        # ===== det的norm vs 其他角色的norm =====
        det_norms = np.linalg.norm(layer_data.get("det", np.zeros((1,1))), axis=1) if "det" in layer_data else [0]
        other_norms = []
        for r, hs in layer_data.items():
            if r != "det":
                other_norms.extend(np.linalg.norm(hs, axis=1).tolist())
        
        layer_result = {
            # 8角色(含det)
            "with_det_joint_pc1": round(float(pca_8.explained_variance_ratio_[0]*100), 2),
            "with_det_delta_pc1": round(float(delta_pca_8.explained_variance_ratio_[0]*100), 2),
            "with_det_delta_eff_dim_95": int(np.searchsorted(np.cumsum(delta_pca_8.explained_variance_ratio_), 0.95) + 1),
            
            # 7角色(去det), 原始空间
            "without_det_joint_pc1": round(float(pca_7.explained_variance_ratio_[0]*100), 2),
            "without_det_delta_pc1": round(float(delta_pca_7.explained_variance_ratio_[0]*100), 2),
            "without_det_delta_eff_dim_95": int(np.searchsorted(np.cumsum(delta_pca_7.explained_variance_ratio_), 0.95) + 1),
            
            # 7角色(去det), L2归一化
            "without_det_normed_joint_pc1": round(float(pca_7_normed.explained_variance_ratio_[0]*100), 2),
            "without_det_normed_delta_pc1": round(float(delta_pca_7_normed.explained_variance_ratio_[0]*100), 2),
            "without_det_normed_delta_top5_pct": [round(float(v*100), 2) for v in delta_pca_7_normed.explained_variance_ratio_[:5]],
            "without_det_normed_delta_eff_dim_95": int(delta_7_normed_eff_95),
            "without_det_normed_delta_pr": round(float(delta_7_normed_pr), 2),
            
            # Norm统计
            "det_mean_norm": round(float(np.mean(det_norms)), 2),
            "other_mean_norm": round(float(np.mean(other_norms)), 2),
            "det_other_norm_ratio": round(float(np.mean(det_norms) / (np.mean(other_norms) + 1e-10)), 2),
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"  Layer {layer_idx}:")
        print(f"    含det: 联合PC1={pca_8.explained_variance_ratio_[0]*100:.1f}%, δ̄_r PC1={delta_pca_8.explained_variance_ratio_[0]*100:.1f}%")
        print(f"    去det(原始): 联合PC1={pca_7.explained_variance_ratio_[0]*100:.1f}%, δ̄_r PC1={delta_pca_7.explained_variance_ratio_[0]*100:.1f}%")
        print(f"    去det(归一化): 联合PC1={pca_7_normed.explained_variance_ratio_[0]*100:.1f}%, δ̄_r PC1={delta_pca_7_normed.explained_variance_ratio_[0]*100:.1f}%, eff_dim_95={delta_7_normed_eff_95}")
        print(f"    det/other norm比: {np.mean(det_norms)/(np.mean(other_norms)+1e-10):.1f}")
    
    return results


# ==================== Exp3: 位置控制实验 ====================

def exp3_position_control(model_name, model, tokenizer, device, layers_to_test):
    """Exp3: 控制token位置——同一词在不同位置"""
    print(f"\n{'='*60}")
    print(f"Exp3: 位置控制实验 (同一词在不同位置)")
    print(f"{'='*60}")
    
    # 设计: 同一个词"king"出现在句子的不同位置
    # pos1: 句首 "King ruled wisely" (king at position 0-1)
    # pos2: 句中 "The king ruled wisely" (king at position 1)
    # pos3: 句后 "They honored king today" (king at position 2-3)
    # pos4: 句末 "The people chose king" (king at last position)
    
    position_data = {
        "pos0": {
            "sentences": [
                "King ruled wisely today",
                "Queen governed fairly always",
                "Doctor worked carefully here",
                "Teacher taught clearly now",
                "Soldier fought bravely then",
                "Artist painted beautifully well",
            ],
            "target_words": ["King", "Queen", "Doctor", "Teacher", "Soldier", "Artist"],
            "expected_pos": 0,
        },
        "pos1": {
            "sentences": [
                "The king ruled wisely today",
                "The queen governed fairly always",
                "The doctor worked carefully here",
                "The teacher taught clearly now",
                "The soldier fought bravely then",
                "The artist painted beautifully well",
            ],
            "target_words": ["king", "queen", "doctor", "teacher", "soldier", "artist"],
            "expected_pos": 1,
        },
        "pos2": {
            "sentences": [
                "They crowned king yesterday",
                "They honored queen today",
                "They visited doctor recently",
                "They thanked teacher warmly",
                "They praised soldier highly",
                "They admired artist greatly",
            ],
            "target_words": ["king", "queen", "doctor", "teacher", "soldier", "artist"],
            "expected_pos": 2,
        },
        "pos3_end": {
            "sentences": [
                "The people chose king",
                "The nation followed queen",
                "The hospital needed doctor",
                "The school hired teacher",
                "The army respected soldier",
                "The gallery featured artist",
            ],
            "target_words": ["king", "queen", "doctor", "teacher", "soldier", "artist"],
            "expected_pos": -1,  # 最后一个内容词
        },
    }
    
    results = {"model": model_name, "exp": 3, "experiment": "position_control", "layers": {}}
    
    for layer_idx in layers_to_test:
        layers = get_layers(model)
        if layer_idx >= len(layers):
            layer_idx = len(layers) - 1
        target_layer = layers[layer_idx]
        
        pos_hs = {}
        pos_norms = {}
        
        for pos_name, pos_info in position_data.items():
            hs_list = []
            norm_list = []
            
            for sent, target_word in zip(pos_info["sentences"], pos_info["target_words"]):
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
                hs_list.append(h_vec)
                norm_list.append(np.linalg.norm(h_vec))
            
            if hs_list:
                pos_hs[pos_name] = np.array(hs_list)
                pos_norms[pos_name] = np.array(norm_list)
        
        if len(pos_hs) < 3:
            continue
        
        # ===== 原始空间: 不同位置的PCA =====
        all_pos_hs = np.vstack(list(pos_hs.values()))
        pos_pca = PCA()
        pos_pca.fit(all_pos_hs)
        
        # ===== L2归一化后: 不同位置的PCA =====
        all_pos_hs_normed = l2_normalize(all_pos_hs)
        pos_pca_normed = PCA()
        pos_pca_normed.fit(all_pos_hs_normed)
        
        # ===== 位置间的范数比 =====
        pos_mean_norms = {k: round(float(np.mean(v)), 2) for k, v in pos_norms.items()}
        
        # ===== 位置间的角度(归一化后) =====
        pos_centers_normed = {k: np.mean(l2_normalize(v), axis=0) for k, v in pos_hs.items()}
        pos_angle_matrix = {}
        pos_keys = list(pos_centers_normed.keys())
        for i in range(len(pos_keys)):
            for j in range(i+1, len(pos_keys)):
                k1, k2 = pos_keys[i], pos_keys[j]
                cos_a = np.dot(pos_centers_normed[k1], pos_centers_normed[k2])
                cos_a = np.clip(cos_a, -1, 1)
                pos_angle_matrix[f"{k1}-{k2}"] = round(float(np.degrees(np.arccos(cos_a))), 1)
        
        # ===== 同一词(nsubj角色)作为参照 =====
        # 用nsubj的king位置1 vs pos1的king位置1, 验证数据一致性
        
        layer_result = {
            "positions_found": list(pos_hs.keys()),
            "position_norms": pos_mean_norms,
            "raw_joint_pc1": round(float(pos_pca.explained_variance_ratio_[0]*100), 2),
            "raw_joint_top3_pct": [round(float(v*100), 2) for v in pos_pca.explained_variance_ratio_[:3]],
            "normed_joint_pc1": round(float(pos_pca_normed.explained_variance_ratio_[0]*100), 2),
            "normed_joint_top3_pct": [round(float(v*100), 2) for v in pos_pca_normed.explained_variance_ratio_[:3]],
            "position_angles_normed": pos_angle_matrix,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"  Layer {layer_idx}:")
        print(f"    位置norms: {pos_mean_norms}")
        print(f"    原始联合PC1={pos_pca.explained_variance_ratio_[0]*100:.1f}%, 归一化联合PC1={pos_pca_normed.explained_variance_ratio_[0]*100:.1f}%")
        print(f"    位置间角度(归一化): {pos_angle_matrix}")
    
    return results


# ==================== Exp4: ε_ℓ残差精细分析 ====================

def exp4_residual_analysis(model_name, model, tokenizer, device, layers_to_test):
    """Exp4: ε = ĥ - <ĥ> 的精细分析"""
    print(f"\n{'='*60}")
    print(f"Exp4: ε_ℓ残差精细分析 (归一化后的方向残余)")
    print(f"{'='*60}")
    
    all_data, role_names = collect_all_data_controlled(model, tokenizer, device, layers_to_test)
    
    results = {"model": model_name, "exp": 4, "experiment": "residual_analysis", "layers": {}}
    
    for layer_idx in sorted(all_data.keys()):
        layer_data = all_data[layer_idx]
        if len(layer_data) < 4:
            continue
        
        # L2归一化
        layer_data_normed = {r: l2_normalize(hs) for r, hs in layer_data.items()}
        
        # 合并所有归一化数据
        all_hs_normed = np.vstack([hs for hs in layer_data_normed.values()])
        all_labels = []
        for role, hs in layer_data_normed.items():
            all_labels.extend([role] * len(hs))
        
        # 计算全局均值
        global_mean_normed = np.mean(all_hs_normed, axis=0)
        
        # ε = ĥ - <ĥ>
        epsilon = all_hs_normed - global_mean_normed
        
        # ε的PCA
        eps_pca = PCA()
        eps_pca.fit(epsilon)
        eps_cumvar = np.cumsum(eps_pca.explained_variance_ratio_)
        
        # ε的有效维度
        eps_eff_dim_80 = np.searchsorted(eps_cumvar, 0.80) + 1
        eps_eff_dim_90 = np.searchsorted(eps_cumvar, 0.90) + 1
        eps_eff_dim_95 = np.searchsorted(eps_cumvar, 0.95) + 1
        
        # Participation ratio
        eps_pr = (np.sum(eps_pca.explained_variance_ratio_))**2 / np.sum(eps_pca.explained_variance_ratio_**2)
        
        # ===== ε的方差分解: 角色间 vs 角色内 =====
        # 角色间方差: 不同角色中心的方差
        role_centers_eps = {}
        for j, (role, _) in enumerate([(r, None) for r in layer_data_normed.keys()]):
            role_hs = layer_data_normed[role]
            role_eps = role_hs - global_mean_normed
            role_centers_eps[role] = np.mean(role_eps, axis=0)
        
        between_var = np.mean(np.sum(np.array(list(role_centers_eps.values()))**2, axis=1))
        
        within_var = 0
        n_within = 0
        for role, hs in layer_data_normed.items():
            center = role_centers_eps[role]
            role_eps = hs - global_mean_normed
            for e in role_eps:
                within_var += np.sum((e - center)**2)
                n_within += 1
        within_var /= n_within if n_within > 0 else 1
        
        total_eps_var = np.mean(np.sum(epsilon**2, axis=1))
        between_pct = between_var / total_eps_var * 100 if total_eps_var > 0 else 0
        within_pct = within_var / total_eps_var * 100 if total_eps_var > 0 else 0
        
        # ===== ε中角色信息的LDA分类 =====
        eps_proj = eps_pca.transform(epsilon)[:, :min(10, eps_pca.n_components_)]
        labels_idx = [list(layer_data_normed.keys()).index(l) for l in all_labels]
        
        if len(set(labels_idx)) >= 2:
            lda = LinearDiscriminantAnalysis()
            try:
                lda.fit(eps_proj, labels_idx)
                eps_lda_acc = round(float(lda.score(eps_proj, labels_idx)), 3)
            except Exception:
                eps_lda_acc = None
        else:
            eps_lda_acc = None
        
        # ===== ε的PC1方向与角色标签的关系 =====
        pc1 = eps_pca.components_[0]
        pc1_projections = epsilon @ pc1
        
        role_pc1_means = {}
        for role in layer_data_normed.keys():
            idx = [i for i, l in enumerate(all_labels) if l == role]
            role_pc1_means[role] = round(float(np.mean(pc1_projections[idx])), 6)
        
        # ===== 逐角色的ε维度 =====
        role_eps_dims = {}
        for role, hs in layer_data_normed.items():
            role_eps = hs - global_mean_normed
            if len(role_eps) >= 3:
                rpca = PCA()
                rpca.fit(role_eps)
                rc = np.cumsum(rpca.explained_variance_ratio_)
                role_eps_dims[role] = {
                    "pc1_pct": round(float(rpca.explained_variance_ratio_[0]*100), 2),
                    "eff_dim_90": int(np.searchsorted(rc, 0.90) + 1),
                }
        
        layer_result = {
            "epsilon_var_pct_of_total": round(float(total_eps_var / np.mean(np.sum(all_hs_normed**2, axis=1)) * 100), 2),
            "epsilon_pc1_pct": round(float(eps_pca.explained_variance_ratio_[0]*100), 2),
            "epsilon_top5_pct": [round(float(v*100), 2) for v in eps_pca.explained_variance_ratio_[:5]],
            "epsilon_cumvar_top5_pct": [round(float(v*100), 1) for v in eps_cumvar[:5]],
            "epsilon_eff_dim_80": int(eps_eff_dim_80),
            "epsilon_eff_dim_90": int(eps_eff_dim_90),
            "epsilon_eff_dim_95": int(eps_eff_dim_95),
            "epsilon_participation_ratio": round(float(eps_pr), 2),
            "between_role_var_pct": round(float(between_pct), 2),
            "within_role_var_pct": round(float(within_pct), 2),
            "epsilon_lda_accuracy": eps_lda_acc,
            "role_pc1_means_in_epsilon": role_pc1_means,
            "role_epsilon_dims": role_eps_dims,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        print(f"  Layer {layer_idx}:")
        print(f"    ε占总量: {total_eps_var / np.mean(np.sum(all_hs_normed**2, axis=1)) * 100:.2f}%")
        print(f"    ε PCA: PC1={eps_pca.explained_variance_ratio_[0]*100:.1f}%, eff_dim_95={eps_eff_dim_95}, PR={eps_pr:.2f}")
        print(f"    ε ANOVA: between={between_pct:.1f}%, within={within_pct:.1f}%")
        print(f"    ε LDA: {eps_lda_acc}")
        print(f"    角色ε维度: {role_eps_dims}")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Phase 25: 单位球面上的语法几何")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4],
                       help="实验编号(1-4)")
    args = parser.parse_args()
    
    model_name = args.model
    exp_num = args.exp
    
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
        print(f"[ccme] {model_name} loaded with 8bit, device={device}")
    else:
        model, tokenizer, device = load_model(model_name)
    
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    layers_to_test = [0]
    if n_layers > 6:
        step = (n_layers - 1) // 6
        layers_to_test = list(range(0, n_layers, step))[:7]
    if n_layers - 1 not in layers_to_test:
        layers_to_test.append(n_layers - 1)
    layers_to_test = sorted(set(layers_to_test))
    
    print(f"\nModel: {model_name}, Layers: {n_layers}, Test layers: {layers_to_test}")
    
    if exp_num == 1:
        results = exp1_normalized_geometry(model_name, model, tokenizer, device, layers_to_test)
    elif exp_num == 2:
        results = exp2_without_det(model_name, model, tokenizer, device, layers_to_test)
    elif exp_num == 3:
        results = exp3_position_control(model_name, model, tokenizer, device, layers_to_test)
    elif exp_num == 4:
        results = exp4_residual_analysis(model_name, model, tokenizer, device, layers_to_test)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5_temp")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"ccme_exp{exp_num}_{model_name}_results.json")
    
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
    
    release_model(model)
    print("模型已释放")


if __name__ == "__main__":
    main()
