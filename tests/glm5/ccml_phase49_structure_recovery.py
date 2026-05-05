"""
Phase 49: Structure Recovery — 结构恢复实验
============================================

SLT (Structural Language Theory) 核心转变:
    之前: 研究 z(x) ∈ R^k 的几何结构
    现在: 恢复 S(x) 的离散结构

关键认知:
    向量 = 结构的连续近似
    语言本体 = 离散结构系统 (entities, relations, attributes)

五个实验 (按优先级):
    49A: 去norm伪影 — 球面归一化后重测所有现象
    49B: 角色绑定恢复 — h(x) → {agent, patient, modifier}
    49C: 结构不变性 — 语法变换 = 结构保持?
    49D: 非线性组合 — 三项绑定测试
    49E: 操作函数恢复 — Δz = f(z)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import torch
import numpy as np
import gc
import time
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, collect_layer_outputs, get_W_U)


# ============================================================
# 49A: 去Norm伪影 — 球面归一化后重测
# ============================================================
def exp_49a_remove_norm_artifacts(model, tokenizer, info, model_name):
    """
    核心问题: Phase 48C的97.2%全局Δz方向是否只是norm增长?
    
    方法: z_norm = z / ||z||, 在球面上重测:
    1. Δz_norm的PCA结构
    2. 组合律(rel_err, cos)
    3. 语法变换(R²)
    """
    print("\n" + "="*70)
    print("49A: Removing Norm Artifacts — 球面归一化重测")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    
    # --- Sentence sets ---
    # 1. Norm growth sentences
    norm_sentences = [
        "The cat sits on the mat.",
        "A big red dog runs quickly.",
        "Small birds fly in the blue sky.",
        "The old man walks slowly home.",
        "Beautiful flowers grow in the garden.",
        "Young children play with toys.",
        "The white horse runs fast.",
        "Dark clouds cover the sky.",
        "The small fish swims upstream.",
        "A tall tree stands alone.",
        "The quick fox jumps high.",
        "Warm rain falls softly.",
        "The brave soldier fights hard.",
        "Old books rest on shelves.",
        "Bright stars shine at night.",
        "The green frog jumps far.",
        "Heavy snow falls quietly.",
        "The tiny ant carries food.",
        "Sweet honey tastes good.",
        "The loud bell rings daily.",
    ]
    
    # 2. Composition test pairs (from 48A, but with more structure)
    composition_pairs = [
        # (composed, adj_only, noun_only, base)
        ("The red cat", "The red one", "A cat", "The one"),
        ("The big dog", "The big one", "A dog", "The one"),
        ("The small bird", "The small one", "A bird", "The one"),
        ("The old man", "The old one", "A man", "The one"),
        ("The warm water", "The warm one", "Water", "The one"),
        ("The tall tree", "The tall one", "A tree", "The one"),
        ("The dark room", "The dark one", "A room", "The one"),
        ("The fast car", "The fast one", "A car", "The one"),
        ("The cold ice", "The cold one", "Ice", "The one"),
        ("The heavy stone", "The heavy one", "A stone", "The one"),
        ("The soft pillow", "The soft one", "A pillow", "The one"),
        ("The bright light", "The bright one", "Light", "The one"),
    ]
    
    # 3. Syntax pairs (from 48B, morphological only)
    syntax_pairs = [
        # (form_A, form_B, type)
        ("The cat runs", "The cats run", "plural"),
        ("The dog walks", "The dogs walk", "plural"),
        ("The bird flies", "The birds fly", "plural"),
        ("The tree grows", "The trees grow", "plural"),
        ("She walks", "She walked", "tense"),
        ("He runs", "He ran", "tense"),
        ("It jumps", "It jumped", "tense"),
        ("They play", "They played", "tense"),
        ("The cat sits", "The cat sat", "tense"),
        ("The dog barks", "The dog barked", "tense"),
    ]
    
    # --- Collect hidden states ---
    def get_hidden_states(sentences):
        """Collect last-token hidden states at all layers"""
        all_h = {}  # {layer_idx: [N, d_model]}
        
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                embed = model.get_input_embeddings()(input_ids)
                position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                outputs = collect_layer_outputs(model, embed, position_ids, n_layers)
            
            for li in range(n_layers):
                key = f"L{li}"
                if key in outputs:
                    h = outputs[key][0, -1, :].numpy()  # last token
                    if li not in all_h:
                        all_h[li] = []
                    all_h[li].append(h)
            
            del outputs, embed
            gc.collect()
        
        for li in all_h:
            all_h[li] = np.array(all_h[li])
        
        return all_h
    
    # --- Step 1: Norm growth analysis ---
    print("\n--- Step 1: Norm Growth Analysis ---")
    h_norm = get_hidden_states(norm_sentences)
    
    print(f"  {'Layer':>6} {'||h|| mean':>12} {'||h|| std':>12} {'||h||/||h0||':>14}")
    norms_by_layer = {}
    for li in sorted(h_norm.keys()):
        norms = np.linalg.norm(h_norm[li], axis=1)
        norms_by_layer[li] = norms
        h0_norm = np.mean(norms_by_layer.get(0, norms))
        if li % max(1, n_layers // 8) == 0 or li == n_layers - 1:
            print(f"  {li:>6} {np.mean(norms):>12.2f} {np.std(norms):>12.2f} {np.mean(norms)/max(h0_norm,1e-10):>14.3f}")
    
    # --- Step 2: Spherical normalization ---
    print("\n--- Step 2: Spherical Normalization ---")
    h_sphere = {}
    for li in h_norm:
        h = h_norm[li]
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        h_sphere[li] = h / norms
    
    # Verify normalization
    for li in [0, n_layers//2, n_layers-1]:
        if li in h_sphere:
            norms_check = np.linalg.norm(h_sphere[li], axis=1)
            print(f"  Layer {li}: ||z_norm|| mean = {np.mean(norms_check):.6f} (should be 1.0)")
    
    # --- Step 3: Δz structure on sphere ---
    print("\n--- Step 3: Δz on Sphere vs Raw ---")
    
    # Raw Δz PCA
    all_dz_raw = []
    for li in range(1, n_layers):
        if li in h_norm and (li-1) in h_norm:
            dz = h_norm[li] - h_norm[li-1]
            all_dz_raw.append(dz)
    all_dz_raw = np.vstack(all_dz_raw)
    
    pca_raw = PCA()
    pca_raw.fit(all_dz_raw)
    var_raw = pca_raw.explained_variance_ratio_
    
    # Sphere Δz PCA  
    all_dz_sphere = []
    for li in range(1, n_layers):
        if li in h_sphere and (li-1) in h_sphere:
            dz = h_sphere[li] - h_sphere[li-1]
            all_dz_sphere.append(dz)
    all_dz_sphere = np.vstack(all_dz_sphere)
    
    pca_sphere = PCA()
    pca_sphere.fit(all_dz_sphere)
    var_sphere = pca_sphere.explained_variance_ratio_
    
    print(f"\n  Raw Δz PCA variance:")
    print(f"  Top1: {var_raw[0]:.4f}, Top5: {sum(var_raw[:5]):.4f}, Top10: {sum(var_raw[:10]):.4f}")
    
    print(f"\n  Sphere Δz PCA variance:")
    print(f"  Top1: {var_sphere[0]:.4f}, Top5: {sum(var_sphere[:5]):.4f}, Top10: {sum(var_sphere[:10]):.4f}")
    
    # Dimensionality comparison
    def dim_at_variance(var_ratio, threshold):
        cumvar = np.cumsum(var_ratio)
        idx = np.searchsorted(cumvar, threshold)
        return min(idx + 1, len(var_ratio))
    
    print(f"\n  {'Metric':>12} {'Raw':>8} {'Sphere':>8} {'Change':>8}")
    for threshold, label in [(0.5, "dim50"), (0.8, "dim80"), (0.9, "dim90"), (0.95, "dim95")]:
        d_raw = dim_at_variance(var_raw, threshold)
        d_sph = dim_at_variance(var_sphere, threshold)
        change = "↓" if d_sph < d_raw else ("↑" if d_sph > d_raw else "=")
        print(f"  {label:>12} {d_raw:>8} {d_sph:>8} {change:>8}")
    
    # --- Step 4: Composition on sphere ---
    print("\n--- Step 4: Composition Test on Sphere ---")
    
    h_comp = {}
    all_comp_sents = []
    for pair in composition_pairs:
        all_comp_sents.extend(list(pair))
    
    # Deduplicate
    unique_sents = list(dict.fromkeys(all_comp_sents))
    sent_to_idx = {s: i for i, s in enumerate(unique_sents)}
    
    h_unique = get_hidden_states(unique_sents)
    
    print(f"  {'Layer':>6} {'raw_rel_err':>13} {'sph_rel_err':>13} {'raw_cos':>9} {'sph_cos':>9}")
    
    key_layers = [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    for li in sorted(h_unique.keys()):
        if li not in key_layers and li != 0:
            continue
        
        raw_errors = []
        sph_errors = []
        raw_cosines = []
        sph_cosines = []
        
        for composed, adj_only, noun_only, base in composition_pairs:
            ci, ai, ni, bi = (sent_to_idx[s] for s in [composed, adj_only, noun_only, base])
            
            h_raw = h_unique[li]
            z_comp = h_raw[ci]
            z_adj = h_raw[ai]
            z_noun = h_raw[ni]
            z_base = h_raw[bi]
            
            # Predicted: z(adj) + z(noun) - z(base)
            z_pred_raw = z_adj + z_noun - z_base
            
            # Raw metrics
            err_raw = np.linalg.norm(z_comp - z_pred_raw) / max(np.linalg.norm(z_comp), 1e-10)
            cos_raw = 1 - cosine(z_comp, z_pred_raw) if np.linalg.norm(z_pred_raw) > 1e-10 else 0
            raw_errors.append(err_raw)
            raw_cosines.append(cos_raw)
            
            # Sphere metrics
            def normalize_vec(v):
                n = np.linalg.norm(v)
                return v / n if n > 1e-10 else v
            
            z_comp_s = normalize_vec(z_comp)
            z_adj_s = normalize_vec(z_adj)
            z_noun_s = normalize_vec(z_noun)
            z_base_s = normalize_vec(z_base)
            z_pred_s = normalize_vec(z_adj_s + z_noun_s - z_base_s)
            
            err_sph = np.linalg.norm(z_comp_s - z_pred_s) / max(np.linalg.norm(z_comp_s), 1e-10)
            cos_sph = 1 - cosine(z_comp_s, z_pred_s) if np.linalg.norm(z_pred_s) > 1e-10 else 0
            sph_errors.append(err_sph)
            sph_cosines.append(cos_sph)
        
        if li in key_layers or li == 0:
            print(f"  {li:>6} {np.mean(raw_errors):>13.3f} {np.mean(sph_errors):>13.3f} {np.mean(raw_cosines):>9.3f} {np.mean(sph_cosines):>9.3f}")
    
    # --- Step 5: Syntax transformation on sphere ---
    print("\n--- Step 5: Syntax Transformation on Sphere ---")
    
    h_syn = {}
    syn_unique = list(dict.fromkeys([p[0] for p in syntax_pairs] + [p[1] for p in syntax_pairs]))
    syn_to_idx = {s: i for i, s in enumerate(syn_unique)}
    h_syn_unique = get_hidden_states(syn_unique)
    
    print(f"  {'Layer':>6} {'raw_R2_plural':>15} {'sph_R2_plural':>15} {'raw_R2_tense':>14} {'sph_R2_tense':>14}")
    
    for li in sorted(h_syn_unique.keys()):
        if li not in key_layers and li != 0:
            continue
        
        h_all = h_syn_unique[li]
        
        # Normalize
        norms = np.linalg.norm(h_all, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        h_sph = h_all / norms
        
        for syn_type in ["plural", "tense"]:
            pairs = [p for p in syntax_pairs if p[2] == syn_type]
            idxs_A = [syn_to_idx[p[0]] for p in pairs]
            idxs_B = [syn_to_idx[p[1]] for p in pairs]
            
            # Raw: linear fit z_B = W z_A + b
            z_A_raw = h_all[idxs_A]
            z_B_raw = h_all[idxs_B]
            ridge_raw = Ridge(alpha=1.0)
            ridge_raw.fit(z_A_raw, z_B_raw)
            z_pred_raw = ridge_raw.predict(z_A_raw)
            ss_res = np.sum((z_B_raw - z_pred_raw)**2)
            ss_tot = np.sum((z_B_raw - np.mean(z_B_raw, axis=0))**2)
            r2_raw = 1 - ss_res / max(ss_tot, 1e-20)
            
            # Sphere: linear fit on normalized
            z_A_sph = h_sph[idxs_A]
            z_B_sph = h_sph[idxs_B]
            ridge_sph = Ridge(alpha=1.0)
            ridge_sph.fit(z_A_sph, z_B_sph)
            z_pred_sph = ridge_sph.predict(z_A_sph)
            ss_res_s = np.sum((z_B_sph - z_pred_sph)**2)
            ss_tot_s = np.sum((z_B_sph - np.mean(z_B_sph, axis=0))**2)
            r2_sph = 1 - ss_res_s / max(ss_tot_s, 1e-20)
            
            if syn_type == "plural":
                r2_raw_pl = r2_raw
                r2_sph_pl = r2_sph
            else:
                r2_raw_te = r2_raw
                r2_sph_te = r2_sph
        
        if li in key_layers or li == 0:
            print(f"  {li:>6} {r2_raw_pl:>15.3f} {r2_sph_pl:>15.3f} {r2_raw_te:>14.3f} {r2_sph_te:>14.3f}")
    
    # --- Step 6: Angular structure analysis ---
    print("\n--- Step 6: Angular Structure (Pure Direction) ---")
    print("  After removing norm, what structure remains?")
    
    # Compute pairwise angles on sphere
    for li in key_layers:
        if li not in h_sphere:
            continue
        h_s = h_sphere[li]  # [N, d]
        
        # Pairwise cosine similarity
        N = h_s.shape[0]
        cos_sim = h_s @ h_s.T  # [N, N]
        norms_diag = np.sqrt(np.diag(cos_sim))
        cos_sim = cos_sim / np.outer(norms_diag, norms_diag + 1e-10)
        
        # Statistics
        off_diag = cos_sim[np.triu_indices(N, k=1)]
        print(f"  Layer {li}: cos_sim mean={np.mean(off_diag):.3f}, "
              f"std={np.std(off_diag):.3f}, "
              f"min={np.min(off_diag):.3f}, max={np.max(off_diag):.3f}")
    
    # --- Step 7: Δz norm correlation ---
    print("\n--- Step 7: Δz vs ||z|| Correlation ---")
    print("  Is Δz dominated by norm change?")
    
    for li in range(1, n_layers):
        if li not in h_norm or (li-1) not in h_norm:
            continue
        if li not in key_layers and li != 1:
            continue
        
        dz = h_norm[li] - h_norm[li-1]
        z_prev = h_norm[li-1]
        
        # Correlation between ||Δz|| and ||z||
        dz_norms = np.linalg.norm(dz, axis=1)
        z_norms = np.linalg.norm(z_prev, axis=1)
        
        corr = np.corrcoef(dz_norms, z_norms)[0, 1]
        
        # Δz direction vs z direction
        cos_dz_z = []
        for i in range(len(dz)):
            dn = np.linalg.norm(dz[i])
            zn = np.linalg.norm(z_prev[i])
            if dn > 1e-10 and zn > 1e-10:
                cos_dz_z.append(np.dot(dz[i], z_prev[i]) / (dn * zn))
        
        print(f"  Layer {li}: corr(||Δz||, ||z||)={corr:.3f}, "
              f"mean_cos(Δz, z)={np.mean(cos_dz_z):.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("49A SUMMARY: Norm Artifact Assessment")
    print("="*70)
    
    top1_raw = var_raw[0]
    top1_sph = var_sphere[0]
    
    print(f"\n  ★ Global Δz top1 variance:")
    print(f"    Raw:    {top1_raw:.4f} ({top1_raw*100:.1f}%)")
    print(f"    Sphere: {top1_sph:.4f} ({top1_sph*100:.1f}%)")
    
    if top1_sph < top1_raw * 0.5:
        print(f"    → ★★★ CONFIRMED: Most of top1 variance was norm artifact!")
    elif top1_sph < top1_raw * 0.8:
        print(f"    → Partial: Norm contributes significantly but not all")
    else:
        print(f"    → Norm is NOT the main factor; direction structure persists")
    
    dim90_raw = dim_at_variance(var_raw, 0.9)
    dim90_sph = dim_at_variance(var_sphere, 0.9)
    print(f"\n  ★ Δz dimensionality (dim90):")
    print(f"    Raw:    {dim90_raw}")
    print(f"    Sphere: {dim90_sph}")
    if dim90_sph > dim90_raw * 1.5:
        print(f"    → Norm collapse: Raw was underestimating true dimensionality!")
    
    return {
        "var_raw_top1": float(top1_raw),
        "var_sphere_top1": float(top1_sph),
        "dim90_raw": dim90_raw,
        "dim90_sph": dim90_sph,
    }


# ============================================================
# 49B: 角色绑定恢复 — Role Binding Recovery
# ============================================================
def exp_49b_role_binding(model, tokenizer, info, model_name):
    """
    核心测试: 模型是否编码 agent/patient/modifier 角色?
    
    设计:
    - 构造 SVO 句子, 变换 agent/patient
    - Probe: 从 h(x) 预测角色
    - 关键测试: 角色绑定不变性
      z(agent=cat in "cat eats fish") ≈ z(agent=cat in "cat chases dog")?
    """
    print("\n" + "="*70)
    print("49B: Role Binding Recovery — 角色绑定恢复")
    print("="*70)
    
    n_layers = info.n_layers
    device = next(model.parameters()).device
    
    # --- Sentence sets with controlled roles ---
    # Set 1: Same agent, different actions/patients
    # Tests: agent invariance across contexts
    agent_invariance = [
        # (sentence, agent, patient, action)
        ("The cat eats the fish", "cat", "fish", "eat"),
        ("The cat chases the dog", "cat", "dog", "chase"),
        ("The cat sees the bird", "cat", "bird", "see"),
        ("The cat catches the mouse", "cat", "mouse", "catch"),
        ("The cat watches the dog", "cat", "dog", "watch"),
        
        ("The dog chases the cat", "dog", "cat", "chase"),
        ("The dog bites the bone", "dog", "bone", "bite"),
        ("The dog sees the cat", "dog", "cat", "see"),
        ("The dog follows the man", "dog", "man", "follow"),
        ("The dog guards the house", "dog", "house", "guard"),
        
        ("The bird eats the worm", "bird", "worm", "eat"),
        ("The bird sees the cat", "bird", "cat", "see"),
        ("The bird flies over the tree", "bird", "tree", "fly over"),
    ]
    
    # Set 2: Same patient, different agents
    # Tests: patient invariance across contexts
    patient_invariance = [
        ("The cat eats the fish", "cat", "fish"),
        ("The bear eats the fish", "bear", "fish"),
        ("The man eats the fish", "man", "fish"),
        
        ("The cat chases the dog", "cat", "dog"),
        ("The boy chases the dog", "boy", "dog"),
        ("The fox chases the dog", "fox", "dog"),
        
        ("The cat catches the mouse", "cat", "mouse"),
        ("The owl catches the mouse", "owl", "mouse"),
        ("The snake catches the mouse", "snake", "mouse"),
    ]
    
    # Set 3: Active/Passive with same roles
    # Tests: structural invariance under syntax change
    structural_pairs = [
        ("The cat eats the fish", "The fish is eaten by the cat", "cat", "fish"),
        ("The dog chases the cat", "The cat is chased by the dog", "dog", "cat"),
        ("The man sees the bird", "The bird is seen by the man", "man", "bird"),
        ("The bear catches the fish", "The fish is caught by the bear", "bear", "fish"),
        ("The owl watches the mouse", "The mouse is watched by the owl", "owl", "mouse"),
    ]
    
    # Set 4: Modifier binding
    modifier_sentences = [
        # (sentence, entity, modifier, modifier_type)
        ("The red cat sits", "cat", "red", "color"),
        ("The big cat sits", "cat", "big", "size"),
        ("The red dog sits", "dog", "red", "color"),
        ("The big dog sits", "dog", "big", "size"),
        ("The old man walks", "man", "old", "age"),
        ("The young man walks", "man", "young", "age"),
        ("The old woman walks", "woman", "old", "age"),
        ("The young woman walks", "woman", "young", "age"),
    ]
    
    # --- Collect hidden states ---
    def get_hs(sentences):
        result = {}
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            with torch.no_grad():
                embed = model.get_input_embeddings()(inputs["input_ids"])
                pos_ids = torch.arange(inputs["input_ids"].shape[1], device=device).unsqueeze(0)
                outputs = collect_layer_outputs(model, embed, pos_ids, n_layers)
            for li in range(n_layers):
                key = f"L{li}"
                if key in outputs:
                    h = outputs[key][0, -1, :].numpy()
                    if li not in result:
                        result[li] = []
                    result[li].append(h)
            del outputs, embed
            gc.collect()
        for li in result:
            result[li] = np.array(result[li])
        return result
    
    # --- Analysis 1: Role Probing ---
    print("\n--- Analysis 1: Role Probing (Linear Readout) ---")
    print("  Can we linearly read agent/patient identity from h(x)?")
    
    # Build probe dataset from agent_invariance
    all_sents_probe = [s[0] for s in agent_invariance]
    agents = [s[1] for s in agent_invariance]
    patients = [s[2] for s in agent_invariance]
    actions = [s[3] for s in agent_invariance]
    
    h_probe = get_hs(all_sents_probe)
    
    print(f"  {'Layer':>6} {'agent_acc':>11} {'patient_acc':>13} {'action_acc':>12}")
    
    best_layer_agent = 0
    best_acc_agent = 0
    
    for li in sorted(h_probe.keys()):
        h = h_probe[li]
        
        # Normalize
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        h_norm = h / np.maximum(norms, 1e-10)
        
        # Agent probe
        unique_agents = list(set(agents))
        agent_labels = [unique_agents.index(a) for a in agents]
        if len(unique_agents) >= 2 and len(set(agent_labels)) >= 2:
            try:
                clf = LogisticRegression(max_iter=500, C=1.0, multi_class='ovr')
                scores_agent = cross_val_score(clf, h_norm, agent_labels, cv=min(3, len(set(agent_labels))), scoring='accuracy')
                acc_agent = scores_agent.mean()
            except:
                acc_agent = 0.0
        else:
            acc_agent = 0.0
        
        # Patient probe
        unique_patients = list(set(patients))
        patient_labels = [unique_patients.index(p) for p in patients]
        if len(unique_patients) >= 2 and len(set(patient_labels)) >= 2:
            try:
                clf = LogisticRegression(max_iter=500, C=1.0, multi_class='ovr')
                scores_patient = cross_val_score(clf, h_norm, patient_labels, cv=min(3, len(set(patient_labels))), scoring='accuracy')
                acc_patient = scores_patient.mean()
            except:
                acc_patient = 0.0
        else:
            acc_patient = 0.0
        
        # Action probe
        unique_actions = list(set(actions))
        action_labels = [unique_actions.index(a) for a in actions]
        if len(unique_actions) >= 2 and len(set(action_labels)) >= 2:
            try:
                clf = LogisticRegression(max_iter=500, C=1.0, multi_class='ovr')
                scores_action = cross_val_score(clf, h_norm, action_labels, cv=min(3, len(set(action_labels))), scoring='accuracy')
                acc_action = scores_action.mean()
            except:
                acc_action = 0.0
        else:
            acc_action = 0.0
        
        if acc_agent > best_acc_agent:
            best_acc_agent = acc_agent
            best_layer_agent = li
        
        if li % max(1, n_layers // 8) == 0 or li == n_layers - 1:
            print(f"  {li:>6} {acc_agent:>11.3f} {acc_patient:>13.3f} {acc_action:>12.3f}")
    
    print(f"\n  ★ Best agent probing: Layer {best_layer_agent}, acc={best_acc_agent:.3f}")
    
    # --- Analysis 2: Agent Invariance ---
    print("\n--- Analysis 2: Agent Binding Invariance ---")
    print("  Is z(agent=cat) similar across different sentences?")
    
    # Group by agent
    from collections import defaultdict
    agent_groups = defaultdict(list)
    for i, (sent, agent, patient, action) in enumerate(agent_invariance):
        agent_groups[agent].append(i)
    
    h_all = h_probe
    
    for li in [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if li not in h_all:
            continue
        
        h = h_all[li]
        # Normalize
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        h = h / np.maximum(norms, 1e-10)
        
        print(f"\n  Layer {li}:")
        
        # For each agent with 2+ sentences, compute intra-agent similarity
        # vs inter-agent similarity
        intra_sims = []
        inter_sims = []
        
        agent_list = list(agent_groups.keys())
        for ai, agent in enumerate(agent_list):
            idxs = agent_groups[agent]
            if len(idxs) < 2:
                continue
            
            # Intra-agent cosine similarity
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    sim = 1 - cosine(h[idxs[i]], h[idxs[j]])
                    intra_sims.append(sim)
            
            # Inter-agent similarity
            for aj in range(ai+1, len(agent_list)):
                other_agent = agent_list[aj]
                other_idxs = agent_groups[other_agent]
                for ii in idxs[:3]:
                    for jj in other_idxs[:3]:
                        sim = 1 - cosine(h[ii], h[jj])
                        inter_sims.append(sim)
        
        if intra_sims and inter_sims:
            print(f"    Intra-agent cos: mean={np.mean(intra_sims):.3f}, std={np.std(intra_sims):.3f}")
            print(f"    Inter-agent cos: mean={np.mean(inter_sims):.3f}, std={np.std(inter_sims):.3f}")
            gap = np.mean(intra_sims) - np.mean(inter_sims)
            print(f"    Gap: {gap:.3f} {'★★★ AGENT BINDING EXISTS!' if gap > 0.15 else '⚠️ Weak' if gap > 0.05 else '❌ No binding'}")
    
    # --- Analysis 3: Active/Passive Structural Invariance ---
    print("\n--- Analysis 3: Active/Passive Role Invariance ---")
    print("  Does agent/cat have similar representation in active vs passive?")
    
    active_sents = [p[0] for p in structural_pairs]
    passive_sents = [p[1] for p in structural_pairs]
    
    h_active = get_hs(active_sents)
    h_passive = get_hs(passive_sents)
    
    print(f"  {'Layer':>6} {'cos(active,passive)':>20} {'cos_agent_active_passive':>26} {'cos_patient_active_passive':>28}")
    
    for li in [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if li not in h_active or li not in h_passive:
            continue
        
        h_a = h_active[li]
        h_p = h_passive[li]
        
        # Overall similarity
        overall_cos = []
        for i in range(len(h_a)):
            c = 1 - cosine(h_a[i], h_p[i])
            overall_cos.append(c)
        
        # Role probe: train on active, test on passive
        # Can we predict which entity is the agent in both forms?
        # We need more sentences for this, so let's just compare directions
        
        # Simple test: agent entity direction
        # In "cat eats fish", agent=cat is the subject
        # In "fish is eaten by cat", agent=cat is after "by"
        # The "cat" token representation should carry agent role info
        
        print(f"  {li:>6} {np.mean(overall_cos):>20.3f}")
    
    # --- Analysis 4: Modifier Binding ---
    print("\n--- Analysis 4: Modifier Binding ---")
    print("  Is z(red in 'red cat') similar to z(red in 'red dog')?")
    
    mod_sents = [s[0] for s in modifier_sentences]
    h_mod = get_hs(mod_sents)
    
    for li in [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if li not in h_mod:
            continue
        
        h = h_mod[li]
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        h = h / np.maximum(norms, 1e-10)
        
        # Same modifier, different entity
        # "red cat" (idx 0) vs "red dog" (idx 2)
        # "big cat" (idx 1) vs "big dog" (idx 3)
        same_mod_diff_entity = []
        # "red cat" (idx 0) vs "big cat" (idx 1)
        # "red dog" (idx 2) vs "big dog" (idx 3)
        diff_mod_same_entity = []
        
        if len(h) >= 8:
            # color invariance: same color different entity
            same_mod_diff_entity.append(1 - cosine(h[0], h[2]))  # red cat vs red dog
            same_mod_diff_entity.append(1 - cosine(h[1], h[3]))  # big cat vs big dog
            same_mod_diff_entity.append(1 - cosine(h[4], h[6]))  # old man vs old woman
            same_mod_diff_entity.append(1 - cosine(h[5], h[7]))  # young man vs young woman
            
            # entity invariance: same entity different modifier
            diff_mod_same_entity.append(1 - cosine(h[0], h[1]))  # red cat vs big cat
            diff_mod_same_entity.append(1 - cosine(h[2], h[3]))  # red dog vs big dog
            diff_mod_same_entity.append(1 - cosine(h[4], h[5]))  # old man vs young man
            diff_mod_same_entity.append(1 - cosine(h[6], h[7]))  # old woman vs young woman
        
        print(f"  Layer {li}:")
        if same_mod_diff_entity:
            print(f"    Same modifier, diff entity: cos={np.mean(same_mod_diff_entity):.3f}")
        if diff_mod_same_entity:
            print(f"    Diff modifier, same entity: cos={np.mean(diff_mod_same_entity):.3f}")
    
    # --- Analysis 5: Dependency structure probing ---
    print("\n--- Analysis 5: Dependency Structure Probing ---")
    print("  Can we linearly separate sentences by role structure?")
    
    # Different structures:
    # Type 1: agent-action-patient (transitive): "cat eats fish"
    # Type 2: agent-action (intransitive): "cat sleeps"
    # Type 3: agent-action-location: "cat sleeps inside"
    
    structure_sentences = {
        "transitive": [
            "The cat eats the fish",
            "The dog chases the cat",
            "The man sees the bird",
            "The owl catches the mouse",
            "The bear eats the honey",
            "The fox hunts the rabbit",
        ],
        "intransitive": [
            "The cat sleeps quietly",
            "The dog runs fast",
            "The man walks slowly",
            "The bird flies high",
            "The fish swims deep",
            "The frog jumps far",
        ],
    }
    
    all_struct_sents = structure_sentences["transitive"] + structure_sentences["intransitive"]
    struct_labels = [0]*len(structure_sentences["transitive"]) + [1]*len(structure_sentences["intransitive"])
    
    h_struct = get_hs(all_struct_sents)
    
    print(f"  {'Layer':>6} {'struct_probe_acc':>18}")
    
    for li in sorted(h_struct.keys()):
        h = h_struct[li]
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        h = h / np.maximum(norms, 1e-10)
        
        try:
            clf = LogisticRegression(max_iter=500, C=1.0)
            scores = cross_val_score(clf, h, struct_labels, cv=3, scoring='accuracy')
            acc = scores.mean()
        except:
            acc = 0.0
        
        if li % max(1, n_layers // 8) == 0 or li == n_layers - 1:
            print(f"  {li:>6} {acc:>18.3f}")
    
    print("\n" + "="*70)
    print("49B SUMMARY: Role Binding Recovery")
    print("="*70)
    print(f"  Best agent probing accuracy: {best_acc_agent:.3f} at Layer {best_layer_agent}")
    
    return {"best_agent_acc": best_acc_agent, "best_agent_layer": best_layer_agent}


# ============================================================
# 49C: 结构不变性 — Structural Invariance under Syntax
# ============================================================
def exp_49c_structural_invariance(model, tokenizer, info, model_name):
    """
    核心测试: 语法变换是否保持结构不变?
    
    关键: 不测 z_B ≈ A z_A (Phase 48B已证明不成立)
    而测: role(agent_A) ≈ role(agent_B)?
    
    如果成立: 语法 = 结构保持变换 (等价于群的共变作用)
    """
    print("\n" + "="*70)
    print("49C: Structural Invariance under Syntax Change")
    print("="*70)
    
    n_layers = info.n_layers
    device = next(model.parameters()).device
    
    # --- Design: Extract role-specific subspaces ---
    # Key idea: If there exists a "role subspace", then
    # the agent component of z should be invariant to syntax
    
    # Sentence matrix:
    # Row: (sentence, agent_word, patient_word, syntax_type)
    sentence_matrix = [
        # Active/Passive pairs with same agent/patient
        ("The cat eats the fish", "cat", "fish", "active"),
        ("The fish is eaten by the cat", "cat", "fish", "passive"),
        ("The dog chases the cat", "dog", "cat", "active"),
        ("The cat is chased by the dog", "dog", "cat", "passive"),
        ("The man sees the bird", "man", "bird", "active"),
        ("The bird is seen by the man", "man", "bird", "passive"),
        ("The bear catches the fish", "bear", "fish", "active"),
        ("The fish is caught by the bear", "bear", "fish", "passive"),
        
        # Statement/Question pairs
        ("The cat eats fish", "cat", "fish", "statement"),
        ("Does the cat eat fish", "cat", "fish", "question"),
        ("The dog chases the cat", "dog", "cat", "statement"),
        ("Does the dog chase the cat", "dog", "cat", "question"),
        ("The man sees the bird", "man", "bird", "statement"),
        ("Does the man see the bird", "man", "bird", "question"),
        
        # Tense pairs (same roles, different tense)
        ("The cat eats the fish", "cat", "fish", "present"),
        ("The cat ate the fish", "cat", "fish", "past"),
        ("The dog chases the cat", "dog", "cat", "present"),
        ("The dog chased the cat", "dog", "cat", "past"),
        ("The man sees the bird", "man", "bird", "present"),
        ("The man saw the bird", "man", "bird", "past"),
        
        # Plural pairs (same roles, different number)
        ("The cat eats the fish", "cat", "fish", "singular"),
        ("The cats eat the fish", "cats", "fish", "plural"),
        ("The dog chases the cat", "dog", "cat", "singular"),
        ("The dogs chase the cat", "dogs", "cat", "plural"),
    ]
    
    # --- Collect hidden states ---
    def get_hs(sentences):
        result = {}
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            with torch.no_grad():
                embed = model.get_input_embeddings()(inputs["input_ids"])
                pos_ids = torch.arange(inputs["input_ids"].shape[1], device=device).unsqueeze(0)
                outputs = collect_layer_outputs(model, embed, pos_ids, n_layers)
            for li in range(n_layers):
                key = f"L{li}"
                if key in outputs:
                    h = outputs[key][0, -1, :].numpy()
                    if li not in result:
                        result[li] = []
                    result[li].append(h)
            del outputs, embed
            gc.collect()
        for li in result:
            result[li] = np.array(result[li])
        return result
    
    all_sents = [s[0] for s in sentence_matrix]
    h_all = get_hs(all_sents)
    
    # --- Analysis 1: Role subspace extraction ---
    print("\n--- Analysis 1: Role Subspace Extraction ---")
    
    # Group by agent identity
    from collections import defaultdict
    agent_to_indices = defaultdict(list)
    for i, (sent, agent, patient, stype) in enumerate(sentence_matrix):
        agent_to_indices[agent].append(i)
    
    # For each layer, find the subspace that best separates agents
    for li in [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if li not in h_all:
            continue
        
        h = h_all[li]
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        h = h / np.maximum(norms, 1e-10)
        
        # LDA to find agent-discriminating directions
        agent_labels = []
        for i, (sent, agent, patient, stype) in enumerate(sentence_matrix):
            agent_labels.append(agent)
        
        unique_agents = list(set(agent_labels))
        if len(unique_agents) < 2:
            continue
        
        y = np.array([unique_agents.index(a) for a in agent_labels])
        
        try:
            lda = LinearDiscriminantAnalysis(n_components=min(len(unique_agents)-1, 5))
            z_lda = lda.fit_transform(h, y)
            
            # Cross-val accuracy
            clf = LogisticRegression(max_iter=500, C=1.0, multi_class='ovr')
            scores = cross_val_score(clf, z_lda, y, cv=min(3, len(unique_agents)), scoring='accuracy')
            agent_acc = scores.mean()
        except:
            agent_acc = 0.0
            z_lda = None
        
        print(f"  Layer {li}: Agent LDA accuracy = {agent_acc:.3f}")
    
    # --- Analysis 2: Role invariance across syntax types ---
    print("\n--- Analysis 2: Role Invariance Across Syntax ---")
    print("  Does agent representation stay stable across syntax changes?")
    
    # Pair types: active/passive, statement/question, present/past, singular/plural
    pair_types = {
        "active/passive": [(0,1), (2,3), (4,5), (6,7)],
        "statement/question": [(8,9), (10,11), (12,13)],
        "present/past": [(14,15), (16,17), (18,19)],
        "singular/plural": [(20,21), (22,23)],
    }
    
    print(f"  {'Layer':>6}", end="")
    for ptype in pair_types:
        print(f"  {ptype:>18}", end="")
    print()
    
    for li in [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if li not in h_all:
            continue
        
        h = h_all[li]
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        h = h / np.maximum(norms, 1e-10)
        
        print(f"  {li:>6}", end="")
        
        for ptype, pairs in pair_types.items():
            cos_sims = []
            for ia, ib in pairs:
                if ia < len(h) and ib < len(h):
                    c = 1 - cosine(h[ia], h[ib])
                    cos_sims.append(c)
            
            if cos_sims:
                # These pairs have SAME agent/patient but different syntax
                # High cos = syntax invariance of overall representation
                print(f"  {np.mean(cos_sims):>18.3f}", end="")
            else:
                print(f"  {'N/A':>18}", end="")
        
        print()
    
    # --- Analysis 3: Role-specific invariance ---
    print("\n--- Analysis 3: Role-Specific Invariance (CRITICAL) ---")
    print("  Is the 'agent component' of z invariant to syntax change?")
    
    # Method: Project onto agent-discriminating subspace, then test invariance
    
    for li in [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if li not in h_all:
            continue
        
        h = h_all[li]
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        h = h / np.maximum(norms, 1e-10)
        
        # Get LDA subspace for agents
        agent_labels = [s[1] for s in sentence_matrix]
        unique_agents = list(set(agent_labels))
        y = np.array([unique_agents.index(a) for a in agent_labels])
        
        try:
            lda = LinearDiscriminantAnalysis(n_components=min(len(unique_agents)-1, 5))
            z_lda = lda.fit_transform(h, y)
        except:
            continue
        
        # Now test: for active/passive pairs with same agent,
        # is z_lda similar?
        print(f"\n  Layer {li}:")
        
        for ptype, pairs in pair_types.items():
            lda_sims = []
            full_sims = []
            
            for ia, ib in pairs:
                if ia < len(z_lda) and ib < len(z_lda):
                    # LDA subspace similarity
                    c_lda = 1 - cosine(z_lda[ia], z_lda[ib]) if len(z_lda[ia]) > 1 else 0
                    # Full space similarity
                    c_full = 1 - cosine(h[ia], h[ib])
                    lda_sims.append(c_lda)
                    full_sims.append(c_full)
            
            if lda_sims:
                print(f"    {ptype:>20}: LDA_cos={np.mean(lda_sims):.3f}, Full_cos={np.mean(full_sims):.3f}")
                
                if np.mean(lda_sims) > 0.8:
                    print(f"      ★★★ AGENT SUBSPACE IS SYNTAX-INVARIANT!")
                elif np.mean(lda_sims) > 0.5:
                    print(f"      ⚠️ Partial invariance")
                else:
                    print(f"      ❌ No invariance — syntax changes agent representation")
    
    # --- Analysis 4: Patient subspace invariance ---
    print("\n--- Analysis 4: Patient Subspace Invariance ---")
    
    for li in [n_layers//2, 3*n_layers//4, n_layers-1]:
        if li not in h_all:
            continue
        
        h = h_all[li]
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        h = h / np.maximum(norms, 1e-10)
        
        # Patient LDA
        patient_labels = [s[2] for s in sentence_matrix]
        unique_patients = list(set(patient_labels))
        y_pat = np.array([unique_patients.index(p) for p in patient_labels])
        
        try:
            lda_pat = LinearDiscriminantAnalysis(n_components=min(len(unique_patients)-1, 5))
            z_lda_pat = lda_pat.fit_transform(h, y_pat)
        except:
            continue
        
        print(f"  Layer {li}:")
        for ptype, pairs in pair_types.items():
            lda_sims = []
            for ia, ib in pairs:
                if ia < len(z_lda_pat) and ib < len(z_lda_pat):
                    c = 1 - cosine(z_lda_pat[ia], z_lda_pat[ib]) if len(z_lda_pat[ia]) > 1 else 0
                    lda_sims.append(c)
            
            if lda_sims:
                print(f"    {ptype:>20}: Patient_LDA_cos={np.mean(lda_sims):.3f}")
    
    print("\n" + "="*70)
    print("49C SUMMARY: Structural Invariance")
    print("="*70)
    
    return {}


# ============================================================
# 49D: 非线性组合 — Three-way Binding
# ============================================================
def exp_49d_nonlinear_composition(model, tokenizer, info, model_name):
    """
    核心测试: 组合 ≠ 加法, 组合 = 绑定
    
    设计:
    - 测试3项组合: "red big cat" vs "red" + "big" + "cat"
    - 测量交互项: interaction(A,B) ≠ 0?
    - 测试属性独立性: effect(red) 独立于 effect(big)?
    """
    print("\n" + "="*70)
    print("49D: Non-linear Composition — Three-way Binding")
    print("="*70)
    
    n_layers = info.n_layers
    device = next(model.parameters()).device
    
    # --- Design: Factorial composition ---
    # 3 colors × 3 sizes × 3 animals = 27 sentences (full factorial)
    colors = ["red", "blue", "green"]
    sizes = ["big", "small", "tall"]
    animals = ["cat", "dog", "bird"]
    
    # Build all combinations
    factorial_sentences = []
    for color in colors:
        for size in sizes:
            for animal in animals:
                factorial_sentences.append(f"The {color} {size} {animal} sits")
    
    # Control sentences (single attribute)
    color_only = [f"The {color} one sits" for color in colors]
    size_only = [f"The {size} one sits" for size in sizes]
    animal_only = [f"The {animal} sits" for animal in animals]
    base_sentence = "The one sits"
    
    all_sentences = factorial_sentences + color_only + size_only + animal_only + [base_sentence]
    
    # --- Collect hidden states ---
    def get_hs(sentences):
        result = {}
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            with torch.no_grad():
                embed = model.get_input_embeddings()(inputs["input_ids"])
                pos_ids = torch.arange(inputs["input_ids"].shape[1], device=device).unsqueeze(0)
                outputs = collect_layer_outputs(model, embed, pos_ids, n_layers)
            for li in range(n_layers):
                key = f"L{li}"
                if key in outputs:
                    h = outputs[key][0, -1, :].numpy()
                    if li not in result:
                        result[li] = []
                    result[li].append(h)
            del outputs, embed
            gc.collect()
        for li in result:
            result[li] = np.array(result[li])
        return result
    
    h_all = get_hs(all_sentences)
    
    n_factorial = len(factorial_sentences)  # 27
    n_color = len(color_only)  # 3
    n_size = len(size_only)  # 3
    n_animal = len(animal_only)  # 3
    
    # Indices
    idx_factorial = list(range(n_factorial))
    idx_color = list(range(n_factorial, n_factorial + n_color))
    idx_size = list(range(n_factorial + n_color, n_factorial + n_color + n_size))
    idx_animal = list(range(n_factorial + n_color + n_size, n_factorial + n_color + n_size + n_animal))
    idx_base = n_factorial + n_color + n_size + n_animal
    
    # --- Analysis 1: Additive model fit ---
    print("\n--- Analysis 1: Additive vs Interaction Model ---")
    print("  Model: z(color,size,animal) = base + f_color + f_size + f_animal + ε")
    print("  Question: Is ε small (additive) or large (non-linear binding)?")
    
    for li in [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if li not in h_all:
            continue
        
        h = h_all[li]
        
        # Get mean representations
        h_base = h[idx_base]  # base sentence
        
        # Color effects: mean(z(color,*,*) - z(*,*,*))
        color_effects = {}
        for ci, color in enumerate(colors):
            # All sentences with this color
            color_idxs = [ci * len(sizes) * len(animals) + si * len(animals) + ai
                          for si in range(len(sizes)) for ai in range(len(animals))]
            color_effects[color] = np.mean(h[color_idxs], axis=0) - np.mean(h[idx_factorial], axis=0)
        
        # Size effects
        size_effects = {}
        for si, size in enumerate(sizes):
            size_idxs = [ci * len(sizes) * len(animals) + si * len(animals) + ai
                          for ci in range(len(colors)) for ai in range(len(animals))]
            size_effects[size] = np.mean(h[size_idxs], axis=0) - np.mean(h[idx_factorial], axis=0)
        
        # Animal effects
        animal_effects = {}
        for ai, animal in enumerate(animals):
            animal_idxs = [ci * len(sizes) * len(animals) + si * len(animals) + ai
                           for ci in range(len(colors)) for si in range(len(sizes))]
            animal_effects[animal] = np.mean(h[animal_idxs], axis=0) - np.mean(h[idx_factorial], axis=0)
        
        # Predict: z_pred = grand_mean + color_effect + size_effect + animal_effect
        grand_mean = np.mean(h[idx_factorial], axis=0)
        
        residuals = []
        for ci, color in enumerate(colors):
            for si, size in enumerate(sizes):
                for ai, animal in enumerate(animals):
                    idx = ci * len(sizes) * len(animals) + si * len(animals) + ai
                    z_pred = grand_mean + color_effects[color] + size_effects[size] + animal_effects[animal]
                    residual = np.linalg.norm(h[idx] - z_pred) / max(np.linalg.norm(h[idx]), 1e-10)
                    residuals.append(residual)
        
        # R² of additive model
        ss_res = 0
        ss_tot = 0
        for ci, color in enumerate(colors):
            for si, size in enumerate(sizes):
                for ai, animal in enumerate(animals):
                    idx = ci * len(sizes) * len(animals) + si * len(animals) + ai
                    z_pred = grand_mean + color_effects[color] + size_effects[size] + animal_effects[animal]
                    ss_res += np.sum((h[idx] - z_pred)**2)
                    ss_tot += np.sum((h[idx] - grand_mean)**2)
        
        r2_additive = 1 - ss_res / max(ss_tot, 1e-20)
        
        print(f"  Layer {li}: Additive model R² = {r2_additive:.3f}, mean rel_residual = {np.mean(residuals):.3f}")
    
    # --- Analysis 2: Two-way interaction terms ---
    print("\n--- Analysis 2: Interaction Terms ---")
    print("  color×size, color×animal, size×animal interactions")
    
    for li in [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if li not in h_all:
            continue
        
        h = h_all[li]
        
        # Compute interaction via ANOVA-like decomposition
        # color×size interaction: 
        # Does the effect of color depend on size?
        
        interactions = {}
        
        for factor1, factor2, n1, n2, names1, names2 in [
            ("color", "size", len(colors), len(sizes), colors, sizes),
            ("color", "animal", len(colors), len(animals), colors, animals),
            ("size", "animal", len(sizes), len(animals), sizes, animals),
        ]:
            # Compute 2-way interaction strength
            # interaction = z(a,b) - z(a,.) - z(.,b) + z(.,.)
            interaction_norms = []
            
            for i1 in range(n1):
                for i2 in range(n2):
                    # z(factor1=i1, factor2=i2) averaged over third factor
                    if factor1 == "color" and factor2 == "size":
                        # Average over animals
                        idxs = [i1 * len(sizes) * len(animals) + i2 * len(animals) + ai
                                for ai in range(len(animals))]
                    elif factor1 == "color" and factor2 == "animal":
                        idxs = [i1 * len(sizes) * len(animals) + si * len(animals) + i2
                                for si in range(len(sizes))]
                    else:  # size × animal
                        idxs = [ci * len(sizes) * len(animals) + i1 * len(animals) + i2
                                for ci in range(len(colors))]
                    
                    z_12 = np.mean(h[idxs], axis=0)
                    
                    # z(factor1=i1, *) averaged
                    if factor1 == "color":
                        idxs_1 = [i1 * len(sizes) * len(animals) + si * len(animals) + ai
                                   for si in range(len(sizes)) for ai in range(len(animals))]
                    else:  # size
                        idxs_1 = [ci * len(sizes) * len(animals) + i1 * len(animals) + ai
                                   for ci in range(len(colors)) for ai in range(len(animals))]
                    z_1 = np.mean(h[idxs_1], axis=0)
                    
                    # z(*, factor2=i2) averaged
                    if factor2 == "size":
                        idxs_2 = [ci * len(sizes) * len(animals) + i2 * len(animals) + ai
                                   for ci in range(len(colors)) for ai in range(len(animals))]
                    elif factor2 == "animal":
                        idxs_2 = [ci * len(sizes) * len(animals) + si * len(animals) + i2
                                   for ci in range(len(colors)) for si in range(len(sizes))]
                    else:
                        idxs_2 = idxs  # fallback
                    
                    z_2 = np.mean(h[idxs_2], axis=0)
                    
                    z_grand = np.mean(h[idx_factorial], axis=0)
                    
                    # Interaction
                    interaction = z_12 - z_1 - z_2 + z_grand
                    interaction_norms.append(np.linalg.norm(interaction))
            
            interactions[f"{factor1}×{factor2}"] = np.mean(interaction_norms)
        
        # Also compute main effect norms for comparison
        main_norms = {}
        for factor, n_f, names, idxs_fn in [
            ("color", len(colors), colors, 
             lambda i: [i * len(sizes) * len(animals) + si * len(animals) + ai
                        for si in range(len(sizes)) for ai in range(len(animals))]),
            ("size", len(sizes), sizes,
             lambda i: [ci * len(sizes) * len(animals) + i * len(animals) + ai
                        for ci in range(len(colors)) for ai in range(len(animals))]),
            ("animal", len(animals), animals,
             lambda i: [ci * len(sizes) * len(animals) + si * len(animals) + i
                        for ci in range(len(colors)) for si in range(len(sizes))]),
        ]:
            effect_norms = []
            for i in range(n_f):
                idxs = idxs_fn(i)
                effect = np.mean(h[idxs], axis=0) - np.mean(h[idx_factorial], axis=0)
                effect_norms.append(np.linalg.norm(effect))
            main_norms[factor] = np.mean(effect_norms)
        
        print(f"  Layer {li}:")
        print(f"    Main effects:  color={main_norms['color']:.3f}, size={main_norms['size']:.3f}, animal={main_norms['animal']:.3f}")
        print(f"    Interactions:  color×size={interactions['color×size']:.3f}, "
              f"color×animal={interactions['color×animal']:.3f}, size×animal={interactions['size×animal']:.3f}")
        
        # Interaction / Main ratio
        for ix_name, m_name in [("color×size", "color"), ("color×animal", "color"), ("size×animal", "size")]:
            ratio = interactions[ix_name] / max(main_norms[m_name], 1e-10)
            print(f"    {ix_name}/{m_name} ratio = {ratio:.3f} {'(significant interaction!)' if ratio > 0.3 else ''}")
    
    # --- Analysis 3: Attribute independence test ---
    print("\n--- Analysis 3: Attribute Independence ---")
    print("  Is the effect of 'red' the same for 'cat' and 'dog'?")
    
    for li in [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if li not in h_all:
            continue
        
        h = h_all[li]
        
        # Compare: effect of red on cat vs effect of red on dog
        # "red big cat" (idx 0*9+0*3+0=0) vs "big cat" is implicit
        # We need: z(red X) - z(X) for different X
        
        # Use full factorial to compute color effects per animal
        for color_idx, color in enumerate(colors):
            effects_per_animal = []
            for animal_idx, animal in enumerate(animals):
                # z(color, *, animal) - z(*, *, animal)
                with_color = [h[color_idx * len(sizes) * len(animals) + si * len(animals) + animal_idx]
                              for si in range(len(sizes))]
                without_color = []
                for ci in range(len(colors)):
                    if ci != color_idx:
                        for si in range(len(sizes)):
                            without_color.append(h[ci * len(sizes) * len(animals) + si * len(animals) + animal_idx])
                
                effect = np.mean(with_color, axis=0) - np.mean(without_color, axis=0)
                effects_per_animal.append(effect)
            
            # Are these effects similar across animals?
            cos_sims = []
            for i in range(len(effects_per_animal)):
                for j in range(i+1, len(effects_per_animal)):
                    n1 = np.linalg.norm(effects_per_animal[i])
                    n2 = np.linalg.norm(effects_per_animal[j])
                    if n1 > 1e-10 and n2 > 1e-10:
                        c = 1 - cosine(effects_per_animal[i], effects_per_animal[j])
                        cos_sims.append(c)
            
            if cos_sims:
                print(f"  Layer {li}, {color}: Cross-animal effect cos = {np.mean(cos_sims):.3f}")
    
    print("\n" + "="*70)
    print("49D SUMMARY: Non-linear Composition")
    print("="*70)
    
    return {}


# ============================================================
# 49E: 操作函数恢复 — Operation Function Recovery
# ============================================================
def exp_49e_operation_functions(model, tokenizer, info, model_name):
    """
    核心升级: Δz = f(z) 而不是 Δz = A z
    
    方法:
    - 收集 (z_l, z_{l+1}) 对
    - 拟合非线性函数 f: z → Δz
    - 测试泛化能力
    - 尝试分解: f = Σ c_i · f_i (有限操作集)
    """
    print("\n" + "="*70)
    print("49E: Operation Function Recovery — Δz = f(z)")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    
    # Diverse sentences for function fitting
    sentences = [
        "The cat sits on the mat.",
        "A big red dog runs quickly.",
        "Small birds fly in the sky.",
        "The old man walks home slowly.",
        "Beautiful flowers grow in the garden.",
        "Young children play with toys.",
        "The white horse runs fast.",
        "Dark clouds cover the sky.",
        "The small fish swims upstream.",
        "A tall tree stands alone.",
        "The quick fox jumps high.",
        "Warm rain falls softly down.",
        "The brave soldier fights hard.",
        "Old books rest on shelves.",
        "Bright stars shine at night.",
        "The green frog jumps far.",
        "Heavy snow falls quietly.",
        "The tiny ant carries food.",
        "Sweet honey tastes very good.",
        "The loud bell rings daily.",
        "Cats and dogs play together.",
        "The river flows through the valley.",
        "A gentle breeze blows softly.",
        "The sun rises in the east.",
        "Music fills the quiet room.",
        "The student reads the book.",
        "The teacher writes on the board.",
        "A chef cooks delicious food.",
        "The artist paints a picture.",
        "The doctor helps the patient.",
    ]
    
    # --- Collect hidden states ---
    def get_hs(sentences):
        result = {}
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            with torch.no_grad():
                embed = model.get_input_embeddings()(inputs["input_ids"])
                pos_ids = torch.arange(inputs["input_ids"].shape[1], device=device).unsqueeze(0)
                outputs = collect_layer_outputs(model, embed, pos_ids, n_layers)
            for li in range(n_layers):
                key = f"L{li}"
                if key in outputs:
                    h = outputs[key][0, -1, :].numpy()
                    if li not in result:
                        result[li] = []
                    result[li].append(h)
            del outputs, embed
            gc.collect()
        for li in result:
            result[li] = np.array(result[li])
        return result
    
    h_all = get_hs(sentences)
    N = len(sentences)
    
    # --- Analysis 1: Linear vs Quadratic fit ---
    print("\n--- Analysis 1: Linear vs Non-linear Function Fit ---")
    print("  Linear:      Δz = A z + b")
    print("  Quadratic:   Δz = A z + b + Σ z_i z_j w_ij")
    print("  Question: Does quadratic improve over linear?")
    
    key_layers = [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    for li in range(1, n_layers):
        if li not in h_all or (li-1) not in h_all:
            continue
        if li not in key_layers:
            continue
        
        z_prev = h_all[li-1]  # [N, d]
        z_next = h_all[li]    # [N, d]
        dz = z_next - z_prev  # [N, d]
        
        # Normalize
        z_prev_norm = np.linalg.norm(z_prev, axis=1, keepdims=True)
        z_prev_n = z_prev / np.maximum(z_prev_norm, 1e-10)
        dz_norm = np.linalg.norm(dz, axis=1, keepdims=True)
        dz_n = dz / np.maximum(dz_norm, 1e-10)
        
        # PCA on z_prev to reduce dimensionality
        pca_z = PCA(n_components=min(40, N-1))
        z_pca = pca_z.fit_transform(z_prev_n)
        pca_dz = PCA(n_components=min(40, N-1))
        dz_pca = pca_dz.fit_transform(dz_n)
        
        # Split train/test
        n_train = int(0.7 * N)
        idx = np.random.RandomState(42).permutation(N)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        
        z_train = z_pca[train_idx]
        dz_train = dz_pca[train_idx]
        z_test = z_pca[test_idx]
        dz_test = dz_pca[test_idx]
        
        # Linear fit
        ridge_lin = Ridge(alpha=1.0)
        ridge_lin.fit(z_train, dz_train)
        dz_pred_lin = ridge_lin.predict(z_test)
        r2_lin = 1 - np.sum((dz_test - dz_pred_lin)**2) / max(np.sum((dz_test - np.mean(dz_test, axis=0))**2), 1e-20)
        
        # Quadratic fit: add z_i * z_j terms (top k dimensions)
        k_quad = min(10, z_pca.shape[1])
        z_train_quad = np.hstack([z_train, z_train[:, :k_quad]**2])
        z_test_quad = np.hstack([z_test, z_test[:, :k_quad]**2])
        
        ridge_quad = Ridge(alpha=1.0)
        ridge_quad.fit(z_train_quad, dz_train)
        dz_pred_quad = ridge_quad.predict(z_test_quad)
        r2_quad = 1 - np.sum((dz_test - dz_pred_quad)**2) / max(np.sum((dz_test - np.mean(dz_test, axis=0))**2), 1e-20)
        
        # Cosine similarity of predictions
        cos_lin = []
        cos_quad = []
        for i in range(len(dz_test)):
            n_test = np.linalg.norm(dz_test[i])
            if n_test < 1e-10:
                continue
            cl = 1 - cosine(dz_test[i], dz_pred_lin[i]) if np.linalg.norm(dz_pred_lin[i]) > 1e-10 else 0
            cq = 1 - cosine(dz_test[i], dz_pred_quad[i]) if np.linalg.norm(dz_pred_quad[i]) > 1e-10 else 0
            cos_lin.append(cl)
            cos_quad.append(cq)
        
        print(f"  Layer {li}:")
        print(f"    Linear:    R²={r2_lin:.3f}, cos={np.mean(cos_lin):.3f}")
        print(f"    Quadratic: R²={r2_quad:.3f}, cos={np.mean(cos_quad):.3f}")
        print(f"    Improvement: ΔR²={r2_quad-r2_lin:.3f} {'★★★ SIGNIFICANT NONLINEARITY!' if r2_quad-r2_lin > 0.1 else '⚠️ Marginal' if r2_quad-r2_lin > 0.03 else 'Negligible'}")
    
    # --- Analysis 2: Function conditioning on semantic class ---
    print("\n--- Analysis 2: Context-Dependent Operations ---")
    print("  Does f(z) depend on sentence type?")
    
    # Simple test: split by sentence characteristic
    # Use short vs long sentences as proxy
    short_sents = [s for s in sentences if len(s.split()) <= 6]
    long_sents = [s for s in sentences if len(s.split()) > 6]
    
    print(f"  Short sentences: {len(short_sents)}, Long sentences: {len(long_sents)}")
    print(f"  (Using sentence length as proxy for structural complexity)")
    
    # --- Analysis 3: Residual vs state-dependent dynamics ---
    print("\n--- Analysis 3: State-Dependent vs Constant Dynamics ---")
    print("  Is Δz ≈ α·z (norm scaling) or truly state-dependent?")
    
    for li in key_layers:
        if li not in h_all or (li-1) not in h_all:
            continue
        
        z_prev = h_all[li-1]
        z_next = h_all[li]
        dz = z_next - z_prev
        
        # Test 1: cos(Δz, z)
        cos_dz_z = []
        for i in range(N):
            dn = np.linalg.norm(dz[i])
            zn = np.linalg.norm(z_prev[i])
            if dn > 1e-10 and zn > 1e-10:
                cos_dz_z.append(np.dot(dz[i], z_prev[i]) / (dn * zn))
        
        # Test 2: Is ||Δz|| proportional to ||z||?
        dz_norms = np.linalg.norm(dz, axis=1)
        z_norms = np.linalg.norm(z_prev, axis=1)
        corr = np.corrcoef(dz_norms, z_norms)[0, 1]
        
        # Test 3: Remove projection of z from Δz, what remains?
        dz_perp = dz.copy()
        for i in range(N):
            zn = np.linalg.norm(z_prev[i])
            if zn > 1e-10:
                proj = np.dot(dz[i], z_prev[i]) / (zn**2) * z_prev[i]
                dz_perp[i] = dz[i] - proj
        
        perp_fraction = np.mean(np.linalg.norm(dz_perp, axis=1)) / max(np.mean(dz_norms), 1e-10)
        
        print(f"  Layer {li}:")
        print(f"    cos(Δz, z): {np.mean(cos_dz_z):.3f}")
        print(f"    corr(||Δz||, ||z||): {corr:.3f}")
        print(f"    Perpendicular fraction: {perp_fraction:.3f}")
        
        if perp_fraction > 0.5:
            print(f"    → ★ Δz is mostly PERPENDICULAR to z → not just norm scaling!")
        elif perp_fraction > 0.3:
            print(f"    → Mixed: both radial and tangential components")
        else:
            print(f"    → Δz is mostly parallel to z → norm scaling dominates")
    
    # --- Analysis 4: Kernel regression for non-linear fit ---
    print("\n--- Analysis 4: Kernel Regression (Non-parametric f) ---")
    from sklearn.kernel_ridge import KernelRidge
    
    for li in key_layers:
        if li not in h_all or (li-1) not in h_all:
            continue
        
        z_prev = h_all[li-1]
        z_next = h_all[li]
        dz = z_next - z_prev
        
        # Use PCA to reduce dimension
        pca = PCA(n_components=min(20, N-1))
        z_pca = pca.fit_transform(z_prev)
        
        # PCA on dz
        pca_dz = PCA(n_components=min(20, N-1))
        dz_pca = pca_dz.fit_transform(dz)
        
        n_train = int(0.7 * N)
        idx = np.random.RandomState(42).permutation(N)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        
        z_train = z_pca[train_idx]
        dz_train = dz_pca[train_idx]
        z_test = z_pca[test_idx]
        dz_test = dz_pca[test_idx]
        
        # Linear baseline
        ridge = Ridge(alpha=1.0)
        ridge.fit(z_train, dz_train)
        dz_pred_ridge = ridge.predict(z_test)
        r2_ridge = 1 - np.sum((dz_test - dz_pred_ridge)**2) / max(np.sum((dz_test - np.mean(dz_test, axis=0))**2), 1e-20)
        
        # Kernel Ridge (RBF)
        try:
            krr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
            krr.fit(z_train, dz_train)
            dz_pred_krr = krr.predict(z_test)
            r2_krr = 1 - np.sum((dz_test - dz_pred_krr)**2) / max(np.sum((dz_test - np.mean(dz_test, axis=0))**2), 1e-20)
        except:
            r2_krr = r2_ridge
        
        print(f"  Layer {li}: Linear R²={r2_ridge:.3f}, KernelRidge R²={r2_krr:.3f}, Δ={r2_krr-r2_ridge:.3f}")
    
    print("\n" + "="*70)
    print("49E SUMMARY: Operation Function Recovery")
    print("="*70)
    
    return {}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 49: Structure Recovery")
    parser.add_argument("--model", type=str, default="deepseek7b",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                       help="Experiment: 0=all, 1=49A, 2=49B, 3=49C, 4=49D, 5=49E")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase 49: Structure Recovery — SLT验证")
    print(f"Model: {args.model}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    
    print(f"  Model: {info.name}, Layers: {info.n_layers}, d_model: {info.d_model}")
    
    results = {}
    
    try:
        if args.exp in [0, 1]:
            results["49A"] = exp_49a_remove_norm_artifacts(model, tokenizer, info, args.model)
        
        if args.exp in [0, 2]:
            results["49B"] = exp_49b_role_binding(model, tokenizer, info, args.model)
        
        if args.exp in [0, 3]:
            results["49C"] = exp_49c_structural_invariance(model, tokenizer, info, args.model)
        
        if args.exp in [0, 4]:
            results["49D"] = exp_49d_nonlinear_composition(model, tokenizer, info, args.model)
        
        if args.exp in [0, 5]:
            results["49E"] = exp_49e_operation_functions(model, tokenizer, info, args.model)
    
    finally:
        release_model(model)
    
    print(f"\n{'='*70}")
    print("Phase 49 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
