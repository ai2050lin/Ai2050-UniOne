"""
CCL-E(250.5): 正交补几何 + LayerNorm因果 + 因果方向流形
============================================================
核心问题: "语法正交补空间"是确定性几何结构还是统计假象?
用户核心洞察: 大脑类似最小作用量原理, 随机可能性极低

三合一实验:
  Exp1 (Phase 4G): 正交补空间的几何结构
    → 投影到PCA正交补后, 不同语法角色的点如何分布?
    → 它们是低维流形还是高维散点?
    → 如果是流形 → 语言有确定性几何结构 (支持最小作用量)

  Exp2 (Phase 4H): LayerNorm前后的语法信息
    → LN前后的语法信息量对比
    → LN是瓶颈还是自然压缩?
    → 如果LN前语法信息更多 → LN是瓶颈 → 最小作用量证据

  Exp3 (Phase 4I): 因果方向流形
    → 收集多组nsubj/dobj的因果方向
    → 它们是否形成低维子空间?
    → 如果是 → 因果方向有结构 → 可以参数化语法角色
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 扩展数据集: 更多的nsubj/dobj句子对, 用于4I =====
DEP_SENTENCES = {
    "nsubj": [
        "The cat sat on the mat",
        "The dog ran through the park",
        "The bird sang a beautiful song",
        "The child played with the toys",
        "The student read the textbook",
        "The teacher explained the lesson",
        "The scientist discovered the formula",
        "The writer published the novel",
        "The cat slept on the sofa",
        "The dog barked at the stranger",
        "The bird flew over the house",
        "The child smiled at the teacher",
        "The student answered the question",
        "The teacher graded the papers",
        "The scientist published the paper",
        "The writer finished the manuscript",
    ],
    "dobj": [
        "She chased the cat away",
        "He found the dog outside",
        "They watched the bird closely",
        "We helped the child today",
        "I praised the student loudly",
        "You thanked the teacher warmly",
        "He remembered the scientist well",
        "She admired the writer greatly",
        "They fed the cat carefully",
        "I walked the dog slowly",
        "We rescued the bird quickly",
        "She comforted the child gently",
        "He encouraged the student daily",
        "They respected the teacher deeply",
        "We honored the scientist proudly",
        "I supported the writer fully",
    ],
}

DEP_TARGET_WORDS = {
    "nsubj": ["cat", "dog", "bird", "child", "student", "teacher", "scientist", "writer",
              "cat", "dog", "bird", "child", "student", "teacher", "scientist", "writer"],
    "dobj": ["cat", "dog", "bird", "child", "student", "teacher", "scientist", "writer",
             "cat", "dog", "bird", "child", "student", "teacher", "scientist", "writer"],
}


def find_token_index(tokens, word):
    word_lower = word.lower()
    for i, tok in enumerate(tokens):
        if word_lower in tok.lower() or tok.lower().startswith(word_lower[:3]):
            return i
    return None


def get_last_layer_hidden(model, tokenizer, device, sentence):
    """获取最后层的hidden states"""
    layers = get_layers(model)
    last_layer = layers[-1]
    
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float()
        else:
            captured['h'] = output.detach().float()
    
    h_handle = last_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output = model(**toks)
        base_logits = output.logits.detach().float()
    
    h_handle.remove()
    
    if 'h' not in captured:
        return None, None, toks
    
    return captured['h'], base_logits, toks


def compute_logits_from_hidden(model, hidden_states):
    """从hidden states通过final norm + lm_head计算logits"""
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        normed = model.model.norm(hidden_states.to(model.model.norm.weight.device).to(model.model.norm.weight.dtype))
    else:
        normed = hidden_states
    
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(normed.to(model.lm_head.weight.dtype))
    else:
        logits = normed
    
    return logits.detach().float()


def collect_dep_representations(model, tokenizer, device, sentences, dep_type, target_words):
    """Collect hidden state representations at the last layer for dep positions."""
    all_reps = []
    
    for sent, target in zip(sentences, target_words):
        hidden_states, base_logits, toks = get_last_layer_hidden(model, tokenizer, device, sent)
        if hidden_states is None:
            continue
        
        input_ids = toks.input_ids
        tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens, target)
        if dep_idx is None:
            continue
        
        h_dep = hidden_states[0, dep_idx, :].cpu().numpy()
        all_reps.append({
            'h': h_dep,
            'dep_type': dep_type,
            'sentence': sent,
            'target': target,
            'dep_idx': dep_idx,
            'hidden_states': hidden_states,
            'base_logits': base_logits,
        })
    
    return all_reps


def collect_multilayer_representations(model, tokenizer, device, sentences, dep_type, target_words):
    """收集多层hidden states (LN前和LN后)"""
    layers = get_layers(model)
    n_layers = len(layers)
    
    # 采样层: 前1/4, 中间, 后1/4, 最后
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    
    all_layer_reps = {li: [] for li in sample_layers}
    
    for sent, target in zip(sentences, target_words):
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens, target)
        if dep_idx is None:
            continue
        
        # Hook: 收集LN前的输出(attention/MLP输出)和LN后的输出
        captured = {}
        hooks = []
        
        for li in sample_layers:
            layer = layers[li]
            
            # Hook在层输出 (LN后)
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            
            hooks.append(layer.register_forward_hook(make_hook(f'L{li}_after')))
            
            # Hook在self_attn输出 (LN前, 即residual stream中的信号)
            # 注意: 这里我们在input_layernorm之前无法直接hook
            # 但我们可以hook self_attn的输出, 这是LN+attn后的结果
            # 更精确: 我们需要hook residual connection
            # 简化方案: 比较层输出 vs 层输入
            
        with torch.no_grad():
            _ = model(**toks)
        
        for h in hooks:
            h.remove()
        
        for li in sample_layers:
            key = f'L{li}_after'
            if key in captured:
                h = captured[key][0, dep_idx, :].numpy()
                all_layer_reps[li].append({
                    'h': h,
                    'dep_type': dep_type,
                    'sentence': sent,
                })
    
    return all_layer_reps, sample_layers


# ================================================================
# Exp1: 正交补空间的几何结构 (Phase 4G)
# ================================================================

def exp1_orthogonal_complement_geometry(model, tokenizer, device, model_info):
    """
    ★★★★★ 正交补空间几何结构
    核心假设: 如果语言背后是最小作用量原理, 语法在正交补中应该形成低维流形
    
    方法:
    1. PCA分解: h = h_PCA + h_orth (h_orth在PCA正交补中)
    2. 分析h_orth的几何:
       a. h_orth的有效秩 (是否有低维结构)
       b. h_orth中nsubj/dobj的LDA分类准确率
       c. h_orth中的聚类结构 (是否形成流形)
       d. h_orth的范数与h_PCA的范数比
    3. 对比随机基线: 随机方向的h_orth是否也有结构?
    """
    print("\n" + "="*70)
    print("Exp1 (Phase 4G): 正交补空间几何结构")
    print("="*70)
    
    # 1. 收集所有dep表示
    print("\n[1] Collecting dep representations...")
    all_reps = []
    for dep_type in ["nsubj", "dobj"]:
        reps = collect_dep_representations(
            model, tokenizer, device,
            DEP_SENTENCES[dep_type][:8],
            dep_type,
            DEP_TARGET_WORDS[dep_type][:8],
        )
        all_reps.extend(reps)
        print(f"  {dep_type}: {len(reps)} representations")
    
    if len(all_reps) < 6:
        print("  ERROR: Too few representations")
        return None
    
    H = np.array([r['h'] for r in all_reps])
    y = np.array([0 if r['dep_type'] == 'nsubj' else 1 for r in all_reps])
    
    nsubj_mask = y == 0
    dobj_mask = y == 1
    
    # 2. PCA分解
    print("\n[2] PCA decomposition...")
    pca = PCA()
    pca.fit(H)
    var_ratio = pca.explained_variance_ratio_
    
    # 找到累积方差达到阈值的维度
    cumvar = np.cumsum(var_ratio)
    k_95 = np.searchsorted(cumvar, 0.95) + 1
    k_99 = np.searchsorted(cumvar, 0.99) + 1
    k_999 = np.searchsorted(cumvar, 0.999) + 1
    
    print(f"  k_95={k_95}, k_99={k_99}, k_999={k_999}")
    print(f"  PC1: {var_ratio[0]*100:.2f}%, top5: {np.sum(var_ratio[:5])*100:.2f}%")
    
    # 3. ★★★ 投影到PCA主空间和正交补
    print("\n[3] Projecting to PCA subspace and orthogonal complement...")
    
    # 使用不同k值, 分析正交补中有多少语法信息
    k_list = [k_95, k_99, min(k_999, H.shape[1] // 2)]
    k_list = sorted(set(k_list))
    
    results = {
        "model": model_info.name,
        "pca_info": {
            "pc1": float(var_ratio[0]),
            "top5": float(np.sum(var_ratio[:5])),
            "k_95": int(k_95),
            "k_99": int(k_99),
            "k_999": int(k_999),
            "d_model": H.shape[1],
        },
        "orthogonal_analysis": [],
    }
    
    for k in k_list:
        print(f"\n  --- k={k} (PCA主空间维度) ---")
        
        # PCA主空间基
        U_k = pca.components_[:k].T  # [d_model, k]
        
        # ★★★ 关键: 投影到正交补
        # h_PCA = U_k @ U_k^T @ h  (在PCA主空间中的投影)
        # h_orth = h - h_PCA  (在正交补中的投影)
        
        H_centered = H - pca.mean_
        H_PCA = H_centered @ U_k @ U_k.T + pca.mean_  # [N, d_model]
        H_orth = H - H_PCA  # [N, d_model]
        
        # 范数比
        pca_norms = np.linalg.norm(H_PCA - pca.mean_, axis=1)
        orth_norms = np.linalg.norm(H_orth - pca.mean_, axis=1)
        norm_ratio = np.mean(orth_norms) / max(np.mean(pca_norms), 1e-10)
        
        print(f"  PCA norm mean: {np.mean(pca_norms):.4f}")
        print(f"  Orth norm mean: {np.mean(orth_norms):.4f}")
        print(f"  Orth/PCA ratio: {norm_ratio:.6f}")
        
        # 4. ★★★ 正交补中的分类能力
        # LDA on h_orth
        lda_orth = LinearDiscriminantAnalysis(n_components=1)
        try:
            lda_orth.fit(H_orth, y)
            score_orth = lda_orth.score(H_orth, y)
        except:
            score_orth = 0.0
        
        # LDA on h_PCA
        lda_pca = LinearDiscriminantAnalysis(n_components=1)
        try:
            lda_pca.fit(H_PCA, y)
            score_pca = lda_pca.score(H_PCA, y)
        except:
            score_pca = 0.0
        
        # LDA on full h
        lda_full = LinearDiscriminantAnalysis(n_components=1)
        lda_full.fit(H, y)
        score_full = lda_full.score(H, y)
        
        print(f"  LDA accuracy: full={score_full:.3f}, PCA={score_pca:.3f}, Orth={score_orth:.3f}")
        
        # 5. ★★★ 正交补的有效秩 (是否有低维结构)
        pca_orth = PCA()
        pca_orth.fit(H_orth)
        var_orth = pca_orth.explained_variance_ratio_
        
        # 有效秩
        var_probs = var_orth / np.sum(var_orth)
        var_probs = var_probs[var_probs > 1e-15]
        entropy_orth = -np.sum(var_probs * np.log(var_probs))
        eff_rank_orth = np.exp(entropy_orth)
        
        # 正交补PC1的方差比
        orth_pc1 = float(var_orth[0]) if len(var_orth) > 0 else 0.0
        orth_top5 = float(np.sum(var_orth[:5])) if len(var_orth) >= 5 else float(np.sum(var_orth))
        
        print(f"  Orth eff_rank: {eff_rank_orth:.1f}")
        print(f"  Orth PC1: {orth_pc1*100:.4f}%, top5: {orth_top5*100:.4f}%")
        
        # 6. ★★★ 正交补中nsubj/dobj的质心距离 vs 簇内距离
        centroid_nsubj_orth = np.mean(H_orth[nsubj_mask], axis=0)
        centroid_dobj_orth = np.mean(H_orth[dobj_mask], axis=0)
        
        inter_dist = np.linalg.norm(centroid_dobj_orth - centroid_nsubj_orth)
        intra_nsubj = np.mean(np.linalg.norm(H_orth[nsubj_mask] - centroid_nsubj_orth, axis=1))
        intra_dobj = np.mean(np.linalg.norm(H_orth[dobj_mask] - centroid_dobj_orth, axis=1))
        intra_mean = (intra_nsubj + intra_dobj) / 2
        
        fisher_ratio = inter_dist / max(intra_mean, 1e-10)
        
        print(f"  Inter-centroid dist: {inter_dist:.4f}")
        print(f"  Intra-cluster dist: nsubj={intra_nsubj:.4f}, dobj={intra_dobj:.4f}")
        print(f"  Fisher ratio: {fisher_ratio:.4f}")
        
        # 7. ★★★ 随机基线: 随机k维子空间的正交补
        # 如果正交补的分类能力显著高于随机 → 语法信息确实在正交补中
        n_random = 10
        random_scores = []
        for _ in range(n_random):
            rand_U = np.random.randn(H.shape[1], k)
            rand_U, _ = np.linalg.qr(rand_U)  # 正交化
            H_rand_orth = H - (H - pca.mean_) @ rand_U @ rand_U.T - pca.mean_
            try:
                lda_rand = LinearDiscriminantAnalysis(n_components=1)
                lda_rand.fit(H_rand_orth, y)
                random_scores.append(lda_rand.score(H_rand_orth, y))
            except:
                random_scores.append(0.5)
        
        random_score_mean = np.mean(random_scores)
        print(f"  Random baseline LDA: {random_score_mean:.3f}")
        print(f"  ★ Orth / Random lift: {score_orth / max(random_score_mean, 0.01):.2f}x")
        
        # 8. ★★★ 正交补中因果方向的投影
        # 收集迁移方向在正交补中的投影比
        migration_dir = centroid_dobj_orth - centroid_nsubj_orth
        migration_norm = np.linalg.norm(migration_dir)
        
        if migration_norm > 1e-10:
            migration_in_orth = np.linalg.norm(
                (migration_dir / migration_norm) @ (np.eye(H.shape[1]) - U_k @ U_k.T)
            )
        else:
            migration_in_orth = 0.0
        
        print(f"  Migration dir in orth complement: {migration_in_orth:.4f}")
        
        k_result = {
            "k": int(k),
            "norm_ratio": float(norm_ratio),
            "lda_accuracy": {"full": float(score_full), "pca": float(score_pca), "orth": float(score_orth)},
            "random_baseline": float(random_score_mean),
            "orth_lift": float(score_orth / max(random_score_mean, 0.01)),
            "eff_rank_orth": float(eff_rank_orth),
            "orth_pc1": orth_pc1,
            "orth_top5": orth_top5,
            "fisher_ratio": float(fisher_ratio),
            "inter_centroid": float(inter_dist),
            "intra_nsubj": float(intra_nsubj),
            "intra_dobj": float(intra_dobj),
            "migration_in_orth": float(migration_in_orth),
        }
        results["orthogonal_analysis"].append(k_result)
    
    # 9. ★★★ 最关键测试: 正交补中的扰动vs PCA主空间中的扰动
    # 如果正交补中的扰动能改变语法 → 语法确实在正交补中
    print("\n[4] Perturbation test: Orth vs PCA subspace...")
    
    k_main = k_99
    U_main = pca.components_[:k_main].T  # [d_model, k_main]
    # 正交补基: 使用SVD获得
    U_orth = pca.components_[k_main:].T  # [d_model, d_model - k_main]
    
    rep0 = all_reps[0]
    hidden_states = rep0['hidden_states']
    dep_idx = rep0['dep_idx']
    base_logits = rep0['base_logits']
    base_probs = torch.softmax(base_logits[0, dep_idx], dim=-1).cpu().numpy()
    
    h_orig = rep0['h']
    h_norm = np.linalg.norm(h_orig)
    
    pert_results = {}
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        # 扰动在PCA主空间方向
        # 找到在PCA主空间中的随机方向
        rand_pca_dir = np.random.randn(k_main)
        rand_pca_dir = rand_pca_dir / np.linalg.norm(rand_pca_dir)
        pert_pca = alpha * h_norm * (U_main @ rand_pca_dir)
        
        # 扰动在正交补方向
        n_orth_dirs = min(20, U_orth.shape[1])
        rand_orth_dir = np.random.randn(n_orth_dirs)
        rand_orth_dir = rand_orth_dir / np.linalg.norm(rand_orth_dir)
        pert_orth = alpha * h_norm * (U_orth[:, :n_orth_dirs] @ rand_orth_dir)
        
        # 应用扰动, 计算KL
        for pert_name, pert_vec in [("pca_random", pert_pca), ("orth_random", pert_orth)]:
            pert_t = torch.tensor(pert_vec, dtype=hidden_states.dtype, device=hidden_states.device)
            mod_h = hidden_states.clone()
            mod_h[0, dep_idx, :] += pert_t
            pert_logits = compute_logits_from_hidden(model, mod_h)
            pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
            kl = float(np.sum(base_probs * np.log(base_probs / (pert_probs + 1e-10) + 1e-10)))
            
            key = f"{pert_name}_a{alpha}"
            pert_results[key] = kl
        
        # ★★★ 关键: 扰动沿LDA方向(主要在正交补中) vs 扰动沿PC1方向(主要在PCA中)
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(H, y)
        lda_dir = lda.coef_[0]
        lda_dir = lda_dir / np.linalg.norm(lda_dir)
        
        pc1_dir = pca.components_[0]
        
        for dir_name, dir_vec in [("lda", lda_dir), ("pc1", pc1_dir)]:
            pert_vec = alpha * h_norm * dir_vec
            pert_t = torch.tensor(pert_vec, dtype=hidden_states.dtype, device=hidden_states.device)
            mod_h = hidden_states.clone()
            mod_h[0, dep_idx, :] += pert_t
            pert_logits = compute_logits_from_hidden(model, mod_h)
            pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
            kl = float(np.sum(base_probs * np.log(base_probs / (pert_probs + 1e-10) + 1e-10)))
            
            key = f"{dir_name}_a{alpha}"
            pert_results[key] = kl
    
    results["perturbation_test"] = pert_results
    
    # 打印perturbation结果
    for alpha in [0.5, 2.0]:
        print(f"  α={alpha}: pca_rand={pert_results.get(f'pca_random_a{alpha}', 0):.4f}, "
              f"orth_rand={pert_results.get(f'orth_random_a{alpha}', 0):.4f}, "
              f"lda={pert_results.get(f'lda_a{alpha}', 0):.4f}, "
              f"pc1={pert_results.get(f'pc1_a{alpha}', 0):.4f}")
    
    return results


# ================================================================
# Exp2: LayerNorm前后的语法信息 (Phase 4H)
# ================================================================

def exp2_layernorm_causal(model, tokenizer, device, model_info):
    """
    ★★★★ LayerNorm前后的语法信息
    核心问题: LN是压缩语法信息的瓶颈, 还是语法信息天然在低能量方向?
    
    方法:
    1. 收集多层(包括LN前和LN后)的hidden states
    2. 每层分别做:
       a. 计算LDA分类准确率 (语法信息量)
       b. 计算有效秩 (维度压缩程度)
       c. 计算norm变异系数 (LN的效果)
    3. 对比LN前后的信息变化
    
    更直接的方案: 
    - 对于最后一层, 手动skip LayerNorm, 直接用h_before_LN计算logits
    - 对比有LN vs 无LN的因果效应
    """
    print("\n" + "="*70)
    print("Exp2 (Phase 4H): LayerNorm因果分析")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = len(layers)
    
    # 1. 收集各层的dep表示
    print("\n[1] Collecting multi-layer representations...")
    
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    
    dep_sents_nsubj = DEP_SENTENCES["nsubj"][:6]
    dep_sents_dobj = DEP_SENTENCES["dobj"][:6]
    dep_words_nsubj = DEP_TARGET_WORDS["nsubj"][:6]
    dep_words_dobj = DEP_TARGET_WORDS["dobj"][:6]
    
    layer_data = {}
    
    for li in sample_layers:
        layer = layers[li]
        all_h = []
        all_y = []
        
        for dep_type, sents, words in [("nsubj", dep_sents_nsubj, dep_words_nsubj),
                                        ("dobj", dep_sents_dobj, dep_words_dobj)]:
            for sent, target in zip(sents, words):
                toks = tokenizer(sent, return_tensors="pt").to(device)
                input_ids = toks.input_ids
                tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
                dep_idx = find_token_index(tokens, target)
                if dep_idx is None:
                    continue
                
                captured = {}
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu()
                        else:
                            captured[key] = output.detach().float().cpu()
                    return hook
                
                h_handle = layer.register_forward_hook(make_hook('h'))
                with torch.no_grad():
                    _ = model(**toks)
                h_handle.remove()
                
                if 'h' in captured:
                    h = captured['h'][0, dep_idx, :].numpy()
                    all_h.append(h)
                    all_y.append(0 if dep_type == "nsubj" else 1)
        
        if len(all_h) >= 4:
            H_layer = np.array(all_h)
            y_layer = np.array(all_y)
            
            # LDA
            lda = LinearDiscriminantAnalysis(n_components=1)
            try:
                lda.fit(H_layer, y_layer)
                lda_score = lda.score(H_layer, y_layer)
            except:
                lda_score = 0.5
            
            # PCA
            pca = PCA()
            pca.fit(H_layer)
            var = pca.explained_variance_ratio_
            var_probs = var / np.sum(var)
            var_probs = var_probs[var_probs > 1e-15]
            entropy = -np.sum(var_probs * np.log(var_probs))
            eff_rank = np.exp(entropy)
            
            # Norm stats
            norms = np.linalg.norm(H_layer, axis=1)
            norm_mean = np.mean(norms)
            norm_cv = np.std(norms) / max(norm_mean, 1e-10)
            
            # ★★★ Norm方向上的LDA: 归一化后的分类
            H_normed = H_layer / np.linalg.norm(H_layer, axis=1, keepdims=True)
            lda_normed = LinearDiscriminantAnalysis(n_components=1)
            try:
                lda_normed.fit(H_normed, y_layer)
                lda_normed_score = lda_normed.score(H_normed, y_layer)
            except:
                lda_normed_score = 0.5
            
            # PCA on normed
            pca_normed = PCA()
            pca_normed.fit(H_normed)
            var_normed = pca_normed.explained_variance_ratio_
            var_probs_n = var_normed / np.sum(var_normed)
            var_probs_n = var_probs_n[var_probs_n > 1e-15]
            entropy_n = -np.sum(var_probs_n * np.log(var_probs_n))
            eff_rank_normed = np.exp(entropy_n)
            
            layer_data[li] = {
                "n_samples": len(all_h),
                "lda_raw": float(lda_score),
                "lda_normed": float(lda_normed_score),
                "eff_rank_raw": float(eff_rank),
                "eff_rank_normed": float(eff_rank_normed),
                "pc1_raw": float(var[0]),
                "pc1_normed": float(var_normed[0]),
                "norm_mean": float(norm_mean),
                "norm_cv": float(norm_cv),
            }
            
            print(f"  L{li}: LDA_raw={lda_score:.3f}, LDA_normed={lda_normed_score:.3f}, "
                  f"eff_rank={eff_rank:.1f}->{eff_rank_normed:.1f}, "
                  f"PC1={var[0]*100:.1f}%->{var_normed[0]*100:.1f}%, "
                  f"norm_cv={norm_cv:.3f}")
    
    # 2. ★★★ 关键测试: Skip LayerNorm的因果效应
    print("\n[2] Skip-LayerNorm causal test...")
    
    # 对最后一层: 直接用pre-LN hidden states vs post-LN hidden states
    last_layer = layers[-1]
    
    # 收集nsubj和dobj的hidden states (pre-LN and post-LN)
    pre_ln_h = []
    post_ln_h = []
    labels = []
    
    for dep_type, sents, words in [("nsubj", dep_sents_nsubj[:4], dep_words_nsubj[:4]),
                                    ("dobj", dep_sents_dobj[:4], dep_words_dobj[:4])]:
        for sent, target in zip(sents, words):
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens, target)
            if dep_idx is None:
                continue
            
            # 获取post-LN hidden states (层输出)
            captured = {}
            def make_hook2(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            
            h_handle = last_layer.register_forward_hook(make_hook2('post_ln'))
            
            with torch.no_grad():
                output = model(**toks)
            
            h_handle.remove()
            
            if 'post_ln' not in captured:
                continue
            
            h_post = captured['post_ln']  # [1, seq, d_model]
            
            # pre-LN: 通过反推得到
            # h_post = LayerNorm(h_pre) + residual  (不精确, 但可以近似)
            # 更精确: hook在LN的input
            # 但很多模型的LN在attn/mlp内部, 不容易直接hook
            
            # 替代方案: 直接用model.model.norm计算
            # h_post经过model.model.norm后送入lm_head
            # 所以: logits = lm_head(norm(h_post))
            # 如果skip final norm: logits = lm_head(h_post)
            
            # 对比: 有final_norm vs 无final_norm的KL
            base_logits = output.logits.detach().float()
            base_probs = torch.softmax(base_logits[0, dep_idx], dim=-1).cpu().numpy()
            
            # Skip final norm: 直接用h_post送入lm_head
            h_post_tensor = captured['post_ln'].to(model.lm_head.weight.dtype).to(device)
            skip_norm_logits = model.lm_head(h_post_tensor).detach().float()
            skip_norm_probs = torch.softmax(skip_norm_logits[0, dep_idx], dim=-1).cpu().numpy()
            
            # 有final norm
            with_norm_logits = compute_logits_from_hidden(model, captured['post_ln'].to(device))
            with_norm_probs = torch.softmax(with_norm_logits[0, dep_idx], dim=-1).cpu().numpy()
            
            # KL(skip_norm || with_norm)
            kl_skip_vs_norm = float(np.sum(with_norm_probs * np.log(
                with_norm_probs / (skip_norm_probs + 1e-10) + 1e-10)))
            
            pre_ln_h.append(h_post[0, dep_idx, :].numpy())
            post_ln_h.append(with_norm_probs)  # 只存probs
            labels.append(0 if dep_type == "nsubj" else 1)
            
            del output, base_logits
            torch.cuda.empty_cache()
    
    # 3. ★★★ LayerNorm方向分析
    print("\n[3] LayerNorm direction analysis...")
    
    # 更精确的方法: 直接比较LayerNorm的输入和输出
    # 收集最后一层LN的input和output
    ln_io_data = []
    
    for dep_type, sents, words in [("nsubj", dep_sents_nsubj[:4], dep_words_nsubj[:4]),
                                    ("dobj", dep_sents_dobj[:4], dep_words_dobj[:4])]:
        for sent, target in zip(sents, words):
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens, target)
            if dep_idx is None:
                continue
            
            # Hook model.model.norm 的 input 和 output
            captured = {}
            
            def make_ln_hook(key):
                def hook(module, input, output):
                    if isinstance(input, tuple):
                        captured[f'{key}_input'] = input[0].detach().float().cpu()
                    if isinstance(output, tuple):
                        captured[f'{key}_output'] = output[0].detach().float().cpu()
                    else:
                        captured[f'{key}_output'] = output.detach().float().cpu()
                return hook
            
            ln = model.model.norm
            ln_handle = ln.register_forward_hook(make_ln_hook('final_ln'))
            
            with torch.no_grad():
                _ = model(**toks)
            
            ln_handle.remove()
            
            if 'final_ln_input' in captured and 'final_ln_output' in captured:
                h_pre = captured['final_ln_input'][0, dep_idx, :].numpy()
                h_post = captured['final_ln_output'][0, dep_idx, :].numpy()
                ln_io_data.append({
                    'h_pre': h_pre,
                    'h_post': h_post,
                    'dep_type': dep_type,
                })
    
    ln_results = {}
    if len(ln_io_data) >= 4:
        H_pre = np.array([d['h_pre'] for d in ln_io_data])
        H_post = np.array([d['h_post'] for d in ln_io_data])
        y_ln = np.array([0 if d['dep_type'] == 'nsubj' else 1 for d in ln_io_data])
        
        # LDA on pre-LN
        lda_pre = LinearDiscriminantAnalysis(n_components=1)
        try:
            lda_pre.fit(H_pre, y_ln)
            score_pre = lda_pre.score(H_pre, y_ln)
        except:
            score_pre = 0.5
        
        # LDA on post-LN
        lda_post = LinearDiscriminantAnalysis(n_components=1)
        try:
            lda_post.fit(H_post, y_ln)
            score_post = lda_post.score(H_post, y_ln)
        except:
            score_post = 0.5
        
        # Pre-LN的norm和方差
        pre_norms = np.linalg.norm(H_pre, axis=1)
        post_norms = np.linalg.norm(H_post, axis=1)
        
        # ★★★ LN前后的方向变化
        # LDA方向在pre-LN和post-LN中的夹角
        lda_pre_dir = lda_pre.coef_[0] if hasattr(lda_pre, 'coef_') else np.zeros(H_pre.shape[1])
        lda_pre_dir = lda_pre_dir / max(np.linalg.norm(lda_pre_dir), 1e-10)
        
        lda_post_dir = lda_post.coef_[0] if hasattr(lda_post, 'coef_') else np.zeros(H_post.shape[1])
        lda_post_dir = lda_post_dir / max(np.linalg.norm(lda_post_dir), 1e-10)
        
        cos_pre_post = float(np.dot(lda_pre_dir, lda_post_dir))
        
        # PCA分析
        pca_pre = PCA()
        pca_pre.fit(H_pre)
        var_pre = pca_pre.explained_variance_ratio_
        
        pca_post = PCA()
        pca_post.fit(H_post)
        var_post = pca_post.explained_variance_ratio_
        
        # 有效秩
        def eff_rank(var_ratio):
            vp = var_ratio / np.sum(var_ratio)
            vp = vp[vp > 1e-15]
            return np.exp(-np.sum(vp * np.log(vp)))
        
        print(f"  Pre-LN:  LDA={score_pre:.3f}, eff_rank={eff_rank(var_pre):.1f}, PC1={var_pre[0]*100:.2f}%")
        print(f"  Post-LN: LDA={score_post:.3f}, eff_rank={eff_rank(var_post):.1f}, PC1={var_post[0]*100:.2f}%")
        print(f"  LDA direction cos(pre,post): {cos_pre_post:.4f}")
        print(f"  Norm: pre_mean={np.mean(pre_norms):.2f}, post_mean={np.mean(post_norms):.2f}")
        
        ln_results = {
            "pre_ln_lda": float(score_pre),
            "post_ln_lda": float(score_post),
            "lda_direction_cos": cos_pre_post,
            "pre_ln_pc1": float(var_pre[0]),
            "post_ln_pc1": float(var_post[0]),
            "pre_ln_eff_rank": float(eff_rank(var_pre)),
            "post_ln_eff_rank": float(eff_rank(var_post)),
            "pre_ln_norm_mean": float(np.mean(pre_norms)),
            "post_ln_norm_mean": float(np.mean(post_norms)),
            "lda_lift": float(score_pre / max(score_post, 0.01)),
        }
    
    results = {
        "model": model_info.name,
        "n_layers": n_layers,
        "layer_analysis": {str(k): v for k, v in layer_data.items()},
        "layernorm_io_analysis": ln_results,
    }
    
    return results


# ================================================================
# Exp3: 因果方向流形 (Phase 4I)
# ================================================================

def exp3_causal_direction_manifold(model, tokenizer, device, model_info):
    """
    ★★★ 因果方向流形
    核心问题: 不同句子对的因果方向是否有低维结构?
    
    如果大脑类似最小作用量原理 → 因果方向应该有结构 (低维流形)
    如果是随机的 → 因果方向应该是高维散点
    
    方法:
    1. 收集多组nsubj/dobj句子对的因果梯度方向
    2. PCA分析这些方向的维度
    3. 检验: 因果方向是否近似于低维子空间
    4. 如果是 → 可以参数化语法角色转换
    """
    print("\n" + "="*70)
    print("Exp3 (Phase 4I): 因果方向流形")
    print("="*70)
    
    layers = get_layers(model)
    last_layer = layers[-1]
    
    # 1. 收集大量因果方向
    print("\n[1] Collecting causal gradient directions...")
    
    # 使用所有16个句子对
    causal_directions = []
    direction_info = []
    
    for dep_type, sents, words in [("nsubj", DEP_SENTENCES["nsubj"], DEP_TARGET_WORDS["nsubj"]),
                                    ("dobj", DEP_SENTENCES["dobj"], DEP_TARGET_WORDS["dobj"])]:
        for sent, target in zip(sents, words):
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens, target)
            if dep_idx is None:
                continue
            
            # Autograd
            captured_h = {}
            grad_h = {}
            
            def capture_and_grad_hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone().detach().requires_grad_(True)
                    captured_h['h'] = h
                    def grad_callback(grad):
                        grad_h['grad'] = grad
                    h.register_hook(grad_callback)
                    return (h,) + output[1:]
                else:
                    h = output.clone().detach().requires_grad_(True)
                    captured_h['h'] = h
                    def grad_callback(grad):
                        grad_h['grad'] = grad
                    h.register_hook(grad_callback)
                    return h
            
            hook_handle = last_layer.register_forward_hook(capture_and_grad_hook)
            
            output = model(**toks)
            logits = output.logits
            
            top_token = torch.argmax(logits[0, dep_idx]).item()
            top_prob = torch.softmax(logits[0, dep_idx].float(), dim=-1)[top_token]
            
            top_prob.backward()
            hook_handle.remove()
            
            if 'grad' in grad_h:
                grad_direction = grad_h['grad'][0, dep_idx, :].detach().float().cpu().numpy()
                grad_norm = np.linalg.norm(grad_direction)
                if grad_norm > 1e-10:
                    grad_dir_normed = grad_direction / grad_norm
                    causal_directions.append(grad_dir_normed)
                    direction_info.append({
                        'dep_type': dep_type,
                        'sentence': sent,
                        'target': target,
                        'grad_norm': float(grad_norm),
                    })
            
            del output, logits
            torch.cuda.empty_cache()
    
    print(f"  Collected {len(causal_directions)} causal directions")
    
    if len(causal_directions) < 4:
        print("  ERROR: Too few directions")
        return None
    
    # 2. PCA on 因果方向矩阵
    print("\n[2] PCA on causal directions...")
    
    D = np.array(causal_directions)  # [n_directions, d_model]
    
    pca_dir = PCA()
    pca_dir.fit(D)
    var_dir = pca_dir.explained_variance_ratio_
    
    # 有效秩
    var_probs = var_dir / np.sum(var_dir)
    var_probs = var_probs[var_probs > 1e-15]
    entropy_dir = -np.sum(var_probs * np.log(var_probs))
    eff_rank_dir = np.exp(entropy_dir)
    
    cumvar_dir = np.cumsum(var_dir)
    k_90 = np.searchsorted(cumvar_dir, 0.90) + 1
    k_95 = np.searchsorted(cumvar_dir, 0.95) + 1
    
    print(f"  Eff rank: {eff_rank_dir:.1f}")
    print(f"  PC1: {var_dir[0]*100:.2f}%, top5: {np.sum(var_dir[:5])*100:.2f}%")
    print(f"  k_90: {k_90}, k_95: {k_95}")
    
    # 3. ★★★ nsubj vs dobj 因果方向的差异
    nsubj_dirs = [d for d, info in zip(causal_directions, direction_info) if info['dep_type'] == 'nsubj']
    dobj_dirs = [d for d, info in zip(causal_directions, direction_info) if info['dep_type'] == 'dobj']
    
    if len(nsubj_dirs) >= 2 and len(dobj_dirs) >= 2:
        nsubj_centroid = np.mean(nsubj_dirs, axis=0)
        dobj_centroid = np.mean(dobj_dirs, axis=0)
        
        # 归一化质心
        nsubj_centroid_norm = nsubj_centroid / max(np.linalg.norm(nsubj_centroid), 1e-10)
        dobj_centroid_norm = dobj_centroid / max(np.linalg.norm(dobj_centroid), 1e-10)
        
        cos_nsubj_dobj = float(np.dot(nsubj_centroid_norm, dobj_centroid_norm))
        
        # 簇内距离
        intra_nsubj = np.mean([np.linalg.norm(d - nsubj_centroid) for d in nsubj_dirs])
        intra_dobj = np.mean([np.linalg.norm(d - dobj_centroid) for d in dobj_dirs])
        
        # 簇间距离
        inter_dist = np.linalg.norm(nsubj_centroid - dobj_centroid)
        
        print(f"\n  nsubj vs dobj causal directions:")
        print(f"    cos(centroids): {cos_nsubj_dobj:.4f}")
        print(f"    Intra: nsubj={intra_nsubj:.4f}, dobj={intra_dobj:.4f}")
        print(f"    Inter: {inter_dist:.4f}")
    else:
        cos_nsubj_dobj = 0.0
        intra_nsubj = 0.0
        intra_dobj = 0.0
        inter_dist = 0.0
    
    # 4. ★★★ 与LDA/PCA方向的夹角
    print("\n[3] Causal direction vs LDA/PCA angles...")
    
    # 收集LDA/PCA方向
    all_reps = []
    for dep_type in ["nsubj", "dobj"]:
        reps = collect_dep_representations(
            model, tokenizer, device,
            DEP_SENTENCES[dep_type][:8], dep_type, DEP_TARGET_WORDS[dep_type][:8],
        )
        all_reps.extend(reps)
    
    H = np.array([r['h'] for r in all_reps])
    y = np.array([0 if r['dep_type'] == 'nsubj' else 1 for r in all_reps])
    
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(H, y)
    lda_dir = lda.coef_[0]
    lda_dir = lda_dir / np.linalg.norm(lda_dir)
    
    pca = PCA()
    pca.fit(H)
    pca_pc1 = pca.components_[0]
    
    # 因果方向与LDA/PCA的平均余弦
    cos_causal_lda = [float(np.dot(d, lda_dir)) for d in causal_directions]
    cos_causal_pca = [float(np.dot(d, pca_pc1)) for d in causal_directions]
    
    print(f"  |cos(causal, LDA)| mean: {np.mean(np.abs(cos_causal_lda)):.4f}")
    print(f"  |cos(causal, PCA)| mean: {np.mean(np.abs(cos_causal_pca)):.4f}")
    
    # 5. ★★★ 随机方向基线: 随机方向的有效秩
    n_random_trials = 20
    random_eff_ranks = []
    for _ in range(n_random_trials):
        rand_dirs = np.random.randn(len(causal_directions), D.shape[1])
        rand_dirs = rand_dirs / np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        pca_rand = PCA()
        pca_rand.fit(rand_dirs)
        var_rand = pca_rand.explained_variance_ratio_
        var_probs_r = var_rand / np.sum(var_rand)
        var_probs_r = var_probs_r[var_probs_r > 1e-15]
        entropy_r = -np.sum(var_probs_r * np.log(var_probs_r))
        random_eff_ranks.append(np.exp(entropy_r))
    
    print(f"\n  Random direction eff_rank: mean={np.mean(random_eff_ranks):.1f}")
    print(f"  Causal direction eff_rank: {eff_rank_dir:.1f}")
    print(f"  ★ Structure lift: {np.mean(random_eff_ranks) / max(eff_rank_dir, 0.1):.2f}x (random/causal)")
    
    # 6. ★★★ 因果方向之间的两两余弦矩阵
    print("\n[4] Pairwise cosine matrix...")
    n_dirs = len(causal_directions)
    cos_matrix = np.zeros((n_dirs, n_dirs))
    for i in range(n_dirs):
        for j in range(n_dirs):
            cos_matrix[i, j] = float(np.dot(causal_directions[i], causal_directions[j]))
    
    # nsubj内部平均余弦
    nsubj_indices = [i for i, info in enumerate(direction_info) if info['dep_type'] == 'nsubj']
    dobj_indices = [i for i, info in enumerate(direction_info) if info['dep_type'] == 'dobj']
    
    nsubj_inner_cos = []
    for i in range(len(nsubj_indices)):
        for j in range(i + 1, len(nsubj_indices)):
            nsubj_inner_cos.append(cos_matrix[nsubj_indices[i], nsubj_indices[j]])
    
    dobj_inner_cos = []
    for i in range(len(dobj_indices)):
        for j in range(i + 1, len(dobj_indices)):
            dobj_inner_cos.append(cos_matrix[dobj_indices[i], dobj_indices[j]])
    
    cross_cos = []
    for i in nsubj_indices:
        for j in dobj_indices:
            cross_cos.append(cos_matrix[i, j])
    
    print(f"  nsubj inner cos: mean={np.mean(np.abs(nsubj_inner_cos)):.4f}" if nsubj_inner_cos else "  nsubj inner: N/A")
    print(f"  dobj inner cos: mean={np.mean(np.abs(dobj_inner_cos)):.4f}" if dobj_inner_cos else "  dobj inner: N/A")
    print(f"  cross cos: mean={np.mean(np.abs(cross_cos)):.4f}" if cross_cos else "  cross: N/A")
    
    results = {
        "model": model_info.name,
        "n_directions": len(causal_directions),
        "eff_rank": float(eff_rank_dir),
        "pc1_variance": float(var_dir[0]),
        "top5_variance": float(np.sum(var_dir[:5])),
        "k_90": int(k_90),
        "k_95": int(k_95),
        "random_eff_rank_mean": float(np.mean(random_eff_ranks)),
        "structure_lift": float(np.mean(random_eff_ranks) / max(eff_rank_dir, 0.1)),
        "nsubj_dobj_cos": float(cos_nsubj_dobj),
        "intra_nsubj": float(intra_nsubj),
        "intra_dobj": float(intra_dobj),
        "inter_dist": float(inter_dist),
        "cos_causal_lda_mean": float(np.mean(np.abs(cos_causal_lda))),
        "cos_causal_pca_mean": float(np.mean(np.abs(cos_causal_pca))),
        "nsubj_inner_cos_mean": float(np.mean(np.abs(nsubj_inner_cos))) if nsubj_inner_cos else 0.0,
        "dobj_inner_cos_mean": float(np.mean(np.abs(dobj_inner_cos))) if dobj_inner_cos else 0.0,
        "cross_cos_mean": float(np.mean(np.abs(cross_cos))) if cross_cos else 0.0,
        "direction_details": direction_info,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="CCL-E: Orthogonal Geometry + LayerNorm + Causal Manifold")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"\nModel: {model_info.name} ({model_info.model_class}), Layers: {model_info.n_layers}, d_model: {model_info.d_model}")
    
    try:
        if args.exp == 1:
            results = exp1_orthogonal_complement_geometry(model, tokenizer, device, model_info)
        elif args.exp == 2:
            results = exp2_layernorm_causal(model, tokenizer, device, model_info)
        elif args.exp == 3:
            results = exp3_causal_direction_manifold(model, tokenizer, device, model_info)
        
        if results:
            out_path = f"tests/glm5_temp/ccle_exp{args.exp}_{args.model}_results.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nResults saved to: {out_path}")
    finally:
        release_model(model)


if __name__ == "__main__":
    main()
