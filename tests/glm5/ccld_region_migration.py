"""
CCL-D(250.4): 区域迁移 + 低秩分析 + 因果方向挖掘 (V2 - 直接修改hidden states版)
====================================================
关键修改: 不用hook, 而是直接:
  1. 前向传播获取最后层hidden states
  2. 在hidden states上添加扰动
  3. 通过model.model.norm + model.lm_head计算perturbed logits
  4. 这样避免了hook可能失效的问题
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
    ],
}

DEP_TARGET_WORDS = {
    "nsubj": ["cat", "dog", "bird", "child", "student", "teacher", "scientist", "writer"],
    "dobj": ["cat", "dog", "bird", "child", "student", "teacher", "scientist", "writer"],
}


def find_token_index(tokens, word):
    word_lower = word.lower()
    for i, tok in enumerate(tokens):
        if word_lower in tok.lower() or tok.lower().startswith(word_lower[:3]):
            return i
    return None


def get_last_layer_hidden(model, tokenizer, device, sentence):
    """获取最后层的hidden states (通过hook, 但只做一次)"""
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
    
    hidden_states = captured['h']  # [1, seq_len, d_model]
    return hidden_states, base_logits, toks


def compute_logits_from_hidden(model, hidden_states):
    """从hidden states通过final norm + lm_head计算logits"""
    # model.model.norm is the final LayerNorm
    # model.lm_head is the output projection
    
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
            'hidden_states': hidden_states,  # 保存完整hidden states用于后续修改
            'base_logits': base_logits,
            'toks': toks,
        })
    
    return all_reps


# ================================================================
# Phase 4D: 区域迁移实验 (V2)
# ================================================================

def exp1_region_migration(model, tokenizer, device, model_info):
    """★★★★★ 区域迁移实验 - 直接修改hidden states版"""
    print("\n" + "="*70)
    print("Phase 4D: 区域迁移实验 (Direct Hidden State Modification)")
    print("="*70)
    
    # 1. 收集所有dep表示
    print("\n[1] Collecting dep representations...")
    all_reps = []
    for dep_type in ["nsubj", "dobj"]:
        reps = collect_dep_representations(
            model, tokenizer, device,
            DEP_SENTENCES[dep_type],
            dep_type,
            DEP_TARGET_WORDS[dep_type],
        )
        all_reps.extend(reps)
        print(f"  {dep_type}: {len(reps)} representations collected")
    
    if len(all_reps) < 4:
        print("  ERROR: Too few representations")
        return None
    
    # 2. 构建矩阵和标签
    H = np.array([r['h'] for r in all_reps])
    y = np.array([0 if r['dep_type'] == 'nsubj' else 1 for r in all_reps])
    
    # 3. PCA分析
    print("\n[2] PCA analysis...")
    pca = PCA()
    pca.fit(H)
    explained_var = pca.explained_variance_ratio_
    print(f"  PC1: {explained_var[0]*100:.3f}%")
    print(f"  Top-10: {np.sum(explained_var[:10])*100:.3f}%")
    
    # 4. 计算各类质心和迁移方向
    nsubj_mask = y == 0
    dobj_mask = y == 1
    
    centroid_nsubj = np.mean(H[nsubj_mask], axis=0)
    centroid_dobj = np.mean(H[dobj_mask], axis=0)
    
    migration_dir = centroid_dobj - centroid_nsubj
    migration_norm = np.linalg.norm(migration_dir)
    print(f"  Migration direction norm: {migration_norm:.4f}")
    
    if migration_norm > 0:
        migration_dir_normalized = migration_dir / migration_norm
    else:
        migration_dir_normalized = np.zeros_like(migration_dir)
    
    # 5. LDA方向
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(H, y)
    lda_dir = lda.coef_[0]
    lda_dir = lda_dir / np.linalg.norm(lda_dir)
    
    pca_dir = pca.components_[0]
    
    # 方向夹角
    cos_migration_lda = float(np.dot(migration_dir_normalized, lda_dir))
    cos_migration_pca = float(np.dot(migration_dir_normalized, pca_dir))
    cos_lda_pca = float(np.dot(lda_dir, pca_dir))
    
    print(f"\n[3] Direction angles:")
    print(f"  Migration||LDA:  {cos_migration_lda:.4f}")
    print(f"  Migration||PCA:  {cos_migration_pca:.4f}")
    print(f"  LDA||PCA:        {cos_lda_pca:.4f}")
    
    # 6. ★★★ 直接修改hidden states做区域迁移
    print("\n[4] Region migration (direct hidden state modification)...")
    
    # 检查hidden states norm
    rep0 = all_reps[0]
    h0 = rep0['hidden_states'][0, rep0['dep_idx'], :]
    print(f"  Hidden state norm at dep position: {torch.norm(h0).item():.2f}")
    print(f"  Migration dir norm: {migration_norm:.4f}")
    
    # ★★★ 使用未归一化的迁移方向 (按范数比例)
    # 用alpha相对于hidden state范数的比例
    h_norm_mean = np.mean([np.linalg.norm(r['h']) for r in all_reps])
    alpha_scale = migration_norm / max(h_norm_mean, 1e-10)
    print(f"  h_norm_mean: {h_norm_mean:.2f}, alpha_scale: {alpha_scale:.4f}")
    
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]  # 相对于h_norm的比例
    
    results = {
        "model": model_info.name,
        "pca_variance": {"pc1": float(explained_var[0]), "top10": float(np.sum(explained_var[:10]))},
        "direction_angles": {
            "migration_lda": cos_migration_lda,
            "migration_pca": cos_migration_pca,
            "lda_pca": cos_lda_pca,
        },
        "h_norm_mean": float(h_norm_mean),
        "migration_dir_norm": float(migration_norm),
        "migration_results": [],
    }
    
    nsubj_reps = [r for r in all_reps if r['dep_type'] == 'nsubj']
    
    for rep in nsubj_reps[:6]:
        sent = rep['sentence']
        target = rep['target']
        dep_idx = rep['dep_idx']
        hidden_states = rep['hidden_states']  # [1, seq_len, d_model]
        base_logits = rep['base_logits']
        
        # 基线probs — 修改h[dep_idx]影响logits[dep_idx], 预测dep_idx+1处的token
        base_probs_dep = torch.softmax(base_logits[0, dep_idx], dim=-1).cpu().numpy()
        
        sent_results = {
            "sentence": sent,
            "target": target,
            "dep_idx": dep_idx,
            "alphas": {},
        }
        
        for alpha in alphas:
            # ★★★ 直接修改hidden states
            # h[dep_idx] += alpha * h_norm * migration_dir_normalized
            perturbation = alpha * h_norm_mean * migration_dir_normalized
            perturbation_t = torch.tensor(perturbation, dtype=hidden_states.dtype, device=hidden_states.device)
            
            modified_h = hidden_states.clone()
            modified_h[0, dep_idx, :] += perturbation_t
            
            # 通过final norm + lm_head计算新logits
            perturbed_logits = compute_logits_from_hidden(model, modified_h)
            
            perturbed_probs_dep = torch.softmax(perturbed_logits[0, dep_idx], dim=-1).cpu().numpy()
            
            kl_dep = float(np.sum(base_probs_dep * np.log(base_probs_dep / (perturbed_probs_dep + 1e-10) + 1e-10)))
            
            # Top token变化
            prob_diff = perturbed_probs_dep - base_probs_dep
            top_inc_idx = np.argsort(prob_diff)[-5:][::-1]
            top_inc = []
            for idx in top_inc_idx:
                if abs(prob_diff[idx]) > 1e-8:
                    top_inc.append({
                        "token": safe_decode(tokenizer, idx),
                        "base_prob": float(base_probs_dep[idx]),
                        "perturbed_prob": float(perturbed_probs_dep[idx]),
                        "delta": float(prob_diff[idx]),
                    })
            
            sent_results["alphas"][str(alpha)] = {
                "kl_dep": kl_dep,
                "top_changes": top_inc[:3],
            }
        
        results["migration_results"].append(sent_results)
        print(f"  '{sent}': KL_dep α=1={sent_results['alphas']['1.0']['kl_dep']:.4f}, "
              f"α=5={sent_results['alphas']['5.0']['kl_dep']:.4f}")
    
    # 7. 方向对比: migration vs LDA vs PCA
    print("\n[5] Direction comparison...")
    rep0 = nsubj_reps[0]
    dep_idx = rep0['dep_idx']
    hidden_states = rep0['hidden_states']
    base_logits = rep0['base_logits']
    base_probs_ref = torch.softmax(base_logits[0, dep_idx], dim=-1).cpu().numpy()
    
    dir_comparison = {}
    for dir_name, dir_vec in [("migration", migration_dir_normalized), 
                               ("lda", lda_dir), 
                               ("pca_pc1", pca_dir),
                               ("migration_unnorm", migration_dir / max(migration_norm, 1e-10))]:
        kl_list = []
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            perturbation = alpha * h_norm_mean * dir_vec
            perturbation_t = torch.tensor(perturbation, dtype=hidden_states.dtype, device=hidden_states.device)
            
            modified_h = hidden_states.clone()
            modified_h[0, dep_idx, :] += perturbation_t
            
            perturbed_logits = compute_logits_from_hidden(model, modified_h)
            perturbed_probs = torch.softmax(perturbed_logits[0, dep_idx], dim=-1).cpu().numpy()
            
            kl = float(np.sum(base_probs_ref * np.log(base_probs_ref / (perturbed_probs + 1e-10) + 1e-10)))
            kl_list.append(kl)
        
        dir_comparison[dir_name] = kl_list
        print(f"  {dir_name}: KL(α=0.5)={kl_list[0]:.4f}, KL(α=1)={kl_list[1]:.4f}, KL(α=5)={kl_list[3]:.4f}")
    
    results["direction_comparison"] = dir_comparison
    
    return results


# ================================================================
# Phase 4E: 低秩表示分析
# ================================================================

def exp2_low_rank_analysis(model, tokenizer, device, model_info):
    """★★★★ 低秩表示分析"""
    print("\n" + "="*70)
    print("Phase 4E: 低秩表示分析")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = len(layers)
    
    test_sentences = [
        "The cat sat on the mat",
        "She found the book on the table",
        "The scientist discovered a new formula",
    ]
    
    results = {"model": model_info.name, "n_layers": n_layers, "layer_analysis": []}
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    
    for layer_idx in sample_layers:
        layer = layers[layer_idx]
        all_h = []
        
        for sent in test_sentences:
            toks = tokenizer(sent, return_tensors="pt").to(device)
            
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
                all_h.append(captured['h'][0].numpy())
        
        if not all_h:
            continue
        
        H = np.vstack(all_h)
        pca = PCA()
        pca.fit(H)
        var = pca.explained_variance_ratio_
        
        norms = np.linalg.norm(H, axis=1)
        var_probs = var / np.sum(var)
        var_probs = var_probs[var_probs > 0]
        entropy = -np.sum(var_probs * np.log(var_probs))
        effective_rank = np.exp(entropy)
        norm_cv = np.std(norms) / max(np.mean(norms), 1e-10)
        
        layer_data = {
            "layer": layer_idx,
            "pc1_variance": float(var[0]),
            "top5_variance": float(np.sum(var[:5])),
            "effective_rank": float(effective_rank),
            "mean_norm": float(np.mean(norms)),
            "norm_cv": float(norm_cv),
        }
        results["layer_analysis"].append(layer_data)
        print(f"  L{layer_idx}: PC1={var[0]*100:.2f}%, eff_rank={effective_rank:.1f}, norm_cv={norm_cv:.4f}")
    
    return results


# ================================================================
# Phase 4F: 因果方向挖掘 (autograd)
# ================================================================

def exp3_causal_direction_mining(model, tokenizer, device, model_info):
    """★★★ 因果方向挖掘 - 使用autograd计算梯度"""
    print("\n" + "="*70)
    print("Phase 4F: 因果方向挖掘 (autograd)")
    print("="*70)
    
    layers = get_layers(model)
    last_layer = layers[-1]
    
    test_pairs = [
        {"nsubj": "The cat sat on the", "dobj": "She chased the cat",
         "nsubj_dep_word": "cat", "dobj_dep_word": "cat"},
        {"nsubj": "The dog ran through the", "dobj": "He found the dog",
         "nsubj_dep_word": "dog", "dobj_dep_word": "dog"},
        {"nsubj": "The bird sang a", "dobj": "They watched the bird",
         "nsubj_dep_word": "bird", "dobj_dep_word": "bird"},
    ]
    
    results = {"model": model_info.name, "gradient_analysis": []}
    
    # 先收集LDA/PCA方向
    all_reps = []
    for dep_type in ["nsubj", "dobj"]:
        reps = collect_dep_representations(
            model, tokenizer, device,
            DEP_SENTENCES[dep_type][:4], dep_type, DEP_TARGET_WORDS[dep_type][:4],
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
    pca_dir = pca.components_[0]
    
    cos_lda_pca = float(np.dot(lda_dir, pca_dir))
    print(f"  LDA||PCA cos: {cos_lda_pca:.4f}")
    
    # h_norm用于缩放alpha
    h_norm_mean = float(np.mean([np.linalg.norm(r['h']) for r in all_reps]))
    
    for pair_idx, pair in enumerate(test_pairs):
        print(f"\n  Pair {pair_idx+1}")
        pair_results = {"nsubj_sent": pair["nsubj"], "dobj_sent": pair["dobj"]}
        
        for dep_type in ["nsubj", "dobj"]:
            sent = pair[dep_type]
            dep_word = pair[f"{dep_type}_dep_word"]
            
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens, dep_word)
            if dep_idx is None:
                continue
            
            # ★★★ autograd计算梯度
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
            
            # ★★★ 注意: 修改h[dep_idx]只影响logits[dep_idx], 不影响logits[next_pos]
            top_token = torch.argmax(logits[0, dep_idx]).item()
            top_prob = torch.softmax(logits[0, dep_idx].float(), dim=-1)[top_token]
            
            top_prob.backward()
            hook_handle.remove()
            
            if 'grad' not in grad_h:
                print(f"    {dep_type}: no gradient")
                continue
            
            # 因果方向 = 梯度方向
            grad_direction = grad_h['grad'][0, dep_idx, :].detach().float().cpu().numpy()
            grad_norm = np.linalg.norm(grad_direction)
            grad_dir_normalized = grad_direction / max(grad_norm, 1e-10)
            
            # 保存基线probs
            base_probs_ref = torch.softmax(logits[0, dep_idx].detach().float(), dim=-1).cpu().numpy()
            
            # 获取hidden states
            hidden_states = captured_h['h'].detach()
            
            del output, logits
            torch.cuda.empty_cache()
            
            # 方向夹角
            cos_causal_lda = float(np.dot(grad_dir_normalized, lda_dir))
            cos_causal_pca = float(np.dot(grad_dir_normalized, pca_dir))
            
            # 效果对比: 直接修改hidden states
            causal_kl_list = []
            lda_kl_list = []
            pca_kl_list = []
            
            for alpha in [0.5, 1.0, 2.0, 5.0]:
                # Causal direction
                pert = alpha * h_norm_mean * grad_dir_normalized
                pert_t = torch.tensor(pert, dtype=hidden_states.dtype, device=hidden_states.device)
                mod_h = hidden_states.clone()
                mod_h[0, dep_idx, :] += pert_t
                pert_logits = compute_logits_from_hidden(model, mod_h)
                pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
                kl = float(np.sum(base_probs_ref * np.log(base_probs_ref / (pert_probs + 1e-10) + 1e-10)))
                causal_kl_list.append(kl)
                
                # LDA direction
                pert = alpha * h_norm_mean * lda_dir
                pert_t = torch.tensor(pert, dtype=hidden_states.dtype, device=hidden_states.device)
                mod_h = hidden_states.clone()
                mod_h[0, dep_idx, :] += pert_t
                pert_logits = compute_logits_from_hidden(model, mod_h)
                pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
                kl = float(np.sum(base_probs_ref * np.log(base_probs_ref / (pert_probs + 1e-10) + 1e-10)))
                lda_kl_list.append(kl)
                
                # PCA direction
                pert = alpha * h_norm_mean * pca_dir
                pert_t = torch.tensor(pert, dtype=hidden_states.dtype, device=hidden_states.device)
                mod_h = hidden_states.clone()
                mod_h[0, dep_idx, :] += pert_t
                pert_logits = compute_logits_from_hidden(model, mod_h)
                pert_probs = torch.softmax(pert_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
                kl = float(np.sum(base_probs_ref * np.log(base_probs_ref / (pert_probs + 1e-10) + 1e-10)))
                pca_kl_list.append(kl)
            
            print(f"    {dep_type}: causal||LDA={cos_causal_lda:.4f}, causal||PCA={cos_causal_pca:.4f}")
            print(f"      KL(α=5): causal={causal_kl_list[-1]:.4f}, lda={lda_kl_list[-1]:.4f}, pca={pca_kl_list[-1]:.4f}")
            
            pair_results[f"{dep_type}_cos_causal_lda"] = cos_causal_lda
            pair_results[f"{dep_type}_cos_causal_pca"] = cos_causal_pca
            pair_results[f"{dep_type}_causal_kl"] = causal_kl_list
            pair_results[f"{dep_type}_lda_kl"] = lda_kl_list
            pair_results[f"{dep_type}_pca_kl"] = pca_kl_list
        
        results["gradient_analysis"].append(pair_results)
    
    # 汇总
    print("\n[Summary] Causal direction mining:")
    for key in ["cos_causal_lda", "cos_causal_pca"]:
        vals = []
        for pa in results["gradient_analysis"]:
            for dt in ["nsubj", "dobj"]:
                k = f"{dt}_{key}"
                if k in pa:
                    vals.append(abs(pa[k]))
        if vals:
            print(f"  |{key}| mean: {np.mean(vals):.4f}")
    
    for dir_name in ["causal", "lda", "pca"]:
        vals = []
        for pa in results["gradient_analysis"]:
            for dt in ["nsubj", "dobj"]:
                k = f"{dt}_{dir_name}_kl"
                if k in pa and pa[k]:
                    vals.append(pa[k][-1])
        if vals:
            print(f"  {dir_name} KL(α=5) mean: {np.mean(vals):.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="CCL-D V2")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"\nModel: {model_info.name} ({model_info.model_class}), Layers: {model_info.n_layers}, d_model: {model_info.d_model}")
    
    try:
        if args.exp == 1:
            results = exp1_region_migration(model, tokenizer, device, model_info)
        elif args.exp == 2:
            results = exp2_low_rank_analysis(model, tokenizer, device, model_info)
        elif args.exp == 3:
            results = exp3_causal_direction_mining(model, tokenizer, device, model_info)
        
        if results:
            out_path = f"tests/glm5_temp/ccld_exp{args.exp}_{args.model}_results.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nResults saved to: {out_path}")
    finally:
        release_model(model)


if __name__ == "__main__":
    main()
