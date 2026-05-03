"""
CCL-C(250.3): PCA空间的语法调制分析
===========================================
CCL-B发现:
  - LDA方向有零因果效应 (LDA/PCA消融比率 ≤ 0.01)
  - PCA方向是因果方向 (消融KL=0.25-0.74)
  - "语法=判别子空间"假设被推翻

新假设:
  语法 = PCA空间中的条件分布
  → 不同依存类型的词, 在PCA空间中有不同的分布
  → 语法角色 = PCA系数的条件概率分布
  → 不是独立方向编码, 而是高维空间中的区域编码

核心实验:
  Exp1: PCA系数的条件分布 — 不同依存类型的PCA投影分布
  Exp2: PCA方向上的因果干预 — 沿PCA方向干预, 测量语法输出变化
  Exp3: 互信息分析 — I(h_dep; dep_type | PCA) vs I(h_dep; dep_type)
"""

import torch
import numpy as np
import json
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp, wasserstein_distance
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model
from tests.glm5.ccxlix_cross_layer_scan import (
    extract_all_layers_for_pair, NATURAL_SENTENCES, TARGET_DEP_TYPES, 
    DEP_TYPE_MAP, parse_with_spacy
)

_DEVICE = "cuda:0"

# 各模型的最后一层
LAST_LAYERS = {
    "qwen3": 35,
    "glm4": 39,
    "deepseek7b": 27,
}


def safe_decode(tokenizer, token_id):
    try:
        result = tokenizer.decode([token_id])
        return result if result is not None else f"<tok_{token_id}>"
    except Exception:
        return f"<tok_{token_id}>"


# ============================================================
# Exp1: PCA系数的条件分布分析
# ============================================================

def exp1_pca_conditional_distribution(model_name, model, tokenizer, device):
    """
    Project dep representations onto PCA space and analyze
    the conditional distribution P(PCA_coeffs | dep_type).
    """
    print(f"\n{'='*70}")
    print(f"Exp1: PCA条件分布分析 ({model_name})")
    print(f"{'='*70}")
    
    last_layer = LAST_LAYERS[model_name]
    
    # Parse and extract features
    pairs = parse_with_spacy(NATURAL_SENTENCES)
    type_sampled = {dt: 0 for dt in TARGET_DEP_TYPES}
    balanced_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in TARGET_DEP_TYPES and type_sampled[dt] < 50:
            balanced_pairs.append(p)
            type_sampled[dt] += 1
    
    print(f"  Using {len(balanced_pairs)} pairs at L{last_layer}")
    
    # Extract dep representations
    type_h_deps = {dt: [] for dt in TARGET_DEP_TYPES}
    all_h_deps = []
    all_y = []
    
    for i, (sent, h_idx, d_idx, dep_type) in enumerate(balanced_pairs):
        if i % 50 == 0:
            print(f"    Extracting {i}/{len(balanced_pairs)}...")
        try:
            feats = extract_all_layers_for_pair(
                model, tokenizer, sent, h_idx, d_idx, [last_layer]
            )
            if last_layer in feats and dep_type in type_h_deps:
                h_dep = feats[last_layer]["h_dep"]
                type_h_deps[dep_type].append(h_dep)
                all_h_deps.append(h_dep)
                all_y.append(TARGET_DEP_TYPES.index(dep_type))
        except:
            continue
    
    all_h_deps = np.array(all_h_deps)
    all_y = np.array(all_y)
    n_samples = len(all_y)
    
    print(f"  Total: {n_samples} dep representations")
    
    if n_samples < 50:
        return None
    
    # Fit PCA
    n_pca = min(50, all_h_deps.shape[1], n_samples - 1)
    pca = PCA(n_components=n_pca)
    pca_coeffs = pca.fit_transform(all_h_deps)  # [n_samples, n_pca]
    
    print(f"  PCA top-10 explained variance: {pca.explained_variance_ratio_[:10]}")
    print(f"  Cumulative top-10: {np.cumsum(pca.explained_variance_ratio_[:10])}")
    
    results = {
        "model": model_name,
        "last_layer": last_layer,
        "n_samples": n_samples,
        "n_pca": n_pca,
        "explained_variance_ratio": pca.explained_variance_ratio_[:20].tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_[:20]).tolist(),
    }
    
    # Analyze conditional distribution per PCA component per dep type
    print(f"\n  === Per-component analysis ===")
    
    type_pca_coeffs = {}
    for dt in TARGET_DEP_TYPES:
        if len(type_h_deps[dt]) >= 5:
            type_pca_coeffs[dt] = pca.transform(np.array(type_h_deps[dt]))
    
    # For each PCA component, test if dep types have different distributions
    component_analysis = {}
    
    for comp_idx in range(min(20, n_pca)):
        comp_data = {}
        
        # Get distribution per type
        type_means = {}
        type_stds = {}
        for dt, coeffs in type_pca_coeffs.items():
            vals = coeffs[:, comp_idx]
            type_means[dt] = float(np.mean(vals))
            type_stds[dt] = float(np.std(vals))
        
        # KS test between pairs of types
        dt_list = list(type_pca_coeffs.keys())
        max_ks_stat = 0
        max_ks_pair = ""
        
        for i in range(len(dt_list)):
            for j in range(i+1, len(dt_list)):
                dt1, dt2 = dt_list[i], dt_list[j]
                vals1 = type_pca_coeffs[dt1][:, comp_idx]
                vals2 = type_pca_coeffs[dt2][:, comp_idx]
                
                try:
                    ks_stat, ks_p = ks_2samp(vals1, vals2)
                    wd = wasserstein_distance(vals1, vals2)
                    
                    if ks_stat > max_ks_stat:
                        max_ks_stat = ks_stat
                        max_ks_pair = f"{dt1}_vs_{dt2}"
                except:
                    ks_stat, ks_p, wd = 0, 1, 0
        
        # Also test: can we classify dep_type from this single component?
        comp_vals = pca_coeffs[:, comp_idx].reshape(-1, 1)
        try:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
            scores = cross_val_score(clf, comp_vals, all_y, cv=cv, scoring='accuracy')
            comp_classify_acc = float(scores.mean())
        except:
            comp_classify_acc = 0
        
        # How much of the total variance is explained by this component
        var_explained = pca.explained_variance_ratio_[comp_idx]
        
        component_analysis[comp_idx] = {
            "var_explained": float(var_explained),
            "type_means": type_means,
            "type_stds": type_stds,
            "max_ks_stat": float(max_ks_stat),
            "max_ks_pair": max_ks_pair,
            "classify_acc": comp_classify_acc,
        }
        
        if comp_idx < 10 or comp_classify_acc > 0.15:
            print(f"  PC{comp_idx}: var={var_explained:.4f}, "
                  f"classify_acc={comp_classify_acc:.3f}, "
                  f"max_KS={max_ks_stat:.3f}({max_ks_pair})")
    
    results["component_analysis"] = component_analysis
    
    # Key question: is grammar info in high-variance or low-variance components?
    # Compute cumulative classification accuracy using first k components
    print(f"\n  === Cumulative classification ===")
    cumulative_results = {}
    
    for k in [1, 2, 3, 5, 7, 10, 20, 50]:
        if k > n_pca:
            break
        
        X_k = pca_coeffs[:, :k]
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0)
            cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
            scores = cross_val_score(clf, X_k, all_y, cv=cv, scoring='accuracy')
            acc = float(scores.mean())
        except:
            acc = 0
        
        cum_var = float(np.sum(pca.explained_variance_ratio_[:k]))
        cumulative_results[k] = {"acc": acc, "cum_var": cum_var}
        print(f"  Top-{k} PCs: acc={acc:.3f}, cum_var={cum_var:.4f}")
    
    results["cumulative_classification"] = cumulative_results
    
    # Also compare: PCA-k classification vs LDA classification
    print(f"\n  === PCA vs LDA classification ===")
    
    # LDA on original space
    n_classes = len(np.unique(all_y))
    n_lda = min(n_classes - 1, 7)
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    X_lda = lda.fit_transform(all_h_deps, all_y)
    
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    lda_scores = cross_val_score(clf, X_lda, all_y, cv=cv, scoring='accuracy')
    
    # PCA-top-k classification
    best_k = max(cumulative_results.keys(), key=lambda k: cumulative_results[k]["acc"])
    pca_acc = cumulative_results[best_k]["acc"]
    
    print(f"  LDA classification: {lda_scores.mean():.3f}")
    print(f"  PCA-top-{best_k} classification: {pca_acc:.3f}")
    print(f"  Ratio (PCA/LDA): {pca_acc/max(lda_scores.mean(), 0.01):.3f}")
    
    results["lda_acc"] = float(lda_scores.mean())
    results["best_pca_k"] = best_k
    results["best_pca_acc"] = pca_acc
    results["pca_lda_ratio"] = float(pca_acc / max(lda_scores.mean(), 0.01))
    
    return results


# ============================================================
# Exp2: PCA方向上的因果干预
# ============================================================

def exp2_pca_causal_intervention(model_name, model, tokenizer, device):
    """
    Intervene along PCA directions and measure effect on output.
    Compare with LDA directions as control.
    """
    print(f"\n{'='*70}")
    print(f"Exp2: PCA方向因果干预 ({model_name})")
    print(f"{'='*70}")
    
    last_layer = LAST_LAYERS[model_name]
    layers = get_layers(model)
    d_model = model.get_input_embeddings().weight.shape[1]
    
    # Extract features
    pairs = parse_with_spacy(NATURAL_SENTENCES)
    type_sampled = {dt: 0 for dt in TARGET_DEP_TYPES}
    balanced_pairs = []
    for p in pairs:
        _, _, _, dt = p
        if dt in TARGET_DEP_TYPES and type_sampled[dt] < 50:
            balanced_pairs.append(p)
            type_sampled[dt] += 1
    
    all_h_deps = []
    all_y = []
    for i, (sent, h_idx, d_idx, dep_type) in enumerate(balanced_pairs):
        if i % 50 == 0:
            print(f"    Extracting {i}/{len(balanced_pairs)}...")
        try:
            feats = extract_all_layers_for_pair(
                model, tokenizer, sent, h_idx, d_idx, [last_layer]
            )
            if last_layer in feats and dep_type in TARGET_DEP_TYPES:
                all_h_deps.append(feats[last_layer]["h_dep"])
                all_y.append(TARGET_DEP_TYPES.index(dep_type))
        except:
            continue
    
    all_h_deps = np.array(all_h_deps)
    all_y = np.array(all_y)
    
    if len(all_y) < 50:
        return None
    
    # Fit PCA
    pca = PCA(n_components=min(20, d_model, len(all_y)-1))
    pca.fit(all_h_deps)
    
    # Also fit LDA for comparison
    all_pair_concats = []
    for i, (sent, h_idx, d_idx, dep_type) in enumerate(balanced_pairs):
        try:
            feats = extract_all_layers_for_pair(
                model, tokenizer, sent, h_idx, d_idx, [last_layer]
            )
            if last_layer in feats and dep_type in TARGET_DEP_TYPES:
                all_pair_concats.append(feats[last_layer]["pair_concat"])
        except:
            continue
    
    if len(all_pair_concats) < 50:
        return None
    
    all_pair_concats = np.array(all_pair_concats)
    n_classes = len(np.unique(all_y[:len(all_pair_concats)]))
    lda = LinearDiscriminantAnalysis(n_components=min(n_classes-1, 7))
    lda.fit(all_pair_concats, all_y[:len(all_pair_concats)])
    lda_dep_dirs = lda.scalings_[d_model:, :]
    from scipy.linalg import orth
    U_lda = orth(lda_dep_dirs)
    
    # Test sentences for intervention
    test_sentences = [
        ["The cat sat on the mat", "cat", "nsubj"],
        ["She ate the apple", "apple", "dobj"],
        ["The red car stopped", "red", "amod"],
        ["She ran quickly home", "quickly", "advmod"],
    ]
    
    alphas = [1, 5, 10, 20]
    
    results = {
        "model": model_name,
        "last_layer": last_layer,
        "pca_top5_var": pca.explained_variance_ratio_[:5].tolist(),
        "alphas": alphas,
        "intervention_results": [],
    }
    
    for sent_data in test_sentences:
        sentence, dep_word, expected_dep = sent_data
        
        toks = tokenizer(sentence, return_tensors="pt").to(device)
        input_ids = toks["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        has_bos = tokens[0] in ['<s>', '<|begin_of_text|>', '<|im_start|>', 'Ċ']
        
        # Find dep token
        offset = 1 if has_bos else 0
        dep_idx = None
        for i, t in enumerate(tokens[offset:], offset):
            if dep_word.lower() in t.lower().strip("▁Ġ").strip():
                dep_idx = i
                break
        
        if dep_idx is None:
            continue
        
        print(f"\n  '{sentence}' dep={dep_word}@{dep_idx}")
        
        # Get base output
        with torch.no_grad():
            base_output = model(**toks)
            base_logits = base_output.logits.detach().float().cpu()
        
        sent_result = {
            "sentence": sentence,
            "dep_word": dep_word,
            "dep_idx": dep_idx,
            "expected_dep": expected_dep,
            "interventions": [],
        }
        
        # Intervene along each PCA direction
        for comp_idx in range(min(7, pca.n_components_)):
            direction = pca.components_[comp_idx]  # [d_model]
            var = pca.explained_variance_ratio_[comp_idx]
            
            for alpha in alphas:
                # Add direction at dep position
                dir_t = torch.tensor(direction * alpha, dtype=torch.float32, device=device)
                
                def make_hook(pos, dir_vec):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].clone()
                            if pos < h.shape[1]:
                                h[0, pos, :] += dir_vec.to(h.dtype)
                            return (h,) + output[1:]
                        else:
                            h = output.clone()
                            if pos < h.shape[1]:
                                h[0, pos, :] += dir_vec.to(h.dtype)
                            return h
                    return hook
                
                hook = layers[-1].register_forward_hook(make_hook(dep_idx, dir_t))
                with torch.no_grad():
                    interv_output = model(**toks)
                    interv_logits = interv_output.logits.detach().float().cpu()
                hook.remove()
                
                # Compute KL
                n_pos = base_logits.shape[1]
                total_kl = 0
                for pos in range(min(n_pos, 15)):
                    bp = torch.softmax(base_logits[0, pos], dim=-1).numpy()
                    ip = torch.softmax(interv_logits[0, pos], dim=-1).numpy()
                    total_kl += float(np.sum(bp * np.log(bp / (ip + 1e-10) + 1e-10)))
                
                interv_data = {
                    "type": "PCA",
                    "component": comp_idx,
                    "var_explained": float(var),
                    "alpha": alpha,
                    "total_kl": float(total_kl),
                }
                sent_result["interventions"].append(interv_data)
                
                if comp_idx == 0 and alpha == alphas[-1]:
                    print(f"    PC{comp_idx}(var={var:.4f}) α={alpha}: KL={total_kl:.4f}")
        
        # Also intervene along LDA directions (control)
        for comp_idx in range(min(7, U_lda.shape[1])):
            direction = U_lda[:, comp_idx]
            
            for alpha in alphas:
                dir_t = torch.tensor(direction * alpha, dtype=torch.float32, device=device)
                
                hook = layers[-1].register_forward_hook(make_hook(dep_idx, dir_t))
                with torch.no_grad():
                    interv_output = model(**toks)
                    interv_logits = interv_output.logits.detach().float().cpu()
                hook.remove()
                
                n_pos = base_logits.shape[1]
                total_kl = 0
                for pos in range(min(n_pos, 15)):
                    bp = torch.softmax(base_logits[0, pos], dim=-1).numpy()
                    ip = torch.softmax(interv_logits[0, pos], dim=-1).numpy()
                    total_kl += float(np.sum(bp * np.log(bp / (ip + 1e-10) + 1e-10)))
                
                interv_data = {
                    "type": "LDA",
                    "component": comp_idx,
                    "alpha": alpha,
                    "total_kl": float(total_kl),
                }
                sent_result["interventions"].append(interv_data)
        
        results["intervention_results"].append(sent_result)
    
    # Summary: PCA vs LDA intervention effect
    pca_kls = []
    lda_kls = []
    for sr in results["intervention_results"]:
        for iv in sr["interventions"]:
            if iv["type"] == "PCA":
                pca_kls.append((iv["component"], iv["var_explained"], iv["alpha"], iv["total_kl"]))
            else:
                lda_kls.append((iv["component"], iv["alpha"], iv["total_kl"]))
    
    print(f"\n  === PCA vs LDA intervention effect ===")
    for alpha in alphas:
        pca_alpha_kls = [kl for (_, _, a, kl) in pca_kls if a == alpha]
        lda_alpha_kls = [kl for (_, a, kl) in lda_kls if a == alpha]
        
        pca_mean = np.mean(pca_alpha_kls) if pca_alpha_kls else 0
        lda_mean = np.mean(lda_alpha_kls) if lda_alpha_kls else 0
        
        results[f"pca_alpha{alpha}_mean_kl"] = float(pca_mean)
        results[f"lda_alpha{alpha}_mean_kl"] = float(lda_mean)
        
        print(f"  α={alpha}: PCA_KL={pca_mean:.4f}, LDA_KL={lda_mean:.4f}, "
              f"PCA/LDA={pca_mean/max(lda_mean,1e-10):.1f}x")
    
    # Per PCA component effect
    print(f"\n  === Per-PCA-component intervention (α={alphas[-1]}) ===")
    for comp_idx in range(min(7, pca.n_components_)):
        comp_kls = [kl for (c, _, a, kl) in pca_kls if c == comp_idx and a == alphas[-1]]
        var = pca.explained_variance_ratio_[comp_idx]
        if comp_kls:
            print(f"  PC{comp_idx}(var={var:.4f}): mean_KL={np.mean(comp_kls):.4f}")
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2])
    args = parser.parse_args()
    
    model_name = args.model
    exp_num = args.exp
    
    print(f"\n{'='*70}")
    print(f"CCL-C(250.3): PCA空间语法调制分析")
    print(f"Model: {model_name}, Exp: {exp_num}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  Model: {model_info.model_class}, layers={model_info.n_layers}, d_model={model_info.d_model}")
    
    try:
        if exp_num == 1:
            result = exp1_pca_conditional_distribution(model_name, model, tokenizer, device)
        elif exp_num == 2:
            result = exp2_pca_causal_intervention(model_name, model, tokenizer, device)
        
        if result is not None:
            out_path = f"tests/glm5_temp/cclc_exp{exp_num}_{model_name}_results.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n  Results saved to {out_path}")
    
    finally:
        release_model(model)
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
