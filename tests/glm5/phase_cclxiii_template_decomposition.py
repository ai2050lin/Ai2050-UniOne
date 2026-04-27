"""
Phase CCLXIII: 模板信号谱分解 — 分析残差流90%模板信号的内部结构
================================================================
核心问题: 模板信号(残差流的90%+)到底是什么? 包含哪些子结构?

3个子实验:
  Exp1: 模板信号PCA/ICA分解 — 对模板均值做PCA+ICA, 看内部结构
  Exp2: 去模板后残余信号分析 — 减去模板后, 残余信号的结构和可解释性
  Exp3: 模板vs内容的方差分解 — 量化"模板"/"内容"/"噪声"各占多少方差

填充: GR-1a(语法编码), KN-1a(概念编码维度), 核心问题1

用法:
  python phase_cclxiii_template_decomposition.py --model qwen3 --exp 1
"""
import argparse, os, sys, json, time, gc
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
)

OUTPUT_DIR = Path("results/causal_fiber")

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "snake"],
    "food":    ["apple", "rice", "bread", "cheese", "salmon", "mango"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "chisel"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "volcano"],
}

TEMPLATES = [
    "The {} is",
    "I saw a {} today",
    "This {} looks",
    "A {} can be",
    "The {} was",
    "My {} is very",
    "Every {} has",
    "That {} seems",
]


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def collect_residuals_at_position(model, tokenizer, device, prompt, target_token_str, n_layers):
    input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
    target_pos = None
    for i, t in enumerate(tokens):
        if target_token_str.lower() in t.lower().replace("▁", "").replace("Ġ", ""):
            target_pos = i
            break
    if target_pos is None:
        target_pos = len(tokens) - 1
    
    captured = {}
    layers = get_layers(model)
    
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
    
    with torch.no_grad():
        try:
            _ = model(input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
    
    for h in hooks:
        h.remove()
    
    residuals = []
    for li in range(n_layers):
        key = f"L{li}"
        if key in captured:
            residuals.append(captured[key][0, target_pos].numpy())
        else:
            residuals.append(None)
    
    return residuals, target_pos


# ============================================================
# Exp1: 模板信号PCA/ICA分解
# ============================================================
def exp1_template_pca_ica(args):
    """对大量句子的残差流做PCA+ICA, 分解模板信号的内部结构"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp1: 模板信号PCA/ICA分解 ({args.model}, {n_layers}层, d={d_model})")
    
    # 收集大量句子的残差流(在概念位置)
    all_words = []
    for cat, words in CONCEPTS.items():
        all_words.extend(words)
    
    # 对每个词×每个模板收集残差
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # 收集残差矩阵: [n_samples, d_model] at each sample layer
    residual_matrices = {li: [] for li in sample_layers}
    sample_labels = []  # (word, template, category)
    
    print(f"  收集残差: {len(all_words)}词 × {len(TEMPLATES)}模板 = {len(all_words)*len(TEMPLATES)}样本")
    
    for word in all_words:
        cat = None
        for c, ws in CONCEPTS.items():
            if word in ws:
                cat = c
                break
        
        for template in TEMPLATES:
            prompt = template.format(word)
            resid, pos = collect_residuals_at_position(
                model, tokenizer, device, prompt, word, n_layers)
            
            for li in sample_layers:
                if resid[li] is not None:
                    residual_matrices[li].append(resid[li])
            
            sample_labels.append({"word": word, "template": template, "category": cat})
    
    # 释放模型
    release_model(model)
    
    # PCA + ICA 分析
    from sklearn.decomposition import PCA, FastICA
    from sklearn.preprocessing import StandardScaler
    
    results = {}
    
    for li in sample_layers:
        X = np.array(residual_matrices[li])
        n_samples = X.shape[0]
        print(f"\n  === L{li}: {n_samples}样本 × {d_model}维 ===")
        
        if n_samples < 10:
            print(f"    样本太少, 跳过")
            continue
        
        # 1. 原始PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=min(30, d_model, n_samples-1))
        X_pca = pca.fit_transform(X_scaled)
        
        # 前10个PC的解释方差
        top_var = pca.explained_variance_ratio_[:10]
        cum_var = np.cumsum(top_var)
        
        print(f"    PCA前10分量解释方差: {[f'{v:.4f}' for v in top_var]}")
        print(f"    累计: {[f'{v:.4f}' for v in cum_var]}")
        
        # 2. 用前5个PC的载荷分析: 哪些因素在驱动?
        # 对每个PC, 分析它是否与类别/模板/词相关
        pc_interpretations = {}
        for pc_idx in range(min(5, X_pca.shape[1])):
            pc_values = X_pca[:, pc_idx]
            
            # 按类别分组
            cat_means = {}
            for ci, label in enumerate(sample_labels):
                c = label["category"]
                if c not in cat_means:
                    cat_means[c] = []
                cat_means[c].append(pc_values[ci])
            
            # 按模板分组
            tmpl_means = {}
            for ci, label in enumerate(sample_labels):
                t = label["template"]
                if t not in tmpl_means:
                    tmpl_means[t] = []
                tmpl_means[t].append(pc_values[ci])
            
            # ANOVA-like: 组间方差 / 总方差
            grand_mean = np.mean(pc_values)
            total_var = np.var(pc_values)
            
            cat_between_var = sum(
                len(v) * (np.mean(v) - grand_mean)**2 
                for v in cat_means.values()
            ) / len(pc_values) if total_var > 0 else 0
            
            tmpl_between_var = sum(
                len(v) * (np.mean(v) - grand_mean)**2 
                for v in tmpl_means.values()
            ) / len(pc_values) if total_var > 0 else 0
            
            cat_r2 = cat_between_var / total_var if total_var > 0 else 0
            tmpl_r2 = tmpl_between_var / total_var if total_var > 0 else 0
            
            pc_interpretations[str(pc_idx)] = {
                "category_r2": round(float(cat_r2), 4),
                "template_r2": round(float(tmpl_r2), 4),
                "var_explained": round(float(top_var[pc_idx]), 4),
                "cat_means": {c: round(float(np.mean(v)), 3) for c, v in cat_means.items()},
                "tmpl_means": {t: round(float(np.mean(v)), 3) for t, v in tmpl_means.items()},
            }
            
            dominant = "category" if cat_r2 > tmpl_r2 else "template"
            print(f"    PC{pc_idx}: var={top_var[pc_idx]:.4f}, "
                  f"cat_R2={cat_r2:.4f}, tmpl_R2={tmpl_r2:.4f} -> {dominant}")
        
        # 3. ICA (在PCA降维后)
        n_ica = min(10, X_pca.shape[1])
        try:
            ica = FastICA(n_components=n_ica, random_state=42, max_iter=500)
            X_ica = ica.fit_transform(X_pca[:, :n_ica])
            
            ica_interpretations = {}
            for ic_idx in range(n_ica):
                ic_values = X_ica[:, ic_idx]
                
                cat_means_ic = {}
                for ci, label in enumerate(sample_labels):
                    c = label["category"]
                    if c not in cat_means_ic:
                        cat_means_ic[c] = []
                    cat_means_ic[c].append(ic_values[ci])
                
                tmpl_means_ic = {}
                for ci, label in enumerate(sample_labels):
                    t = label["template"]
                    if t not in tmpl_means_ic:
                        tmpl_means_ic[t] = []
                    tmpl_means_ic[t].append(ic_values[ci])
                
                grand_mean_ic = np.mean(ic_values)
                total_var_ic = np.var(ic_values)
                
                cat_between_ic = sum(
                    len(v) * (np.mean(v) - grand_mean_ic)**2 
                    for v in cat_means_ic.values()
                ) / len(ic_values) if total_var_ic > 0 else 0
                
                tmpl_between_ic = sum(
                    len(v) * (np.mean(v) - grand_mean_ic)**2 
                    for v in tmpl_means_ic.values()
                ) / len(ic_values) if total_var_ic > 0 else 0
                
                cat_r2_ic = cat_between_ic / total_var_ic if total_var_ic > 0 else 0
                tmpl_r2_ic = tmpl_between_ic / total_var_ic if total_var_ic > 0 else 0
                
                ica_interpretations[str(ic_idx)] = {
                    "category_r2": round(float(cat_r2_ic), 4),
                    "template_r2": round(float(tmpl_r2_ic), 4),
                }
            
            ica_summary = {
                "n_components": n_ica,
                "interpretations": ica_interpretations,
                "mean_cat_r2": round(float(np.mean([v["category_r2"] for v in ica_interpretations.values()])), 4),
                "mean_tmpl_r2": round(float(np.mean([v["template_r2"] for v in ica_interpretations.values()])), 4),
            }
        except Exception as e:
            ica_summary = {"error": str(e)}
        
        results[str(li)] = {
            "n_samples": n_samples,
            "pca_top10_var": [round(float(v), 6) for v in top_var],
            "pca_cum_var_10": round(float(cum_var[-1]), 4),
            "pc_interpretations": pc_interpretations,
            "ica_summary": ica_summary,
        }
    
    out_dir = OUTPUT_DIR / f"{args.model}_cclxiii"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "phase": "CCLXIII",
        "exp": 1,
        "date": datetime.now().isoformat(),
        "model": args.model,
        "model_info": {"n_layers": n_layers, "d_model": d_model},
        "results": results,
    }
    
    with open(out_dir / "exp1_template_pca_ica.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  === Exp1 总结 ===")
    for li_str, r in sorted(results.items(), key=lambda x: int(x[0])):
        print(f"  L{li_str}: PCA10累计={r['pca_cum_var_10']:.4f}")
        for pc_idx, pci in r.get("pc_interpretations", {}).items():
            dominant = "CAT" if pci["category_r2"] > pci["template_r2"] else "TMPL"
            print(f"    PC{pc_idx}: var={pci['var_explained']:.4f}, "
                  f"cat_R2={pci['category_r2']:.4f}, tmpl_R2={pci['template_r2']:.4f} -> {dominant}")
    
    return output


# ============================================================
# Exp2: 去模板后残余信号分析
# ============================================================
def exp2_residual_after_denoising(args):
    """减去模板均值后, 分析残余信号的可解释性"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp2: 去模板后残余信号分析 ({args.model})")
    
    all_words = []
    for cat, words in CONCEPTS.items():
        all_words.extend(words)
    
    template = "The {} is"
    mid_layer = n_layers // 2
    
    # 收集所有概念的残差
    resid_by_word = {}
    for word in all_words:
        prompt = template.format(word)
        resid, pos = collect_residuals_at_position(
            model, tokenizer, device, prompt, word, n_layers)
        resid_by_word[word] = resid
    
    # 收集同一概念在不同模板下的残差(用于去模板)
    multi_template_resids = {}
    test_words = all_words[:12]  # 前12个词
    for word in test_words:
        multi_template_resids[word] = {}
        for tmpl in TEMPLATES[:4]:
            prompt = tmpl.format(word)
            resid, pos = collect_residuals_at_position(
                model, tokenizer, device, prompt, word, n_layers)
            multi_template_resids[word][tmpl] = resid
    
    release_model(model)
    
    results = {}
    
    # 方法1: 减全局均值
    for li in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if resid_by_word[all_words[0]][li] is None:
            continue
        
        all_vecs = [resid_by_word[w][li] for w in all_words if resid_by_word[w][li] is not None]
        if not all_vecs:
            continue
        
        global_mean = np.mean(all_vecs, axis=0)
        
        # 去模板后的残余
        residuals_denoised = {w: v - global_mean for w, v in resid_by_word.items() if v[li] is not None}
        
        # 残余的类别可分性
        cat_centroids = {}
        for cat, words in CONCEPTS.items():
            cat_vecs = [residuals_denoised[w] for w in words if w in residuals_denoised]
            if cat_vecs:
                cat_centroids[cat] = np.mean(cat_vecs, axis=0)
        
        # 类内/类间距离
        intra_dists = []
        inter_dists = []
        cats = list(cat_centroids.keys())
        
        for cat in cats:
            words = [w for w in CONCEPTS[cat] if w in residuals_denoised]
            for w in words:
                intra_dists.append(np.linalg.norm(residuals_denoised[w] - cat_centroids[cat]))
        
        for i, c1 in enumerate(cats):
            for j, c2 in enumerate(cats):
                if i >= j:
                    continue
                inter_dists.append(np.linalg.norm(cat_centroids[c1] - cat_centroids[c2]))
        
        # 类间cos
        inter_cos = []
        for i, c1 in enumerate(cats):
            for j, c2 in enumerate(cats):
                if i >= j:
                    continue
                inter_cos.append(proper_cos(cat_centroids[c1], cat_centroids[c2]))
        
        # 信号强度比: 残余范数 / 原始范数
        mean_orig_norm = float(np.mean([np.linalg.norm(v) for v in all_vecs]))
        mean_resid_norm = float(np.mean([np.linalg.norm(v) for v in residuals_denoised.values()]))
        
        # PCA维度
        from sklearn.decomposition import PCA
        X_resid = np.array(list(residuals_denoised.values()))
        pca = PCA(n_components=min(10, X_resid.shape[0]-1, d_model))
        pca.fit(X_resid)
        
        # 95%方差需要的维度
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        dim_95 = int(np.searchsorted(cum_var, 0.95) + 1) if len(cum_var) > 0 else 0
        dim_90 = int(np.searchsorted(cum_var, 0.90) + 1) if len(cum_var) > 0 else 0
        
        results[str(li)] = {
            "mean_orig_norm": round(mean_orig_norm, 2),
            "mean_resid_norm": round(mean_resid_norm, 2),
            "signal_ratio": round(mean_resid_norm / mean_orig_norm, 4) if mean_orig_norm > 0 else 0,
            "mean_intra_dist": round(float(np.mean(intra_dists)), 2),
            "mean_inter_dist": round(float(np.mean(inter_dists)), 2),
            "intra_inter_ratio": round(float(np.mean(intra_dists)) / float(np.mean(inter_dists)), 4) if inter_dists else 0,
            "mean_inter_cos": round(float(np.mean(inter_cos)), 4),
            "pca_dim_90": dim_90,
            "pca_dim_95": dim_95,
            "pca_top5_var": [round(float(v), 4) for v in pca.explained_variance_ratio_[:5]],
        }
        
        print(f"  L{li}: signal_ratio={results[str(li)]['signal_ratio']:.4f}, "
              f"intra/inter={results[str(li)]['intra_inter_ratio']:.4f}, "
              f"inter_cos={results[str(li)]['mean_inter_cos']:.4f}, "
              f"pca_dim95={dim_95}")
    
    # 方法2: 减模板特异均值(同模板下的全局均值)
    template_analysis = {}
    for li in [n_layers//2, n_layers-1]:
        # 收集同模板下所有词的残差
        tmpl_means = {}
        for tmpl in TEMPLATES[:4]:
            tmpl_vecs = []
            for word in test_words:
                if word in multi_template_resids and tmpl in multi_template_resids[word]:
                    if multi_template_resids[word][tmpl][li] is not None:
                        tmpl_vecs.append(multi_template_resids[word][tmpl][li])
            if tmpl_vecs:
                tmpl_means[tmpl] = np.mean(tmpl_vecs, axis=0)
        
        # 模板间cos: 不同模板的全局均值有多相似?
        tmpl_cos = {}
        tmpls = list(tmpl_means.keys())
        for i in range(len(tmpls)):
            for j in range(i+1, len(tmpls)):
                c = proper_cos(tmpl_means[tmpls[i]], tmpl_means[tmpls[j]])
                tmpl_cos[f"{tmpls[i]}_vs_{tmpls[j]}"] = round(c, 4)
        
        template_analysis[str(li)] = {
            "template_cos": tmpl_cos,
            "mean_template_cos": round(float(np.mean(list(tmpl_cos.values()))), 4) if tmpl_cos else 0,
        }
        
        print(f"  L{li} 模板间cos: mean={template_analysis[str(li)]['mean_template_cos']:.4f}")
    
    out_dir = OUTPUT_DIR / f"{args.model}_cclxiii"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "phase": "CCLXIII",
        "exp": 2,
        "date": datetime.now().isoformat(),
        "model": args.model,
        "model_info": {"n_layers": n_layers, "d_model": d_model},
        "results": results,
        "template_analysis": template_analysis,
    }
    
    with open(out_dir / "exp2_residual_denoising.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    return output


# ============================================================
# Exp3: 方差分解: 模板 vs 内容 vs 噪声
# ============================================================
def exp3_variance_decomposition(args):
    """量化方差来源: 模板(句式) vs 内容(类别+词) vs 噪声(残差)"""
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp3: 方差分解 ({args.model})")
    
    all_words = []
    word_cats = {}
    for cat, words in CONCEPTS.items():
        for w in words:
            all_words.append(w)
            word_cats[w] = cat
    
    # 收集: 词 × 模板 × 层 的残差
    resid_data = {}  # (word, template) -> [layer0, layer1, ...]
    
    for word in all_words:
        for tmpl in TEMPLATES[:6]:  # 6个模板
            prompt = tmpl.format(word)
            resid, pos = collect_residuals_at_position(
                model, tokenizer, device, prompt, word, n_layers)
            resid_data[(word, tmpl)] = resid
    
    release_model(model)
    
    # 方差分解 at each layer
    results = {}
    
    for li in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        # 构建3D矩阵: [n_words, n_templates, d_model]
        mat = np.zeros((len(all_words), len(TEMPLATES[:6]), d_model))
        valid = np.ones((len(all_words), len(TEMPLATES[:6])), dtype=bool)
        
        for wi, word in enumerate(all_words):
            for ti, tmpl in enumerate(TEMPLATES[:6]):
                key = (word, tmpl)
                if key in resid_data and resid_data[key][li] is not None:
                    mat[wi, ti] = resid_data[key][li]
                else:
                    valid[wi, ti] = False
        
        # 全局均值
        valid_vecs = mat[valid]
        grand_mean = np.mean(valid_vecs, axis=0)
        
        # 总方差
        total_var = float(np.mean(np.var(valid_vecs, axis=0)))
        
        if total_var < 1e-10:
            results[str(li)] = {"total_var": 0}
            continue
        
        # 模板主效应: 每个模板的均值 - 全局均值
        tmpl_means = np.zeros((len(TEMPLATES[:6]), d_model))
        for ti in range(len(TEMPLATES[:6])):
            tmpl_vecs = mat[:, ti][valid[:, ti]]
            tmpl_means[ti] = np.mean(tmpl_vecs, axis=0) - grand_mean
        tmpl_var = float(np.mean(np.var(tmpl_means, axis=0)) * len(TEMPLATES[:6]))
        
        # 类别主效应: 每个类别的均值 - 全局均值
        cat_centroids = {}
        for cat in CONCEPTS.keys():
            cat_words = [w for w in all_words if word_cats[w] == cat]
            cat_vecs = []
            for w in cat_words:
                for ti in range(len(TEMPLATES[:6])):
                    if valid[all_words.index(w), ti]:
                        cat_vecs.append(mat[all_words.index(w), ti])
            if cat_vecs:
                cat_centroids[cat] = np.mean(cat_vecs, axis=0) - grand_mean
        
        cat_var_vecs = np.array(list(cat_centroids.values()))
        cat_var = float(np.mean(np.var(cat_var_vecs, axis=0)) * len(cat_centroids))
        
        # 词内方差(同类内不同词的差异)
        within_cat_var = 0
        for cat in CONCEPTS.keys():
            cat_words = [w for w in all_words if word_cats[w] == cat]
            cat_vecs = []
            for w in cat_words:
                for ti in range(len(TEMPLATES[:6])):
                    if valid[all_words.index(w), ti]:
                        cat_vecs.append(mat[all_words.index(w), ti])
            if cat_vecs:
                cat_mean = np.mean(cat_vecs, axis=0)
                within_cat_var += float(np.mean(np.var(cat_vecs, axis=0)))
        within_cat_var /= len(CONCEPTS)
        
        # 残差(交互+噪声)
        residual_var = total_var - tmpl_var - cat_var - within_cat_var
        residual_var = max(residual_var, 0)
        
        # 百分比
        results[str(li)] = {
            "total_var": round(total_var, 2),
            "template_var_pct": round(tmpl_var / total_var * 100, 1),
            "category_var_pct": round(cat_var / total_var * 100, 1),
            "within_cat_var_pct": round(within_cat_var / total_var * 100, 1),
            "residual_var_pct": round(residual_var / total_var * 100, 1),
            "template_var_abs": round(tmpl_var, 2),
            "category_var_abs": round(cat_var, 2),
        }
        
        print(f"  L{li}: total={total_var:.2f}, "
              f"tmpl={tmpl_var/total_var*100:.1f}%, "
              f"cat={cat_var/total_var*100:.1f}%, "
              f"within_cat={within_cat_var/total_var*100:.1f}%, "
              f"residual={residual_var/total_var*100:.1f}%")
    
    out_dir = OUTPUT_DIR / f"{args.model}_cclxiii"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "phase": "CCLXIII",
        "exp": 3,
        "date": datetime.now().isoformat(),
        "model": args.model,
        "model_info": {"n_layers": n_layers, "d_model": d_model},
        "results": results,
    }
    
    with open(out_dir / "exp3_variance_decomposition.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Phase CCLXIII: 模板信号谱分解")
    print(f"Model: {args.model}, Exp: {args.exp}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    if args.exp == 1:
        exp1_template_pca_ica(args)
    elif args.exp == 2:
        exp2_residual_after_denoising(args)
    elif args.exp == 3:
        exp3_variance_decomposition(args)
