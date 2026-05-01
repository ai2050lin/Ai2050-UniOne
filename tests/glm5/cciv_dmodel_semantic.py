"""
CCIV(304): Residual Stream (d_model空间)语义方向分析
====================================================
前置发现: W_down有效秩~430, n_inter空间扰动被压缩, 方向不特异。
现在直接在d_model空间操作, 绕过W_down瓶颈。

实验设计:
  Step 1: 收集4类别在各层的residual stream (d_model维)
  Step 2: ANOVA分析d_model每个维度的类别区分力
  Step 3: PCA分析d_model空间的类别结构
  Step 4: Hook perturb d_model语义方向 vs 随机方向
  Step 5: 直接修改residual stream做因果干预

用法:
  python cciv_dmodel_semantic.py --model qwen3
  python cciv_dmodel_semantic.py --model glm4
  python cciv_dmodel_semantic.py --model deepseek7b
"""
import argparse, os, sys, time, gc, json
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict
from scipy import stats
from scipy.sparse.linalg import svds

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "cciv_dmodel_semantic_log.txt"

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "lion", "bear", "fish", "snake", "whale",
                "rabbit", "deer", "fox", "wolf", "tiger", "monkey", "elephant", "dolphin", "parrot", "duck"],
    "food":    ["apple", "rice", "bread", "cheese", "pizza", "banana", "mango", "pasta", "salad", "steak",
                "soup", "cake", "cookie", "grape", "lemon", "peach", "corn", "bean", "pepper", "onion"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "screw", "pliers", "chisel", "level", "ruler",
                "shovel", "axe", "clamp", "welder", "plane", "anvil", "lathe", "forge", "drill", "mallet"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "valley", "canyon", "island", "meadow", "glacier",
                "volcano", "waterfall", "swamp", "tundra", "prairie", "cliff", "reef", "lagoon", "cave", "ridge"],
}

TEMPLATE = "The {} is"

N_CONCEPTS_PER_CAT = 15  # 每类15个词, 4类=60个样本


def log_f(msg="", end="\n"):
    print(msg, end=end, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + end)


def collect_residual_stream(model, tokenizer, device, model_info):
    """收集4类别在各层的residual stream (last token position)"""
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    categories = list(CONCEPTS.keys())
    all_words = []
    word_cats = []
    for cat in categories:
        words = CONCEPTS[cat][:N_CONCEPTS_PER_CAT]
        all_words.extend(words)
        word_cats.extend([cat] * len(words))
    
    n_words = len(all_words)
    log_f(f"  Collecting residual stream for {n_words} words ({N_CONCEPTS_PER_CAT}/cat)")
    
    # 存储各层residual stream
    # residual_stream[layer_idx] = [n_words, d_model]
    residual_stream = {}
    
    # 逐词forward, 用hook收集每层输出
    for wi, word in enumerate(all_words):
        prompt = TEMPLATE.format(word)
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        
        captured = {}
        hooks = []
        
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        for li in range(n_layers):
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(**toks)
            except Exception as e:
                log_f(f"  Forward failed for '{word}': {e}")
        
        for h in hooks:
            h.remove()
        
        # Extract last token position
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                vec = captured[key][0, -1, :].numpy()  # [d_model]
                if li not in residual_stream:
                    residual_stream[li] = []
                residual_stream[li].append(vec)
        
        if (wi + 1) % 15 == 0:
            log_f(f"    {wi+1}/{n_words} words done")
    
    # Convert to arrays
    for li in residual_stream:
        residual_stream[li] = np.array(residual_stream[li])  # [n_words, d_model]
    
    return residual_stream, all_words, word_cats, categories


def anova_dmodel(residual_stream, word_cats, categories, sample_layers):
    """对d_model每个维度做ANOVA, 找类别区分维度"""
    log_f("\n  === Step 2: d_model ANOVA ===")
    
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    cat_indices = defaultdict(list)
    for wi, cat in enumerate(word_cats):
        cat_indices[cat].append(wi)
    
    results = {}
    
    for li in sample_layers:
        if li not in residual_stream:
            continue
        h = residual_stream[li]  # [n_words, d_model]
        d_model = h.shape[1]
        
        # 4-group ANOVA per dimension
        groups = [h[cat_indices[c]] for c in categories]  # list of [n_per_cat, d_model]
        
        sig_count_05 = 0
        sig_count_001 = 0
        max_f = 0
        max_f_dim = 0
        top_dims = []  # (F, dim, eta_sq, per_cat_means)
        
        for dim in range(d_model):
            dim_vals = [g[:, dim] for g in groups]
            F, p = stats.f_oneway(*dim_vals)
            
            if p < 0.05:
                sig_count_05 += 1
            if p < 0.001:
                sig_count_001 += 1
            
            if F > max_f:
                max_f = F
                max_f_dim = dim
            
            # eta_sq for top dims
            if p < 0.001:
                ss_between = sum(len(g) * (g[:, dim].mean() - h[:, dim].mean())**2 
                                for g in groups)
                ss_total = np.sum((h[:, dim] - h[:, dim].mean())**2)
                eta_sq = ss_between / max(ss_total, 1e-10)
                per_cat_means = {c: float(g[:, dim].mean()) for c, g in zip(categories, groups)}
                top_dims.append((float(F), dim, float(eta_sq), per_cat_means))
        
        top_dims.sort(reverse=True)
        
        log_f(f"  L{li}: sig(p<0.05)={sig_count_05}/{d_model} ({100*sig_count_05/d_model:.1f}%), "
              f"sig(p<0.001)={sig_count_001}, max_F={max_f:.1f} (dim{max_f_dim})")
        
        # Print top-5 dims
        for rank, (F, dim, eta_sq, means) in enumerate(top_dims[:5]):
            means_str = ", ".join([f"{c}={means[c]:.2f}" for c in categories])
            log_f(f"    #{rank+1} dim{dim}: F={F:.1f}, eta_sq={eta_sq:.4f}, means=[{means_str}]")
        
        results[li] = {
            "sig_05": sig_count_05,
            "sig_001": sig_count_001,
            "total_dims": d_model,
            "max_F": float(max_f),
            "max_F_dim": int(max_f_dim),
            "top_dims": [(F, dim, eta_sq, means) for F, dim, eta_sq, means in top_dims[:20]],
        }
    
    return results


def pca_dmodel(residual_stream, word_cats, categories, sample_layers):
    """PCA分析d_model空间的类别结构"""
    log_f("\n  === Step 3: d_model PCA ===")
    
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    cat_indices = defaultdict(list)
    for wi, cat in enumerate(word_cats):
        cat_indices[cat].append(wi)
    
    results = {}
    
    for li in sample_layers:
        if li not in residual_stream:
            continue
        h = residual_stream[li]  # [n_words, d_model]
        n_words, d_model = h.shape
        
        # Center
        h_centered = h - h.mean(axis=0, keepdims=True)
        
        # SVD for PCA
        # h_centered = U @ S @ Vt, Vt rows = PC directions
        n_components = min(50, min(h_centered.shape) - 1)
        U, s, Vt = np.linalg.svd(h_centered, full_matrices=False)
        
        explained_var_ratio = (s**2) / np.sum(s**2)
        cum_var = np.cumsum(explained_var_ratio)
        
        log_f(f"  L{li}: top-5 var={explained_var_ratio[:5].round(4).tolist()}, "
              f"cum(10)={cum_var[9]:.4f}, cum(50)={cum_var[min(49,n_components-1)]:.4f}")
        
        # Project onto top PCs and compute per-category means
        proj = h_centered @ Vt[:10].T  # [n_words, 10]
        for pc in range(min(5, 10)):
            cat_means = {c: float(proj[cat_indices[c], pc].mean()) for c in categories}
            means_str = ", ".join([f"{c}={cat_means[c]:.2f}" for c in categories])
            log_f(f"    PC{pc}: {means_str}")
        
        # ANOVA on PC projections
        log_f(f"    PC ANOVA:")
        for pc in range(min(10, n_components)):
            groups = [proj[cat_indices[c], pc] for c in categories]
            F, p = stats.f_oneway(*groups)
            if p < 0.01:
                log_f(f"      PC{pc}: F={F:.1f}, p={p:.4e}")
        
        results[li] = {
            "explained_var_ratio": explained_var_ratio[:20].tolist(),
            "cum_var_10": float(cum_var[9]),
            "pc_directions": Vt[:10],  # [10, d_model]
        }
    
    return results


def hook_perturb_dmodel(model, tokenizer, device, model_info, 
                         residual_stream, word_cats, categories,
                         anova_results, pca_results, sample_layers):
    """Hook perturb在d_model空间: 语义方向 vs 随机方向"""
    log_f("\n  === Step 4: d_model Hook Perturb (semantic vs random) ===")
    
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # Test words (separate from training)
    test_words = {
        "animals": ["tiger", "eagle", "whale", "fox", "deer"],
        "food":    ["mango", "steak", "soup", "grape", "peach"],
        "tools":   ["pliers", "chisel", "shovel", "clamp", "anvil"],
        "nature":  ["glacier", "canyon", "meadow", "swamp", "cliff"],
    }
    
    EPS_LIST = [0.5, 1.0, 2.0]
    N_RANDOM = 5  # 随机方向数量
    
    results = {}
    
    for li in sample_layers:
        if li not in anova_results:
            continue
        
        log_f(f"\n  --- L{li} ---")
        
        # 获取语义方向
        # 方向1: ANOVA top维度方向 (单维度)
        top_anova = anova_results[li]["top_dims"]
        if len(top_anova) == 0:
            log_f(f"    No significant ANOVA dims, skipping")
            continue
        
        # 方向2: LDA方向 (多维度组合)
        # 用top-50 ANOVA维度做LDA
        top50_dims = [d for _, d, _, _ in top_anova[:50]]
        
        # 方向3: PC1方向
        pc_dirs = pca_results[li]["pc_directions"]  # [10, d_model]
        
        # 构建方向集
        directions = {}
        
        # (a) ANOVA top-1 维度方向 (单位向量)
        dim1 = top_anova[0][1]
        dir_anova1 = np.zeros(d_model)
        dir_anova1[dim1] = 1.0
        directions["anova_top1"] = dir_anova1
        
        # (b) ANOVA top-5 维度的加权组合
        dir_anova5 = np.zeros(d_model)
        for rank, (F, dim, eta_sq, means) in enumerate(top_anova[:5]):
            dir_anova5[dim] = F
        dir_anova5 = dir_anova5 / max(np.linalg.norm(dir_anova5), 1e-10)
        directions["anova_top5"] = dir_anova5
        
        # (c) LDA方向
        h = residual_stream[li]  # [n_words, d_model]
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        cat_indices = defaultdict(list)
        for wi, cat in enumerate(word_cats):
            cat_indices[cat].append(wi)
        
        n_lda_dims = min(50, len(top50_dims))
        lda_dims = top50_dims[:n_lda_dims]
        h_sub = h[:, lda_dims]  # [n_words, n_lda_dims]
        
        # LDA: maximize between-class / within-class
        n_classes = len(categories)
        overall_mean = h_sub.mean(axis=0)
        
        S_w = np.zeros((n_lda_dims, n_lda_dims))
        S_b = np.zeros((n_lda_dims, n_lda_dims))
        for cat in categories:
            idx = cat_indices[cat]
            class_mean = h_sub[idx].mean(axis=0)
            S_b += len(idx) * np.outer(class_mean - overall_mean, class_mean - overall_mean)
            for x in h_sub[idx]:
                diff = (x - class_mean).reshape(-1, 1)
                S_w += diff @ diff.T
        
        try:
            S_w_reg = S_w + 1e-6 * np.eye(n_lda_dims)  # regularize
            S_w_inv = np.linalg.pinv(S_w_reg)
            eigvals, eigvecs = np.linalg.eig(S_w_inv @ S_b)
            sort_idx = np.argsort(-np.real(eigvals))
            lda_dir_sub = np.real(eigvecs[:, sort_idx[0]])  # [n_lda_dims]
            
            dir_lda = np.zeros(d_model)
            for i, dim in enumerate(lda_dims):
                dir_lda[dim] = lda_dir_sub[i]
            dir_lda = dir_lda / max(np.linalg.norm(dir_lda), 1e-10)
            directions["lda"] = dir_lda
        except Exception as e:
            log_f(f"    LDA failed: {e}, skipping")
        
        # (d) PC1方向
        directions["pc1"] = pc_dirs[0] / max(np.linalg.norm(pc_dirs[0]), 1e-10)
        
        # (e) 随机方向 (d_model维)
        np.random.seed(42)
        for ri in range(N_RANDOM):
            rand_dir = np.random.randn(d_model)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            directions[f"random_{ri}"] = rand_dir
        
        log_f(f"    Direction set: {list(directions.keys())}")
        
        # 测试每个方向
        results[li] = {}
        
        for dir_name, direction in directions.items():
            results[li][dir_name] = {}
            
            for eps in EPS_LIST:
                top1_changes = 0
                total_tests = 0
                delta_norms = []
                logit_coss = []
                
                for cat, words in test_words.items():
                    for word in words:
                        prompt = TEMPLATE.format(word)
                        toks = tokenizer(prompt, return_tensors="pt").to(device)
                        
                        # Baseline forward
                        with torch.no_grad():
                            base_out = model(**toks)
                            base_logits = base_out.logits[0, -1].float().cpu().numpy()
                            base_top1 = int(np.argmax(base_logits))
                        
                        # Hook perturb: modify residual stream at layer li
                        captured = {}
                        hooks = []
                        
                        def make_perturb_hook(key, direction_np, eps_val, d_m):
                            def hook(module, input, output):
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()
                                # Add perturbation to last token position
                                dir_tensor = torch.tensor(direction_np, dtype=out.dtype, device=out.device)
                                out[0, -1, :] += eps_val * dir_tensor
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return hook
                        
                        hooks.append(layers[li].register_forward_hook(
                            make_perturb_hook(f"L{li}", direction, eps, d_model)
                        ))
                        
                        with torch.no_grad():
                            pert_out = model(**toks)
                            pert_logits = pert_out.logits[0, -1].float().cpu().numpy()
                            pert_top1 = int(np.argmax(pert_logits))
                        
                        for h in hooks:
                            h.remove()
                        
                        # Measure
                        delta_logits = pert_logits - base_logits
                        delta_norm = float(np.linalg.norm(delta_logits))
                        
                        base_norm = float(np.linalg.norm(base_logits))
                        pert_norm = float(np.linalg.norm(pert_logits))
                        if base_norm > 0 and pert_norm > 0:
                            cos = float(np.dot(pert_logits, base_logits) / (pert_norm * base_norm))
                        else:
                            cos = 1.0
                        
                        if pert_top1 != base_top1:
                            top1_changes += 1
                        total_tests += 1
                        delta_norms.append(delta_norm)
                        logit_coss.append(cos)
                
                avg_delta = float(np.mean(delta_norms))
                avg_cos = float(np.mean(logit_coss))
                top1_rate = top1_changes / max(total_tests, 1)
                
                results[li][dir_name][str(eps)] = {
                    "top1_rate": float(top1_rate),
                    "avg_delta_norm": avg_delta,
                    "avg_logit_cos": avg_cos,
                    "n_tests": total_tests,
                }
                
                is_semantic = dir_name not in [f"random_{ri}" for ri in range(N_RANDOM)]
                label = "SEM" if is_semantic else "RND"
                log_f(f"    {label} {dir_name} eps={eps}: top1={top1_rate:.3f} "
                      f"||dL||={avg_delta:.1f} cos={avg_cos:.4f}")
    
    return results


def causal_intervention_dmodel(model, tokenizer, device, model_info, 
                                residual_stream, word_cats, categories,
                                sample_layers):
    """Step 5: 直接在d_model空间做swap因果干预"""
    log_f("\n  === Step 5: d_model Causal Intervention (swap) ===")
    
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 4类别各取5个词
    swap_words = {
        "animals": ["dog", "cat", "horse", "eagle", "shark"],
        "food":    ["apple", "rice", "bread", "cheese", "pizza"],
        "tools":   ["hammer", "knife", "saw", "drill", "wrench"],
        "nature":  ["mountain", "river", "ocean", "forest", "desert"],
    }
    
    cats = list(swap_words.keys())
    results = {}
    
    for li in sample_layers:
        log_f(f"\n  --- L{li} ---")
        
        # 收集各词在L{li}的residual stream
        word_h = {}
        for cat in cats:
            for word in swap_words[cat]:
                prompt = TEMPLATE.format(word)
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                
                captured = {}
                hooks = []
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu()
                        else:
                            captured[key] = output.detach().float().cpu()
                    return hook
                
                hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
                
                with torch.no_grad():
                    _ = model(**toks)
                
                for h in hooks:
                    h.remove()
                
                if f"L{li}" in captured:
                    word_h[word] = captured[f"L{li}"][0, -1, :].numpy()  # [d_model]
        
        log_f(f"    Collected {len(word_h)} word representations")
        
        # Swap实验: 替换residual stream
        # 取animals→food, food→tools, tools→nature, nature→animals的swap对
        swap_pairs = []
        for ci in range(len(cats)):
            src_cat = cats[ci]
            tgt_cat = cats[(ci + 1) % len(cats)]
            for si in range(min(3, len(swap_words[src_cat]))):
                src_word = swap_words[src_cat][si]
                tgt_word = swap_words[tgt_cat][si]
                if src_word in word_h and tgt_word in word_h:
                    swap_pairs.append((src_word, tgt_word, src_cat, tgt_cat))
        
        top1_changes = 0
        total = 0
        cat_changes = defaultdict(int)
        cat_totals = defaultdict(int)
        
        for src_word, tgt_word, src_cat, tgt_cat in swap_pairs:
            # Baseline
            prompt = TEMPLATE.format(src_word)
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                base_out = model(**toks)
                base_logits = base_out.logits[0, -1].float().cpu().numpy()
                base_top1 = int(np.argmax(base_logits))
            
            # Hook: swap residual stream at L{li}
            tgt_h_tensor = torch.tensor(word_h[tgt_word], dtype=torch.float16, device=device)
            
            def make_swap_hook(tgt_h):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                    else:
                        out = output.clone()
                    out[0, -1, :] = tgt_h.to(out.dtype)
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                return hook
            
            hook = layers[li].register_forward_hook(make_swap_hook(tgt_h_tensor))
            
            with torch.no_grad():
                swap_out = model(**toks)
                swap_logits = swap_out.logits[0, -1].float().cpu().numpy()
                swap_top1 = int(np.argmax(swap_logits))
            
            hook.remove()
            
            if swap_top1 != base_top1:
                top1_changes += 1
                cat_changes[f"{src_cat}→{tgt_cat}"] += 1
            total += 1
            cat_totals[f"{src_cat}→{tgt_cat}"] += 1
        
        rate = top1_changes / max(total, 1)
        log_f(f"    d_model swap: {top1_changes}/{total} = {rate:.3f}")
        
        for pair_key in sorted(cat_totals.keys()):
            if cat_totals[pair_key] > 0:
                r = cat_changes[pair_key] / cat_totals[pair_key]
                log_f(f"      {pair_key}: {cat_changes[pair_key]}/{cat_totals[pair_key]} = {r:.3f}")
        
        results[li] = {
            "top1_changes": top1_changes,
            "total": total,
            "rate": float(rate),
            "per_pair": {k: {"changes": cat_changes[k], "total": cat_totals[k]} for k in cat_totals},
        }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    t0 = time.time()
    
    # Clear log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"CCIV(304) d_model Semantic Analysis - {model_name}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    log_f(f"=== CCIV(304) d_model Semantic Analysis - {model_name} ===")
    
    # Load model
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    log_f(f"Model: {model_name}, L={model_info.n_layers}, d={model_info.d_model}, "
          f"n_inter={model_info.intermediate_size}")
    
    # Sample layers
    nl = model_info.n_layers
    sample_layers = [0, nl//4, nl//2, 3*nl//4, nl-1]
    # Add mid layers
    for x in [nl//3, 2*nl//3]:
        if x not in sample_layers:
            sample_layers.append(x)
    sample_layers = sorted(set(sample_layers))
    log_f(f"Sample layers: {sample_layers}")
    
    # Step 1: Collect residual stream
    log_f("\n=== Step 1: Collect Residual Stream ===")
    residual_stream, all_words, word_cats, categories = collect_residual_stream(
        model, tokenizer, device, model_info
    )
    
    for li in sample_layers:
        if li in residual_stream:
            h = residual_stream[li]
            log_f(f"  L{li}: shape={h.shape}, norm={np.linalg.norm(h, axis=1).mean():.2f}")
    
    # Step 2: ANOVA
    anova_results = anova_dmodel(residual_stream, word_cats, categories, sample_layers)
    
    # Step 3: PCA
    pca_results = pca_dmodel(residual_stream, word_cats, categories, sample_layers)
    
    # Save intermediate results
    save_data = {
        "model": model_name,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
        "n_inter": model_info.intermediate_size,
        "sample_layers": sample_layers,
    }
    
    # Save ANOVA summary
    anova_summary = {}
    for li, ar in anova_results.items():
        anova_summary[str(li)] = {
            "sig_05": ar["sig_05"],
            "sig_001": ar["sig_001"],
            "total_dims": ar["total_dims"],
            "max_F": ar["max_F"],
            "max_F_dim": ar["max_F_dim"],
            "top5_dims": [(F, dim, eta_sq, means) for F, dim, eta_sq, means in ar["top_dims"][:5]],
        }
    save_data["anova"] = anova_summary
    
    # Save PCA summary
    pca_summary = {}
    for li, pr in pca_results.items():
        pca_summary[str(li)] = {
            "explained_var_ratio": pr["explained_var_ratio"],
            "cum_var_10": pr["cum_var_10"],
        }
    save_data["pca"] = pca_summary
    
    # Step 4: Hook perturb
    perturb_results = hook_perturb_dmodel(
        model, tokenizer, device, model_info,
        residual_stream, word_cats, categories,
        anova_results, pca_results, sample_layers
    )
    
    # Save perturb summary
    perturb_summary = {}
    for li, dirs in perturb_results.items():
        perturb_summary[str(li)] = {}
        for dir_name, eps_data in dirs.items():
            perturb_summary[str(li)][dir_name] = eps_data
    save_data["perturb"] = perturb_summary
    
    # Step 5: Causal intervention
    intervention_results = causal_intervention_dmodel(
        model, tokenizer, device, model_info,
        residual_stream, word_cats, categories, sample_layers
    )
    
    # Save intervention summary
    intervention_summary = {}
    for li, ir in intervention_results.items():
        intervention_summary[str(li)] = ir
    save_data["intervention"] = intervention_summary
    
    # Save JSON
    json_path = TEMP_DIR / f"cciv_dmodel_semantic_{model_name}.json"
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj
    
    save_data = convert(save_data)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log_f(f"\nResults saved to {json_path}")
    
    # Release model
    release_model(model)
    gc.collect()
    
    elapsed = time.time() - t0
    log_f(f"\nTotal time: {elapsed/60:.1f} min")
    log_f(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
