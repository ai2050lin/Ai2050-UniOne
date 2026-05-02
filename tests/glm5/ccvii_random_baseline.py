"""
CCVII(357): 随机基线实验 — 简化版
===================================
核心问题: Embedding→Residual的~90°正交是数学必然性还是训练结果?

实验:
  1. 数学随机: 两个独立N×D随机矩阵的PCA Procrustes角度
  2. 随机Transformer: 随机初始化模型的前向传播
  3. 结构化随机: 真实embedding + 随机层权重

理论预测:
  在R^D中, 两个独立的随机K维子空间, 当K<<D时, Procrustes旋转角度→90°
  因为K个奇异值σ_i ~ sqrt(χ²) → 在N,K→∞时, E[tr(R)] = 0 → θ=arccos(-1/(K-1))→90°
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
from scipy.linalg import svd
from scipy.stats import pearsonr, ttest_1samp
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# CCVI训练模型的已知结果 (N=50)
TRAINED_RESULTS = {
    "qwen3": {"d_model": 2560, "animal50": {"theta": 91.9, "beta_emb": 0.169}, "vehicle50": {"theta": 89.7, "beta_emb": 0.349}},
    "glm4": {"d_model": 4096, "animal50": {"theta": 90.5, "beta_emb": 0.230}, "vehicle50": {"theta": 91.0, "beta_emb": 0.238}},
    "deepseek7b": {"d_model": 3584, "animal50": {"theta": 89.9, "beta_emb": 0.149}, "vehicle50": {"theta": 91.4, "beta_emb": 0.218}},
}


def procrustes_align(X, Y):
    """Orthogonal Procrustes"""
    M = X.T @ Y
    U, sigma, Vt = svd(M, full_matrices=False)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Y_pred = X @ R
    error = np.sum((Y - Y_pred)**2) / np.sum(Y**2) if np.sum(Y**2) > 0 else 1.0
    return R, sigma, error


def compute_rotation_angle(R):
    """计算旋转矩阵的旋转角度"""
    K = R.shape[0]
    trace_val = np.trace(R)
    cos_angle = np.clip((trace_val - 1) / max(K - 1, 1), -1, 1)
    angle_deg = np.degrees(np.arccos(cos_angle))
    return angle_deg, float(trace_val)


def compute_pca(points):
    """PCA"""
    N, d = points.shape
    K = min(N - 1, d)
    mean = points.mean(axis=0)
    centered = points - mean
    U, S, Vt = svd(centered, full_matrices=False)
    scores = U[:, :K] * S[:K]
    return scores, Vt[:K, :], S[:K], mean, K


# ============================================================
# Part 1: 数学随机模拟
# ============================================================

def mathematical_random_simulation():
    """
    核心测试: 两个独立的随机N×D点云, PCA后Procrustes旋转角度是多少?
    
    如果随机基线也给出~90°, 则90°正交是维度诅咒的结果
    如果随机基线给出0°或180°, 则90°是训练的特异性
    """
    print(f"\n{'='*70}")
    print(f"Part 1: 数学随机模拟")
    print(f"{'='*70}")
    
    np.random.seed(42)
    
    # 关键配置: 与训练模型相同的N和D
    configs = [
        # (N, D, model_name, domain)
        (50, 2560, "qwen3", "animal50"),
        (50, 2560, "qwen3", "vehicle50"),
        (50, 4096, "glm4", "animal50"),
        (50, 4096, "glm4", "vehicle50"),
        (50, 3584, "deepseek7b", "animal50"),
        (50, 3584, "deepseek7b", "vehicle50"),
        # 额外: 不同N和D
        (10, 2560, None, None),
        (30, 2560, None, None),
        (50, 512, None, None),
        (50, 1024, None, None),
        (100, 2560, None, None),
    ]
    
    n_trials = 300  # 增加到300次以获得更稳定的分布
    
    results = []
    
    for N, D, model_name, domain in configs:
        K = N - 1
        angles = []
        beta_embs = []
        r_dists = []
        
        for trial in range(n_trials):
            # 生成两个独立的随机点云
            X = np.random.randn(N, D)
            Y = np.random.randn(N, D)
            
            # PCA
            X_scores, _, _, _, _ = compute_pca(X)
            Y_scores, _, _, _, _ = compute_pca(Y)
            
            # Procrustes
            X_c = X_scores - X_scores.mean(axis=0)
            Y_c = Y_scores - Y_scores.mean(axis=0)
            
            R, sigma, error = procrustes_align(X_c, Y_c)
            angle_deg, _ = compute_rotation_angle(R)
            angles.append(angle_deg)
            
            # β_emb: X的cosine距离 vs Y的euclidean距离
            norms_X = np.linalg.norm(X, axis=1, keepdims=True)
            norms_X = np.maximum(norms_X, 1e-10)
            X_norm = X / norms_X
            cos_sim = X_norm @ X_norm.T
            cos_dist = 1.0 - cos_sim
            np.fill_diagonal(cos_dist, 0.0)
            
            current_dist = squareform(pdist(Y_scores, metric='euclidean'))
            upper = np.triu_indices(N, k=1)
            r_beta, _ = pearsonr(cos_dist[upper], current_dist[upper])
            beta_embs.append(r_beta)
            
            # r_dist
            dist_from = squareform(pdist(X_c, metric='euclidean'))
            dist_to = squareform(pdist(Y_c, metric='euclidean'))
            r_dist, _ = pearsonr(dist_from[upper], dist_to[upper])
            r_dists.append(r_dist)
        
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        mean_beta = np.mean(beta_embs)
        mean_rdist = np.mean(r_dists)
        
        result = {
            "N": N, "D": D, "K": K, "D_over_N": D/N,
            "n_trials": n_trials,
            "mean_theta": float(mean_angle),
            "std_theta": float(std_angle),
            "mean_beta_emb": float(mean_beta),
            "mean_r_dist": float(mean_rdist),
            "p5": float(np.percentile(angles, 5)),
            "p50": float(np.percentile(angles, 50)),
            "p95": float(np.percentile(angles, 95)),
        }
        results.append(result)
        
        # 与训练模型对比
        label = f"{model_name}/{domain}" if model_name else f"N={N},D={D}"
        trained_theta = TRAINED_RESULTS[model_name][domain]["theta"] if model_name else None
        trained_beta = TRAINED_RESULTS[model_name][domain]["beta_emb"] if model_name else None
        
        print(f"  {label:<25s}: random θ={mean_angle:.1f}°±{std_angle:.1f}° "
              f"[{result['p5']:.1f}°-{result['p95']:.1f}°], "
              f"β={mean_beta:+.3f}, r_dist={mean_rdist:.3f}", end="")
        
        if trained_theta is not None:
            in_ci = result['p5'] <= trained_theta <= result['p95']
            z = (trained_theta - mean_angle) / std_angle if std_angle > 0 else 0
            print(f" | trained θ={trained_theta:.1f}°, z={z:.2f}, "
                  f"{'在CI内' if in_ci else '★超出CI★'}, "
                  f"trained β={trained_beta:+.3f}")
        else:
            print()
    
    # 保存
    output = {"experiment": "CCVII_Part1", "n_trials": n_trials, "results": results}
    out_path = TEMP / "ccvii_math_random_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {out_path}")
    
    return output


# ============================================================
# Part 2: 随机Transformer模型
# ============================================================

def random_transformer_experiment(model_name):
    """随机初始化的Transformer模型"""
    print(f"\n{'='*70}")
    print(f"Part 2: 随机Transformer — {model_name}")
    print(f"{'='*70}")
    
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    
    cfg_path = {
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(cfg_path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建随机模型 (先CPU, 再CUDA)
    print(f"  创建随机初始化模型...")
    config = AutoConfig.from_pretrained(cfg_path, trust_remote_code=True, local_files_only=True)
    random_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    # 先保持CPU, 等前向传播时再移到CUDA
    random_model.eval()
    device = "cpu"
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # 检查是否有足够GPU内存
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        model_size_gb = sum(p.numel() * p.element_size() for p in random_model.parameters()) / (1024**3)
        print(f"  模型大小: {model_size_gb:.1f}GB, GPU空闲: {free_mem:.1f}GB")
        if free_mem > model_size_gb * 2:
            random_model = random_model.to("cuda")
            device = "cuda"
            print(f"  模型已移至CUDA")
        else:
            print(f"  GPU内存不足, 使用CPU (会较慢)")
    device = next(random_model.parameters()).device
    
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    print(f"  d_model={d_model}, n_layers={n_layers}, device={device}")
    
    from tests.glm5.ccvi_large_n_rotation import ANIMAL50, VEHICLE50
    
    layer_candidates = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    
    all_results = {}
    
    for domain_name, categories in [("animal50", ANIMAL50), ("vehicle50", VEHICLE50)]:
        cat_names = list(categories.keys())
        N = len(cat_names)
        
        print(f"\n  --- {domain_name} (N={N}) ---")
        
        # Embedding中心
        embed_layer = random_model.get_input_embeddings()
        W_E = embed_layer.weight.detach().float().cpu().numpy()
        
        emb_centers = {}
        for cat_name, words in categories.items():
            embeddings = []
            for word in words[:3]:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                if not token_ids: continue
                word_emb = np.mean(W_E[token_ids], axis=0)
                embeddings.append(word_emb)
            if embeddings:
                emb_centers[cat_name] = np.mean(embeddings, axis=0)
        
        # Residual中心
        layers = get_layers(random_model)
        layer_centers = {"emb": emb_centers}
        
        for layer_idx in layer_candidates:
            print(f"  L{layer_idx}...", end=" ", flush=True)
            t0 = time.time()
            
            res_centers = {}
            for cat_name, words in categories.items():
                residuals = []
                for word in words[:3]:
                    prompt = f"The word is {word}"
                    toks = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        inputs_embeds = embed_layer(toks.input_ids)
                        captured = {}
                        def make_hook(key):
                            def hook(module, input, output):
                                if isinstance(output, tuple):
                                    captured[key] = output[0].detach().float().cpu().numpy()
                                else:
                                    captured[key] = output.detach().float().cpu().numpy()
                            return hook
                        hook = layers[layer_idx].register_forward_hook(make_hook(f"L{layer_idx}"))
                        try:
                            _ = random_model(inputs_embeds=inputs_embeds)
                        except: pass
                        hook.remove()
                        if f"L{layer_idx}" in captured:
                            residuals.append(captured[f"L{layer_idx}"][0, -1, :])
                if residuals:
                    res_centers[cat_name] = np.mean(residuals, axis=0)
            
            common_cats = [n for n in cat_names if n in res_centers]
            if len(common_cats) < 10:
                print(f"skip ({len(common_cats)})")
                continue
            cat_names = common_cats
            N = len(cat_names)
            layer_centers[f"L{layer_idx}"] = res_centers
            print(f"OK N={N} ({time.time()-t0:.1f}s)")
        
        # PCA + Procrustes
        layer_pcas = {}
        for lk, centers in layer_centers.items():
            pts = np.array([centers[n] for n in cat_names])
            sc, _, _, _, K = compute_pca(pts)
            layer_pcas[lk] = {"scores": sc, "K": K}
        
        emb_scores = layer_pcas["emb"]["scores"]
        direct_results = []
        
        for lk in list(layer_pcas.keys())[1:]:
            X = emb_scores
            Y = layer_pcas[lk]["scores"]
            K = layer_pcas[lk]["K"]
            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)
            
            R, sigma, error = procrustes_align(X_c, Y_c)
            angle_deg, _ = compute_rotation_angle(R)
            
            # β_emb
            norms_e = np.linalg.norm(np.array([emb_centers[n] for n in cat_names]), axis=1, keepdims=True)
            norms_e = np.maximum(norms_e, 1e-10)
            e_norm = np.array([emb_centers[n] for n in cat_names]) / norms_e
            cos_sim = e_norm @ e_norm.T
            cos_dist = 1.0 - cos_sim
            np.fill_diagonal(cos_dist, 0.0)
            cur_dist = squareform(pdist(layer_pcas[lk]["scores"], metric='euclidean'))
            upper = np.triu_indices(N, k=1)
            r_beta, _ = pearsonr(cos_dist[upper], cur_dist[upper])
            
            dist_from = squareform(pdist(X_c, metric='euclidean'))
            dist_to = squareform(pdist(Y_c, metric='euclidean'))
            r_dist, _ = pearsonr(dist_from[upper], dist_to[upper])
            
            direct_results.append({
                "layer_key": lk, "theta": float(angle_deg), "beta_emb": float(r_beta),
                "r_dist": float(r_dist), "error": float(error), "N": N, "K": K,
            })
            print(f"    emb→{lk}: θ={angle_deg:.1f}°, β={r_beta:+.3f}, r_dist={r_dist:.3f}")
        
        all_results[domain_name] = {"direct": direct_results, "N": N}
    
    # 汇总对比
    print(f"\n  --- {model_name} 随机模型 vs 训练模型 ---")
    for domain_name in ["animal50", "vehicle50"]:
        if domain_name not in all_results: continue
        direct = all_results[domain_name]["direct"]
        if not direct: continue
        avg_theta = np.mean([d["theta"] for d in direct])
        avg_beta = np.mean([d["beta_emb"] for d in direct])
        trained = TRAINED_RESULTS[model_name][domain_name]
        print(f"  {domain_name}: random θ={avg_theta:.1f}° β={avg_beta:+.3f} | "
              f"trained θ={trained['theta']:.1f}° β={trained['beta_emb']:+.3f} | "
              f"Δθ={avg_theta-trained['theta']:+.1f}° Δβ={avg_beta-trained['beta_emb']:+.3f}")
    
    del random_model
    torch.cuda.empty_cache()
    gc.collect()
    
    output = {"experiment": "CCVII_Part2", "model": model_name, "d_model": d_model, "results": all_results}
    out_path = TEMP / f"ccvii_random_tf_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"保存到: {out_path}")
    return output


# ============================================================
# Part 3: 结构化随机 — 真实embedding + 随机层权重
# ============================================================

def structural_random_experiment(model_name):
    """真实embedding + 随机层权重"""
    print(f"\n{'='*70}")
    print(f"Part 3: 结构化随机 (真实emb+随机层) — {model_name}")
    print(f"{'='*70}")
    
    import torch
    from tests.glm5.ccvi_large_n_rotation import ANIMAL50, VEHICLE50
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().float().cpu().numpy()
    
    # 收集embedding中心
    emb_centers = {}
    for domain_name, categories in [("animal50", ANIMAL50), ("vehicle50", VEHICLE50)]:
        for cat_name, words in categories.items():
            embeddings = []
            for word in words[:3]:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                if not token_ids: continue
                word_emb = np.mean(W_E[token_ids], axis=0)
                embeddings.append(word_emb)
            if embeddings:
                emb_centers[(domain_name, cat_name)] = np.mean(embeddings, axis=0)
    
    # 随机化层权重
    print(f"  随机化层权重...")
    layers = get_layers(model)
    for layer in layers:
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(layer.self_attn, name):
                w = getattr(layer.self_attn, name).weight
                getattr(layer.self_attn, name).weight.data = torch.randn_like(w) * 0.02
        if hasattr(layer, "mlp"):
            for name in ["up_proj", "down_proj", "gate_proj", "gate_up_proj"]:
                if hasattr(layer.mlp, name):
                    w = getattr(layer.mlp, name).weight
                    getattr(layer.mlp, name).weight.data = torch.randn_like(w) * 0.02
    
    layer_candidates = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    all_results = {}
    
    for domain_name, categories in [("animal50", ANIMAL50), ("vehicle50", VEHICLE50)]:
        cat_names = list(categories.keys())
        N = len(cat_names)
        
        print(f"\n  --- {domain_name} (N={N}) ---")
        
        domain_emb = {cn: emb_centers[(domain_name, cn)] for cn in cat_names if (domain_name, cn) in emb_centers}
        layer_centers = {"emb": domain_emb}
        
        for layer_idx in layer_candidates:
            print(f"  L{layer_idx}...", end=" ", flush=True)
            t0 = time.time()
            
            res_centers = {}
            for cat_name in cat_names:
                words = categories[cat_name]
                residuals = []
                for word in words[:3]:
                    prompt = f"The word is {word}"
                    toks = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        inputs_embeds = embed_layer(toks.input_ids)
                        captured = {}
                        def make_hook(key):
                            def hook(module, input, output):
                                if isinstance(output, tuple):
                                    captured[key] = output[0].detach().float().cpu().numpy()
                                else:
                                    captured[key] = output.detach().float().cpu().numpy()
                            return hook
                        hook = layers[layer_idx].register_forward_hook(make_hook(f"L{layer_idx}"))
                        try:
                            _ = model(inputs_embeds=inputs_embeds)
                        except: pass
                        hook.remove()
                        if f"L{layer_idx}" in captured:
                            residuals.append(captured[f"L{layer_idx}"][0, -1, :])
                if residuals:
                    res_centers[cat_name] = np.mean(residuals, axis=0)
            
            common_cats = [n for n in cat_names if n in res_centers and n in domain_emb]
            if len(common_cats) < 10:
                print(f"skip ({len(common_cats)})")
                continue
            cat_names = common_cats
            N = len(cat_names)
            # 更新emb centers到common子集
            layer_centers["emb"] = {n: domain_emb[n] for n in cat_names}
            layer_centers[f"L{layer_idx}"] = res_centers
            print(f"OK N={N} ({time.time()-t0:.1f}s)")
        
        # PCA + Procrustes
        layer_pcas = {}
        for lk, centers in layer_centers.items():
            pts = np.array([centers[n] for n in cat_names])
            sc, _, _, _, K = compute_pca(pts)
            layer_pcas[lk] = {"scores": sc, "K": K}
        
        emb_scores = layer_pcas["emb"]["scores"]
        direct_results = []
        
        for lk in list(layer_pcas.keys())[1:]:
            X = emb_scores
            Y = layer_pcas[lk]["scores"]
            K = layer_pcas[lk]["K"]
            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)
            
            R, sigma, error = procrustes_align(X_c, Y_c)
            angle_deg, _ = compute_rotation_angle(R)
            
            norms_e = np.linalg.norm(np.array([layer_centers["emb"][n] for n in cat_names]), axis=1, keepdims=True)
            norms_e = np.maximum(norms_e, 1e-10)
            e_norm = np.array([layer_centers["emb"][n] for n in cat_names]) / norms_e
            cos_sim = e_norm @ e_norm.T
            cos_dist = 1.0 - cos_sim
            np.fill_diagonal(cos_dist, 0.0)
            cur_dist = squareform(pdist(layer_pcas[lk]["scores"], metric='euclidean'))
            upper = np.triu_indices(N, k=1)
            r_beta, _ = pearsonr(cos_dist[upper], cur_dist[upper])
            
            dist_from = squareform(pdist(X_c, metric='euclidean'))
            dist_to = squareform(pdist(Y_c, metric='euclidean'))
            r_dist, _ = pearsonr(dist_from[upper], dist_to[upper])
            
            direct_results.append({
                "layer_key": lk, "theta": float(angle_deg), "beta_emb": float(r_beta),
                "r_dist": float(r_dist), "error": float(error), "N": N, "K": K,
            })
            print(f"    emb→{lk}: θ={angle_deg:.1f}°, β={r_beta:+.3f}, r_dist={r_dist:.3f}")
        
        all_results[domain_name] = {"direct": direct_results, "N": N}
    
    # 汇总对比
    print(f"\n  --- {model_name} 结构化随机 vs 训练模型 ---")
    for domain_name in ["animal50", "vehicle50"]:
        if domain_name not in all_results: continue
        direct = all_results[domain_name]["direct"]
        if not direct: continue
        avg_theta = np.mean([d["theta"] for d in direct])
        avg_beta = np.mean([d["beta_emb"] for d in direct])
        trained = TRAINED_RESULTS[model_name][domain_name]
        print(f"  {domain_name}: struct-random θ={avg_theta:.1f}° β={avg_beta:+.3f} | "
              f"trained θ={trained['theta']:.1f}° β={trained['beta_emb']:+.3f} | "
              f"Δθ={avg_theta-trained['theta']:+.1f}° Δβ={avg_beta-trained['beta_emb']:+.3f}")
    
    release_model(model)
    gc.collect()
    
    output = {"experiment": "CCVII_Part3", "model": model_name, "d_model": d_model, "results": all_results}
    out_path = TEMP / f"ccvii_struct_random_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"保存到: {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default="all",
                       choices=["math", "random", "structural", "all"])
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    t0 = time.time()
    
    if args.part in ["math", "all"]:
        mathematical_random_simulation()
    
    if args.part in ["random", "all"]:
        random_transformer_experiment(args.model)
    
    if args.part in ["structural", "all"]:
        structural_random_experiment(args.model)
    
    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
