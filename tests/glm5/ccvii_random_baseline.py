"""
CCVII(357): 随机基线实验 — 区分90°正交是数学必然性还是训练结果
=================================================================
★★★★★ 核心问题:
  CCV/CCVI发现Embedding→Residual旋转角度~90°
  这是高维空间的数学必然性(维度诅咒)? 还是Transformer训练的结果?

★★★★★ 实验设计:
  Part 1: 数学随机模拟 (无需GPU, 快速)
    - N个随机点在D维空间, 应用随机正交变换
    - PCA + Procrustes, 计算旋转角度
    - 重复100次, 得到旋转角度的分布
    - 对比: 训练模型的实际旋转角度(来自CCVI)

  Part 2: 随机Transformer模型 (需要GPU)
    - 从config创建随机初始化的模型
    - 运行与CCVI相同的分析管道
    - 对比: 训练模型 vs 随机模型

  Part 3: 结构化随机 (关键中间态)
    - 使用训练模型的embedding(真实语义), 但随机化层权重
    - 测试: embedding几何结构是否影响旋转角度?

★★★★★ 理论预测:
  随机矩阵论: 如果X, Y是D维空间中两个独立的N点云(N<<D),
  则Procrustes旋转角度→90° (当D/N→∞时)
  
  对于N=50, D=2560: D/N=51.2, 预期θ≈90°
  对于N=50, D=4096: D/N=81.9, 预期θ更接近90°

用法:
  python ccvii_random_baseline.py --part math    # 数学模拟(快速)
  python ccvii_random_baseline.py --part random --model qwen3  # 随机模型
  python ccvii_random_baseline.py --part structural --model qwen3  # 结构化随机
  python ccvii_random_baseline.py --part all --model qwen3  # 全部
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
from scipy.linalg import svd, orthogonal_procrustes
from scipy.stats import pearsonr, ttest_1samp
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# CCVI训练模型的已知结果 (N=50)
TRAINED_RESULTS = {
    "qwen3": {
        "d_model": 2560,
        "animal50": {"theta": 91.9, "beta_emb": 0.169},
        "vehicle50": {"theta": 89.7, "beta_emb": 0.349},
    },
    "glm4": {
        "d_model": 4096,
        "animal50": {"theta": 90.5, "beta_emb": 0.230},
        "vehicle50": {"theta": 91.0, "beta_emb": 0.238},
    },
    "deepseek7b": {
        "d_model": 3584,
        "animal50": {"theta": 89.9, "beta_emb": 0.149},
        "vehicle50": {"theta": 91.4, "beta_emb": 0.218},
    },
}

# ============================================================
# Part 1: 数学随机模拟
# ============================================================

def mathematical_random_simulation():
    """
    数学随机模拟: 测试随机点云+随机正交变换的旋转角度
    
    核心假设: 如果θ=90°是维度诅咒的结果, 
    那么随机数据也应该给出θ≈90°
    """
    print(f"\n{'='*70}")
    print(f"Part 1: 数学随机模拟")
    print(f"{'='*70}")
    
    np.random.seed(42)
    
    # 参数设置
    N_values = [10, 20, 30, 50, 100]  # 类别数
    D_values = [256, 512, 1024, 2560, 4096]  # 模型维度
    n_trials = 200  # 每个配置的试验次数
    
    results = []
    
    for N in N_values:
        for D in D_values:
            K = N - 1  # PCA维度
            angles = []
            beta_embs = []
            r_dists = []
            spectra_n_gt90 = []
            spectra_mean_rot = []
            
            for trial in range(n_trials):
                # 1. 生成随机点云 X: N×D
                # 使用多元正态分布, 模拟embedding的统计特性
                X = np.random.randn(N, D)
                
                # 2. 生成随机正交变换 Q: D×D
                # 使用QR分解生成均匀分布的正交矩阵
                A = np.random.randn(D, D)
                Q, _ = np.linalg.qr(A)
                
                # 3. 应用变换: Y = X @ Q
                Y = X @ Q
                
                # 4. PCA
                X_centered = X - X.mean(axis=0)
                Y_centered = Y - Y.mean(axis=0)
                
                Ux, Sx, Vtx = svd(X_centered, full_matrices=False)
                Uy, Sy, Vty = svd(Y_centered, full_matrices=False)
                
                X_scores = Ux[:, :K] * Sx[:K]  # N×K
                Y_scores = Uy[:, :K] * Sy[:K]  # N×K
                
                # 5. Procrustes对齐
                X_c = X_scores - X_scores.mean(axis=0)
                Y_c = Y_scores - Y_scores.mean(axis=0)
                
                M = X_c.T @ Y_c
                Um, sigma_m, Vtm = svd(M, full_matrices=False)
                R = Vtm.T @ Um.T
                
                # 确保是旋转(行列式=+1)
                if np.linalg.det(R) < 0:
                    Vtm[-1, :] *= -1
                    R = Vtm.T @ Um.T
                
                # 6. 旋转角度
                trace_val = np.trace(R)
                cos_angle = np.clip((trace_val - 1) / max(K - 1, 1), -1, 1)
                angle_deg = np.degrees(np.arccos(cos_angle))
                angles.append(angle_deg)
                
                # 7. β_emb (embedding距离 vs 变换后距离的相关性)
                dist_X = squareform(pdist(X_scores, metric='euclidean'))
                dist_Y = squareform(pdist(Y_scores, metric='euclidean'))
                
                upper = np.triu_indices(N, k=1)
                r_dist, _ = pearsonr(dist_X[upper], dist_Y[upper])
                r_dists.append(r_dist)
                
                # β_emb: 原始X的cosine距离 vs Y的euclidean距离
                norms_X = np.linalg.norm(X, axis=1, keepdims=True)
                norms_X = np.maximum(norms_X, 1e-10)
                X_norm = X / norms_X
                cos_sim = X_norm @ X_norm.T
                cos_dist = 1.0 - cos_sim
                np.fill_diagonal(cos_dist, 0.0)
                
                r_beta, _ = pearsonr(cos_dist[upper], dist_Y[upper])
                beta_embs.append(r_beta)
                
                # 8. 旋转频谱
                eigenvalues = np.linalg.eigvals(R)
                rot_angles_spec = []
                processed = set()
                for i, ev in enumerate(eigenvalues):
                    if i in processed:
                        continue
                    if np.isreal(ev):
                        real_val = np.real(ev)
                        if real_val < -0.5:
                            rot_angles_spec.append(180.0)
                        processed.add(i)
                    else:
                        theta = np.degrees(np.arccos(np.clip(np.real(ev), -1, 1)))
                        rot_angles_spec.append(theta)
                        for j in range(i+1, len(eigenvalues)):
                            if j not in processed:
                                if np.abs(np.real(eigenvalues[j]) - np.real(ev)) < 0.01 and \
                                   np.abs(np.abs(np.imag(eigenvalues[j])) - np.abs(np.imag(ev))) < 0.01:
                                    processed.add(j)
                                    break
                        processed.add(i)
                
                n_gt90 = sum(1 for a in rot_angles_spec if a > 90)
                mean_rot = np.mean(rot_angles_spec) if rot_angles_spec else 0
                spectra_n_gt90.append(n_gt90)
                spectra_mean_rot.append(mean_rot)
            
            # 汇总
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)
            mean_beta = np.mean(beta_embs)
            mean_rdist = np.mean(r_dists)
            mean_n_gt90 = np.mean(spectra_n_gt90)
            mean_mean_rot = np.mean(spectra_mean_rot)
            
            # t-test: 角度是否显著≠90°?
            t_stat, p_val = ttest_1samp(angles, 90.0)
            
            result = {
                "N": N, "D": D, "K": K, "D_over_N": D / N,
                "n_trials": n_trials,
                "mean_theta": float(mean_angle),
                "std_theta": float(std_angle),
                "mean_beta_emb": float(mean_beta),
                "mean_r_dist": float(mean_rdist),
                "mean_n_gt90": float(mean_n_gt90),
                "mean_mean_rot": float(mean_mean_rot),
                "t_test_vs_90": {"t": float(t_stat), "p": float(p_val)},
                "theta_percentiles": {
                    "p5": float(np.percentile(angles, 5)),
                    "p25": float(np.percentile(angles, 25)),
                    "p50": float(np.percentile(angles, 50)),
                    "p75": float(np.percentile(angles, 75)),
                    "p95": float(np.percentile(angles, 95)),
                }
            }
            results.append(result)
            
            print(f"  N={N:3d}, D={D:4d}, D/N={D/N:5.1f}: "
                  f"θ={mean_angle:.2f}°±{std_angle:.2f}°, "
                  f"β={mean_beta:+.3f}, r_dist={mean_rdist:.3f}, "
                  f"n_gt90={mean_n_gt90:.1f}/{K}, "
                  f"t-test vs 90°: t={t_stat:.2f}, p={p_val:.4f}")
    
    # 详细分析: 固定N=50, 改变D
    print(f"\n--- 详细分析: N=50, 改变D ---")
    n50_results = [r for r in results if r["N"] == 50]
    for r in n50_results:
        trained_theta = None
        if r["D"] == 2560:
            trained_theta = TRAINED_RESULTS["qwen3"]["animal50"]["theta"]
            trained_beta = TRAINED_RESULTS["qwen3"]["animal50"]["beta_emb"]
            model_name = "Qwen3"
        elif r["D"] == 4096:
            trained_theta = TRAINED_RESULTS["glm4"]["animal50"]["theta"]
            trained_beta = TRAINED_RESULTS["glm4"]["animal50"]["beta_emb"]
            model_name = "GLM4"
        elif r["D"] == 3584:
            trained_theta = TRAINED_RESULTS["deepseek7b"]["animal50"]["theta"]
            trained_beta = TRAINED_RESULTS["deepseek7b"]["animal50"]["beta_emb"]
            model_name = "DS7B"
        else:
            model_name = "N/A"
        
        if trained_theta is not None:
            z_score = (trained_theta - r["mean_theta"]) / r["std_theta"] if r["std_theta"] > 0 else 0
            print(f"  D={r['D']:4d} ({model_name}): "
                  f"random={r['mean_theta']:.2f}°±{r['std_theta']:.2f}°, "
                  f"trained={trained_theta:.1f}°, "
                  f"z={z_score:.2f}, "
                  f"β: random={r['mean_beta_emb']:+.3f} vs trained={trained_beta:+.3f}")
    
    # 详细分析: 固定D=2560, 改变N
    print(f"\n--- 详细分析: D=2560, 改变N ---")
    d2560_results = [r for r in results if r["D"] == 2560]
    for r in d2560_results:
        print(f"  N={r['N']:3d} (K={r['K']:3d}): "
              f"θ={r['mean_theta']:.2f}°±{r['std_theta']:.2f}°, "
              f"p5={r['theta_percentiles']['p5']:.2f}°, "
              f"p95={r['theta_percentiles']['p95']:.2f}°, "
              f"β={r['mean_beta_emb']:+.3f}, "
              f"n_gt90={r['mean_n_gt90']:.1f}/{r['K']}")
    
    # 保存
    output = {
        "experiment": "CCVII_Part1_Mathematical_Random",
        "n_trials": n_trials,
        "results": results,
    }
    out_path = TEMP / "ccvii_math_random_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nPart 1 结果已保存到: {out_path}")
    
    return output


# ============================================================
# Part 2: 随机Transformer模型
# ============================================================

def random_transformer_experiment(model_name):
    """
    随机Transformer模型: 从config创建随机初始化模型, 运行CCVI管道
    
    关键: 随机模型的embedding没有语义, 但几何结构仍然可以被分析
    """
    print(f"\n{'='*70}")
    print(f"Part 2: 随机Transformer模型 — {model_name}")
    print(f"{'='*70}")
    
    from tests.glm5.ccvi_large_n_rotation import (
        ANIMAL50, VEHICLE50, DOMAINS_LARGE,
        get_category_centers_embedding, compute_pca, procrustes_align,
        compute_rotation_angle, compute_rotation_spectrum,
        compute_cosine_dist_matrix,
    )
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    
    cfg_path = {
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }[model_name]
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg_path, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建随机初始化模型
    print(f"  创建随机初始化模型...")
    config = AutoConfig.from_pretrained(cfg_path, trust_remote_code=True, local_files_only=True)
    
    # 使用float16减少内存
    random_model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    if torch.cuda.is_available():
        random_model = random_model.to("cuda")
    random_model.eval()
    device = next(random_model.parameters()).device
    
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    print(f"  随机模型: d_model={d_model}, n_layers={n_layers}, device={device}")
    
    # 选择5个关键层
    layer_candidates = sorted(set([
        0,
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 1,
    ]))
    print(f"  测试层: {layer_candidates}")
    
    all_results = {}
    
    for domain_name, categories in [("animal50", ANIMAL50), ("vehicle50", VEHICLE50)]:
        cat_names = list(categories.keys())
        N = len(cat_names)
        K = N - 1
        
        print(f"\n  --- 领域: {domain_name} (N={N}) ---")
        
        # 1. Embedding中心 (使用随机模型的embedding)
        print(f"  收集embedding中心(随机模型)...")
        embed_layer = random_model.get_input_embeddings()
        W_E = embed_layer.weight.detach().float().cpu().numpy()
        
        emb_centers = {}
        for cat_name, words in categories.items():
            embeddings = []
            for word in words[:3]:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 0:
                    continue
                word_emb = np.mean(W_E[token_ids], axis=0)
                embeddings.append(word_emb)
            if len(embeddings) > 0:
                emb_centers[cat_name] = np.mean(embeddings, axis=0)
        
        print(f"  得到{len(emb_centers)}个embedding中心")
        
        # 2. 各Residual层中心
        layers = get_layers(random_model)
        layer_centers = {"emb": emb_centers}
        
        for layer_idx in layer_candidates:
            print(f"  收集L{layer_idx}中心(随机模型)...", end=" ", flush=True)
            t0 = time.time()
            
            res_centers = {}
            for cat_name, words in categories.items():
                residuals = []
                for word in words[:3]:
                    prompt = f"The word is {word}"
                    toks = tokenizer(prompt, return_tensors="pt").to(device)
                    input_ids = toks.input_ids
                    
                    with torch.no_grad():
                        inputs_embeds = embed_layer(input_ids)
                        
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
                        except Exception as e:
                            print(f"forward error: {e}")
                        hook.remove()
                        
                        if f"L{layer_idx}" in captured:
                            res = captured[f"L{layer_idx}"][0, -1, :]
                            residuals.append(res)
                
                if len(residuals) > 0:
                    res_centers[cat_name] = np.mean(residuals, axis=0)
            
            common_cats = [n for n in cat_names if n in res_centers]
            if len(common_cats) < 10:
                print(f"跳过(只有{len(common_cats)}个)")
                continue
            
            cat_names = common_cats
            N = len(cat_names)
            K = N - 1
            layer_centers[f"L{layer_idx}"] = res_centers
            print(f"OK (N={N}, {time.time()-t0:.1f}s)")
        
        # 3. PCA
        layer_pcas = {}
        for layer_key, centers in layer_centers.items():
            points = np.array([centers[name] for name in cat_names])
            pca = compute_pca(points)
            layer_pcas[layer_key] = pca
        
        # 4. Procrustes (emb→各层)
        emb_pca = layer_pcas["emb"]
        direct_results = []
        
        for layer_key in list(layer_pcas.keys())[1:]:
            X = emb_pca["scores"]
            Y = layer_pcas[layer_key]["scores"]
            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)
            
            R, scale, error, sigma = procrustes_align(X_c, Y_c)
            angle_deg, trace_val = compute_rotation_angle(R)
            rot_spectrum = compute_rotation_spectrum(R)
            
            # β_emb
            emb_dist = compute_cosine_dist_matrix(emb_centers, cat_names)
            current_dist = squareform(pdist(layer_pcas[layer_key]["scores"], metric='euclidean'))
            
            upper = np.triu_indices(N, k=1)
            r_emb, p_emb = pearsonr(emb_dist[upper], current_dist[upper])
            
            # r_dist
            dist_from = squareform(pdist(X_c, metric='euclidean'))
            dist_to = squareform(pdist(Y_c, metric='euclidean'))
            r_dist, _ = pearsonr(dist_from[upper], dist_to[upper])
            
            n_gt90 = sum(1 for a in rot_spectrum if a > 90)
            
            result = {
                "layer_key": layer_key,
                "direct_angle_deg": float(angle_deg),
                "beta_emb": float(r_emb),
                "r_dist": float(r_dist),
                "error": float(error),
                "N": N,
                "K": K,
                "n_rotations_gt90": n_gt90,
                "mean_rotation": float(np.mean(rot_spectrum)) if rot_spectrum else 0,
            }
            direct_results.append(result)
            
            print(f"    emb→{layer_key}: θ={angle_deg:.1f}°, "
                  f"β_emb={r_emb:+.3f}, r_dist={r_dist:.3f}, "
                  f"n_gt90={n_gt90}/{len(rot_spectrum)}")
        
        all_results[domain_name] = {
            "direct": direct_results,
            "N": N,
            "K": K,
        }
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"Part 2 汇总 — {model_name} (随机模型)")
    print(f"{'='*70}")
    
    for domain_name in ["animal50", "vehicle50"]:
        if domain_name not in all_results:
            continue
        data = all_results[domain_name]
        direct = data["direct"]
        if not direct:
            continue
        
        avg_angle = np.mean([d["direct_angle_deg"] for d in direct])
        avg_beta = np.mean([d["beta_emb"] for d in direct])
        avg_rdist = np.mean([d["r_dist"] for d in direct])
        avg_n_gt90 = np.mean([d["n_rotations_gt90"] for d in direct])
        
        # 对比训练模型
        trained = TRAINED_RESULTS[model_name][domain_name]
        
        print(f"  {domain_name}:")
        print(f"    随机模型: θ={avg_angle:.1f}°, β={avg_beta:+.3f}, r_dist={avg_rdist:.3f}")
        print(f"    训练模型: θ={trained['theta']:.1f}°, β={trained['beta_emb']:+.3f}")
        print(f"    差异: Δθ={avg_angle-trained['theta']:+.1f}°, Δβ={avg_beta-trained['beta_emb']:+.3f}")
    
    # 释放
    del random_model
    torch.cuda.empty_cache()
    gc.collect()
    
    output = {
        "experiment": "CCVII_Part2_Random_Transformer",
        "model": model_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "domain_results": all_results,
    }
    out_path = TEMP / f"ccvii_random_transformer_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nPart 2 结果已保存到: {out_path}")
    
    return output


# ============================================================
# Part 3: 结构化随机 — 真实embedding + 随机层权重
# ============================================================

def structural_random_experiment(model_name):
    """
    结构化随机: 使用训练模型的embedding, 但随机化层权重
    
    测试: embedding的语义几何结构是否影响旋转角度?
    如果是: 说明训练的embedding有助于保持几何结构
    如果否: 说明旋转角度完全由维度和变换类型决定
    """
    print(f"\n{'='*70}")
    print(f"Part 3: 结构化随机 — {model_name}")
    print(f"{'='*70}")
    
    import torch
    from tests.glm5.ccvi_large_n_rotation import (
        ANIMAL50, VEHICLE50,
        compute_pca, procrustes_align,
        compute_rotation_angle, compute_rotation_spectrum,
        compute_cosine_dist_matrix,
    )
    
    # 1. 加载训练模型, 获取embedding
    print(f"  加载训练模型获取embedding...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().float().cpu().numpy()
    
    # 获取类别embedding中心 (使用训练模型的embedding)
    emb_centers_animal = {}
    for cat_name, words in ANIMAL50.items():
        embeddings = []
        for word in words[:3]:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            word_emb = np.mean(W_E[token_ids], axis=0)
            embeddings.append(word_emb)
        if len(embeddings) > 0:
            emb_centers_animal[cat_name] = np.mean(embeddings, axis=0)
    
    emb_centers_vehicle = {}
    for cat_name, words in VEHICLE50.items():
        embeddings = []
        for word in words[:3]:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            word_emb = np.mean(W_E[token_ids], axis=0)
            embeddings.append(word_emb)
        if len(embeddings) > 0:
            emb_centers_vehicle[cat_name] = np.mean(embeddings, axis=0)
    
    print(f"  得到 animal50={len(emb_centers_animal)}个, vehicle50={len(emb_centers_vehicle)}个中心")
    
    # 2. 随机化模型权重 (保留embedding)
    print(f"  随机化层权重(保留embedding)...")
    layers = get_layers(model)
    
    # 保存原始权重(不需要, 只需要embedding)
    # 随机化每层的权重
    for layer in layers:
        # 随机化attention权重
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(layer.self_attn, name):
                w = getattr(layer.self_attn, name).weight
                getattr(layer.self_attn, name).weight.data = torch.randn_like(w)
        
        # 随机化MLP权重
        if hasattr(layer, "mlp"):
            for name in ["up_proj", "down_proj", "gate_proj", "gate_up_proj"]:
                if hasattr(layer.mlp, name):
                    w = getattr(layer.mlp, name).weight
                    getattr(layer.mlp, name).weight.data = torch.randn_like(w) * 0.02
    
    print(f"  层权重已随机化")
    
    # 3. 收集residual中心 (使用随机层权重 + 真实embedding)
    layer_candidates = sorted(set([
        0,
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 1,
    ]))
    
    all_results = {}
    
    for domain_name, emb_centers, categories in [
        ("animal50", emb_centers_animal, ANIMAL50),
        ("vehicle50", emb_centers_vehicle, VEHICLE50),
    ]:
        cat_names = list(emb_centers.keys())
        N = len(cat_names)
        K = N - 1
        
        print(f"\n  --- 领域: {domain_name} (N={N}) ---")
        
        layer_centers = {"emb": emb_centers}
        
        for layer_idx in layer_candidates:
            print(f"  收集L{layer_idx}(结构化随机)...", end=" ", flush=True)
            t0 = time.time()
            
            res_centers = {}
            for cat_name in cat_names:
                words = categories[cat_name]
                residuals = []
                for word in words[:3]:
                    prompt = f"The word is {word}"
                    toks = tokenizer(prompt, return_tensors="pt").to(device)
                    input_ids = toks.input_ids
                    
                    with torch.no_grad():
                        inputs_embeds = embed_layer(input_ids)
                        
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
                        except Exception as e:
                            print(f"forward error: {e}")
                        hook.remove()
                        
                        if f"L{layer_idx}" in captured:
                            res = captured[f"L{layer_idx}"][0, -1, :]
                            residuals.append(res)
                
                if len(residuals) > 0:
                    res_centers[cat_name] = np.mean(residuals, axis=0)
            
            common_cats = [n for n in cat_names if n in res_centers]
            if len(common_cats) < 10:
                print(f"跳过(只有{len(common_cats)}个)")
                continue
            
            cat_names = common_cats
            N = len(cat_names)
            K = N - 1
            layer_centers[f"L{layer_idx}"] = res_centers
            print(f"OK (N={N}, {time.time()-t0:.1f}s)")
        
        # PCA + Procrustes
        layer_pcas = {}
        for layer_key, centers in layer_centers.items():
            points = np.array([centers[name] for name in cat_names])
            pca = compute_pca(points)
            layer_pcas[layer_key] = pca
        
        emb_pca = layer_pcas["emb"]
        direct_results = []
        
        for layer_key in list(layer_pcas.keys())[1:]:
            X = emb_pca["scores"]
            Y = layer_pcas[layer_key]["scores"]
            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)
            
            R, scale, error, sigma = procrustes_align(X_c, Y_c)
            angle_deg, trace_val = compute_rotation_angle(R)
            rot_spectrum = compute_rotation_spectrum(R)
            
            emb_dist = compute_cosine_dist_matrix(emb_centers, cat_names)
            current_dist = squareform(pdist(layer_pcas[layer_key]["scores"], metric='euclidean'))
            upper = np.triu_indices(N, k=1)
            r_emb, p_emb = pearsonr(emb_dist[upper], current_dist[upper])
            
            dist_from = squareform(pdist(X_c, metric='euclidean'))
            dist_to = squareform(pdist(Y_c, metric='euclidean'))
            r_dist, _ = pearsonr(dist_from[upper], dist_to[upper])
            
            n_gt90 = sum(1 for a in rot_spectrum if a > 90)
            
            result = {
                "layer_key": layer_key,
                "direct_angle_deg": float(angle_deg),
                "beta_emb": float(r_emb),
                "r_dist": float(r_dist),
                "error": float(error),
                "N": N,
                "K": K,
                "n_rotations_gt90": n_gt90,
                "mean_rotation": float(np.mean(rot_spectrum)) if rot_spectrum else 0,
            }
            direct_results.append(result)
            
            print(f"    emb→{layer_key}: θ={angle_deg:.1f}°, "
                  f"β_emb={r_emb:+.3f}, r_dist={r_dist:.3f}, "
                  f"n_gt90={n_gt90}/{len(rot_spectrum)}")
        
        all_results[domain_name] = {
            "direct": direct_results,
            "N": N,
            "K": K,
        }
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"Part 3 汇总 — {model_name} (结构化随机: 真实emb + 随机层)")
    print(f"{'='*70}")
    
    for domain_name in ["animal50", "vehicle50"]:
        if domain_name not in all_results:
            continue
        data = all_results[domain_name]
        direct = data["direct"]
        if not direct:
            continue
        
        avg_angle = np.mean([d["direct_angle_deg"] for d in direct])
        avg_beta = np.mean([d["beta_emb"] for d in direct])
        avg_rdist = np.mean([d["r_dist"] for d in direct])
        
        trained = TRAINED_RESULTS[model_name][domain_name]
        
        print(f"  {domain_name}:")
        print(f"    结构化随机: θ={avg_angle:.1f}°, β={avg_beta:+.3f}, r_dist={avg_rdist:.3f}")
        print(f"    训练模型:   θ={trained['theta']:.1f}°, β={trained['beta_emb']:+.3f}")
        print(f"    差异: Δθ={avg_angle-trained['theta']:+.1f}°, Δβ={avg_beta-trained['beta_emb']:+.3f}")
    
    release_model(model)
    gc.collect()
    
    output = {
        "experiment": "CCVII_Part3_Structural_Random",
        "model": model_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "domain_results": all_results,
    }
    out_path = TEMP / f"ccvii_structural_random_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nPart 3 结果已保存到: {out_path}")
    
    return output


# ============================================================
# 跨实验综合比较
# ============================================================

def cross_experiment_comparison(math_results, random_tf_results=None, structural_results=None):
    """
    跨实验综合比较
    """
    print(f"\n{'='*70}")
    print(f"CCVII 综合比较")
    print(f"{'='*70}")
    
    # 从数学模拟中提取关键数据
    math_data = {}
    for r in math_results["results"]:
        key = f"N{r['N']}_D{r['D']}"
        math_data[key] = r
    
    print("\n--- 1. 数学随机 vs 训练模型 ---")
    print(f"{'配置':<20} {'随机θ':>10} {'训练θ':>10} {'Δθ':>8} {'随机β':>10} {'训练β':>10} {'Δβ':>8}")
    print("-" * 80)
    
    for model_name, mdata in TRAINED_RESULTS.items():
        D = mdata["d_model"]
        for domain in ["animal50", "vehicle50"]:
            key = f"N50_D{D}"
            if key in math_data:
                mr = math_data[key]
                trained_theta = mdata[domain]["theta"]
                trained_beta = mdata[domain]["beta_emb"]
                delta_theta = mr["mean_theta"] - trained_theta
                delta_beta = mr["mean_beta_emb"] - trained_beta
                
                print(f"{model_name}/{domain:<10} {mr['mean_theta']:>9.1f}° {trained_theta:>9.1f}° {delta_theta:>+7.1f}° "
                      f"{mr['mean_beta_emb']:>+9.3f} {trained_beta:>+9.3f} {delta_beta:>+7.3f}")
    
    print("\n--- 2. D/N比率与θ的关系 ---")
    n50_data = [(r["D_over_N"], r["mean_theta"], r["std_theta"]) 
                for r in math_results["results"] if r["N"] == 50]
    n50_data.sort(key=lambda x: x[0])
    
    for d_over_n, mean_th, std_th in n50_data:
        bar = "█" * int(mean_th - 80)
        print(f"  D/N={d_over_n:5.1f}: θ={mean_th:.2f}°±{std_th:.2f}° {bar}")
    
    print("\n--- 3. 关键结论 ---")
    # 检查: 训练模型的θ是否在随机分布的95%CI内?
    conclusions = []
    for model_name, mdata in TRAINED_RESULTS.items():
        D = mdata["d_model"]
        key = f"N50_D{D}"
        if key in math_data:
            mr = math_data[key]
            for domain in ["animal50", "vehicle50"]:
                trained_theta = mdata[domain]["theta"]
                p5 = mr["theta_percentiles"]["p5"]
                p95 = mr["theta_percentiles"]["p95"]
                
                in_ci = p5 <= trained_theta <= p95
                z = (trained_theta - mr["mean_theta"]) / mr["std_theta"] if mr["std_theta"] > 0 else 0
                
                conclusions.append({
                    "model": model_name,
                    "domain": domain,
                    "trained_theta": trained_theta,
                    "random_mean": mr["mean_theta"],
                    "random_std": mr["std_theta"],
                    "z_score": z,
                    "in_95ci": in_ci,
                })
                
                status = "在95%CI内" if in_ci else "★超出95%CI★"
                print(f"  {model_name}/{domain}: trained={trained_theta:.1f}°, "
                      f"random={mr['mean_theta']:.1f}°±{mr['std_theta']:.1f}°, "
                      f"z={z:.2f} → {status}")
    
    # 汇总判断
    n_in_ci = sum(1 for c in conclusions if c["in_95ci"])
    n_total = len(conclusions)
    
    print(f"\n  ★★★★★ 总结: {n_in_ci}/{n_total}个配置的θ在随机95%CI内")
    
    if n_in_ci == n_total:
        print("  → 所有训练模型的θ都在随机分布的95%CI内")
        print("  → 90°正交是数学必然性(维度诅咒), 不是训练特异性!")
    elif n_in_ci >= n_total * 0.7:
        print("  → 大部分训练模型的θ在随机分布的95%CI内")
        print("  → 90°正交主要是数学必然性, 但训练可能有微弱调节")
    else:
        print("  → 多数训练模型的θ超出随机分布的95%CI")
        print("  → 训练显著改变了旋转角度!")
    
    # β_emb比较
    print("\n--- 4. β_emb: 随机 vs 训练 ---")
    for model_name, mdata in TRAINED_RESULTS.items():
        D = mdata["d_model"]
        key = f"N50_D{D}"
        if key in math_data:
            mr = math_data[key]
            for domain in ["animal50", "vehicle50"]:
                trained_beta = mdata[domain]["beta_emb"]
                random_beta = mr["mean_beta_emb"]
                print(f"  {model_name}/{domain}: "
                      f"random β={random_beta:+.3f}, trained β={trained_beta:+.3f}, "
                      f"Δβ={trained_beta-random_beta:+.3f}")
    
    print("\n  ★★★★★ β_emb差异:")
    print("  → 如果trained β >> random β: 训练使得几何结构被更好保持")
    print("  → 如果trained β ≈ random β: 几何结构保持与训练无关")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default="all",
                       choices=["math", "random", "structural", "all", "compare"])
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    t0 = time.time()
    
    math_results = None
    random_tf_results = None
    structural_results = None
    
    if args.part in ["math", "all"]:
        math_results = mathematical_random_simulation()
    
    if args.part in ["random", "all"]:
        random_tf_results = random_transformer_experiment(args.model)
    
    if args.part in ["structural", "all"]:
        structural_results = structural_random_experiment(args.model)
    
    if args.part == "compare":
        # 加载已有的数学模拟结果
        math_path = TEMP / "ccvii_math_random_results.json"
        if math_path.exists():
            with open(math_path, "r", encoding="utf-8") as f:
                math_results = json.load(f)
        
        # 加载随机Transformer结果
        rtf_path = TEMP / f"ccvii_random_transformer_{args.model}_results.json"
        if rtf_path.exists():
            with open(rtf_path, "r", encoding="utf-8") as f:
                random_tf_results = json.load(f)
        
        # 加载结构化随机结果
        sr_path = TEMP / f"ccvii_structural_random_{args.model}_results.json"
        if sr_path.exists():
            with open(sr_path, "r", encoding="utf-8") as f:
                structural_results = json.load(f)
    
    if math_results is not None:
        cross_experiment_comparison(math_results, random_tf_results, structural_results)
    
    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
