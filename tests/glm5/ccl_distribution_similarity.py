"""
CCL(350): 分布相似性假说的直接验证
======================================
★★★★★ CCXLIX核心发现:
  "对比组织"假说被推翻! 所有对立领域都显示正相关!
  修正假说: 几何距离反映分布相似性(共现模式), 不是语义对立性

★★★★★ 本实验核心验证:
  1. 用模型token embedding的cosine相似度作为分布相似性度量
  2. 用手定义语义维度距离作为语义距离度量
  3. 比较哪个更好地预测残差空间中的几何距离
  4. 用偏相关分离两者的独立贡献

★★★★★ 设计改进:
  - N=8类别/领域 → 28对/领域 → 充足统计功效
  - 4个领域: emotion8, animal8, color8, evaluation8
  - 多层分析: 浅层vs深层, 检查组织原则是否随层变化

用法:
  python ccl_distribution_similarity.py --model qwen3
  python ccl_distribution_similarity.py --model glm4
  python ccl_distribution_similarity.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
from scipy.stats import pearsonr, spearmanr

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# ============================================================
# 语义类别定义 — N=8类别, 4领域
# ============================================================

DOMAINS = {
    "emotion8": {
        "categories": {
            "happy":  ["happy", "joy", "cheerful", "glad", "pleased", "delight"],
            "sad":    ["sad", "sorrow", "unhappy", "gloomy", "miserable", "grief"],
            "angry":  ["angry", "furious", "rage", "mad", "hostile", "outrage"],
            "scared": ["scared", "afraid", "fearful", "terrified", "anxious", "panic"],
            "surprised": ["surprised", "amazed", "astonished", "shocked", "stunned", "startled"],
            "disgusted": ["disgusted", "revolted", "repulsed", "nauseated", "sickened", "appalled"],
            "proud":  ["proud", "honored", "dignified", "triumphant", "arrogant", "boastful"],
            "calm":   ["calm", "peaceful", "serene", "tranquil", "relaxed", "composed"],
        },
        # VAD (valence, arousal, dominance) - 标准化到[-1,1]
        "semantic_dims": {
            "happy":     [ 0.81,  0.51,  0.46],
            "sad":       [-0.81, -0.58, -0.43],
            "angry":     [-0.64,  0.83,  0.60],
            "scared":    [-0.63,  0.83, -0.36],
            "surprised": [ 0.32,  0.83, -0.15],
            "disgusted": [-0.60,  0.35,  0.12],
            "proud":     [ 0.68,  0.50,  0.78],
            "calm":      [ 0.56, -0.49,  0.32],
        },
        "dim_names": ["valence", "arousal", "dominance"],
    },
    "animal8": {
        "categories": {
            "canine":  ["dog", "wolf", "fox", "coyote", "jackal", "hound"],
            "feline":  ["cat", "lion", "tiger", "leopard", "cheetah", "panther"],
            "bird":    ["eagle", "sparrow", "owl", "parrot", "penguin", "hawk"],
            "fish":    ["salmon", "trout", "shark", "tuna", "cod", "bass"],
            "insect":  ["ant", "bee", "butterfly", "beetle", "wasp", "moth"],
            "reptile": ["snake", "lizard", "turtle", "crocodile", "iguana", "gecko"],
            "primate": ["monkey", "ape", "gorilla", "chimpanzee", "baboon", "lemur"],
            "rodent":  ["mouse", "rat", "squirrel", "hamster", "rabbit", "guinea"],
        },
        # (size:1-5, warm_blooded:0/1, habitat:land=1/water=0/air=0.5, carnivore:0-1)
        "semantic_dims": {
            "canine":  [3, 1, 1.0, 1.0],
            "feline":  [3, 1, 1.0, 1.0],
            "bird":    [2, 1, 0.5, 0.3],
            "fish":    [2, 0, 0.0, 0.8],
            "insect":  [1, 0, 1.0, 0.2],
            "reptile": [2, 0, 1.0, 0.7],
            "primate": [3, 1, 1.0, 0.3],
            "rodent":  [1, 1, 1.0, 0.1],
        },
        "dim_names": ["size", "warm_blooded", "habitat", "carnivore"],
    },
    "color8": {
        "categories": {
            "red":    ["red", "crimson", "scarlet", "ruby", "maroon", "cherry"],
            "blue":   ["blue", "azure", "cobalt", "navy", "sapphire", "indigo"],
            "green":  ["green", "emerald", "lime", "olive", "jade", "forest"],
            "yellow": ["yellow", "golden", "amber", "lemon", "mustard", "saffron"],
            "orange": ["orange", "tangerine", "coral", "peach", "apricot", "copper"],
            "purple": ["purple", "violet", "lavender", "plum", "magenta", "mauve"],
            "pink":   ["pink", "rose", "fuchsia", "blush", "carnation", "salmon"],
            "brown":  ["brown", "chocolate", "tan", "bronze", "chestnut", "mahogany"],
        },
        # 近似RGB (归一化到0-1)
        "semantic_dims": {
            "red":    [0.90, 0.05, 0.05],
            "blue":   [0.10, 0.10, 0.90],
            "green":  [0.05, 0.55, 0.05],
            "yellow": [0.95, 0.95, 0.10],
            "orange": [0.95, 0.55, 0.05],
            "purple": [0.50, 0.05, 0.55],
            "pink":   [0.95, 0.55, 0.60],
            "brown":  [0.55, 0.30, 0.05],
        },
        "dim_names": ["R", "G", "B"],
    },
    "evaluation8": {
        "categories": {
            "excellent": ["excellent", "outstanding", "superb", "magnificent", "brilliant", "exceptional"],
            "good":      ["good", "nice", "fine", "decent", "pleasant", "satisfactory"],
            "amazing":   ["amazing", "incredible", "wonderful", "fantastic", "marvelous", "extraordinary"],
            "perfect":   ["perfect", "flawless", "ideal", "impeccable", "supreme", "ultimate"],
            "terrible":  ["terrible", "awful", "dreadful", "horrible", "atrocious", "appalling"],
            "bad":       ["bad", "poor", "inferior", "lousy", "rotten", "deficient"],
            "horrific":  ["horrific", "shocking", "ghastly", "gruesome", "nightmarish", "horrifying"],
            "mediocre":  ["mediocre", "average", "ordinary", "passable", "adequate", "acceptable"],
        },
        # (valence: -1 to +1, intensity: 1-5)
        "semantic_dims": {
            "excellent": [ 1.0, 4],
            "good":      [ 0.5, 2],
            "amazing":   [ 1.0, 4],
            "perfect":   [ 1.0, 5],
            "terrible":  [-1.0, 4],
            "bad":       [-0.5, 2],
            "horrific":  [-1.0, 5],
            "mediocre":  [ 0.0, 1],
        },
        "dim_names": ["valence", "intensity"],
    },
}


def get_category_centers_residual(model, tokenizer, device, categories, layer_idx):
    """在指定层收集残差中心"""
    layers = get_layers(model)
    embed_layer = model.get_input_embeddings()
    
    cat_centers = {}
    
    for cat_name, words in categories.items():
        residuals = []
        for word in words:
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
                _ = model(inputs_embeds=inputs_embeds)
                hook.remove()
                
                if f"L{layer_idx}" in captured:
                    # 取最后一个token的残差
                    res = captured[f"L{layer_idx}"][0, -1, :]
                    residuals.append(res)
        
        if len(residuals) > 0:
            cat_centers[cat_name] = np.mean(residuals, axis=0)
    
    return cat_centers


def get_category_centers_embedding(model, tokenizer, categories):
    """从token embedding层获取类别中心(分布相似性度量)"""
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().float().cpu().numpy()  # [vocab, d_model]
    
    cat_centers = {}
    
    for cat_name, words in categories.items():
        embeddings = []
        for word in words:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            # 多子词取均值
            word_emb = np.mean(W_E[token_ids], axis=0)
            embeddings.append(word_emb)
        
        if len(embeddings) > 0:
            cat_centers[cat_name] = np.mean(embeddings, axis=0)
    
    return cat_centers


def compute_distance_matrix(centers, cat_names):
    """计算类别中心之间的距离矩阵"""
    N = len(cat_names)
    points = np.array([centers[name] for name in cat_names])  # [N, d]
    
    # SVD投影到(N-1)维(仅对残差空间)
    # 对于embedding空间, 直接用cosine距离
    
    # 欧氏距离
    dists = squareform(pdist(points, metric='euclidean'))
    
    return dists


def compute_cosine_distance_matrix(centers, cat_names):
    """计算cosine距离矩阵 (1 - cosine_similarity)"""
    N = len(cat_names)
    points = np.array([centers[name] for name in cat_names])  # [N, d]
    
    # 归一化
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points_norm = points / norms
    
    # cosine similarity矩阵
    cos_sim = points_norm @ points_norm.T
    # cosine distance = 1 - similarity
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    
    return cos_dist


def compute_semantic_distance_matrix(semantic_dims, cat_names):
    """计算语义距离矩阵(在语义维度空间中的Euclidean距离)"""
    N = len(cat_names)
    points = np.array([semantic_dims[name] for name in cat_names], dtype=float)
    
    # 标准化每个维度到[0,1]
    for j in range(points.shape[1]):
        vmin, vmax = points[:, j].min(), points[:, j].max()
        if vmax > vmin:
            points[:, j] = (points[:, j] - vmin) / (vmax - vmin)
    
    dists = squareform(pdist(points, metric='euclidean'))
    return dists


def flatten_upper_tri(mat, N):
    """提取上三角元素(不含对角线)"""
    indices = np.triu_indices(N, k=1)
    return mat[indices]


def partial_corr_numpy(x, y, z):
    """
    计算偏相关 corr(x, y | z)
    即在控制z后, x和y的相关性
    """
    from numpy.linalg import lstsq
    
    # x_residual = x - z的线性预测
    z_mat = np.column_stack([z, np.ones(len(z))])
    coef_x, _, _, _ = lstsq(z_mat, x, rcond=None)
    x_resid = x - z_mat @ coef_x
    
    coef_y, _, _, _ = lstsq(z_mat, y, rcond=None)
    y_resid = y - z_mat @ coef_y
    
    # 残差之间的相关
    if np.std(x_resid) < 1e-10 or np.std(y_resid) < 1e-10:
        return 0.0, 1.0
    
    r, p = pearsonr(x_resid, y_resid)
    return r, p


def run_experiment(model_name):
    """运行单个模型的完整实验"""
    print(f"\n{'='*70}")
    print(f"CCL: 分布相似性假说直接验证 - {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"  模型: {info.model_class}, d_model={info.d_model}, n_layers={n_layers}")
    
    # 选择测试层: 浅/中/深
    layer_candidates = [
        max(1, n_layers // 6),       # 浅层 ~17%
        n_layers // 3,               # 中浅层 ~33%
        n_layers // 2,               # 中层 ~50%
        2 * n_layers // 3,           # 中深层 ~67%
        min(n_layers - 2, 5 * n_layers // 6),  # 深层 ~83%
    ]
    layer_candidates = sorted(set(layer_candidates))
    print(f"  测试层: {layer_candidates}")
    
    all_results = {}
    
    for domain_name, domain_def in DOMAINS.items():
        categories = domain_def["categories"]
        semantic_dims = domain_def["semantic_dims"]
        cat_names = list(categories.keys())
        N = len(cat_names)
        
        print(f"\n--- 领域: {domain_name} (N={N}, {N*(N-1)//2}对) ---")
        
        # 1. 计算语义距离矩阵(固定, 不依赖层)
        sem_dist_mat = compute_semantic_distance_matrix(semantic_dims, cat_names)
        sem_dists = flatten_upper_tri(sem_dist_mat, N)
        
        # 2. 计算embedding距离矩阵(固定, 不依赖层)
        emb_centers = get_category_centers_embedding(model, tokenizer, categories)
        if len(emb_centers) != N:
            print(f"  WARNING: 只得到{len(emb_centers)}/{N}个类别的embedding中心, 跳过")
            continue
        emb_dist_mat = compute_cosine_distance_matrix(emb_centers, cat_names)
        emb_dists = flatten_upper_tri(emb_dist_mat, N)
        
        # 3. 逐层计算几何距离并分析
        domain_results = {"semantic_dims": domain_def["dim_names"]}
        
        for layer_idx in layer_candidates:
            print(f"  层L{layer_idx}...", end=" ", flush=True)
            
            # 收集残差中心
            res_centers = get_category_centers_residual(model, tokenizer, device, categories, layer_idx)
            if len(res_centers) != N:
                print(f"只得到{len(res_centers)}/{N}个类别中心, 跳过")
                continue
            
            # SVD投影到(N-1)维
            points = np.array([res_centers[name] for name in cat_names])
            U, S, Vt = svd(points, full_matrices=False)
            D = min(N - 1, points.shape[1])
            points_proj = U[:, :D] @ np.diag(S[:D])  # [N, D]
            
            # 几何距离矩阵
            geo_dist_mat = squareform(pdist(points_proj, metric='euclidean'))
            geo_dists = flatten_upper_tri(geo_dist_mat, N)
            
            # === 核心分析 ===
            
            # A. 几何 vs 语义
            r_geo_sem, p_geo_sem = pearsonr(geo_dists, sem_dists)
            rho_geo_sem, p_rho_sem = spearmanr(geo_dists, sem_dists)
            
            # B. 几何 vs embedding (分布相似性)
            r_geo_emb, p_geo_emb = pearsonr(geo_dists, emb_dists)
            rho_geo_emb, p_rho_emb = spearmanr(geo_dists, emb_dists)
            
            # C. embedding vs 语义 (两者之间的相关, 控制变量)
            r_emb_sem, p_emb_sem = pearsonr(emb_dists, sem_dists)
            
            # D. 偏相关: 控制 embedding 后, 几何 vs 语义
            pr_geo_sem_ctrl_emb, pp_geo_sem = partial_corr_numpy(geo_dists, sem_dists, emb_dists)
            
            # E. 偏相关: 控制 语义 后, 几何 vs embedding
            pr_geo_emb_ctrl_sem, pp_geo_emb = partial_corr_numpy(geo_dists, emb_dists, sem_dists)
            
            # F. 哪个预测变量更强?
            # 直接比较 r^2
            r2_sem = r_geo_sem**2
            r2_emb = r_geo_emb**2
            winner = "embedding" if r2_emb > r2_sem else "semantic"
            
            # G. 多元回归: geo ~ sem + emb
            from numpy.linalg import lstsq
            X = np.column_stack([sem_dists, emb_dists, np.ones(len(sem_dists))])
            coeffs, _, _, _ = lstsq(X, geo_dists, rcond=None)
            beta_sem, beta_emb = coeffs[0], coeffs[1]
            
            # 预测R^2
            pred = X @ coeffs
            ss_res = np.sum((geo_dists - pred)**2)
            ss_tot = np.sum((geo_dists - np.mean(geo_dists))**2)
            R2_total = 1 - ss_res / max(ss_tot, 1e-10)
            
            layer_key = f"L{layer_idx}"
            domain_results[layer_key] = {
                "r_geo_sem": float(r_geo_sem), "p_geo_sem": float(p_geo_sem),
                "rho_geo_sem": float(rho_geo_sem), "p_rho_sem": float(p_rho_sem),
                "r_geo_emb": float(r_geo_emb), "p_geo_emb": float(p_geo_emb),
                "rho_geo_emb": float(rho_geo_emb), "p_rho_emb": float(p_rho_emb),
                "r_emb_sem": float(r_emb_sem), "p_emb_sem": float(p_emb_sem),
                "pr_geo_sem_ctrl_emb": float(pr_geo_sem_ctrl_emb),
                "pp_geo_sem": float(pp_geo_sem),
                "pr_geo_emb_ctrl_sem": float(pr_geo_emb_ctrl_sem),
                "pp_geo_emb": float(pp_geo_emb),
                "r2_sem": float(r2_sem), "r2_emb": float(r2_emb),
                "winner": winner,
                "beta_sem": float(beta_sem), "beta_emb": float(beta_emb),
                "R2_total": float(R2_total),
            }
            
            sig_star = "*" if p_geo_emb < 0.05 else ""
            sig_star2 = "*" if p_geo_sem < 0.05 else ""
            print(f"r(geo,emb)={r_geo_emb:+.3f}{sig_star} r(geo,sem)={r_geo_sem:+.3f}{sig_star2} "
                  f"winner={winner} R2={R2_total:.3f}")
        
        all_results[domain_name] = domain_results
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # 保存结果
    out_path = TEMP / f"ccl_{model_name}_results.json"
    
    def make_serializable(obj):
        if obj is None: return None
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.bool_)): return float(obj) if not isinstance(obj, np.bool_) else bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, (bool, int, float, str)): return obj
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [make_serializable(x) for x in obj]
        if isinstance(obj, tuple): return str(obj)
        return str(obj)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(all_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    # === 汇总分析 ===
    print(f"\n{'='*70}")
    print(f"汇总分析 - {model_name}")
    print(f"{'='*70}")
    
    for domain_name, domain_res in all_results.items():
        print(f"\n--- {domain_name} ---")
        print(f"  {'层':>6} | {'r(geo,emb)':>10} | {'r(geo,sem)':>10} | {'偏r(emb|sem)':>12} | {'偏r(sem|emb)':>12} | {'winner':>8} | {'R2':>6}")
        print(f"  {'-'*80}")
        
        for key, val in domain_res.items():
            if key.startswith("L"):
                layer_num = key[1:]
                sig_emb = "*" if val.get("pp_geo_emb", 1) < 0.05 else ""
                sig_sem = "*" if val.get("pp_geo_sem", 1) < 0.05 else ""
                print(f"  {layer_num:>6} | {val['r_geo_emb']:+10.3f} | {val['r_geo_sem']:+10.3f} | "
                      f"{val['pr_geo_emb_ctrl_sem']:+12.3f}{sig_emb} | "
                      f"{val['pr_geo_sem_ctrl_emb']:+12.3f}{sig_sem} | "
                      f"{val['winner']:>8} | {val['R2_total']:6.3f}")
    
    # === 跨领域总结 ===
    print(f"\n{'='*70}")
    print(f"跨领域总结 - {model_name}")
    print(f"{'='*70}")
    
    # 对每个层, 统计所有领域中embedding胜出vs semantic胜出的次数
    for key in ["L" + str(l) for l in layer_candidates]:
        emb_wins = 0
        sem_wins = 0
        ties = 0
        for domain_name, domain_res in all_results.items():
            if key in domain_res:
                w = domain_res[key]["winner"]
                if w == "embedding": emb_wins += 1
                elif w == "semantic": sem_wins += 1
                else: ties += 1
        
        total = emb_wins + sem_wins + ties
        if total > 0:
            print(f"  {key}: embedding胜={emb_wins}, semantic胜={sem_wins}, 平={ties} (共{total}领域)")
    
    # 找出最佳预测层(跨领域平均R2最高)
    best_layer = None
    best_avg_r2 = -1
    for key in ["L" + str(l) for l in layer_candidates]:
        r2s = []
        for domain_name, domain_res in all_results.items():
            if key in domain_res:
                r2s.append(domain_res[key]["R2_total"])
        if len(r2s) > 0:
            avg_r2 = np.mean(r2s)
            if avg_r2 > best_avg_r2:
                best_avg_r2 = avg_r2
                best_layer = key
    
    print(f"\n  最佳预测层: {best_layer} (跨领域平均R2={best_avg_r2:.3f})")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    args = parser.parse_args()
    
    t0 = time.time()
    results = run_experiment(args.model)
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}秒")
