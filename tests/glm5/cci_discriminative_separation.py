"""
CCI(351): 判别性分离效应的直接验证
====================================
★★★★★ CCL核心发现:
  GLM4/DS7B中, animal领域 r(geo,emb)<0
  → embedding相似的动物在几何上更远(判别性分离)

★★★★★ 本实验验证策略:
  实验1: 类别对层面的分离测试
    - 对每个类别对, 计算embedding相似度和几何距离
    - 如果判别性分离存在: 高相似度对的几何距离应偏大
    - 用"分离指数"量化: sep = (d_geo - d_geo_expected) / d_geo_expected
    - d_geo_expected = 基于embedding相似度的线性预测

  实验2: 难度梯度测试
    - 设计3个"近义对"(需要精细区分) 和 3个"远义对"(不需要区分)
    - 预测: 近义对的分离指数应更高
    - 领域: animal(近义: canine-feline, 远义: canine-fish)

  实验3: 层间演变
    - 分离效应应在哪些层出现?
    - 预测: 浅层可能没有分离, 中深层开始出现

  实验4: 分离方向分析
    - 分离是沿哪个方向发生的?
    - 沿语义轴方向分离? 还是沿正交方向?
    - 如果沿语义轴 → 语义驱动
    - 如果沿正交方向 → 分布/功能驱动

  实验5: 模型间对比
    - Qwen3: 之前未显示分离效应, 是否在某些领域/层有?
    - GLM4: 分离最强, 哪些领域最明显?

用法:
  python cci_discriminative_separation.py --model qwen3
  python cci_discriminative_separation.py --model glm4
  python cci_discriminative_separation.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
from scipy.stats import pearsonr, spearmanr, ttest_ind

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# ============================================================
# 领域定义 — 含近义/远义标注
# ============================================================

DOMAINS = {
    "animal10": {
        "categories": {
            "dog":      ["dog", "puppy", "hound", "canine", "pooch", "mutt"],
            "cat":      ["cat", "kitten", "feline", "tomcat", "pussy", "kitty"],
            "wolf":     ["wolf", "werewolf", "lupine", "coyote", "jackal", "husky"],
            "lion":     ["lion", "tiger", "leopard", "cheetah", "panther", "cougar"],
            "bird":     ["bird", "sparrow", "robin", "finch", "wren", "swallow"],
            "eagle":    ["eagle", "hawk", "falcon", "vulture", "osprey", "condor"],
            "fish":     ["fish", "trout", "salmon", "bass", "perch", "cod"],
            "shark":    ["shark", "whale", "dolphin", "porpoise", "orca", "narwhal"],
            "snake":    ["snake", "serpent", "viper", "cobra", "python", "adder"],
            "lizard":   ["lizard", "gecko", "iguana", "chameleon", "salamander", "newt"],
        },
        # 近义对(需要精细区分): dog-cat, wolf-lion, bird-eagle, fish-shark, snake-lizard
        # 远义对(不需要区分): dog-fish, cat-bird, wolf-snake, eagle-shark, lion-lizard
        "near_pairs": [
            ("dog", "cat"), ("wolf", "lion"), ("bird", "eagle"),
            ("fish", "shark"), ("snake", "lizard"),
        ],
        "far_pairs": [
            ("dog", "fish"), ("cat", "bird"), ("wolf", "snake"),
            ("eagle", "shark"), ("lion", "lizard"),
        ],
    },
    "emotion10": {
        "categories": {
            "happy":    ["happy", "joyful", "cheerful", "glad", "pleased", "delighted"],
            "sad":      ["sad", "sorrowful", "unhappy", "gloomy", "miserable", "depressed"],
            "angry":    ["angry", "furious", "enraged", "irate", "hostile", "livid"],
            "scared":   ["scared", "afraid", "fearful", "terrified", "anxious", "frightened"],
            "calm":     ["calm", "peaceful", "serene", "tranquil", "relaxed", "composed"],
            "excited":  ["excited", "thrilled", "elated", "eager", "enthusiastic", "energetic"],
            "proud":    ["proud", "honored", "dignified", "triumphant", "boastful", "arrogant"],
            "ashamed":  ["ashamed", "embarrassed", "guilty", "humiliated", "remorseful", "contrite"],
            "surprised":["surprised", "amazed", "astonished", "shocked", "stunned", "startled"],
            "disgusted":["disgusted", "revolted", "repulsed", "nauseated", "appalled", "sickened"],
        },
        # 近义对(需要区分): happy-excited, sad-ashamed, angry-disgusted, scared-surprised, calm-proud
        # 远义对: happy-sad, angry-calm, scared-proud, excited-disgusted, surprised-ashamed
        "near_pairs": [
            ("happy", "excited"), ("sad", "ashamed"), ("angry", "disgusted"),
            ("scared", "surprised"), ("calm", "proud"),
        ],
        "far_pairs": [
            ("happy", "sad"), ("angry", "calm"), ("scared", "proud"),
            ("excited", "disgusted"), ("surprised", "ashamed"),
        ],
    },
    "profession10": {
        "categories": {
            "doctor":   ["doctor", "physician", "surgeon", "medic", "clinician", "healer"],
            "nurse":    ["nurse", "caregiver", "medic", "attendant", "practitioner", "orderly"],
            "teacher":  ["teacher", "instructor", "educator", "professor", "tutor", "lecturer"],
            "student":  ["student", "pupil", "scholar", "learner", "undergraduate", "trainee"],
            "chef":     ["chef", "cook", "culinary", "baker", "caterer", "pastry"],
            "waiter":   ["waiter", "server", "bartender", "barista", "attendant", "host"],
            "artist":   ["artist", "painter", "sculptor", "illustrator", "designer", "creator"],
            "musician": ["musician", "singer", "guitarist", "pianist", "drummer", "vocalist"],
            "lawyer":   ["lawyer", "attorney", "barrister", "counsel", "advocate", "solicitor"],
            "judge":    ["judge", "magistrate", "justice", "arbitrator", "referee", "adjudicator"],
        },
        # 近义对: doctor-nurse, teacher-student, chef-waiter, artist-musician, lawyer-judge
        # 远义对: doctor-artist, nurse-musician, teacher-chef, student-lawyer, waiter-judge
        "near_pairs": [
            ("doctor", "nurse"), ("teacher", "student"), ("chef", "waiter"),
            ("artist", "musician"), ("lawyer", "judge"),
        ],
        "far_pairs": [
            ("doctor", "artist"), ("nurse", "musician"), ("teacher", "chef"),
            ("student", "lawyer"), ("waiter", "judge"),
        ],
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
                    res = captured[f"L{layer_idx}"][0, -1, :]
                    residuals.append(res)
        
        if len(residuals) > 0:
            cat_centers[cat_name] = np.mean(residuals, axis=0)
    
    return cat_centers


def get_category_centers_embedding(model, tokenizer, categories):
    """从token embedding层获取类别中心"""
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().float().cpu().numpy()
    
    cat_centers = {}
    
    for cat_name, words in categories.items():
        embeddings = []
        for word in words:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            word_emb = np.mean(W_E[token_ids], axis=0)
            embeddings.append(word_emb)
        
        if len(embeddings) > 0:
            cat_centers[cat_name] = np.mean(embeddings, axis=0)
    
    return cat_centers


def compute_pairwise_distances(centers, cat_names):
    """计算Euclidean距离矩阵"""
    points = np.array([centers[name] for name in cat_names])
    dists = squareform(pdist(points, metric='euclidean'))
    return dists


def compute_pairwise_cosine_dist(centers, cat_names):
    """计算cosine距离矩阵"""
    points = np.array([centers[name] for name in cat_names])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points_norm = points / norms
    cos_sim = points_norm @ points_norm.T
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    return cos_dist


def run_experiment(model_name):
    """运行单个模型的完整实验"""
    print(f"\n{'='*70}")
    print(f"CCI: 判别性分离效应的直接验证 - {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"  模型: {info.model_class}, d_model={info.d_model}, n_layers={n_layers}")
    
    # 选择测试层: 密集采样中层(分离效应最可能出现的区域)
    layer_candidates = sorted(set([
        1,                                          # 极浅
        max(2, n_layers // 6),                      # 浅
        n_layers // 4,                              # 中浅
        n_layers // 3,                              # 中
        5 * n_layers // 12,                         # 中中
        n_layers // 2,                              # 中
        7 * n_layers // 12,                         # 中中
        2 * n_layers // 3,                          # 中深
        3 * n_layers // 4,                          # 深中
        min(n_layers - 2, 5 * n_layers // 6),       # 深
    ]))
    print(f"  测试层: {layer_candidates}")
    
    all_results = {}
    
    for domain_name, domain_def in DOMAINS.items():
        categories = domain_def["categories"]
        near_pairs = domain_def["near_pairs"]
        far_pairs = domain_def["far_pairs"]
        cat_names = list(categories.keys())
        N = len(cat_names)
        
        print(f"\n--- 领域: {domain_name} (N={N}) ---")
        
        # 1. 计算embedding距离(固定)
        emb_centers = get_category_centers_embedding(model, tokenizer, categories)
        if len(emb_centers) != N:
            print(f"  WARNING: 只得到{len(emb_centers)}/{N}个类别, 跳过")
            continue
        emb_dist_mat = compute_pairwise_cosine_dist(emb_centers, cat_names)
        
        domain_results = {}
        
        for layer_idx in layer_candidates:
            print(f"  L{layer_idx}...", end=" ", flush=True)
            
            # 收集残差中心
            res_centers = get_category_centers_residual(model, tokenizer, device, categories, layer_idx)
            if len(res_centers) != N:
                print(f"跳过(只有{len(res_centers)}个类别)")
                continue
            
            # SVD投影
            points = np.array([res_centers[name] for name in cat_names])
            U, S, Vt = svd(points, full_matrices=False)
            D = min(N - 1, points.shape[1])
            points_proj = U[:, :D] @ np.diag(S[:D])
            
            # 几何距离矩阵
            geo_dist_mat = compute_pairwise_distances(
                {name: points_proj[i] for i, name in enumerate(cat_names)},
                cat_names
            )
            
            # === 实验1: 全局分离测试 ===
            # 对于所有对, 计算r(geo, emb)
            upper_idx = np.triu_indices(N, k=1)
            geo_flat = geo_dist_mat[upper_idx]
            emb_flat = emb_dist_mat[upper_idx]
            
            r_geo_emb, p_geo_emb = pearsonr(geo_flat, emb_flat)
            
            # 线性预测: geo = a * emb + b
            from numpy.linalg import lstsq
            X = np.column_stack([emb_flat, np.ones(len(emb_flat))])
            coeffs, _, _, _ = lstsq(X, geo_flat, rcond=None)
            a_coeff, b_coeff = coeffs
            geo_predicted = a_coeff * emb_flat + b_coeff
            
            # 分离指数: (actual - predicted) / predicted
            # 正值 = 几何距离大于预测 = 被推开(分离)
            # 负值 = 几何距离小于预测 = 被拉近(聚合)
            separation_index = (geo_flat - geo_predicted) / np.maximum(np.abs(geo_predicted), 1e-10)
            
            # === 实验2: 近义对 vs 远义对 ===
            near_sep_values = []
            far_sep_values = []
            near_geo_dists = []
            far_geo_dists = []
            near_emb_sims = []
            far_emb_sims = []
            
            for pair in near_pairs:
                i, j = cat_names.index(pair[0]), cat_names.index(pair[1])
                pair_idx = -1
                for k in range(len(upper_idx[0])):
                    if (upper_idx[0][k] == min(i,j) and upper_idx[1][k] == max(i,j)):
                        pair_idx = k
                        break
                if pair_idx >= 0:
                    near_sep_values.append(separation_index[pair_idx])
                    near_geo_dists.append(geo_flat[pair_idx])
                    near_emb_sims.append(1 - emb_flat[pair_idx])  # cosine similarity
            
            for pair in far_pairs:
                i, j = cat_names.index(pair[0]), cat_names.index(pair[1])
                pair_idx = -1
                for k in range(len(upper_idx[0])):
                    if (upper_idx[0][k] == min(i,j) and upper_idx[1][k] == max(i,j)):
                        pair_idx = k
                        break
                if pair_idx >= 0:
                    far_sep_values.append(separation_index[pair_idx])
                    far_geo_dists.append(geo_flat[pair_idx])
                    far_emb_sims.append(1 - emb_flat[pair_idx])
            
            # t-test: 近义对 vs 远义对的分离指数
            if len(near_sep_values) >= 2 and len(far_sep_values) >= 2:
                t_stat, t_pval = ttest_ind(near_sep_values, far_sep_values)
            else:
                t_stat, t_pval = 0, 1.0
            
            # === 实验3: 高embedding相似度对的分离分析 ===
            # 将所有对按embedding相似度排序, 分析top-5相似对的分离指数
            sorted_by_emb = sorted(range(len(emb_flat)), key=lambda k: emb_flat[k])
            top5_similar = sorted_by_emb[:5]  # 最相似的5对
            top5_dissimilar = sorted_by_emb[-5:]  # 最不相似的5对
            
            avg_sep_similar = np.mean(separation_index[top5_similar])
            avg_sep_dissimilar = np.mean(separation_index[top5_dissimilar])
            
            # === 实验4: 分离方向分析 ===
            # 对于近义对, 计算分离方向
            separation_directions = {}
            for pair in near_pairs:
                cat_i, cat_j = pair
                idx_i, idx_j = cat_names.index(cat_i), cat_names.index(cat_j)
                
                # 分离向量: 从embedding预测的位置到实际位置的方向
                # 简化: 用两中心的连线方向
                center_i = points_proj[idx_i]
                center_j = points_proj[idx_j]
                
                # 连线方向(从i到j)
                line_dir = center_j - center_i
                line_norm = np.linalg.norm(line_dir)
                if line_norm > 1e-10:
                    line_dir = line_dir / line_norm
                
                # 实际距离
                actual_dist = np.linalg.norm(center_j - center_i)
                
                separation_directions[f"{cat_i}-{cat_j}"] = {
                    "actual_dist": float(actual_dist),
                    "line_dir_norm": float(line_norm),
                }
            
            layer_key = f"L{layer_idx}"
            domain_results[layer_key] = {
                # 全局
                "r_geo_emb": float(r_geo_emb), "p_geo_emb": float(p_geo_emb),
                "n_pairs": int(len(geo_flat)),
                # 近义vs远义
                "near_sep_mean": float(np.mean(near_sep_values)) if near_sep_values else 0,
                "far_sep_mean": float(np.mean(far_sep_values)) if far_sep_values else 0,
                "near_geo_mean": float(np.mean(near_geo_dists)) if near_geo_dists else 0,
                "far_geo_mean": float(np.mean(far_geo_dists)) if far_geo_dists else 0,
                "near_emb_sim_mean": float(np.mean(near_emb_sims)) if near_emb_sims else 0,
                "far_emb_sim_mean": float(np.mean(far_emb_sims)) if far_emb_sims else 0,
                "t_stat": float(t_stat), "t_pval": float(t_pval),
                # 高相似度对分析
                "avg_sep_top5_similar": float(avg_sep_similar),
                "avg_sep_top5_dissimilar": float(avg_sep_dissimilar),
                # 分离方向
                "separation_directions": separation_directions,
            }
            
            sep_indicator = "SEP" if avg_sep_similar > 0 else "CLP"
            print(f"r={r_geo_emb:+.3f} near_sep={np.mean(near_sep_values):+.3f} "
                  f"far_sep={np.mean(far_sep_values):+.3f} top5_sim={avg_sep_similar:+.3f}({sep_indicator})")
        
        all_results[domain_name] = domain_results
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # 保存结果
    out_path = TEMP / f"cci_{model_name}_results.json"
    
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
        print(f"  {'层':>6} | {'r(geo,emb)':>10} | {'近义sep':>8} | {'远义sep':>8} | "
              f"{'近义geo':>8} | {'远义geo':>8} | {'近义sim':>8} | {'远义sim':>8} | {'t-p':>6}")
        print(f"  {'-'*100}")
        
        for key in sorted(domain_res.keys()):
            if key.startswith("L"):
                val = domain_res[key]
                sig = "*" if val["t_pval"] < 0.05 else ""
                print(f"  {key[1:]:>6} | {val['r_geo_emb']:+10.3f} | {val['near_sep_mean']:+8.3f} | "
                      f"{val['far_sep_mean']:+8.3f} | {val['near_geo_mean']:+8.3f} | "
                      f"{val['far_geo_mean']:+8.3f} | {val['near_emb_sim_mean']:+8.3f} | "
                      f"{val['far_emb_sim_mean']:+8.3f} | {val['t_pval']:.3f}{sig}")
    
    # === 核心指标: 判别性分离是否存在于近义对中 ===
    print(f"\n{'='*70}")
    print(f"核心判定: 近义对是否被推开(分离)?")
    print(f"{'='*70}")
    
    for domain_name, domain_res in all_results.items():
        sep_layers = 0
        total_layers = 0
        for key, val in domain_res.items():
            if key.startswith("L"):
                total_layers += 1
                if val["near_sep_mean"] > 0 and val["avg_sep_top5_similar"] > 0:
                    sep_layers += 1
        
        if total_layers > 0:
            sep_pct = sep_layers / total_layers * 100
            print(f"  {domain_name}: {sep_layers}/{total_layers}层显示近义对分离({sep_pct:.0f}%)")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    args = parser.parse_args()
    
    t0 = time.time()
    results = run_experiment(args.model)
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}秒")
