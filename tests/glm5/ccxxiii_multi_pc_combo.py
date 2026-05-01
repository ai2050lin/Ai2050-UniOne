"""
CCXXIII(323): 推理方向的多PC线性组合
======================================================================
CCXXI发现"大"方向≈PC0/PC1(cos=0.75-0.80), 但不等于任何单一PC。
20%的差异来自哪里? 本实验: "大"方向 = Σ α_k * PC_k?

实验设计:
  1. 收集比较推理残差, PCA分解
  2. "大"方向在PC空间的投影: big_dir = Σ (big_dir · PC_k) * PC_k
  3. 最优线性组合: 最小二乘拟合 big_dir = Σ α_k * PC_k
  4. 残差分析: 拟合后残差的方向和大小
  5. Perturb测试: 用最优组合方向perturb vs 单一PC perturb

用法:
  python ccxxiii_multi_pc_combo.py --model qwen3
  python ccxxiii_multi_pc_combo.py --model glm4
  python ccxxiii_multi_pc_combo.py --model deepseek7b
"""
import argparse, os, sys, json, gc
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy import stats

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model, get_W_U

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxiii_multi_pc_combo_log.txt"

# 比较词对
SIZE_PAIRS = [
    ("elephant", "mouse"), ("whale", "fish"), ("horse", "cat"),
    ("lion", "rabbit"), ("eagle", "sparrow"), ("bear", "fox"),
    ("cow", "chicken"), ("shark", "crab"), ("tiger", "rat"),
    ("mountain", "hill"), ("tree", "bush"), ("house", "shed"),
    ("bus", "car"), ("ship", "boat"), ("plane", "kite"),
]

WEIGHT_PAIRS = [
    ("iron", "feather"), ("rock", "leaf"), ("steel", "paper"),
    ("gold", "cotton"), ("lead", "silk"), ("stone", "grass"),
    ("concrete", "foam"), ("brick", "straw"), ("copper", "wool"),
]

SPEED_PAIRS = [
    ("cheetah", "turtle"), ("falcon", "snail"), ("horse", "slug"),
    ("rocket", "cart"), ("jet", "boat"), ("leopard", "worm"),
    ("eagle", "ant"), ("tiger", "sloth"), ("deer", "beetle"),
]

TEMPLATE_COMPARE = "The {} is bigger than the {}"

# Habitat词汇(用于perturb测试)
WORDS_BY_HABITAT = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit", "fox", "deer"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle", "crab", "seal", "squid", "lobster"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow", "falcon", "pigeon", "robin"],
}

TEMPLATE_HABITAT = "The {} lives in the"

HABITAT_TOKENS = {
    "land": ["land", "ground", "earth", "field", "forest", "plains"],
    "ocean": ["ocean", "sea", "water", "river", "lake", "deep"],
    "sky": ["sky", "air", "trees", "mountains", "heights", "clouds"],
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    log(f"\n{'='*70}\nCCXXIII(323): 推理方向的多PC线性组合 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    W_U = get_W_U(model)
    
    # ===== Step 1: 收集比较推理残差 =====
    log("\n--- Step 1: 收集比较推理残差 ---")
    
    compare_resids = {li: [] for li in test_layers}
    
    all_pairs = SIZE_PAIRS + WEIGHT_PAIRS + SPEED_PAIRS
    for big, small in all_pairs:
        prompt = TEMPLATE_COMPARE.format(big, small)
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = toks.input_ids.shape[1]
        last_pos = seq_len - 1
        
        captured = {}
        def mk_hook(k):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured[k] = o[0, last_pos, :].detach().float().cpu().numpy()
            return hook
        
        hooks = [layers[li].register_forward_hook(mk_hook(f"L{li}")) for li in test_layers]
        with torch.no_grad():
            _ = model(**toks)
        for h in hooks:
            h.remove()
        
        for li in test_layers:
            if f"L{li}" in captured:
                compare_resids[li].append(captured[f"L{li}"])
    
    log(f"  收集了 {len(compare_resids[test_layers[0]])} 个比较残差")
    
    # ===== Step 2: 多PC线性组合分析 =====
    log("\n--- Step 2: 多PC线性组合分析 ---")
    
    for li in test_layers:
        X = np.array(compare_resids[li])
        n_samples = X.shape[0]
        if n_samples < 3:
            continue
        
        # PCA
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # "大"方向: size比较的均值方向
        size_only = compare_resids[li][:len(SIZE_PAIRS)]
        if len(size_only) < 3:
            continue
        
        big_dir_mean = np.mean(size_only, axis=0) - mean
        big_dir_norm = np.linalg.norm(big_dir_mean)
        if big_dir_norm < 1e-10:
            continue
        big_dir = big_dir_mean / big_dir_norm
        
        # 2a: 投影到PC空间
        # big_dir = Σ (big_dir · PC_k) * PC_k (在PC子空间内)
        # 取前K个PC
        for K in [1, 3, 5, 10, 20, 50]:
            # 在前K个PC上的投影
            coeffs = Vt[:K] @ big_dir  # [K]
            reconstructed = Vt[:K].T @ coeffs  # [d_model]
            
            # 余弦相似度
            cos_recon = float(np.dot(big_dir, reconstructed) / 
                            (np.linalg.norm(reconstructed) + 1e-10))
            
            # 投影能量比
            recon_energy = float(np.linalg.norm(reconstructed)**2)
            big_energy = float(np.linalg.norm(big_dir)**2)
            energy_ratio = recon_energy / big_energy if big_energy > 0 else 0
            
            key = f"pc_combo_K{K}_L{li}"
            results[key] = {
                "layer": li,
                "n_pcs": K,
                "cos_reconstruction": round(cos_recon, 4),
                "energy_ratio": round(energy_ratio, 4),
                "top_coeffs": [round(float(c), 4) for c in coeffs[:min(K, 10)]],
                "coeff_squared_ratio": [round(float(c**2) / (np.sum(coeffs**2) + 1e-10), 4) for c in coeffs[:min(K, 10)]],
            }
            
            log(f"  L{li} K={K}: cos_recon={cos_recon:.4f}, energy_ratio={energy_ratio:.4f}")
        
        # 2b: 最优线性组合(最小二乘)
        # 找 α 使得 ||big_dir - Vt[:K].T @ α|| 最小
        # α = Vt[:K] @ big_dir (这就是投影, 已经是最优了)
        # 但我们可以检查: 是否存在非投影的更好组合?
        # 答案: 不存在, 因为PC是正交基, 投影就是最优
        
        # 2c: 残差方向分析
        # 残差 = big_dir - reconstructed (K=50)
        K_full = min(50, Vt.shape[0])
        coeffs_full = Vt[:K_full] @ big_dir
        reconstructed_full = Vt[:K_full].T @ coeffs_full
        residual = big_dir - reconstructed_full
        residual_norm = np.linalg.norm(residual)
        big_norm = np.linalg.norm(big_dir)
        
        # 残差方向: 不在PC子空间中的部分
        # 这在数学上应该=0(因为PC是完备基), 但由于数值误差可能有微小残差
        # 但如果用少量样本估计PCA, PC可能不完备
        # 实际上, X_centered的SVD最多有min(n_samples, d_model)个非零奇异值
        # 所以如果n_samples < d_model, 确实存在不在PC子空间中的方向!
        
        cos_residual = float(np.dot(big_dir, residual) / (residual_norm * big_norm + 1e-10)) if residual_norm > 1e-10 else 0
        
        results[f"residual_analysis_L{li}"] = {
            "layer": li,
            "n_samples": n_samples,
            "n_pcs_available": min(n_samples, d_model),
            "big_dir_norm": round(float(big_norm), 4),
            "residual_norm": round(float(residual_norm), 6),
            "residual_ratio": round(float(residual_norm / big_norm), 6) if big_norm > 0 else 0,
        }
        
        log(f"  L{li}: residual_norm={residual_norm:.6f}, residual_ratio={residual_norm/big_norm:.6f}")
    
    # ===== Step 3: 非线性残差分析 =====
    log("\n--- Step 3: 非线性残差分析 ---")
    
    # "大"方向不在PC子空间中的部分(如果存在)
    # 但更可能的是: "大"方向在PC子空间中, 但cos=0.80说明
    # big_dir不是PC方向的简单组合, 而是多个PC的"斜向"组合
    # 检查: big_dir的前10个PC系数的分布
    
    for li in test_layers:
        X = np.array(compare_resids[li])
        if X.shape[0] < 3:
            continue
        
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        size_only = compare_resids[li][:len(SIZE_PAIRS)]
        if len(size_only) < 3:
            continue
        
        big_dir_mean = np.mean(size_only, axis=0) - mean
        big_dir_norm = np.linalg.norm(big_dir_mean)
        if big_dir_norm < 1e-10:
            continue
        big_dir = big_dir_mean / big_dir_norm
        
        # big_dir在前50个PC上的系数
        coeffs = Vt[:min(50, Vt.shape[0])] @ big_dir
        
        # 系数分布
        coeff_sq = coeffs ** 2
        total_energy = np.sum(coeff_sq)
        
        # 有效维度(基于系数分布)
        eff_dim = 1.0 / np.sum((coeff_sq / total_energy) ** 2) if total_energy > 0 else 0
        
        # Top-K累计能量
        sorted_energy = np.sort(coeff_sq)[::-1]
        cum_energy = np.cumsum(sorted_energy) / total_energy
        
        results[f"coeff_dist_L{li}"] = {
            "layer": li,
            "eff_dim": round(float(eff_dim), 2),
            "top1_ratio": round(float(sorted_energy[0] / total_energy), 4) if len(sorted_energy) > 0 else 0,
            "top3_ratio": round(float(cum_energy[2]), 4) if len(cum_energy) >= 3 else 0,
            "top5_ratio": round(float(cum_energy[4]), 4) if len(cum_energy) >= 5 else 0,
            "top10_ratio": round(float(cum_energy[9]), 4) if len(cum_energy) >= 10 else 0,
            "top20_ratio": round(float(cum_energy[19]), 4) if len(cum_energy) >= 20 else 0,
            "top5_abs_coeffs": [round(float(abs(c)), 4) for c in coeffs[:5]],
        }
        
        log(f"  L{li}: eff_dim={eff_dim:.2f}, top1={sorted_energy[0]/total_energy:.4f}, "
            f"top3={cum_energy[2]:.4f}, top5={cum_energy[4]:.4f}, top10={cum_energy[9]:.4f}")
    
    # ===== Step 4: Perturb测试 - 多PC组合 vs 单一PC =====
    log("\n--- Step 4: Perturb测试 ---")
    
    li_last = test_layers[-1]
    X = np.array(compare_resids[li_last])
    if X.shape[0] >= 3:
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        size_only = compare_resids[li_last][:len(SIZE_PAIRS)]
        big_dir_mean = np.mean(size_only, axis=0) - mean
        big_dir_norm = np.linalg.norm(big_dir_mean)
        if big_dir_norm > 1e-10:
            big_dir = big_dir_mean / big_dir_norm
            
            # 找bigger/smaller tokens
            bigger_tokens = ["bigger", "larger", "greater", "huge", "enormous"]
            smaller_tokens = ["smaller", "tiny", "little", "miniature", "minute"]
            
            bigger_ids = [tokenizer.encode(t, add_special_tokens=False)[0] 
                         for t in bigger_tokens if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
            smaller_ids = [tokenizer.encode(t, add_special_tokens=False)[0] 
                          for t in smaller_tokens if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
            
            test_prompt = "The elephant is bigger than the mouse. The whale is"
            toks = tokenizer(test_prompt, return_tensors="pt").to(device)
            
            # Baseline
            with torch.no_grad():
                base_logits = model(**toks).logits[0, -1, :].detach().float().cpu().numpy()
            
            base_bigger = np.mean([base_logits[tid] for tid in bigger_ids if tid < len(base_logits)])
            base_smaller = np.mean([base_logits[tid] for tid in smaller_ids if tid < len(base_logits)]) if smaller_ids else 0
            base_diff = base_bigger - base_smaller
            
            # 测试3种perturb方向
            alpha = 5.0
            perturb_dirs = {
                "pc0_only": Vt[0],
                "top3_combo": Vt[:3].T @ (Vt[:3] @ big_dir),  # 前3个PC的组合
                "full_big_dir": big_dir,  # 完整"大"方向
            }
            
            for dir_name, perturb_dir in perturb_dirs.items():
                pnorm = np.linalg.norm(perturb_dir)
                if pnorm < 1e-10:
                    continue
                perturb_dir = perturb_dir / pnorm
                
                perturbed = [False]
                def perturb_hook(m, inp, out):
                    if perturbed[0]:
                        return
                    perturbed[0] = True
                    o = out[0] if isinstance(out, tuple) else out
                    o_new = o.clone()
                    dir_tensor = torch.tensor(perturb_dir, dtype=o.dtype, device=device)
                    o_new[0, -1, :] += alpha * dir_tensor
                    if isinstance(out, tuple):
                        return (o_new,) + out[1:]
                    return o_new
                
                hook = layers[li_last].register_forward_hook(perturb_hook)
                with torch.no_grad():
                    perturb_logits = model(**tokenizer(test_prompt, return_tensors="pt").to(device)).logits[0, -1, :].detach().float().cpu().numpy()
                hook.remove()
                
                perturb_bigger = np.mean([perturb_logits[tid] for tid in bigger_ids if tid < len(perturb_logits)])
                perturb_smaller = np.mean([perturb_logits[tid] for tid in smaller_ids if tid < len(perturb_logits)]) if smaller_ids else 0
                perturb_diff = perturb_bigger - perturb_smaller
                
                results[f"perturb_{dir_name}_L{li_last}"] = {
                    "direction": dir_name,
                    "alpha": alpha,
                    "base_diff": round(float(base_diff), 3),
                    "perturb_diff": round(float(perturb_diff), 3),
                    "shift": round(float(perturb_diff - base_diff), 3),
                    "base_bigger": round(float(base_bigger), 3),
                    "perturb_bigger": round(float(perturb_bigger), 3),
                    "bigger_shift": round(float(perturb_bigger - base_bigger), 3),
                    "base_smaller": round(float(base_smaller), 3),
                    "perturb_smaller": round(float(perturb_smaller), 3),
                    "smaller_shift": round(float(perturb_smaller - base_smaller), 3),
                }
                
                log(f"  {dir_name}: base_diff={base_diff:.3f} → perturb_diff={perturb_diff:.3f} (shift={perturb_diff-base_diff:.3f})")
            
            # 保存baseline
            results["perturb_baseline"] = {
                "base_bigger": round(float(base_bigger), 3),
                "base_smaller": round(float(base_smaller), 3),
                "base_diff": round(float(base_diff), 3),
            }
    
    # ===== Step 5: 不同比较维度的方向对比 =====
    log("\n--- Step 5: 不同比较维度的方向对比 ---")
    
    # 分别计算size/weight/speed的"大"方向, 看它们在PC空间中的位置
    for li in [test_layers[0], test_layers[len(test_layers)//2], test_layers[-1]]:
        X = np.array(compare_resids[li])
        if X.shape[0] < 3:
            continue
        
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        dim_dirs = {}
        for dim_name, n_pairs, pairs in [
            ("size", len(SIZE_PAIRS), SIZE_PAIRS),
            ("weight", len(WEIGHT_PAIRS), WEIGHT_PAIRS),
            ("speed", len(SPEED_PAIRS), SPEED_PAIRS),
        ]:
            dim_resids = compare_resids[li][:n_pairs] if dim_name == "size" else \
                         compare_resids[li][len(SIZE_PAIRS):len(SIZE_PAIRS)+n_pairs] if dim_name == "weight" else \
                         compare_resids[li][len(SIZE_PAIRS)+len(WEIGHT_PAIRS):]
            
            if len(dim_resids) < 3:
                continue
            
            dir_vec = np.mean(dim_resids, axis=0) - mean
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-10:
                continue
            dim_dirs[dim_name] = dir_vec / norm
        
        # 各维度方向的PC系数
        for dim_name, dir_vec in dim_dirs.items():
            coeffs = Vt[:min(20, Vt.shape[0])] @ dir_vec
            results[f"dim_pc_coeffs_{dim_name}_L{li}"] = {
                "layer": li,
                "dimension": dim_name,
                "top5_abs_coeffs": [round(float(abs(c)), 4) for c in coeffs[:5]],
                "top5_signed_coeffs": [round(float(c), 4) for c in coeffs[:5]],
            }
        
        # 维度间的余弦相似度
        dim_names = list(dim_dirs.keys())
        for i, d1 in enumerate(dim_names):
            for j, d2 in enumerate(dim_names):
                if i < j:
                    cos_val = float(np.dot(dim_dirs[d1], dim_dirs[d2]))
                    results[f"dim_cos_{d1}_{d2}_L{li}"] = {
                        "layer": li,
                        "dim1": d1,
                        "dim2": d2,
                        "cos": round(cos_val, 4),
                    }
                    log(f"  L{li}: cos({d1}, {d2})={cos_val:.4f}")
    
    # 保存结果
    out_path = TEMP / f"ccxxiii_multi_pc_combo_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "d_model": d_model, "n_layers": n_layers, "results": results}, f, ensure_ascii=False, indent=2)
    log(f"\n结果保存到: {out_path}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    if args.model == "qwen3":
        with open(LOG, "w", encoding="utf-8") as f:
            f.write("")
    
    run(args.model)
