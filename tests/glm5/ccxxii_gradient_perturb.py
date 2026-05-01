"""
CCXXII(322): 梯度引导perturb + 蒸馏展开因果分析
======================================================================
CCXVIII发现head perturb全0%, 可能是d_head太小或方向不准。
CCXVII发现DS7B L27展开到14维, 但展开是否恢复信息还是引入噪声?

实验1: 梯度引导perturb
  - 用logit梯度计算最优perturb方向(而非均值差)
  - 在residual空间perturb, 但用梯度引导选择方向
  - 对比梯度引导 vs SVD方向 perturb

实验2: 蒸馏展开因果分析
  - perturb DS7B L26(展开前) vs L27(展开后) 的方向
  - 测试: perturb L27的"展开维度"(第4-5维)是否破坏语义
  - 测试: perturb L27的"核心维度"(第1-3维)是否与L26一致

设计:
  - 目标: 让输出偏向特定habitat(land/ocean/sky)
  - 方法1: 梯度引导 - 计算d(logit_habitat)/d(residual), 沿梯度方向perturb
  - 方法2: SVD perturb - 沿habitat SVD主方向perturb
  - 方法3: 展开维度perturb - 只perturb DS7B L27的第4-5维

用法:
  python ccxxii_gradient_perturb.py --model qwen3
  python ccxxii_gradient_perturb.py --model glm4
  python ccxxii_gradient_perturb.py --model deepseek7b
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
LOG = TEMP / "ccxxii_gradient_perturb_log.txt"

# 词汇和属性
WORDS_BY_HABITAT = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit", "fox", "deer"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle", "crab", "seal", "squid", "lobster"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow", "falcon", "pigeon", "robin"],
}

TEMPLATE = "The {} lives in the"

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
    
    log(f"\n{'='*70}\nCCXXII(322): 梯度引导perturb + 蒸馏展开因果分析 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    W_U = get_W_U(model)
    
    # 找habitat token IDs
    habitat_token_ids = {}
    for hab, tokens in HABITAT_TOKENS.items():
        ids = []
        for t in tokens:
            tok_ids = tokenizer.encode(" " + t, add_special_tokens=False)
            if len(tok_ids) > 0:
                ids.append(tok_ids[0])
        habitat_token_ids[hab] = ids
    
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    # ===== Part 1: 梯度引导perturb =====
    log("\n--- Part 1: 梯度引导perturb ---")
    
    # 1a: 收集habitat残差
    log("\n  1a: 收集habitat残差...")
    habitat_resids = {li: {"land": [], "ocean": [], "sky": []} for li in test_layers}
    
    for hab, words in WORDS_BY_HABITAT.items():
        for word in words:
            prompt = TEMPLATE.format(word)
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
                    habitat_resids[li][hab].append(captured[f"L{li}"])
    
    # 1b: 计算habitat SVD方向
    log("\n  1b: 计算habitat SVD方向...")
    
    for li in [test_layers[0], test_layers[-1]]:
        all_vecs = []
        labels = []
        for ci, hab in enumerate(["land", "ocean", "sky"]):
            all_vecs.extend(habitat_resids[li][hab])
            labels.extend([ci] * len(habitat_resids[li][hab]))
        
        if len(all_vecs) < 6:
            continue
        
        X = np.array(all_vecs)
        mean = X.mean(axis=0)
        X_centered = X - mean
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 计算每个habitat的方向(与SVD主轴对齐)
        hab_dirs = {}
        for hab in ["land", "ocean", "sky"]:
            hab_vecs = np.array(habitat_resids[li][hab])
            hab_mean = hab_vecs.mean(axis=0) - mean
            norm = np.linalg.norm(hab_mean)
            if norm > 1e-10:
                hab_dirs[hab] = hab_mean / norm
        
        # 1c: 梯度引导perturb
        log(f"\n  1c: 梯度引导perturb (L{li})...")
        
        # 用一个ocean词做测试
        test_word = "whale"
        prompt = TEMPLATE.format(test_word)
        
        for target_hab in ["land", "ocean", "sky"]:
            # 方法1: W_U梯度方向 (线性近似: d(logit_hab)/d(residual) ≈ W_U[target_ids].mean(axis=0))
            # 不需要backward, 节省GPU内存
            target_ids = habitat_token_ids.get(target_hab, [])
            if len(target_ids) == 0:
                continue
            
            # W_U的target_ids行均值 = logit关于residual的梯度方向
            grad_dir_np = W_U[target_ids].mean(axis=0)
            grad_norm = np.linalg.norm(grad_dir_np)
            if grad_norm < 1e-10:
                log(f"    {target_hab}: W_U梯度≈0, 跳过")
                continue
            grad_dir = grad_dir_np / grad_norm
            
            # 方法2: SVD方向
            if target_hab in hab_dirs:
                svd_dir = hab_dirs[target_hab]
            else:
                svd_dir = None
            
            # 测试两种perturb方法
            for method_name, perturb_dir in [("gradient", grad_dir), ("svd", svd_dir)]:
                if perturb_dir is None:
                    continue
                
                for alpha in [2.0, 5.0, 10.0]:
                    # Perturb at target layer
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
                    
                    hook = layers[li].register_forward_hook(perturb_hook)
                    with torch.no_grad():
                        perturb_logits = model(**tokenizer(prompt, return_tensors="pt").to(device)).logits[0, -1, :].detach().float().cpu().numpy()
                    hook.remove()
                    
                    # 计算各habitat的logit shift
                    shifts = {}
                    for hab in ["land", "ocean", "sky"]:
                        base_logits_hab = []
                        perturb_logits_hab = []
                        for tid in habitat_token_ids.get(hab, []):
                            if tid < len(perturb_logits):
                                perturb_logits_hab.append(float(perturb_logits[tid]))
                        
                        # Baseline
                        with torch.no_grad():
                            base_logits = model(**tokenizer(prompt, return_tensors="pt").to(device)).logits[0, -1, :].detach().float().cpu().numpy()
                        
                        for tid in habitat_token_ids.get(hab, []):
                            if tid < len(base_logits):
                                base_logits_hab.append(float(base_logits[tid]))
                        
                        if base_logits_hab and perturb_logits_hab:
                            shifts[hab] = round(float(np.mean(perturb_logits_hab) - np.mean(base_logits_hab)), 4)
                    
                    # 判断成功: target_hab shift > 其他hab shift
                    target_shift = shifts.get(target_hab, 0)
                    other_shifts = [shifts.get(h, 0) for h in ["land", "ocean", "sky"] if h != target_hab]
                    max_other = max(other_shifts) if other_shifts else 0
                    success = target_shift > max_other and target_shift > 0
                    
                    key = f"perturb_{method_name}_{target_hab}_L{li}_a{alpha}"
                    results[key] = {
                        "method": method_name,
                        "target_habitat": target_hab,
                        "layer": li,
                        "alpha": alpha,
                        "shifts": shifts,
                        "target_shift": target_shift,
                        "max_other_shift": max_other,
                        "success": success,
                    }
                    
                    status = "OK" if success else "FAIL"
                    log(f"    {status} {method_name} {target_hab} L{li} a={alpha}: shifts={shifts}")
    
    # ===== Part 2: 蒸馏展开因果分析 (DS7B特有) =====
    log("\n--- Part 2: 蒸馏展开因果分析 ---")
    
    if model_name == "deepseek7b":
        # 分析L26(展开前, 4维) vs L27(展开后, 5维)
        for li in [26, 27]:
            all_vecs = []
            labels = []
            for ci, hab in enumerate(["land", "ocean", "sky"]):
                all_vecs.extend(habitat_resids[li][hab])
                labels.extend([ci] * len(habitat_resids[li][hab]))
            
            if len(all_vecs) < 6:
                continue
            
            X = np.array(all_vecs)
            mean = X.mean(axis=0)
            X_centered = X - mean
            
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            log(f"  DS7B L{li}: top5奇异值 = {[round(float(s), 2) for s in S[:5]]}")
            
            # 测试perturb核心维度(1-3) vs 展开维度(4-5)
            test_word = "whale"
            prompt = TEMPLATE.format(test_word)
            
            for dim_range, dim_name in [((0, 3), "core_1to3"), ((3, 5), "expand_4to5")]:
                start, end = dim_range
                
                # 构建perturb方向: 只用这些PC维度
                # 取ocean方向在这些PC上的投影
                hab_vecs = np.array(habitat_resids[li]["ocean"])
                hab_mean = hab_vecs.mean(axis=0) - mean
                
                # 投影到指定PC
                proj_coeffs = Vt[start:end] @ hab_mean
                perturb_dir = Vt[start:end].T @ proj_coeffs
                norm = np.linalg.norm(perturb_dir)
                if norm < 1e-10:
                    continue
                perturb_dir = perturb_dir / norm
                
                for alpha in [2.0, 5.0, 10.0]:
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
                    
                    hook = layers[li].register_forward_hook(perturb_hook)
                    with torch.no_grad():
                        perturb_logits = model(**tokenizer(prompt, return_tensors="pt").to(device)).logits[0, -1, :].detach().float().cpu().numpy()
                    hook.remove()
                    
                    # Baseline
                    with torch.no_grad():
                        base_logits = model(**tokenizer(prompt, return_tensors="pt").to(device)).logits[0, -1, :].detach().float().cpu().numpy()
                    
                    # 计算ocean shift
                    ocean_ids = habitat_token_ids.get("ocean", [])
                    base_ocean = np.mean([float(base_logits[tid]) for tid in ocean_ids if tid < len(base_logits)])
                    perturb_ocean = np.mean([float(perturb_logits[tid]) for tid in ocean_ids if tid < len(perturb_logits)])
                    
                    key = f"ds7b_expand_{dim_name}_L{li}_a{alpha}"
                    results[key] = {
                        "dim_range": dim_name,
                        "layer": li,
                        "alpha": alpha,
                        "ocean_shift": round(float(perturb_ocean - base_ocean), 4),
                    }
                    log(f"  DS7B {dim_name} L{li} α={alpha}: ocean_shift={perturb_ocean - base_ocean:.4f}")
        
        # L26 vs L27的PC一致性
        log("\n  L26 vs L27 PC一致性:")
        for li1, li2 in [(26, 27)]:
            X1 = np.array([v for hab in ["land", "ocean", "sky"] for v in habitat_resids[li1][hab]])
            X2 = np.array([v for hab in ["land", "ocean", "sky"] for v in habitat_resids[li2][hab]])
            
            if len(X1) < 6 or len(X2) < 6:
                continue
            
            m1 = X1.mean(axis=0)
            m2 = X2.mean(axis=0)
            
            U1, S1, Vt1 = np.linalg.svd(X1 - m1, full_matrices=False)
            U2, S2, Vt2 = np.linalg.svd(X2 - m2, full_matrices=False)
            
            # PC对齐
            pc_alignment = []
            for k in range(min(5, Vt1.shape[0], Vt2.shape[0])):
                cos_val = float(np.abs(np.dot(Vt1[k], Vt2[k])))
                pc_alignment.append(round(cos_val, 4))
            
            log(f"  L{li1} vs L{li2} PC对齐: {pc_alignment}")
            results["ds7b_pc_alignment"] = {
                "layer1": li1,
                "layer2": li2,
                "pc_alignment": pc_alignment,
            }
    else:
        log(f"  跳过蒸馏分析(非DS7B)")
    
    # ===== 保存结果 =====
    out_path = TEMP / f"ccxxii_gradient_perturb_{model_name}.json"
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
