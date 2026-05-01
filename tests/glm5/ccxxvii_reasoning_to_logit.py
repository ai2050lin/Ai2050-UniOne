"""
CCXXVII(327): 推理方向→logit的因果路径
======================================================================
CCXXIII发现: perturb"大"方向效果极弱(shift<0.05)。但推理方向确实存在(eff_dim≈2-4)。
关键问题: 推理方向如何影响最终输出? 通过什么路径?

实验设计:
  1. 找到"bigger"vs"smaller"的logit方向(W_U行向量差)
  2. 测量推理方向与logit方向的余弦
  3. 追踪推理方向在每层到W_U行空间的投影比
  4. Perturb推理方向, 看哪些token的logit变化最大(top-k分析)
  5. 分解: 推理方向通过注意力路径vs MLP路径的影响
  6. 直接在最终残差+L0残差上perturb, 看logit变化

用法:
  python ccxxvii_reasoning_to_logit.py --model qwen3
  python ccxxvii_reasoning_to_logit.py --model glm4
  python ccxxvii_reasoning_to_logit.py --model deepseek7b
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
LOG = TEMP / "ccxxvii_reasoning_to_logit_log.txt"

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

# 比较相关token
BIGGER_TOKENS = ["bigger", "larger", "greater", "huge", "enormous", "giant", "massive", "immense"]
SMALLER_TOKENS = ["smaller", "tiny", "little", "miniature", "minute", "diminutive", "petite", "micro"]
HEAVIER_TOKENS = ["heavier", "denser", "weightier", "massive", "ponderous"]
LIGHTER_TOKENS = ["lighter", "feathery", "airy", "weightless", "buoyant"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_token_ids(tokenizer, words, vocab_size):
    """获取token ID列表"""
    ids = []
    for w in words:
        tids = tokenizer.encode(w, add_special_tokens=False)
        if len(tids) == 1 and tids[0] < vocab_size:
            ids.append(tids[0])
        elif len(tids) > 1:
            # 多token词, 跳过
            pass
    return ids


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    W_U = get_W_U(model)  # [vocab_size, d_model]
    vocab_size = W_U.shape[0]
    
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    log(f"\n{'='*70}\nCCXXVII(327): 推理方向→logit的因果路径 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}, vocab_size={vocab_size}")
    log(f"{'='*70}")
    
    results = {}
    
    # ===== Step 1: Logit方向分析 =====
    log("\n--- Step 1: Logit方向分析 ---")
    
    # "bigger" vs "smaller"的logit方向
    bigger_ids = get_token_ids(tokenizer, BIGGER_TOKENS, vocab_size)
    smaller_ids = get_token_ids(tokenizer, SMALLER_TOKENS, vocab_size)
    heavier_ids = get_token_ids(tokenizer, HEAVIER_TOKENS, vocab_size)
    lighter_ids = get_token_ids(tokenizer, LIGHTER_TOKENS, vocab_size)
    
    log(f"  bigger_ids: {bigger_ids} (from {BIGGER_TOKENS})")
    log(f"  smaller_ids: {smaller_ids} (from {SMALLER_TOKENS})")
    log(f"  heavier_ids: {heavier_ids} (from {HEAVIER_TOKENS})")
    log(f"  lighter_ids: {lighter_ids} (from {LIGHTER_TOKENS})")
    
    # W_U行向量均值差作为logit方向
    if bigger_ids and smaller_ids:
        bigger_emb = W_U[bigger_ids].mean(axis=0)
        smaller_emb = W_U[smaller_ids].mean(axis=0)
        logit_dir_size = bigger_emb - smaller_emb
        logit_dir_size_norm = np.linalg.norm(logit_dir_size)
        if logit_dir_size_norm > 1e-10:
            logit_dir_size = logit_dir_size / logit_dir_size_norm
    else:
        logit_dir_size = np.zeros(d_model)
    
    if heavier_ids and lighter_ids:
        heavier_emb = W_U[heavier_ids].mean(axis=0)
        lighter_emb = W_U[lighter_ids].mean(axis=0)
        logit_dir_weight = heavier_emb - lighter_emb
        logit_dir_weight_norm = np.linalg.norm(logit_dir_weight)
        if logit_dir_weight_norm > 1e-10:
            logit_dir_weight = logit_dir_weight / logit_dir_weight_norm
    else:
        logit_dir_weight = np.zeros(d_model)
    
    # logit方向之间的余弦
    cos_logit = float(np.dot(logit_dir_size, logit_dir_weight))
    log(f"  cos(logit_size, logit_weight) = {cos_logit:.4f}")
    results["logit_direction_alignment"] = {
        "cos_size_weight": round(cos_logit, 4),
        "n_bigger_tokens": len(bigger_ids),
        "n_smaller_tokens": len(smaller_ids),
        "n_heavier_tokens": len(heavier_ids),
        "n_lighter_tokens": len(lighter_ids),
    }
    
    # ===== Step 2: 收集推理残差, 计算推理方向 =====
    log("\n--- Step 2: 收集推理残差 ---")
    
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
    
    # ===== Step 3: 推理方向与logit方向的对齐 =====
    log("\n--- Step 3: 推理方向与logit方向的对齐 ---")
    
    for li in test_layers:
        X = np.array(compare_resids[li])
        if X.shape[0] < 3:
            continue
        
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # size推理方向
        size_only = compare_resids[li][:len(SIZE_PAIRS)]
        if len(size_only) >= 3:
            size_dir = np.mean(size_only, axis=0) - mean
            sn = np.linalg.norm(size_dir)
            if sn > 1e-10:
                size_dir = size_dir / sn
            else:
                size_dir = np.zeros(d_model)
        else:
            size_dir = np.zeros(d_model)
        
        # weight推理方向
        weight_only = compare_resids[li][len(SIZE_PAIRS):len(SIZE_PAIRS)+len(WEIGHT_PAIRS)]
        if len(weight_only) >= 3:
            weight_dir = np.mean(weight_only, axis=0) - mean
            wn = np.linalg.norm(weight_dir)
            if wn > 1e-10:
                weight_dir = weight_dir / wn
            else:
                weight_dir = np.zeros(d_model)
        else:
            weight_dir = np.zeros(d_model)
        
        # 推理方向 vs logit方向的余弦
        cos_size_logit = float(np.dot(size_dir, logit_dir_size)) if np.linalg.norm(size_dir) > 1e-10 else 0
        cos_weight_logit = float(np.dot(weight_dir, logit_dir_weight)) if np.linalg.norm(weight_dir) > 1e-10 else 0
        cos_size_logit_weight = float(np.dot(size_dir, logit_dir_weight)) if np.linalg.norm(size_dir) > 1e-10 else 0
        cos_weight_logit_size = float(np.dot(weight_dir, logit_dir_size)) if np.linalg.norm(weight_dir) > 1e-10 else 0
        
        # 推理方向在W_U行空间的投影比
        # W_U行空间 = W_U的行张成的空间 (在R^d_model中)
        # 用SVD: W_U^T = U S Vt, W_U行空间基 = U的列
        from scipy.sparse.linalg import svds
        k_svd = min(100, min(d_model, vocab_size) - 2)
        try:
            U_wut, s_wut, _ = svds(W_U.T.astype(np.float32), k=k_svd)
            U_wut = np.asarray(U_wut, dtype=np.float64)
        except:
            U_wut = None
        
        proj_info = {}
        for dir_name, dir_vec in [("size", size_dir), ("weight", weight_dir)]:
            dn = np.linalg.norm(dir_vec)
            if dn < 1e-10:
                proj_info[dir_name] = {"norm": 0, "wu_proj_ratio": 0}
                continue
            
            if U_wut is not None:
                proj_coeffs = U_wut.T @ dir_vec
                proj_energy = np.sum(proj_coeffs ** 2)
                wu_proj_ratio = min(proj_energy / max(dn ** 2, 1e-20), 1.0)
            else:
                wu_proj_ratio = 0
            
            proj_info[dir_name] = {
                "norm": round(float(dn), 4),
                "wu_proj_ratio": round(float(wu_proj_ratio), 4),
            }
        
        key = f"reasoning_logit_align_L{li}"
        results[key] = {
            "layer": li,
            "cos_size_to_logit_size": round(cos_size_logit, 4),
            "cos_weight_to_logit_weight": round(cos_weight_logit, 4),
            "cos_size_to_logit_weight": round(cos_size_logit_weight, 4),
            "cos_weight_to_logit_size": round(cos_weight_logit_size, 4),
            "size_wu_proj_ratio": proj_info.get("size", {}).get("wu_proj_ratio", 0),
            "weight_wu_proj_ratio": proj_info.get("weight", {}).get("wu_proj_ratio", 0),
        }
        
        log(f"  L{li}: cos(size,logit_size)={cos_size_logit:.4f}, cos(weight,logit_weight)={cos_weight_logit:.4f}")
        log(f"        cos(size,logit_weight)={cos_size_logit_weight:.4f}, cos(weight,logit_size)={cos_weight_logit_size:.4f}")
        log(f"        wu_proj: size={proj_info.get('size',{}).get('wu_proj_ratio',0):.4f}, weight={proj_info.get('weight',{}).get('wu_proj_ratio',0):.4f}")
    
    # ===== Step 4: Perturb推理方向 - Top-K logit变化分析 =====
    log("\n--- Step 4: Perturb推理方向 - Top-K logit变化 ---")
    
    # 在最后一层perturb, 看logit变化
    li_last = n_layers - 1
    
    X = np.array(compare_resids[li_last])
    if X.shape[0] >= 3:
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        size_only = compare_resids[li_last][:len(SIZE_PAIRS)]
        size_dir = np.mean(size_only, axis=0) - mean
        sn = np.linalg.norm(size_dir)
        if sn > 1e-10:
            size_dir = size_dir / sn
        else:
            size_dir = None
        
        if size_dir is not None:
            test_prompt = "The elephant is bigger than the mouse"
            toks = tokenizer(test_prompt, return_tensors="pt").to(device)
            
            # Baseline logits
            with torch.no_grad():
                base_logits = model(**toks).logits[0, -1, :].detach().float().cpu().numpy()
            
            # Perturb: 在最后一层残差加方向
            alpha = 5.0
            perturbed = [False]
            perturb_dir_tensor = torch.tensor(size_dir, dtype=torch.float32, device=device)
            
            def perturb_hook(m, inp, out):
                if perturbed[0]:
                    return
                perturbed[0] = True
                o = out[0] if isinstance(out, tuple) else out
                o_new = o.clone()
                o_new[0, -1, :] += alpha * perturb_dir_tensor.to(o.dtype)
                if isinstance(out, tuple):
                    return (o_new,) + out[1:]
                return o_new
            
            hook = layers[li_last].register_forward_hook(perturb_hook)
            with torch.no_grad():
                perturb_logits = model(**tokenizer(test_prompt, return_tensors="pt").to(device)).logits[0, -1, :].detach().float().cpu().numpy()
            hook.remove()
            
            # Logit变化
            delta_logits = perturb_logits - base_logits
            
            # Top-K变化的token
            top_k_incr = np.argsort(delta_logits)[-20:][::-1]
            top_k_decr = np.argsort(delta_logits)[:20]
            
            top_incr_tokens = [(tokenizer.decode([tid]), round(float(delta_logits[tid]), 4)) for tid in top_k_incr if tid < vocab_size]
            top_decr_tokens = [(tokenizer.decode([tid]), round(float(delta_logits[tid]), 4)) for tid in top_k_decr if tid < vocab_size]
            
            # 关键token的logit变化
            key_tokens = {}
            for tok_name, tok_ids in [
                ("bigger", bigger_ids), ("smaller", smaller_ids),
                ("heavier", heavier_ids), ("lighter", lighter_ids),
            ]:
                if tok_ids:
                    mean_shift = np.mean([delta_logits[tid] for tid in tok_ids if tid < len(delta_logits)])
                    key_tokens[tok_name] = round(float(mean_shift), 4)
            
            results["perturb_topk_analysis"] = {
                "layer": li_last,
                "alpha": alpha,
                "direction": "size_reasoning",
                "key_token_shifts": key_tokens,
                "top10_increase": top_incr_tokens[:10],
                "top10_decrease": top_decr_tokens[:10],
                "total_logit_norm": round(float(np.linalg.norm(delta_logits)), 4),
                "max_incr": round(float(delta_logits.max()), 4),
                "max_decr": round(float(delta_logits.min()), 4),
            }
            
            log(f"  Perturb size_dir at L{li_last}, alpha={alpha}:")
            log(f"    key shifts: {key_tokens}")
            log(f"    total_norm={np.linalg.norm(delta_logits):.4f}, max_incr={delta_logits.max():.4f}, max_decr={delta_logits.min():.4f}")
    
    # ===== Step 5: 推理方向的层间传播 =====
    log("\n--- Step 5: 推理方向的层间传播 ---")
    
    # 在L0注入size方向, 追踪每层的残差变化
    if size_dir is not None:
        test_prompt_inject = "The elephant is bigger than the mouse"
        toks = tokenizer(test_prompt_inject, return_tensors="pt").to(device)
        seq_len = toks.input_ids.shape[1]
        last_pos = seq_len - 1
        
        # 在L0的残差注入size方向
        alpha_inject = 3.0
        inject_layer = 0
        
        # 收集注入后各层输出
        captured_inject = {}
        def mk_inject_hook(k):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured_inject[k] = o[0, last_pos, :].detach().float().cpu().numpy()
            return hook
        
        # 先收集baseline
        captured_base = {}
        def mk_base_hook(k):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured_base[k] = o[0, last_pos, :].detach().float().cpu().numpy()
            return hook
        
        hooks_base = [layers[li].register_forward_hook(mk_base_hook(f"L{li}")) for li in test_layers]
        with torch.no_grad():
            _ = model(**toks)
        for h in hooks_base:
            h.remove()
        
        # 注入perturb
        inject_done = [False]
        def inject_hook(m, inp, out):
            if inject_done[0]:
                return
            inject_done[0] = True
            o = out[0] if isinstance(out, tuple) else out
            o_new = o.clone()
            dir_tensor = torch.tensor(size_dir, dtype=o.dtype, device=device)
            o_new[0, last_pos, :] += alpha_inject * dir_tensor
            if isinstance(out, tuple):
                return (o_new,) + out[1:]
            return o_new
        
        hooks_inject = [layers[inject_layer].register_forward_hook(inject_hook)]
        hooks_read = [layers[li].register_forward_hook(mk_inject_hook(f"L{li}")) for li in test_layers if li > inject_layer]
        
        with torch.no_grad():
            _ = model(**tokenizer(test_prompt_inject, return_tensors="pt").to(device))
        
        for h in hooks_inject + hooks_read:
            h.remove()
        
        # 计算注入后的残差变化在size方向上的投影
        propagation = {}
        for li in test_layers:
            if f"L{li}" in captured_base and f"L{li}" in captured_inject:
                delta = captured_inject[f"L{li}"] - captured_base[f"L{li}"]
                delta_norm = np.linalg.norm(delta)
                
                # 在size方向上的投影
                if delta_norm > 1e-10:
                    cos_to_size = float(np.dot(delta / delta_norm, size_dir))
                    proj_on_size = float(np.dot(delta, size_dir))
                else:
                    cos_to_size = 0
                    proj_on_size = 0
                
                propagation[f"L{li}"] = {
                    "delta_norm": round(float(delta_norm), 4),
                    "cos_to_size_dir": round(cos_to_size, 4),
                    "proj_on_size_dir": round(proj_on_size, 4),
                }
                
                log(f"  L{li}: delta_norm={delta_norm:.4f}, cos_to_size={cos_to_size:.4f}, proj={proj_on_size:.4f}")
        
        results["reasoning_propagation"] = {
            "inject_layer": inject_layer,
            "alpha": alpha_inject,
            "propagation": propagation,
        }
    
    # ===== Step 6: 推理方向vs PC0/PC1到W_U行空间的投影比 =====
    log("\n--- Step 6: PC方向到W_U行空间的投影比 ---")
    
    if U_wut is not None:
        for li in test_layers:
            X = np.array(compare_resids[li])
            if X.shape[0] < 3:
                continue
            
            mean = X.mean(axis=0)
            X_centered = X - mean
            U_svd, S_svd, Vt_svd = np.linalg.svd(X_centered, full_matrices=False)
            
            pc_proj = {}
            for pc_idx in range(min(5, Vt_svd.shape[0])):
                pc_dir = Vt_svd[pc_idx]
                proj_coeffs = U_wut.T @ pc_dir
                proj_energy = np.sum(proj_coeffs ** 2)
                pc_norm = np.linalg.norm(pc_dir)
                wu_ratio = min(proj_energy / max(pc_norm ** 2, 1e-20), 1.0)
                
                pc_proj[f"PC{pc_idx}"] = round(float(wu_ratio), 4)
            
            results[f"pc_to_wu_proj_L{li}"] = {
                "layer": li,
                "pc_wu_ratios": pc_proj,
            }
            
            log(f"  L{li}: PC→W_U投影比 = {pc_proj}")
    
    # 保存结果
    out_path = TEMP / f"ccxxvii_reasoning_to_logit_{model_name}.json"
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
