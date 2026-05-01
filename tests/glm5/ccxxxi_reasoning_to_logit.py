"""
CCXXXI(331): 推理→Logit因果路径分析 + size vs weight方向分解
======================================================================
核心问题1: 为什么"大"方向perturb效果极弱(<0.05), 但W_U梯度perturb 83%成功?
核心问题2: size vs weight方向相反(cos=-0.75)的原因是什么?

假设:
  - "bigger"方向在残差空间中, 但经过LayerNorm+后续层后, 被旋转/缩放
  - W_U梯度方向 = 最终残差空间中的最优logit方向
  - "bigger"方向 ≠ W_U梯度方向, 因为中间有非线性变换

实验设计:
  Part 1: 逐层perturb效果测试 — 在每层分别用3种方向perturb, 测量logit shift
    方向1: "bigger"语义方向 (比较残差均值差)
    方向2: W_U梯度方向 (d(logit)/d(residual)线性近似)
    方向3: bigger方向经LayerNorm变换后的方向

  Part 2: LayerNorm旋转效应
    - 计算bigger方向在LayerNorm前后的方向变化
    - 测量: bigger方向经过LayerNorm后与W_U行空间的对齐度变化

  Part 3: size vs weight方向分解
    - 收集: "A is big" vs "A is heavy" (纯属性)
    - 收集: "A is bigger than B" vs "A is heavier than B" (比较)
    - 分解: 比较方向 = 完整比较残差 - 纯属性残差
    - 测试: 分离后的"比较操作"方向在size/weight间是否一致

  Part 4: 信息流追踪 — Embedding→L0→L1
    - 追踪语义方差在各层的传播
    - 计算相邻层的PC对齐

用法:
  python ccxxxi_reasoning_to_logit.py --model qwen3
  python ccxxxi_reasoning_to_logit.py --model glm4
  python ccxxxi_reasoning_to_logit.py --model deepseek7b
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
LOG = TEMP / "ccxxxi_reasoning_to_logit_log.txt"

# ===== 词汇定义 =====
# 比较词对
SIZE_COMPARE = [
    ("elephant", "mouse"), ("whale", "fish"), ("horse", "cat"),
    ("lion", "rabbit"), ("bear", "fox"), ("cow", "chicken"),
    ("shark", "crab"), ("tiger", "rat"), ("eagle", "sparrow"),
    ("mountain", "hill"), ("tree", "bush"), ("bus", "car"),
]

WEIGHT_COMPARE = [
    ("iron", "feather"), ("rock", "leaf"), ("steel", "paper"),
    ("gold", "cotton"), ("lead", "silk"), ("stone", "grass"),
    ("concrete", "foam"), ("brick", "straw"), ("copper", "wool"),
]

SPEED_COMPARE = [
    ("cheetah", "turtle"), ("falcon", "snail"), ("horse", "slug"),
    ("rocket", "cart"), ("jet", "boat"), ("leopard", "worm"),
    ("eagle", "ant"), ("tiger", "sloth"), ("deer", "beetle"),
]

# 纯属性词汇 (不带比较)
SIZE_ATTR_WORDS = ["elephant", "mountain", "whale", "building", "giant",
                   "mouse", "ant", "grain", "atom", "dot"]
WEIGHT_ATTR_WORDS = ["iron", "lead", "stone", "steel", "concrete",
                     "feather", "air", "bubble", "smoke", "dust"]
SPEED_ATTR_WORDS = ["cheetah", "rocket", "lightning", "falcon", "bullet",
                    "snail", "turtle", "sloth", "glacier", "molasses"]

# Habitat词汇
WORDS_BY_HABITAT = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit", "fox", "deer"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle", "crab", "seal", "squid", "lobster"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow", "falcon", "pigeon", "robin"],
}

# 模板
TEMPLATE_COMPARE = "The {} is bigger than the {}"
TEMPLATE_WEIGHT_COMPARE = "The {} is heavier than the {}"
TEMPLATE_SIZE_ATTR = "The {} is very big"
TEMPLATE_WEIGHT_ATTR = "The {} is very heavy"
TEMPLATE_SPEED_ATTR = "The {} is very fast"
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


def collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers):
    """收集各层残差(最后一个token位置)"""
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
    return captured


def simulate_layernorm(vec, ln_weight, ln_bias=None, eps=1e-5):
    """模拟LayerNorm变换(对单个向量)"""
    mean = vec.mean()
    var = vec.var()
    normalized = (vec - mean) / np.sqrt(var + eps)
    if ln_weight is not None:
        normalized = normalized * ln_weight
    if ln_bias is not None:
        normalized = normalized + ln_bias
    return normalized


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n{'='*70}\nCCXXXI(331): 推理→Logit因果路径 + size/weight分解 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    W_U = get_W_U(model)  # [vocab_size, d_model]
    
    # 找habitat token IDs
    habitat_token_ids = {}
    for hab, tokens in HABITAT_TOKENS.items():
        ids = []
        for t in tokens:
            tok_ids = tokenizer.encode(" " + t, add_special_tokens=False)
            if len(tok_ids) > 0:
                ids.append(tok_ids[0])
        habitat_token_ids[hab] = ids
    
    # 找bigger/smaller token IDs
    bigger_tokens = ["bigger", "larger", "greater", "huge", "enormous", "massive", "giant", "immense"]
    smaller_tokens = ["smaller", "tiny", "little", "miniature", "minute", "tiny", "petite", "diminutive"]
    heavier_tokens = ["heavier", "weightier", "denser", "ponderous", "massive"]
    lighter_tokens = ["lighter", "feathery", "weightless", "airy", "buoyant"]
    
    bigger_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in bigger_tokens 
                 if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    smaller_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in smaller_tokens 
                   if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    heavier_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in heavier_tokens 
                   if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    lighter_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in lighter_tokens 
                   if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    # ====================================================================
    # Part 1: 逐层perturb效果测试
    # ====================================================================
    log("\n" + "="*60)
    log("Part 1: 逐层perturb效果 — bigger方向 vs W_U梯度方向")
    log("="*60)
    
    # 1a: 收集size比较残差
    log("\n  1a: 收集size比较残差...")
    size_compare_resids = {li: [] for li in test_layers}
    
    for big, small in SIZE_COMPARE:
        prompt = TEMPLATE_COMPARE.format(big, small)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                size_compare_resids[li].append(cap[f"L{li}"])
    
    # 1b: 收集weight比较残差
    log("  1b: 收集weight比较残差...")
    weight_compare_resids = {li: [] for li in test_layers}
    
    for heavy, light in WEIGHT_COMPARE:
        prompt = TEMPLATE_WEIGHT_COMPARE.format(heavy, light)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                weight_compare_resids[li].append(cap[f"L{li}"])
    
    # 1c: 收集habitat残差(用于W_U梯度方向计算)
    log("  1c: 收集habitat残差...")
    habitat_resids = {li: {"land": [], "ocean": [], "sky": []} for li in test_layers}
    
    for hab, words in WORDS_BY_HABITAT.items():
        for word in words:
            prompt = TEMPLATE_HABITAT.format(word)
            cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
            for li in test_layers:
                if f"L{li}" in cap:
                    habitat_resids[li][hab].append(cap[f"L{li}"])
    
    # 1d: 计算各层方向
    log("  1d: 计算各层方向...")
    
    for li in test_layers:
        # "bigger"语义方向
        if len(size_compare_resids[li]) < 3:
            continue
        
        # 用habitat残差作为中性基线(避免size/weight联合均值导致cos=-1的artifact)
        hab_all = []
        for hab in ["land", "ocean", "sky"]:
            hab_all.extend(habitat_resids[li].get(hab, []))
        
        if len(hab_all) < 6:
            continue
        
        baseline_mean = np.mean(hab_all, axis=0)  # 中性基线
        
        # bigger方向 = size比较均值 - 中性基线
        size_mean = np.mean(size_compare_resids[li], axis=0)
        bigger_dir = size_mean - baseline_mean
        bigger_norm = np.linalg.norm(bigger_dir)
        if bigger_norm < 1e-10:
            continue
        bigger_dir = bigger_dir / bigger_norm
        
        # weight方向 = weight比较均值 - 中性基线
        if len(weight_compare_resids[li]) >= 3:
            weight_mean = np.mean(weight_compare_resids[li], axis=0)
            weight_dir = weight_mean - baseline_mean
            weight_norm = np.linalg.norm(weight_dir)
            if weight_norm > 1e-10:
                weight_dir = weight_dir / weight_norm
            else:
                weight_dir = None
        else:
            weight_dir = None
        
        # W_U梯度方向 (bigger tokens)
        wu_grad_bigger = W_U[bigger_ids].mean(axis=0) if bigger_ids else None
        if wu_grad_bigger is not None:
            wu_norm = np.linalg.norm(wu_grad_bigger)
            if wu_norm > 1e-10:
                wu_grad_bigger = wu_grad_bigger / wu_norm
            else:
                wu_grad_bigger = None
        
        # W_U梯度方向 (heavier tokens)
        wu_grad_heavier = W_U[heavier_ids].mean(axis=0) if heavier_ids else None
        if wu_grad_heavier is not None:
            wu_norm = np.linalg.norm(wu_grad_heavier)
            if wu_norm > 1e-10:
                wu_grad_heavier = wu_grad_heavier / wu_norm
            else:
                wu_grad_heavier = None
        
        # 计算方向间的余弦
        dir_cosines = {}
        dir_cosines["cos(bigger_dir, wu_grad_bigger)"] = round(float(np.dot(bigger_dir, wu_grad_bigger)), 4) if wu_grad_bigger is not None else None
        
        if weight_dir is not None:
            dir_cosines["cos(weight_dir, wu_grad_heavier)"] = round(float(np.dot(weight_dir, wu_grad_heavier)), 4) if wu_grad_heavier is not None else None
            dir_cosines["cos(bigger_dir, weight_dir)"] = round(float(np.dot(bigger_dir, weight_dir)), 4)
        
        # bigger方向在W_U行空间中的投影比
        from scipy.sparse.linalg import svds
        W_U_T = W_U.T.astype(np.float32)
        k_svd = min(50, min(W_U_T.shape) - 2)
        U_wu, s_wu, _ = svds(W_U_T, k=k_svd)
        U_wu = np.asarray(U_wu, dtype=np.float64)
        
        proj_bigger = U_wu @ (U_wu.T @ bigger_dir)
        proj_ratio_bigger = float(np.linalg.norm(proj_bigger)**2 / np.linalg.norm(bigger_dir)**2)
        dir_cosines["bigger_in_WU_row_ratio"] = round(proj_ratio_bigger, 4)
        
        if weight_dir is not None:
            proj_weight = U_wu @ (U_wu.T @ weight_dir)
            proj_ratio_weight = float(np.linalg.norm(proj_weight)**2 / np.linalg.norm(weight_dir)**2)
            dir_cosines["weight_in_WU_row_ratio"] = round(proj_ratio_weight, 4)
        
        # W_U梯度方向在W_U行空间中的投影比(应该≈1)
        if wu_grad_bigger is not None:
            proj_wu_grad = U_wu @ (U_wu.T @ wu_grad_bigger)
            proj_ratio_wu_grad = float(np.linalg.norm(proj_wu_grad)**2 / np.linalg.norm(wu_grad_bigger)**2)
            dir_cosines["wu_grad_in_WU_row_ratio"] = round(proj_ratio_wu_grad, 4)
        
        results[f"dir_cosines_L{li}"] = {
            "layer": li,
            **dir_cosines,
        }
        log(f"  L{li}: {dir_cosines}")
    
    # 1e: 逐层perturb测试
    log("\n  1e: 逐层perturb测试...")
    
    test_prompt = "The elephant is bigger than the mouse"
    
    # Baseline logits
    with torch.no_grad():
        base_logits = model(**tokenizer(test_prompt, return_tensors="pt").to(device)).logits[0, -1, :].detach().float().cpu().numpy()
    
    base_bigger = np.mean([base_logits[tid] for tid in bigger_ids if tid < len(base_logits)])
    base_smaller = np.mean([base_logits[tid] for tid in smaller_ids if tid < len(smaller_ids)]) if smaller_ids else 0
    base_diff = base_bigger - base_smaller
    
    log(f"  Baseline: bigger={base_bigger:.3f}, smaller={base_smaller:.3f}, diff={base_diff:.3f}")
    
    perturb_results = {}
    alpha = 5.0
    
    for li in test_layers:
        if len(size_compare_resids[li]) < 3:
            continue
        
        # 方向1: bigger语义方向 (用habitat基线)
        hab_all = []
        for hab in ["land", "ocean", "sky"]:
            hab_all.extend(habitat_resids[li].get(hab, []))
        if len(hab_all) < 6:
            continue
        baseline_mean = np.mean(hab_all, axis=0)
        
        size_mean = np.mean(size_compare_resids[li], axis=0)
        bigger_dir = size_mean - baseline_mean
        bigger_norm = np.linalg.norm(bigger_dir)
        if bigger_norm < 1e-10:
            continue
        bigger_dir = bigger_dir / bigger_norm
        
        # 方向2: W_U梯度方向(bigger)
        wu_grad = W_U[bigger_ids].mean(axis=0) if bigger_ids else None
        if wu_grad is not None:
            wu_norm = np.linalg.norm(wu_grad)
            if wu_norm > 1e-10:
                wu_grad = wu_grad / wu_norm
        
        # 方向3: bigger方向在W_U行空间的投影
        proj_in_wu = U_wu @ (U_wu.T @ bigger_dir)
        proj_norm = np.linalg.norm(proj_in_wu)
        if proj_norm > 1e-10:
            proj_dir = proj_in_wu / proj_norm
        else:
            proj_dir = None
        
        directions = {
            "bigger_semantic": bigger_dir,
            "wu_gradient": wu_grad,
            "bigger_WU_projection": proj_dir,
        }
        
        for dir_name, perturb_dir in directions.items():
            if perturb_dir is None:
                continue
            
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
                perturb_logits = model(**tokenizer(test_prompt, return_tensors="pt").to(device)).logits[0, -1, :].detach().float().cpu().numpy()
            hook.remove()
            
            perturb_bigger = np.mean([perturb_logits[tid] for tid in bigger_ids if tid < len(perturb_logits)])
            perturb_smaller = np.mean([perturb_logits[tid] for tid in smaller_ids if tid < len(perturb_logits)]) if smaller_ids else 0
            perturb_diff = perturb_bigger - perturb_smaller
            
            key = f"perturb_{dir_name}_L{li}"
            perturb_results[key] = {
                "direction": dir_name,
                "layer": li,
                "alpha": alpha,
                "base_diff": round(float(base_diff), 4),
                "perturb_diff": round(float(perturb_diff), 4),
                "shift": round(float(perturb_diff - base_diff), 4),
                "bigger_shift": round(float(perturb_bigger - base_bigger), 4),
                "smaller_shift": round(float(perturb_smaller - base_smaller), 4),
            }
            
            log(f"  L{li} {dir_name}: diff={base_diff:.3f}→{perturb_diff:.3f} (shift={perturb_diff-base_diff:.4f})")
    
    results["perturb_comparison"] = perturb_results
    
    # ====================================================================
    # Part 2: LayerNorm旋转效应
    # ====================================================================
    log("\n" + "="*60)
    log("Part 2: LayerNorm旋转效应")
    log("="*60)
    
    # 获取最后层的LayerNorm权重
    li_last = n_layers - 1
    last_layer = layers[li_last]
    
    # 获取input_layernorm权重
    ln_weight = None
    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(last_layer, ln_name):
            ln = getattr(last_layer, ln_name)
            if hasattr(ln, "weight"):
                ln_weight = ln.weight.detach().cpu().float().numpy()
            break
    
    if ln_weight is not None and len(size_compare_resids[li_last]) >= 3:
        # 取一个实际的残差向量作为"基线"
        base_resid = size_compare_resids[li_last][0]
        
        # 模拟: bigger方向在LayerNorm前后的方向变化
        hab_all_ln = []
        for hab in ["land", "ocean", "sky"]:
            hab_all_ln.extend(habitat_resids[li_last].get(hab, []))
        if len(hab_all_ln) < 6:
            hab_all_ln = size_compare_resids[li_last] + weight_compare_resids[li_last]
        baseline_mean_ln = np.mean(hab_all_ln, axis=0)
        size_mean = np.mean(size_compare_resids[li_last], axis=0)
        bigger_dir = size_mean - baseline_mean_ln
        bigger_norm = np.linalg.norm(bigger_dir)
        if bigger_norm > 1e-10:
            bigger_dir = bigger_dir / bigger_norm
            
            # LayerNorm对方向的影响:
            # LayerNorm(x) = γ * (x - μ) / σ
            # 对于方向向量d, LayerNorm(x + αd) ≈ LayerNorm(x) + α * J_ln * d
            # 其中J_ln是LayerNorm在x处的Jacobian
            
            # 简化: 用数值差分估计LayerNorm对方向的作用
            # 取base_resid, 加减bigger_dir, 看LayerNorm后的差
            resid_plus = base_resid + 0.1 * bigger_dir
            resid_minus = base_resid - 0.1 * bigger_dir
            
            ln_plus = simulate_layernorm(resid_plus, ln_weight)
            ln_minus = simulate_layernorm(resid_minus, ln_weight)
            
            ln_diff = ln_plus - ln_minus  # LayerNorm后的方向变化
            ln_diff_norm = np.linalg.norm(ln_diff)
            if ln_diff_norm > 1e-10:
                ln_dir = ln_diff / ln_diff_norm
            else:
                ln_dir = None
            
            if ln_dir is not None:
                # LayerNorm旋转后的方向与原方向的关系
                cos_before_after = float(np.dot(bigger_dir, ln_dir))
                
                # LayerNorm旋转后的方向与W_U梯度方向的关系
                wu_grad = W_U[bigger_ids].mean(axis=0) if bigger_ids else None
                if wu_grad is not None:
                    wu_norm = np.linalg.norm(wu_grad)
                    if wu_norm > 1e-10:
                        wu_grad = wu_grad / wu_norm
                    cos_ln_wu = float(np.dot(ln_dir, wu_grad))
                else:
                    cos_ln_wu = None
                
                # LayerNorm旋转后方向在W_U行空间的投影
                proj_ln = U_wu @ (U_wu.T @ ln_dir)
                proj_ratio_ln = float(np.linalg.norm(proj_ln)**2 / np.linalg.norm(ln_dir)**2)
                
                results["layernorm_rotation"] = {
                    "layer": li_last,
                    "cos_bigger_before_after": round(cos_before_after, 4),
                    "cos_ln_dir_wu_grad": round(cos_ln_wu, 4) if cos_ln_wu is not None else None,
                    "ln_dir_in_WU_row_ratio": round(proj_ratio_ln, 4),
                    "bigger_dir_in_WU_row_ratio": round(proj_ratio_bigger, 4) if 'proj_ratio_bigger' in dir() else None,
                }
                
                log(f"  L{li_last}: cos(bigger_before, bigger_after_LN)={cos_before_after:.4f}")
                log(f"  L{li_last}: cos(bigger_after_LN, wu_grad)={cos_ln_wu:.4f}" if cos_ln_wu is not None else "")
                log(f"  L{li_last}: LN后方向在W_U行空间投影比={proj_ratio_ln:.4f}")
                log(f"  L{li_last}: LN前方向在W_U行空间投影比={proj_ratio_bigger:.4f}" if 'proj_ratio_bigger' in dir() else "")
    
    # 对更多层做LayerNorm分析
    log("\n  各层LayerNorm权重统计:")
    ln_stats = {}
    for li in test_layers:
        layer = layers[li]
        ln_w = None
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                if hasattr(ln, "weight"):
                    ln_w = ln.weight.detach().cpu().float().numpy()
                break
        
        if ln_w is not None:
            ln_stats[f"L{li}"] = {
                "mean": round(float(ln_w.mean()), 4),
                "std": round(float(ln_w.std()), 4),
                "min": round(float(ln_w.min()), 4),
                "max": round(float(ln_w.max()), 4),
                "pct_below_0.5": round(float((ln_w < 0.5).mean()), 4),
                "pct_above_2.0": round(float((ln_w > 2.0).mean()), 4),
            }
            log(f"  L{li}: mean={ln_w.mean():.4f}, std={ln_w.std():.4f}, pct<0.5={float((ln_w<0.5).mean()):.3f}, pct>2.0={float((ln_w>2.0).mean()):.3f}")
    
    results["layernorm_stats"] = ln_stats
    
    # ====================================================================
    # Part 3: size vs weight方向分解
    # ====================================================================
    log("\n" + "="*60)
    log("Part 3: size vs weight方向分解")
    log("="*60)
    
    # 3a: 收集纯属性残差
    log("  3a: 收集纯属性残差...")
    
    size_attr_resids = {li: [] for li in test_layers}
    for word in SIZE_ATTR_WORDS:
        prompt = TEMPLATE_SIZE_ATTR.format(word)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                size_attr_resids[li].append(cap[f"L{li}"])
    
    weight_attr_resids = {li: [] for li in test_layers}
    for word in WEIGHT_ATTR_WORDS:
        prompt = TEMPLATE_WEIGHT_ATTR.format(word)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                weight_attr_resids[li].append(cap[f"L{li}"])
    
    speed_attr_resids = {li: [] for li in test_layers}
    for word in SPEED_ATTR_WORDS:
        prompt = TEMPLATE_SPEED_ATTR.format(word)
        cap = collect_resid_at_layers(model, tokenizer, device, layers, prompt, test_layers)
        for li in test_layers:
            if f"L{li}" in cap:
                speed_attr_resids[li].append(cap[f"L{li}"])
    
    # 3b: 方向分解分析
    log("  3b: 方向分解分析...")
    
    for li in test_layers:
        if len(size_compare_resids[li]) < 3 or len(weight_compare_resids[li]) < 3:
            continue
        if len(size_attr_resids[li]) < 3 or len(weight_attr_resids[li]) < 3:
            continue
        
        # 合并所有残差计算全局均值 (用habitat作为中性基线+所有残差)
        hab_for_baseline = []
        for hab in ["land", "ocean", "sky"]:
            hab_for_baseline.extend(habitat_resids[li].get(hab, []))
        
        all_resids = (hab_for_baseline + size_compare_resids[li] + weight_compare_resids[li] + 
                     size_attr_resids[li] + weight_attr_resids[li] + speed_attr_resids[li])
        if len(all_resids) < 10:
            continue
        all_mean = np.mean(all_resids, axis=0)
        
        # 比较方向 (size)
        size_comp_mean = np.mean(size_compare_resids[li], axis=0)
        size_comp_dir = size_comp_mean - all_mean
        scn = np.linalg.norm(size_comp_dir)
        if scn > 1e-10:
            size_comp_dir = size_comp_dir / scn
        else:
            continue
        
        # 比较方向 (weight)
        weight_comp_mean = np.mean(weight_compare_resids[li], axis=0)
        weight_comp_dir = weight_comp_mean - all_mean
        wcn = np.linalg.norm(weight_comp_dir)
        if wcn > 1e-10:
            weight_comp_dir = weight_comp_dir / wcn
        else:
            continue
        
        # 纯属性方向 (size)
        size_attr_mean = np.mean(size_attr_resids[li], axis=0)
        size_attr_dir = size_attr_mean - all_mean
        san = np.linalg.norm(size_attr_dir)
        if san > 1e-10:
            size_attr_dir = size_attr_dir / san
        else:
            size_attr_dir = None
        
        # 纯属性方向 (weight)
        weight_attr_mean = np.mean(weight_attr_resids[li], axis=0)
        weight_attr_dir = weight_attr_mean - all_mean
        wan = np.linalg.norm(weight_attr_dir)
        if wan > 1e-10:
            weight_attr_dir = weight_attr_dir / wan
        else:
            weight_attr_dir = None
        
        # 纯属性方向 (speed)
        if len(speed_attr_resids[li]) >= 3:
            speed_attr_mean = np.mean(speed_attr_resids[li], axis=0)
            speed_attr_dir = speed_attr_mean - all_mean
            spn = np.linalg.norm(speed_attr_dir)
            if spn > 1e-10:
                speed_attr_dir = speed_attr_dir / spn
            else:
                speed_attr_dir = None
        else:
            speed_attr_dir = None
        
        # 关键余弦计算
        cosines = {}
        
        # 比较方向之间的余弦
        cosines["cos(size_compare, weight_compare)"] = round(float(np.dot(size_comp_dir, weight_comp_dir)), 4)
        
        # 纯属性方向之间的余弦
        if size_attr_dir is not None and weight_attr_dir is not None:
            cosines["cos(size_attr, weight_attr)"] = round(float(np.dot(size_attr_dir, weight_attr_dir)), 4)
        
        if size_attr_dir is not None and speed_attr_dir is not None:
            cosines["cos(size_attr, speed_attr)"] = round(float(np.dot(size_attr_dir, speed_attr_dir)), 4)
        
        if weight_attr_dir is not None and speed_attr_dir is not None:
            cosines["cos(weight_attr, speed_attr)"] = round(float(np.dot(weight_attr_dir, speed_attr_dir)), 4)
        
        # 比较 vs 纯属性
        if size_attr_dir is not None:
            cosines["cos(size_compare, size_attr)"] = round(float(np.dot(size_comp_dir, size_attr_dir)), 4)
        
        if weight_attr_dir is not None:
            cosines["cos(weight_compare, weight_attr)"] = round(float(np.dot(weight_comp_dir, weight_attr_dir)), 4)
        
        # 分解: 比较方向 - 纯属性方向 = "比较操作"方向?
        # 如果size_compare ≈ size_attr + operation, weight_compare ≈ weight_attr + operation
        # 则size_compare - size_attr ≈ weight_compare - weight_attr ≈ operation
        if size_attr_dir is not None and weight_attr_dir is not None:
            # "比较操作"方向 (size维度)
            op_dir_size = size_comp_dir - np.dot(size_comp_dir, size_attr_dir) * size_attr_dir
            op_norm = np.linalg.norm(op_dir_size)
            if op_norm > 1e-10:
                op_dir_size = op_dir_size / op_norm
            else:
                op_dir_size = None
            
            # "比较操作"方向 (weight维度)
            op_dir_weight = weight_comp_dir - np.dot(weight_comp_dir, weight_attr_dir) * weight_attr_dir
            op_norm = np.linalg.norm(op_dir_weight)
            if op_norm > 1e-10:
                op_dir_weight = op_dir_weight / op_norm
            else:
                op_dir_weight = None
            
            # 比较操作方向在size/weight间的一致性
            if op_dir_size is not None and op_dir_weight is not None:
                cos_op = float(np.dot(op_dir_size, op_dir_weight))
                cosines["cos(operation_from_size, operation_from_weight)"] = round(cos_op, 4)
        
        results[f"direction_decomposition_L{li}"] = {
            "layer": li,
            **cosines,
        }
        log(f"  L{li}: {cosines}")
    
    # ====================================================================
    # Part 4: 信息流追踪 — Embedding→L0→L1
    # ====================================================================
    log("\n" + "="*60)
    log("Part 4: 信息流追踪 — Embedding→L0→L1")
    log("="*60)
    
    # 4a: Embedding矩阵SVD
    log("  4a: Embedding矩阵分析...")
    embed_weight = model.get_input_embeddings().weight.detach().cpu().float().numpy()  # [vocab, d_model]
    
    # SVD of embedding (采样避免内存溢出)
    vocab_size, d_emb = embed_weight.shape
    n_sample = min(5000, vocab_size)  # 采样5000个词
    rng = np.random.RandomState(42)
    idx = rng.choice(vocab_size, n_sample, replace=False)
    embed_sample = embed_weight[idx].astype(np.float32)
    embed_sample_centered = embed_sample - embed_sample.mean(axis=0)
    U_emb, S_emb, Vt_emb = np.linalg.svd(embed_sample_centered, full_matrices=False)
    S_emb_full_est = S_emb  # 采样SVD的奇异值(近似)
    
    # 有效维度
    total_energy = np.sum(S_emb_full_est**2)
    cum_energy = np.cumsum(S_emb_full_est**2) / total_energy
    
    eff_dim_f10 = int(np.sum(S_emb_full_est > 10))
    eff_dim_90 = int(np.searchsorted(cum_energy, 0.90)) + 1
    eff_dim_99 = int(np.searchsorted(cum_energy, 0.99)) + 1
    
    # 熵有效维度
    p = (S_emb_full_est**2) / total_energy
    entropy = -np.sum(p * np.log(p + 1e-30))
    eff_dim_entropy = np.exp(entropy)
    
    results["embedding_svd"] = {
        "shape": list(embed_weight.shape),
        "sample_size": n_sample,
        "eff_dim_F10": eff_dim_f10,
        "eff_dim_90pct": eff_dim_90,
        "eff_dim_99pct": eff_dim_99,
        "eff_dim_entropy": round(float(eff_dim_entropy), 2),
        "top10_sv": [round(float(s), 2) for s in S_emb_full_est[:10]],
    }
    log(f"  Embedding: eff_dim(F>10)={eff_dim_f10}, eff_dim(90%)={eff_dim_90}, eff_dim(entropy)={eff_dim_entropy:.1f}")
    
    # 4b: L0→L1 PC对齐 (用habitat残差)
    log("  4b: L0→L1 PC对齐...")
    
    for li1, li2 in [(0, 1), (1, 2), (0, n_layers//2), (n_layers-2, n_layers-1)]:
        if li1 >= n_layers or li2 >= n_layers:
            continue
        
        X1_list = []
        X2_list = []
        for hab in ["land", "ocean", "sky"]:
            if li1 in habitat_resids and li2 in habitat_resids:
                X1_list.extend(habitat_resids[li1].get(hab, []))
                X2_list.extend(habitat_resids[li2].get(hab, []))
        
        if len(X1_list) < 6 or len(X2_list) < 6:
            continue
        
        X1 = np.array(X1_list)
        X2 = np.array(X2_list)
        
        m1 = X1.mean(axis=0)
        m2 = X2.mean(axis=0)
        
        U1, S1, Vt1 = np.linalg.svd(X1 - m1, full_matrices=False)
        U2, S2, Vt2 = np.linalg.svd(X2 - m2, full_matrices=False)
        
        # PC对齐
        pc_alignment = []
        for k in range(min(5, Vt1.shape[0], Vt2.shape[0])):
            cos_val = float(np.abs(np.dot(Vt1[k], Vt2[k])))
            pc_alignment.append(round(cos_val, 4))
        
        # 子空间重叠
        K = min(5, Vt1.shape[0], Vt2.shape[0])
        subspace1 = Vt1[:K]  # [K, d_model]
        subspace2 = Vt2[:K]
        
        # 子空间重叠 = mean(|cos(PC1_k, PC2_projected)|)
        proj_matrix = subspace2.T @ subspace2  # 投影到subspace2
        overlap = 0
        for k in range(K):
            proj = proj_matrix @ subspace1[k]
            cos_val = float(np.dot(subspace1[k], proj) / (np.linalg.norm(proj) + 1e-10))
            overlap += abs(cos_val)
        overlap = overlap / K
        
        results[f"pc_alignment_L{li1}_L{li2}"] = {
            "layer1": li1,
            "layer2": li2,
            "pc_alignment": pc_alignment,
            "subspace_overlap_K5": round(float(overlap), 4),
        }
        log(f"  L{li1}→L{li2}: PC对齐={pc_alignment}, 子空间重叠={overlap:.4f}")
    
    # 4c: 语义方差比 (各层)
    log("  4c: 语义方差比 (各层)...")
    
    for li in test_layers:
        all_hab = []
        for hab in ["land", "ocean", "sky"]:
            all_hab.extend(habitat_resids[li].get(hab, []))
        
        if len(all_hab) < 6:
            continue
        
        X = np.array(all_hab)
        total_var = float(np.var(X))
        
        # 组间方差
        grand_mean = X.mean(axis=0)
        between_var = 0
        for hab in ["land", "ocean", "sky"]:
            hab_vecs = habitat_resids[li].get(hab, [])
            if len(hab_vecs) >= 2:
                hab_mean = np.mean(hab_vecs, axis=0)
                between_var += len(hab_vecs) * float(np.dot(hab_mean - grand_mean, hab_mean - grand_mean))
        between_var /= len(X)
        
        semantic_ratio = between_var / total_var if total_var > 1e-10 else 0
        F_ratio = between_var / (total_var - between_var + 1e-10)
        
        results[f"semantic_ratio_L{li}"] = {
            "layer": li,
            "semantic_ratio": round(float(semantic_ratio), 4),
            "F_ratio": round(float(F_ratio), 4),
            "total_var": round(float(total_var), 4),
        }
        log(f"  L{li}: semantic_ratio={semantic_ratio:.4f}, F_ratio={F_ratio:.4f}")
    
    # ====================================================================
    # Part 5: 深层perturb — 用W_U梯度方向对比bigger/heavier
    # ====================================================================
    log("\n" + "="*60)
    log("Part 5: W_U梯度perturb — bigger vs heavier方向")
    log("="*60)
    
    # 在最后层测试: W_U梯度方向(bigger)和W_U梯度方向(heavier)
    for li in [n_layers - 2, n_layers - 1]:
        test_prompts = [
            ("bigger_test", "The elephant is bigger than the mouse"),
            ("heavier_test", "The iron is heavier than the feather"),
        ]
        
        for prompt_name, prompt in test_prompts:
            # Baseline
            with torch.no_grad():
                base_logits = model(**tokenizer(prompt, return_tensors="pt").to(device)).logits[0, -1, :].detach().float().cpu().numpy()
            
            base_b = np.mean([base_logits[tid] for tid in bigger_ids if tid < len(base_logits)]) if bigger_ids else 0
            base_s = np.mean([base_logits[tid] for tid in smaller_ids if tid < len(base_logits)]) if smaller_ids else 0
            base_h = np.mean([base_logits[tid] for tid in heavier_ids if tid < len(base_logits)]) if heavier_ids else 0
            base_l = np.mean([base_logits[tid] for tid in lighter_ids if tid < len(base_logits)]) if lighter_ids else 0
            
            for grad_name, grad_ids_list in [("bigger", bigger_ids), ("heavier", heavier_ids)]:
                grad_dir = W_U[grad_ids_list].mean(axis=0)
                grad_norm = np.linalg.norm(grad_dir)
                if grad_norm < 1e-10:
                    continue
                grad_dir = grad_dir / grad_norm
                
                alpha_test = 5.0
                perturbed = [False]
                def perturb_hook(m, inp, out):
                    if perturbed[0]:
                        return
                    perturbed[0] = True
                    o = out[0] if isinstance(out, tuple) else out
                    o_new = o.clone()
                    dir_tensor = torch.tensor(grad_dir, dtype=o.dtype, device=device)
                    o_new[0, -1, :] += alpha_test * dir_tensor
                    if isinstance(out, tuple):
                        return (o_new,) + out[1:]
                    return o_new
                
                hook = layers[li].register_forward_hook(perturb_hook)
                with torch.no_grad():
                    perturb_logits = model(**tokenizer(prompt, return_tensors="pt").to(device)).logits[0, -1, :].detach().float().cpu().numpy()
                hook.remove()
                
                perturb_b = np.mean([perturb_logits[tid] for tid in bigger_ids if tid < len(perturb_logits)]) if bigger_ids else 0
                perturb_s = np.mean([perturb_logits[tid] for tid in smaller_ids if tid < len(perturb_logits)]) if smaller_ids else 0
                perturb_h = np.mean([perturb_logits[tid] for tid in heavier_ids if tid < len(perturb_logits)]) if heavier_ids else 0
                perturb_l = np.mean([perturb_logits[tid] for tid in lighter_ids if tid < len(perturb_logits)]) if lighter_ids else 0
                
                key = f"wu_perturb_{grad_name}_on_{prompt_name}_L{li}"
                results[key] = {
                    "gradient": grad_name,
                    "prompt": prompt_name,
                    "layer": li,
                    "alpha": alpha_test,
                    "bigger_shift": round(float(perturb_b - base_b), 4),
                    "smaller_shift": round(float(perturb_s - base_s), 4),
                    "heavier_shift": round(float(perturb_h - base_h), 4),
                    "lighter_shift": round(float(perturb_l - base_l), 4),
                    "bigger_smaller_diff_shift": round(float((perturb_b - perturb_s) - (base_b - base_s)), 4),
                    "heavier_lighter_diff_shift": round(float((perturb_h - perturb_l) - (base_h - base_l)), 4),
                }
                log(f"  L{li} {grad_name}_grad on {prompt_name}: bigger_shift={perturb_b-base_b:.4f}, heavier_shift={perturb_h-base_h:.4f}")
    
    # ====================================================================
    # 保存结果
    # ====================================================================
    out_path = TEMP / f"ccxxxi_reasoning_to_logit_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "d_model": d_model,
            "n_layers": n_layers,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
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
