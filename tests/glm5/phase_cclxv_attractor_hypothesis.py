"""
Phase CCLXV: 吸引子假设验证
================================================
核心假设: 残差流空间是吸引子场
  - 每个概念 = 一个吸引子盆地
  - 浅层 = 过渡区(可替换, 高switch_rate)
  - 深层 = 盆地内部(自然回归, 低switch_rate)
  - INV-6(加法传播) = 轨迹方程
  - INV-27/30/31 = 过渡区/盆地边/吸引子收敛的行为表现

4个实验:
  Exp1: 局部收缩率(Jacobian谱范数估计)
        → 如果深层谱范数<1, 证明深层是收缩映射(吸引子)
  Exp2: 盆地边界测绘(概念插值的分水岭)
        → 如果存在sharp boundary, 证明吸引子盆地有明确边界
  Exp3: 扰动几何衰减验证
        → 如果log(||δ_l||)线性于l, 证明是几何衰减(线性吸引子)
  Exp4: 轨迹速度场(逐层残差变化量)
        → 如果深层速度→0, 证明接近不动点

否定预测:
  - 如果深层谱范数>1 → 不是吸引子, 是膨胀映射
  - 如果没有sharp boundary → 不是离散吸引子盆地
  - 如果不是几何衰减 → 可能是非线性吸引子或混沌

用法:
  python phase_cclxv_attractor_hypothesis.py --model qwen3 --exp 1
  python phase_cclxv_attractor_hypothesis.py --model qwen3 --exp all
"""
import argparse, os, sys, json, time, gc
# 强制stdout使用utf-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
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
    "animals": ["dog", "cat", "horse", "eagle", "shark", "snake",
                "lion", "bear", "whale", "dolphin", "rabbit", "deer"],
    "food":    ["apple", "rice", "bread", "cheese", "salmon", "mango",
                "grape", "banana", "pasta", "pizza", "cookie", "steak"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "chisel",
                "pliers", "ruler", "level", "clamp", "file", "shovel"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "volcano",
                "canyon", "glacier", "meadow", "island", "valley", "cliff"],
}

TEMPLATES = [
    "The {} is",
    "I saw a {} today",
]


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def json_serialize(obj):
    """Convert numpy types to Python native for JSON serialization"""
    if isinstance(obj, dict):
        return {str(k): json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serialize(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return obj
    elif obj is None:
        return None
    return obj


def collect_full_residuals(model, tokenizer, device, prompt, target_token_str, n_layers):
    """收集所有层的残差流向量(最后一个token位置)"""
    input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
    
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
    
    # 取最后一个token位置
    last_pos = input_ids.shape[1] - 1
    
    residuals = {}
    for li in range(n_layers):
        key = f"L{li}"
        if key in captured:
            residuals[li] = captured[key][0, last_pos].numpy()
    
    return residuals, last_pos


def get_top_k_tokens(logits, tokenizer, k=5):
    probs = torch.softmax(logits[0, -1], dim=-1)
    top_k = torch.topk(probs, k)
    result = []
    for i in range(k):
        tok_id = top_k.indices[i].item()
        tok_str = tokenizer.decode([tok_id]).strip()
        prob = top_k.values[i].item()
        result.append({"token": tok_str, "id": tok_id, "prob": prob})
    return result


# ============================================================
# Exp1: 局部收缩率估计 (Jacobian谱范数的有限差分近似)
# ============================================================
def exp1_contraction_ratio(args):
    """
    核心方法: 对每层, 在自然残差流上加随机扰动δ, 测量输出扰动δ'的缩放比
    
    contraction_ratio(l) = ||δ_{l+1}|| / ||δ_l||
    
    如果 contraction_ratio < 1: 该层是局部收缩的 (吸引子性质)
    如果 contraction_ratio > 1: 该层是局部膨胀的
    
    更精确: 对多层传播
    multi_step_ratio(l1, l2) = ||δ_{l2}|| / ||δ_{l1}||
    如果深层 < 浅层: 整体趋势是收缩 → 支持吸引子假设
    """
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp1: 局部收缩率估计 ({args.model}, {n_layers}L, d={d_model})")
    print(f"  方法: 加随机扰动δ, 测量每层输出的扰动缩放比")
    
    template = "The {} is"
    all_concepts = []
    for cat, words in CONCEPTS.items():
        all_concepts.extend(words)
    
    # 随机选择15个概念(控制计算量)
    rng = np.random.RandomState(42)
    selected = rng.choice(all_concepts, size=min(15, len(all_concepts)), replace=False)
    
    # 扰动强度: 从小到大3个尺度
    epsilons = [0.01, 0.05, 0.1]  # 相对于残差流norm的比例
    
    n_perturbations = 10  # 每个概念每个eps做10次随机扰动取平均
    
    # 收集结果: concept -> eps -> [per_layer_ratios]
    all_results = {}
    
    layers_list = get_layers(model)
    
    for ci, concept in enumerate(selected):
        print(f"\n  [{ci+1}/{len(selected)}] {concept}")
        
        prompt = template.format(concept)
        input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1
        
        # 1. 收集baseline残差流
        resid_baseline = {}
        
        def make_baseline_hook(li):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    resid_baseline[li] = output[0][0, last_pos].detach().float().cpu().numpy()
                else:
                    resid_baseline[li] = output[0, last_pos].detach().float().cpu().numpy()
            return hook
        
        hooks_bl = []
        for li in range(n_layers):
            hooks_bl.append(layers_list[li].register_forward_hook(make_baseline_hook(li)))
        
        with torch.no_grad():
            _ = model(input_ids)
        
        for h in hooks_bl:
            h.remove()
        
        if len(resid_baseline) < n_layers:
            print(f"    Skip: only got {len(resid_baseline)} layers")
            continue
        
        # 2. 对每个扰动强度, 做多次扰动
        concept_data = {}
        
        for eps_frac in epsilons:
            per_layer_perturb_norms = defaultdict(list)  # li -> [||δ_out||]
            
            for trial in range(n_perturbations):
                # 在L0加随机扰动, 追踪到所有层
                # 随机方向
                random_dir = rng.randn(d_model)
                random_dir = random_dir / np.linalg.norm(random_dir)
                
                # 扰动量 = eps_frac * ||h_0||
                h0_norm = np.linalg.norm(resid_baseline[0])
                delta_magnitude = eps_frac * h0_norm
                delta_vec = delta_magnitude * random_dir
                
                perturbed_captured = {}
                
                # L0: 注入扰动
                model_dtype = next(model.parameters()).dtype
                delta_tensor = torch.tensor(delta_vec, dtype=model_dtype, device=device)
                
                def make_inject_hook():
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            out = output[0].clone()
                            out[0, last_pos] = out[0, last_pos] + delta_tensor
                            return (out,) + output[1:]
                        else:
                            out = output.clone()
                            out[0, last_pos] = out[0, last_pos] + delta_tensor
                            return out
                    return hook
                
                def make_capture_hook(li):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            perturbed_captured[li] = output[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            perturbed_captured[li] = output[0, last_pos].detach().float().cpu().numpy()
                    return hook
                
                h_inject = layers_list[0].register_forward_hook(make_inject_hook())
                hooks_cap = []
                for li in range(1, n_layers):
                    hooks_cap.append(layers_list[li].register_forward_hook(make_capture_hook(li)))
                
                with torch.no_grad():
                    try:
                        _ = model(input_ids)
                    except:
                        pass
                
                h_inject.remove()
                for h in hooks_cap:
                    h.remove()
                
                # 计算每层扰动残差 = perturbed - baseline
                for li in range(1, n_layers):
                    if li in perturbed_captured and li in resid_baseline:
                        delta_out = perturbed_captured[li] - resid_baseline[li]
                        perturb_norm = float(np.linalg.norm(delta_out))
                        per_layer_perturb_norms[li].append(perturb_norm)
                
                # L0: 扰动本身
                per_layer_perturb_norms[0].append(delta_magnitude)
            
            # 计算平均扰动范数和收缩率
            avg_norms = {}
            contraction_ratios = {}
            for li in sorted(per_layer_perturb_norms.keys()):
                avg_norms[li] = float(np.mean(per_layer_perturb_norms[li]))
                if li > 0 and (li-1) in avg_norms and avg_norms[li-1] > 1e-10:
                    # 单步收缩率: ||δ_l|| / ||δ_{l-1}||
                    contraction_ratios[li] = avg_norms[li] / avg_norms[li-1]
            
            # 多步收缩率: ||δ_l|| / ||δ_0||
            multi_step_ratios = {}
            if 0 in avg_norms and avg_norms[0] > 1e-10:
                for li in sorted(avg_norms.keys()):
                    multi_step_ratios[li] = avg_norms[li] / avg_norms[0]
            
            concept_data[eps_frac] = {
                "avg_perturb_norm": {str(k): round(v, 2) for k, v in sorted(avg_norms.items())},
                "single_step_ratio": {str(k): round(v, 4) for k, v in sorted(contraction_ratios.items())},
                "multi_step_ratio": {str(k): round(v, 4) for k, v in sorted(multi_step_ratios.items())},
            }
        
        all_results[concept] = concept_data
        
        # 打印关键信息
        for eps_frac in epsilons:
            ms = concept_data[eps_frac]["multi_step_ratio"]
            if "0" in ms and str(n_layers-1) in ms:
                final_ratio = ms[str(n_layers-1)]
                print(f"    eps={eps_frac}: multi_step_ratio L0→L{n_layers-1} = {final_ratio:.4f}")
    
    # 3. 聚合分析
    print("\n  === Exp1 聚合分析 ===")
    
    for eps_frac in epsilons:
        # 收集所有概念在每个层的multi_step_ratio
        layer_ratios = defaultdict(list)
        for concept, data in all_results.items():
            ms = data[eps_frac]["multi_step_ratio"]
            for layer_str, ratio in ms.items():
                layer_ratios[int(layer_str)].append(ratio)
        
        # 计算均值和标准差
        avg_by_layer = {}
        std_by_layer = {}
        for li in sorted(layer_ratios.keys()):
            vals = layer_ratios[li]
            avg_by_layer[li] = float(np.mean(vals))
            std_by_layer[li] = float(np.std(vals))
        
        # 关键指标
        shallow_ratios = [avg_by_layer[l] for l in range(1, min(6, n_layers)) if l in avg_by_layer]
        deep_ratios = [avg_by_layer[l] for l in range(n_layers-5, n_layers) if l in avg_by_layer]
        
        avg_shallow = np.mean(shallow_ratios) if shallow_ratios else 0
        avg_deep = np.mean(deep_ratios) if deep_ratios else 0
        
        # 单步收缩率统计
        single_step_all = defaultdict(list)
        for concept, data in all_results.items():
            ss = data[eps_frac]["single_step_ratio"]
            for layer_str, ratio in ss.items():
                single_step_all[int(layer_str)].append(ratio)
        
        avg_single_step = {}
        for li in sorted(single_step_all.keys()):
            avg_single_step[li] = float(np.mean(single_step_all[li]))
        
        shallow_ss = [avg_single_step[l] for l in range(1, min(6, n_layers)) if l in avg_single_step]
        deep_ss = [avg_single_step[l] for l in range(n_layers-5, n_layers) if l in avg_single_step]
        
        avg_shallow_ss = np.mean(shallow_ss) if shallow_ss else 0
        avg_deep_ss = np.mean(deep_ss) if deep_ss else 0
        
        print(f"\n  eps={eps_frac}:")
        print(f"    Multi-step: shallow(L1-L5) avg={avg_shallow:.4f}, deep(L{n_layers-5}-L{n_layers-1}) avg={avg_deep:.4f}")
        print(f"    Single-step: shallow avg={avg_shallow_ss:.4f}, deep avg={avg_deep_ss:.4f}")
        print(f"    Attractor prediction: deep < 1.0 (contraction), shallow may > 1.0")
        
        # 几何衰减拟合: log(ratio(l)) = a*l + b
        layers_for_fit = sorted([l for l in avg_by_layer.keys() if l > 0])
        if len(layers_for_fit) > 2:
            log_ratios = [np.log(max(avg_by_layer[l], 1e-10)) for l in layers_for_fit]
            fit = np.polyfit(layers_for_fit, log_ratios, 1)
            slope = fit[0]
            # slope < 0 意味着扰动随层指数衰减 → 吸引子
            print(f"    Geometric decay fit: log(ratio) = {slope:.6f}*l + {fit[1]:.4f}")
            print(f"    slope < 0 → perturbation decays → SUPPORTS attractor")
            print(f"    slope > 0 → perturbation grows → REJECTS attractor")
    
    # 保存结果
    out_path = OUTPUT_DIR / f"{args.model}_cclxv" / "exp1_contraction_ratio.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "phase": "CCLXV",
        "model": args.model,
        "experiment": "exp1_contraction_ratio",
        "timestamp": datetime.now().isoformat(),
        "n_concepts": len(all_results),
        "epsilons": epsilons,
        "n_perturbations": n_perturbations,
        "per_concept_data": all_results,
        "aggregation": {
            str(eps): {
                "avg_multi_step_by_layer": {str(k): round(v, 4) for k, v in sorted(avg_by_layer.items())},
                "avg_single_step_by_layer": {str(k): round(v, 4) for k, v in sorted(avg_single_step.items())},
                "shallow_multi_step_avg": round(avg_shallow, 4),
                "deep_multi_step_avg": round(avg_deep, 4),
                "shallow_single_step_avg": round(avg_shallow_ss, 4),
                "deep_single_step_avg": round(avg_deep_ss, 4),
            }
            for eps, avg_by_layer, avg_single_step in [
                (epsilons[0], avg_by_layer, avg_single_step),
            ]
        },
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
        print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp2: 盆地边界测绘 (概念插值的分水岭)
# ============================================================
def exp2_basin_boundary(args):
    """
    核心方法: 在两个概念的残差流之间插值, 找到输出翻转的临界点
    
    h_interp(α) = (1-α) * h_A + α * h_B  (在某层l)
    
    如果吸引子盆地存在, 应该存在一个 α* ∈ (0,1) 使得:
    - α < α*: 输出偏向概念A
    - α > α*: 输出偏向概念B
    - 边界sharpness = |dP(A)/dα| at α* 越大, 盆地边界越清晰
    
    对比不同层:
    - 浅层: α* 处可能渐变(盆地边界模糊, 在过渡区)
    - 深层: α* 处可能阶跃(盆地边界清晰, 在盆地内部)
    """
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp2: 盆地边界测绘 ({args.model}, {n_layers}L, d={d_model})")
    print(f"  方法: 概念对插值, 找输出翻转临界点和边界sharpness")
    
    template = "The {} is"
    
    # 跨类别概念对
    cross_pairs = [
        ("dog", "apple"), ("cat", "hammer"), ("horse", "rice"),
        ("eagle", "ocean"), ("shark", "desert"), ("snake", "cheese"),
        ("lion", "grape"), ("bear", "knife"), ("whale", "saw"),
        ("rabbit", "mango"), ("deer", "drill"), ("dolphin", "bread"),
    ]
    
    # 同类别概念对(控制组: 类别内也应能区分)
    same_pairs = [
        ("dog", "cat"), ("apple", "mango"), ("hammer", "knife"),
        ("mountain", "river"), ("lion", "bear"), ("grape", "banana"),
    ]
    
    # 测试的层: 均匀采样
    test_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in test_layers:
        test_layers.append(n_layers - 1)
    
    # 插值点
    alphas = np.linspace(0, 1, 21)  # 21个点, 步长0.05
    
    all_results = {}
    layers_list = get_layers(model)
    
    for pair_type, pairs in [("cross", cross_pairs), ("same", same_pairs)]:
        for pi, (concept_A, concept_B) in enumerate(pairs):
            pair_key = f"{pair_type}:{concept_A}↔{concept_B}"
            print(f"\n  [{pair_type}] {concept_A} ↔ {concept_B}")
            
            # 1. 收集两个概念在每层的残差流
            prompt_A = template.format(concept_A)
            prompt_B = template.format(concept_B)
            
            resid_A, _ = collect_full_residuals(model, tokenizer, device, prompt_A, concept_A, n_layers)
            resid_B, _ = collect_full_residuals(model, tokenizer, device, prompt_B, concept_B, n_layers)
            
            # 2. 获取两个概念的token id (用于判断输出)
            token_id_A = tokenizer.encode(concept_A, add_special_tokens=False)
            token_id_B = tokenizer.encode(concept_B, add_special_tokens=False)
            
            pair_data = {}
            
            for li in test_layers:
                if li not in resid_A or li not in resid_B:
                    continue
                
                h_A = resid_A[li]
                h_B = resid_B[li]
                
                # 3. 在该层插值, patch到该层, 看最终输出
                alpha_results = []
                
                for alpha in alphas:
                    # 插值向量
                    h_interp = (1 - alpha) * h_A + alpha * h_B
                    
                    # 构造patch: 用h_interp替换layer li的输出
                    input_ids_A = tokenizer(prompt_A, return_tensors="pt").to(device).input_ids
                    last_pos = input_ids_A.shape[1] - 1
                    
                    model_dtype = next(model.parameters()).dtype
                    h_interp_tensor = torch.tensor(h_interp, dtype=model_dtype, device=device)
                    
                    # Hook: 在li层替换残差流
                    def make_patch_hook(target_pos, replacement_vec):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                out = output[0].clone()
                                out[0, target_pos] = replacement_vec
                                return (out,) + output[1:]
                            else:
                                out = output.clone()
                                out[0, target_pos] = replacement_vec
                                return out
                        return hook
                    
                    # 注册hook: 在li层替换, 在最后一层之后获取logits
                    h_patch = layers_list[li].register_forward_hook(
                        make_patch_hook(last_pos, h_interp_tensor))
                    
                    with torch.no_grad():
                        try:
                            logits = model(input_ids_A).logits
                        except:
                            h_patch.remove()
                            continue
                    
                    h_patch.remove()
                    
                    # 获取概率
                    probs = torch.softmax(logits[0, -1], dim=-1)
                    
                    # 获取concept_A和concept_B的概率
                    prob_A = 0.0
                    for tid in token_id_A:
                        if tid < len(probs):
                            prob_A += probs[tid].item()
                    
                    prob_B = 0.0
                    for tid in token_id_B:
                        if tid < len(probs):
                            prob_B += probs[tid].item()
                    
                    alpha_results.append({
                        "alpha": round(float(alpha), 3),
                        "prob_A": round(prob_A, 6),
                        "prob_B": round(prob_B, 6),
                        "prob_diff": round(prob_A - prob_B, 6),
                    })
                
                # 4. 找到分水岭: prob_diff改变符号的alpha
                watershed_alpha = None
                boundary_sharpness = 0.0
                for i in range(len(alpha_results) - 1):
                    d1 = alpha_results[i]["prob_diff"]
                    d2 = alpha_results[i+1]["prob_diff"]
                    if d1 * d2 < 0:  # 符号改变
                        # 线性插值找零点
                        a1, a2 = alpha_results[i]["alpha"], alpha_results[i+1]["alpha"]
                        watershed_alpha = a1 + (a2 - a1) * abs(d1) / (abs(d1) + abs(d2))
                        # 边界sharpness: |d(prob_diff)/d(alpha)|
                        boundary_sharpness = abs(d2 - d1) / (a2 - a1)
                        break
                
                # 如果没有符号改变, 记录最终偏移方向
                if watershed_alpha is None:
                    final_diff = alpha_results[-1]["prob_diff"] if alpha_results else 0
                    watershed_alpha = -1  # 标记为未找到
                    boundary_sharpness = 0
                
                pair_data[str(li)] = {
                    "watershed_alpha": round(watershed_alpha, 4) if watershed_alpha != -1 else None,
                    "boundary_sharpness": round(boundary_sharpness, 4),
                    "alpha_curve": alpha_results,
                }
                
                ws_str = f"α*={watershed_alpha:.3f}" if watershed_alpha != -1 else "no boundary"
                print(f"    L{li:2d}: {ws_str}, sharpness={boundary_sharpness:.4f}")
            
            all_results[pair_key] = pair_data
    
    # 5. 聚合分析
    print("\n  === Exp2 聚合分析 ===")
    
    # 按层聚合sharpness
    layer_sharpness = defaultdict(list)
    layer_watershed = defaultdict(list)
    
    for pair_key, pair_data in all_results.items():
        for layer_str, data in pair_data.items():
            if data["watershed_alpha"] is not None:
                layer_sharpness[int(layer_str)].append(data["boundary_sharpness"])
                layer_watershed[int(layer_str)].append(data["watershed_alpha"])
    
    print(f"\n  {'Layer':>6} | {'Avg Sharpness':>14} | {'Avg α*':>8} | {'N pairs':>7}")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*8}-+-{'-'*7}")
    for li in sorted(layer_sharpness.keys()):
        avg_sharp = np.mean(layer_sharpness[li])
        avg_ws = np.mean(layer_watershed[li]) if layer_watershed[li] else 0
        n = len(layer_sharpness[li])
        print(f"  L{li:>4} | {avg_sharp:>14.4f} | {avg_ws:>8.4f} | {n:>7}")
    
    shallow_sharp = [np.mean(layer_sharpness[l]) for l in range(min(6, n_layers)) if l in layer_sharpness]
    deep_sharp = [np.mean(layer_sharpness[l]) for l in range(max(0, n_layers-6), n_layers) if l in layer_sharpness]
    
    if shallow_sharp and deep_sharp:
        print(f"\n  Shallow avg sharpness: {np.mean(shallow_sharp):.4f}")
        print(f"  Deep avg sharpness: {np.mean(deep_sharp):.4f}")
        print(f"  Prediction: deep > shallow (deeper layers have sharper basins)")
    
    # 保存结果
    out_path = OUTPUT_DIR / f"{args.model}_cclxv" / "exp2_basin_boundary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "phase": "CCLXV",
        "model": args.model,
        "experiment": "exp2_basin_boundary",
        "timestamp": datetime.now().isoformat(),
        "cross_pairs": cross_pairs,
        "same_pairs": same_pairs,
        "test_layers": test_layers,
        "alphas": alphas.tolist(),
        "per_pair_data": all_results,
        "aggregation": {
            "layer_avg_sharpness": {str(k): round(np.mean(v), 4) for k, v in sorted(layer_sharpness.items())},
            "layer_avg_watershed": {str(k): round(np.mean(v), 4) for k, v in sorted(layer_watershed.items())},
            "shallow_avg_sharpness": round(np.mean(shallow_sharp), 4) if shallow_sharp else 0,
            "deep_avg_sharpness": round(np.mean(deep_sharp), 4) if deep_sharp else 0,
        },
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp3: 扰动几何衰减验证
# ============================================================
def exp3_geometric_decay(args):
    """
    核心方法: 在L0注入概念差分, 逐层追踪差分方向的对齐度和衰减
    
    如果残差流是线性吸引子:
      ||δ_l|| = ||δ_0|| * r^l  (几何衰减)
      即 log(||δ_l||) = log(||δ_0||) + l * log(r)
      如果log(r) < 0, 即r < 1 → 几何衰减
    
    同时检查差分方向保持度(delta_cos):
      如果delta_cos一直很高 → 扰动方向没变, 只是缩放 → 线性吸引子
      如果delta_cos快速下降 → 扰动方向旋转了 → 非线性吸引子
    """
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp3: 扰动几何衰减验证 ({args.model}, {n_layers}L, d={d_model})")
    print(f"  方法: 差分注入, 追踪||δ||的逐层衰减+方向保持度")
    
    template = "The {} is"
    
    # 概念对: 跨类别和同类别
    all_pairs = [
        ("dog", "apple"), ("cat", "hammer"), ("horse", "rice"),
        ("eagle", "ocean"), ("shark", "desert"), ("snake", "cheese"),
        ("dog", "cat"), ("apple", "mango"), ("hammer", "knife"),
        ("mountain", "river"),
    ]
    
    all_results = {}
    layers_list = get_layers(model)
    
    for source, target in all_pairs:
        print(f"\n  {source} → {target}")
        
        prompt_source = template.format(source)
        prompt_target = template.format(target)
        
        # 1. 收集source和target的原始残差流
        resid_source, _ = collect_full_residuals(model, tokenizer, device, prompt_source, source, n_layers)
        resid_target, _ = collect_full_residuals(model, tokenizer, device, prompt_target, target, n_layers)
        
        if 0 not in resid_source or 0 not in resid_target:
            print(f"    Skip: missing L0 residual")
            continue
        
        # 2. 计算原始差分方向(在每层)
        original_deltas = {}
        original_delta_dirs = {}
        for li in range(n_layers):
            if li in resid_source and li in resid_target:
                delta = resid_source[li] - resid_target[li]
                original_deltas[li] = delta
                delta_norm = np.linalg.norm(delta)
                if delta_norm > 1e-10:
                    original_delta_dirs[li] = delta / delta_norm
        
        # 3. 在L0注入差分: h_target_patched = h_target + delta_source_L0
        delta_L0 = resid_source[0] - resid_target[0]
        delta_L0_norm = np.linalg.norm(delta_L0)
        delta_L0_dir = delta_L0 / (delta_L0_norm + 1e-10)
        
        input_ids_target = tokenizer(prompt_target, return_tensors="pt").to(device).input_ids
        last_pos = input_ids_target.shape[1] - 1
        
        # 收集干预后每层的残差流
        perturbed_captured = {}
        
        model_dtype = next(model.parameters()).dtype
        delta_tensor = torch.tensor(delta_L0, dtype=model_dtype, device=device)
        
        def make_inject_hook():
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    out[0, last_pos] = out[0, last_pos] + delta_tensor
                    return (out,) + output[1:]
                else:
                    out = output.clone()
                    out[0, last_pos] = out[0, last_pos] + delta_tensor
                    return out
            return hook
        
        def make_capture_hook(li):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    perturbed_captured[li] = output[0][0, last_pos].detach().float().cpu().numpy()
                else:
                    perturbed_captured[li] = output[0, last_pos].detach().float().cpu().numpy()
            return hook
        
        h_inject = layers_list[0].register_forward_hook(make_inject_hook())
        hooks_cap = []
        for li in range(1, n_layers):
            hooks_cap.append(layers_list[li].register_forward_hook(make_capture_hook(li)))
        
        with torch.no_grad():
            try:
                logits = model(input_ids_target).logits
            except:
                pass
        
        h_inject.remove()
        for h in hooks_cap:
            h.remove()
        
        # 4. 核心分析: 干预残差 vs 原始target残差
        per_layer_delta_norm = {}
        per_layer_delta_cos = {}  # 干预delta方向 vs 原始L0 delta方向
        per_layer_delta_attr_cos = {}  # 干预delta方向 vs 该层原始delta方向
        
        for li in sorted(perturbed_captured.keys()):
            if li not in resid_target:
                continue
            
            # 干预后的差分 = perturbed - original_target
            current_delta = perturbed_captured[li] - resid_target[li]
            current_delta_norm = np.linalg.norm(current_delta)
            per_layer_delta_norm[li] = current_delta_norm
            
            # 方向保持度 vs L0 delta方向
            if current_delta_norm > 1e-10:
                per_layer_delta_cos[li] = proper_cos(current_delta / current_delta_norm, delta_L0_dir)
            
            # 方向保持度 vs 该层原始delta方向
            if li in original_delta_dirs and current_delta_norm > 1e-10:
                per_layer_delta_attr_cos[li] = proper_cos(current_delta / current_delta_norm, original_delta_dirs[li])
        
        # 5. 几何衰减拟合
        layers_for_fit = sorted([l for l in per_layer_delta_norm.keys() if l > 0 and per_layer_delta_norm[l] > 1e-10])
        geometric_fit = None
        if len(layers_for_fit) > 3:
            log_norms = [np.log(per_layer_delta_norm[l]) for l in layers_for_fit]
            fit = np.polyfit(layers_for_fit, log_norms, 1)
            # fit[0] = slope, fit[1] = intercept
            # slope < 0 → 几何衰减
            r_value = np.exp(fit[0])  # 每层的衰减比
            r_squared = np.corrcoef(layers_for_fit, log_norms)[0, 1] ** 2
            geometric_fit = {
                "slope": round(fit[0], 6),
                "intercept": round(fit[1], 4),
                "r_per_layer": round(r_value, 6),  # r < 1 → 衰减
                "r_squared": round(r_squared, 4),
            }
        
        result = {
            "source": source,
            "target": target,
            "delta_norm_L0": round(delta_L0_norm, 2),
            "per_layer_delta_norm": {str(k): round(v, 2) for k, v in sorted(per_layer_delta_norm.items())},
            "per_layer_delta_cos_vs_L0": {str(k): round(v, 4) for k, v in sorted(per_layer_delta_cos.items())},
            "per_layer_delta_cos_vs_original": {str(k): round(v, 4) for k, v in sorted(per_layer_delta_attr_cos.items())},
            "geometric_fit": geometric_fit,
        }
        
        all_results[f"{source}->{target}"] = result
        
        # 打印
        gf = geometric_fit
        if gf:
            print(f"    Geometric fit: r={gf['r_per_layer']:.4f}, R2={gf['r_squared']:.4f}")
            print(f"    {'r<1 → geometric decay (attractor)' if gf['r_per_layer'] < 1 else 'r>1 → expansion (NOT attractor)'}")
        print(f"    Delta cos vs L0: L1={per_layer_delta_cos.get(1, 0):.3f}, "
              f"L{n_layers//2}={per_layer_delta_cos.get(n_layers//2, 0):.3f}, "
              f"L{n_layers-1}={per_layer_delta_cos.get(n_layers-1, 0):.3f}")
    
    # 聚合
    print("\n  === Exp3 聚合分析 ===")
    
    all_r_values = []
    all_r_squared = []
    for key, data in all_results.items():
        if data["geometric_fit"]:
            all_r_values.append(data["geometric_fit"]["r_per_layer"])
            all_r_squared.append(data["geometric_fit"]["r_squared"])
    
    if all_r_values:
        avg_r = np.mean(all_r_values)
        avg_r2 = np.mean(all_r_squared)
        print(f"  Avg r_per_layer = {avg_r:.4f} (r<1 → geometric decay → SUPPORTS attractor)")
        print(f"  Avg R² = {avg_r2:.4f} (high → geometric decay is a good model)")
        
        # 检查是否所有r < 1
        all_below_1 = all(r < 1 for r in all_r_values)
        print(f"  All r < 1? {all_below_1}")
    
    # 方向保持度分析
    shallow_delta_cos = []
    deep_delta_cos = []
    for key, data in all_results.items():
        dcs = data["per_layer_delta_cos_vs_L0"]
        for l in range(1, 6):
            if str(l) in dcs:
                shallow_delta_cos.append(dcs[str(l)])
        for l in range(n_layers - 5, n_layers):
            if str(l) in dcs:
                deep_delta_cos.append(dcs[str(l)])
    
    if shallow_delta_cos and deep_delta_cos:
        print(f"\n  Delta direction alignment vs L0:")
        print(f"    Shallow(L1-L5) avg cos = {np.mean(shallow_delta_cos):.4f}")
        print(f"    Deep(L{n_layers-5}-L{n_layers-1}) avg cos = {np.mean(deep_delta_cos):.4f}")
        print(f"    If deep cos still high → perturbation direction preserved → linear attractor")
        print(f"    If deep cos drops → direction rotates → nonlinear attractor")
    
    # 保存
    out_path = OUTPUT_DIR / f"{args.model}_cclxv" / "exp3_geometric_decay.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "phase": "CCLXV",
        "model": args.model,
        "experiment": "exp3_geometric_decay",
        "timestamp": datetime.now().isoformat(),
        "pairs": all_pairs,
        "per_pair_data": all_results,
        "aggregation": {
            "avg_r_per_layer": round(np.mean(all_r_values), 4) if all_r_values else None,
            "avg_r_squared": round(np.mean(all_r_squared), 4) if all_r_squared else None,
            "all_r_below_1": all(r < 1 for r in all_r_values) if all_r_values else None,
            "shallow_delta_cos_avg": round(np.mean(shallow_delta_cos), 4) if shallow_delta_cos else None,
            "deep_delta_cos_avg": round(np.mean(deep_delta_cos), 4) if deep_delta_cos else None,
        },
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp4: 轨迹速度场 (逐层残差变化量)
# ============================================================
def exp4_trajectory_velocity(args):
    """
    核心方法: 测量残差流在每层的"速度" = ||h_{l+1} - h_l||
    
    如果深层接近不动点(吸引子):
      velocity → 0 as l → n_layers
    
    同时测量:
      velocity_direction_change = cos(h_{l+1}-h_l, h_{l+2}-h_{l+1})
      如果深层方向变化大 → 可能在极限环附近振荡
      如果深层方向一致 → 接近点吸引子
    """
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"\n  Exp4: 轨迹速度场 ({args.model}, {n_layers}L, d={d_model})")
    print(f"  方法: 逐层残差流变化量||h(l+1)-h(l)||, 检查深层是否趋于0")
    
    template = "The {} is"
    all_concepts = []
    for cat, words in CONCEPTS.items():
        all_concepts.extend(words)
    
    # 测试48个概念(4类×12)
    all_results = {}
    
    for concept in all_concepts:
        prompt = template.format(concept)
        
        # 收集所有层残差流
        residuals, _ = collect_full_residuals(model, tokenizer, device, prompt, concept, n_layers)
        
        if len(residuals) < n_layers:
            continue
        
        # 计算速度
        velocities = {}
        velocity_cos = {}  # 相邻速度方向的一致性
        
        for li in range(n_layers - 1):
            if li in residuals and li + 1 in residuals:
                delta = residuals[li + 1] - residuals[li]
                velocities[li] = float(np.linalg.norm(delta))
        
        for li in range(n_layers - 2):
            if li in velocities and li + 1 in velocities:
                d1 = residuals[li + 1] - residuals[li]
                d2 = residuals[li + 2] - residuals[li + 1]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-10 and n2 > 1e-10:
                    velocity_cos[li] = proper_cos(d1 / n1, d2 / n2)
        
        # 残差流norm本身
        resid_norms = {li: float(np.linalg.norm(residuals[li])) for li in sorted(residuals.keys())}
        
        all_results[concept] = {
            "velocities": {str(k): round(v, 2) for k, v in sorted(velocities.items())},
            "velocity_cos": {str(k): round(v, 4) for k, v in sorted(velocity_cos.items())},
            "resid_norms": {str(k): round(v, 2) for k, v in sorted(resid_norms.items())},
        }
    
    # 聚合分析
    print("\n  === Exp4 聚合分析 ===")
    
    # 速度的层平均
    layer_velocities = defaultdict(list)
    layer_velocity_cos = defaultdict(list)
    layer_norms = defaultdict(list)
    
    for concept, data in all_results.items():
        for layer_str, vel in data["velocities"].items():
            layer_velocities[int(layer_str)].append(vel)
        for layer_str, cos in data["velocity_cos"].items():
            layer_velocity_cos[int(layer_str)].append(cos)
        for layer_str, norm in data["resid_norms"].items():
            layer_norms[int(layer_str)].append(norm)
    
    # 打印关键层
    print(f"\n  {'Layer':>6} | {'Avg Velocity':>12} | {'Avg Resid Norm':>14} | {'Vel/Norm':>8} | {'Avg Vel Cos':>11}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*14}-+-{'-'*8}-+-{'-'*11}")
    
    key_layers = [0, 1, 2, 5, 10, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-5, n_layers-3, n_layers-2, n_layers-1]
    key_layers = sorted(set([l for l in key_layers if 0 <= l < n_layers]))
    
    for li in key_layers:
        avg_vel = np.mean(layer_velocities[li]) if li in layer_velocities else 0
        avg_norm = np.mean(layer_norms[li]) if li in layer_norms else 0
        vel_norm_ratio = avg_vel / avg_norm if avg_norm > 1e-10 else 0
        avg_cos = np.mean(layer_velocity_cos[li]) if li in layer_velocity_cos else 0
        print(f"  L{li:>4} | {avg_vel:>12.2f} | {avg_norm:>14.2f} | {vel_norm_ratio:>8.4f} | {avg_cos:>11.4f}")
    
    # 关键判断: 深层速度是否→0
    shallow_vel = [np.mean(layer_velocities[l]) for l in range(min(6, n_layers-1)) if l in layer_velocities]
    deep_vel = [np.mean(layer_velocities[l]) for l in range(max(0, n_layers-6), n_layers-1) if l in layer_velocities]
    
    shallow_norm = [np.mean(layer_norms[l]) for l in range(min(6, n_layers)) if l in layer_norms]
    deep_norm = [np.mean(layer_norms[l]) for l in range(max(0, n_layers-6), n_layers) if l in layer_norms]
    
    avg_shallow_vel = np.mean(shallow_vel) if shallow_vel else 0
    avg_deep_vel = np.mean(deep_vel) if deep_vel else 0
    avg_shallow_norm = np.mean(shallow_norm) if shallow_norm else 0
    avg_deep_norm = np.mean(deep_norm) if deep_norm else 0
    
    # 归一化速度 (velocity / norm)
    shallow_vel_ratio = avg_shallow_vel / avg_shallow_norm if avg_shallow_norm > 0 else 0
    deep_vel_ratio = avg_deep_vel / avg_deep_norm if avg_deep_norm > 0 else 0
    
    print(f"\n  Shallow (L0-L5): avg velocity={avg_shallow_vel:.2f}, rel_velocity={shallow_vel_ratio:.4f}")
    print(f"  Deep (L{n_layers-6}-L{n_layers-1}): avg velocity={avg_deep_vel:.2f}, rel_velocity={deep_vel_ratio:.4f}")
    print(f"\n  Attractor prediction:")
    print(f"    Deep velocity < Shallow velocity → SUPPORTS approach to fixed point")
    print(f"    Deep velocity ≈ Shallow velocity → NOT approaching fixed point")
    print(f"    Deep velocity > Shallow velocity → REJECTS attractor (expanding)")
    
    # 速度衰减拟合
    layers_for_fit = sorted(layer_velocities.keys())
    if len(layers_for_fit) > 3:
        avg_v = [np.mean(layer_velocities[l]) for l in layers_for_fit]
        # 归一化
        avg_n = [np.mean(layer_norms[l]) for l in layers_for_fit]
        rel_v = [v/n if n > 1e-10 else 0 for v, n in zip(avg_v, avg_n)]
        
        log_rel_v = [np.log(max(rv, 1e-10)) for rv in rel_v]
        fit = np.polyfit(layers_for_fit, log_rel_v, 1)
        print(f"\n  Relative velocity decay fit: slope = {fit[0]:.6f}")
        print(f"    slope < 0 → velocity decreasing toward fixed point → SUPPORTS attractor")
    
    # 保存
    out_path = OUTPUT_DIR / f"{args.model}_cclxv" / "exp4_trajectory_velocity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "phase": "CCLXV",
        "model": args.model,
        "experiment": "exp4_trajectory_velocity",
        "timestamp": datetime.now().isoformat(),
        "n_concepts": len(all_results),
        "per_concept_data": {k: v for k, v in list(all_results.items())[:5]},  # 只保存前5个概念
        "aggregation": {
            "layer_avg_velocity": {str(k): round(np.mean(v), 2) for k, v in sorted(layer_velocities.items())},
            "layer_avg_norm": {str(k): round(np.mean(v), 2) for k, v in sorted(layer_norms.items())},
            "layer_avg_velocity_cos": {str(k): round(np.mean(v), 4) for k, v in sorted(layer_velocity_cos.items())},
            "shallow_avg_velocity": round(avg_shallow_vel, 2),
            "deep_avg_velocity": round(avg_deep_vel, 2),
            "shallow_rel_velocity": round(shallow_vel_ratio, 4),
            "deep_rel_velocity": round(deep_vel_ratio, 4),
        },
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase CCLXV: Attractor Hypothesis Verification")
    parser.add_argument("--model", type=str, default="qwen3",
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="1",
                        choices=["1", "2", "3", "4", "all"])
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Phase CCLXV: 吸引子假设验证")
    print(f"Model: {args.model}")
    print(f"{'='*60}")
    
    results = {}
    
    if args.exp in ["1", "all"]:
        print(f"\n{'='*60}")
        print(f"  Exp1: 局部收缩率 (Jacobian谱范数估计)")
        print(f"{'='*60}")
        r = exp1_contraction_ratio(args)
        results["exp1"] = {
            "avg_r_per_layer": r.get("aggregation", {}).get("0", {}).get("avg_single_step_by_layer", {}),
            "shallow_ss_avg": r.get("aggregation", {}).get("0", {}).get("shallow_single_step_avg"),
            "deep_ss_avg": r.get("aggregation", {}).get("0", {}).get("deep_single_step_avg"),
        }
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if args.exp in ["2", "all"]:
        print(f"\n{'='*60}")
        print(f"  Exp2: 盆地边界测绘")
        print(f"{'='*60}")
        r = exp2_basin_boundary(args)
        results["exp2"] = {
            "shallow_sharpness": r.get("aggregation", {}).get("shallow_avg_sharpness"),
            "deep_sharpness": r.get("aggregation", {}).get("deep_avg_sharpness"),
        }
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if args.exp in ["3", "all"]:
        print(f"\n{'='*60}")
        print(f"  Exp3: 扰动几何衰减验证")
        print(f"{'='*60}")
        r = exp3_geometric_decay(args)
        results["exp3"] = {
            "avg_r_per_layer": r.get("aggregation", {}).get("avg_r_per_layer"),
            "avg_r_squared": r.get("aggregation", {}).get("avg_r_squared"),
            "all_r_below_1": r.get("aggregation", {}).get("all_r_below_1"),
        }
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if args.exp in ["4", "all"]:
        print(f"\n{'='*60}")
        print(f"  Exp4: 轨迹速度场")
        print(f"{'='*60}")
        r = exp4_trajectory_velocity(args)
        results["exp4"] = {
            "shallow_rel_velocity": r.get("aggregation", {}).get("shallow_rel_velocity"),
            "deep_rel_velocity": r.get("aggregation", {}).get("deep_rel_velocity"),
        }
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 最终判定
    print(f"\n{'='*60}")
    print(f"  CCLXV 吸引子假设最终判定 ({args.model})")
    print(f"{'='*60}")
    
    verdict = "INCONCLUSIVE"
    evidence_for = 0
    evidence_against = 0
    
    # Exp1判定: 深层单步收缩率 < 1?
    if "exp1" in results:
        deep_ss = results["exp1"].get("deep_ss_avg")
        shallow_ss = results["exp1"].get("shallow_ss_avg")
        if deep_ss is not None and shallow_ss is not None:
            if deep_ss < 1.0:
                evidence_for += 1
                print(f"  Exp1: deep single_step_ratio={deep_ss:.4f} < 1.0 → SUPPORTS attractor ✓")
            else:
                evidence_against += 1
                print(f"  Exp1: deep single_step_ratio={deep_ss:.4f} >= 1.0 → REJECTS attractor ✗")
    
    # Exp2判定: 深层sharpness > 浅层?
    if "exp2" in results:
        deep_sharp = results["exp2"].get("deep_sharpness")
        shallow_sharp = results["exp2"].get("shallow_sharpness")
        if deep_sharp is not None and shallow_sharp is not None:
            if deep_sharp > shallow_sharp:
                evidence_for += 1
                print(f"  Exp2: deep sharpness={deep_sharp:.4f} > shallow={shallow_sharp:.4f} → SUPPORTS attractor ✓")
            else:
                evidence_against += 1
                print(f"  Exp2: deep sharpness={deep_sharp:.4f} <= shallow={shallow_sharp:.4f} → REJECTS attractor ✗")
    
    # Exp3判定: r_per_layer < 1?
    if "exp3" in results:
        avg_r = results["exp3"].get("avg_r_per_layer")
        all_below = results["exp3"].get("all_r_below_1")
        if avg_r is not None:
            if avg_r < 1.0:
                evidence_for += 1
                print(f"  Exp3: avg r_per_layer={avg_r:.4f} < 1.0 → geometric decay → SUPPORTS attractor ✓")
            else:
                evidence_against += 1
                print(f"  Exp3: avg r_per_layer={avg_r:.4f} >= 1.0 → NOT geometric decay → REJECTS attractor ✗")
    
    # Exp4判定: 深层速度 < 浅层?
    if "exp4" in results:
        deep_vel = results["exp4"].get("deep_rel_velocity")
        shallow_vel = results["exp4"].get("shallow_rel_velocity")
        if deep_vel is not None and shallow_vel is not None:
            if deep_vel < shallow_vel:
                evidence_for += 1
                print(f"  Exp4: deep rel_velocity={deep_vel:.4f} < shallow={shallow_vel:.4f} → approaching fixed point ✓")
            else:
                evidence_against += 1
                print(f"  Exp4: deep rel_velocity={deep_vel:.4f} >= shallow={shallow_vel:.4f} → NOT approaching fixed point ✗")
    
    # 综合判定
    if evidence_for >= 3:
        verdict = "STRONGLY SUPPORTED"
    elif evidence_for >= 2:
        verdict = "SUPPORTED"
    elif evidence_for == evidence_against:
        verdict = "MIXED EVIDENCE"
    elif evidence_against >= 3:
        verdict = "STRONGLY REJECTED"
    elif evidence_against >= 2:
        verdict = "REJECTED"
    
    print(f"\n  Evidence FOR attractor: {evidence_for}/4")
    print(f"  Evidence AGAINST attractor: {evidence_against}/4")
    print(f"\n  >>> VERDICT: {verdict} <<<")
    
    # 保存判定
    verdict_path = OUTPUT_DIR / f"{args.model}_cclxv" / "verdict.json"
    with open(verdict_path, 'w', encoding='utf-8') as f:
        json.dump({
            "phase": "CCLXV",
            "model": args.model,
            "verdict": verdict,
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"\n  Verdict saved to {verdict_path}")


if __name__ == "__main__":
    main()
