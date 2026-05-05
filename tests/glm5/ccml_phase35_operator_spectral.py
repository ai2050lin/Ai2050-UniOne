"""
CCML(Phase 35): 算子谱分析 — 从几何直觉到算子谱
====================================================

Phase 34批评分析 (3条全部正确):

  ❌ 硬伤1: CV小≠Jacobian是常数 ✔正确
     证据: amp_ratio(eps=1.0/eps=0.1)=0.45, 远离1.0!
     → 在更大扰动尺度下, Jacobian显著非线性
     → 只能说"在当前扰动尺度下局部近似线性", 不是全局线性
     → 需要测: 不同数据分布+不同扰动尺度下的Jacobian

  ❌ 硬伤2: 方向旋转≠没有语义方向 ✔正确
     如果函数g(h)在"补偿"方向旋转, 语义可以保持稳定
     类比: 向量旋转+分类器同步旋转 → 分类结果不变
     正确结论: "没有跨层不变的线性方向" (≠ "没有语义结构")
     → 需要测: 语义函数g(h_l)是否跨层守恒

  ❌ 硬伤3: lm_head方向强是trivial的 ✔正确
     logit = W_U · h → 沿W_U^T方向扰动当然增加logit (定义成立)
     这不是"发现因果机制", 只是确认计算图
     真正的问题: 为什么早期层注入的扰动, 经过几十层后仍然对W_U有投影?
     → 需要测: 扰动在W_U子空间中的投影如何随层变化

Phase 35核心任务:
  35A: ★★★★★★★★★ Jacobian SVD — 哪些方向被放大/压缩? 有效秩?
    → 直接计算每层Jacobian的SVD
    → J = U Σ V^T: V=输入基, Σ=放大/压缩, U=输出基
    → 关键: 哪些输入方向被保留+放大+对齐到W_U

  35B: ★★★★★★★ W_U子空间对齐 — 扰动如何变成"可读信号"?
    → 测量扰动delta在W_U行空间中的投影比, 随层变化
    → 如果投影比不降(甚至升) → Jacobian链在"保护"W_U可读子空间
    → 如果投影比降 → Jacobian链是"随机旋转"(信号被稀释)

  35C: ★★★★★★★ 语义守恒测试 — g(h_l)是否跨层一致?
    → 在每层训练probe, 然后测试: probe_l(h_{l+k})是否≈probe_l(h_l)?
    → 如果语义守恒: 即使方向旋转, 函数值保持 → 存在某种不变量
    → 如果语义不守恒: 语义在逐层变化/细化 → 信息在重分配

关键数学框架:
  Jacobian SVD: J_l = U_l Σ_l V_l^T
    - V_l: 输入方向基 (哪些输入方向被"读取")
    - Σ_l: 奇异值谱 (每个方向被放大/压缩多少)
    - U_l: 输出方向基 (信号被映射到哪里)
    
  W_U对齐: proj_ratio(l) = ||Proj_{W_U}(delta_l)||^2 / ||delta_l||^2
    - 如果proj_ratio随层上升 → Jacobian链在"对齐"到W_U
    - 如果proj_ratio随层下降 → 信号在W_U不可读子空间中"泄漏"

  语义守恒: R²(l, l+k) = corr(probe_l(h_l), probe_l(h_{l+k}))
    - 如果R²≈1 → 语义是跨层不变量
    - 如果R²<1 → 语义在逐层变化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from collections import defaultdict
import time

from model_utils import (load_model, get_layers, get_model_info, release_model,
                         safe_decode, MODEL_CONFIGS, get_W_U)


# ============================================================================
# 数据定义 (同Phase 34, 但增加更多概念)
# ============================================================================

CONCEPT_DATASET = {
    "apple":      {"edible":1, "animacy":0, "size":0.5},
    "orange":     {"edible":1, "animacy":0, "size":0.5},
    "banana":     {"edible":1, "animacy":0, "size":0.5},
    "strawberry": {"edible":1, "animacy":0, "size":0.2},
    "grape":      {"edible":1, "animacy":0, "size":0.2},
    "cherry":     {"edible":1, "animacy":0, "size":0.2},
    "lemon":      {"edible":1, "animacy":0, "size":0.3},
    "mango":      {"edible":1, "animacy":0, "size":0.5},
    "peach":      {"edible":1, "animacy":0, "size":0.4},
    "pear":       {"edible":1, "animacy":0, "size":0.5},
    "watermelon": {"edible":1, "animacy":0, "size":0.9},
    "pineapple":  {"edible":1, "animacy":0, "size":0.7},
    "blueberry":  {"edible":1, "animacy":0, "size":0.1},
    "coconut":    {"edible":1, "animacy":0, "size":0.6},
    "tomato":     {"edible":1, "animacy":0, "size":0.3},
    "kiwi":       {"edible":1, "animacy":0, "size":0.2},
    "plum":       {"edible":1, "animacy":0, "size":0.3},
    "fig":        {"edible":1, "animacy":0, "size":0.2},
    "lime":       {"edible":1, "animacy":0, "size":0.2},
    "melon":      {"edible":1, "animacy":0, "size":0.7},
    "dog":        {"edible":0, "animacy":1, "size":0.5},
    "cat":        {"edible":0, "animacy":1, "size":0.3},
    "elephant":   {"edible":0, "animacy":1, "size":1.0},
    "eagle":      {"edible":0, "animacy":1, "size":0.5},
    "salmon":     {"edible":1, "animacy":1, "size":0.5},
    "horse":      {"edible":0, "animacy":1, "size":0.8},
    "cow":        {"edible":1, "animacy":1, "size":0.8},
    "pig":        {"edible":1, "animacy":1, "size":0.6},
    "bird":       {"edible":0, "animacy":1, "size":0.2},
    "fish":       {"edible":1, "animacy":1, "size":0.3},
    "snake":      {"edible":0, "animacy":1, "size":0.5},
    "frog":       {"edible":0, "animacy":1, "size":0.2},
    "bee":        {"edible":0, "animacy":1, "size":0.1},
    "ant":        {"edible":0, "animacy":1, "size":0.05},
    "bear":       {"edible":0, "animacy":1, "size":0.9},
    "rabbit":     {"edible":0, "animacy":1, "size":0.3},
    "deer":       {"edible":0, "animacy":1, "size":0.7},
    "whale":      {"edible":0, "animacy":1, "size":1.0},
    "chicken":    {"edible":1, "animacy":1, "size":0.3},
    "shark":      {"edible":0, "animacy":1, "size":0.8},
    "hammer":     {"edible":0, "animacy":0, "size":0.5},
    "knife":      {"edible":0, "animacy":0, "size":0.3},
    "chair":      {"edible":0, "animacy":0, "size":0.6},
    "shirt":      {"edible":0, "animacy":0, "size":0.4},
    "car":        {"edible":0, "animacy":0, "size":1.0},
    "book":       {"edible":0, "animacy":0, "size":0.4},
    "shoe":       {"edible":0, "animacy":0, "size":0.3},
    "ball":       {"edible":0, "animacy":0, "size":0.3},
    "cup":        {"edible":0, "animacy":0, "size":0.2},
    "pen":        {"edible":0, "animacy":0, "size":0.2},
    "table":      {"edible":0, "animacy":0, "size":0.7},
    "door":       {"edible":0, "animacy":0, "size":0.8},
    "rock":       {"edible":0, "animacy":0, "size":0.5},
    "key":        {"edible":0, "animacy":0, "size":0.1},
    "plate":      {"edible":0, "animacy":0, "size":0.4},
    "bottle":     {"edible":0, "animacy":0, "size":0.3},
    "clock":      {"edible":0, "animacy":0, "size":0.3},
    "lamp":       {"edible":0, "animacy":0, "size":0.4},
    "tree":       {"edible":0, "animacy":0, "size":1.0},
    "flower":     {"edible":0, "animacy":0, "size":0.2},
    "cloud":      {"edible":0, "animacy":0, "size":1.0},
    "water":      {"edible":1, "animacy":0, "size":0.5},
    "fire":       {"edible":0, "animacy":0, "size":0.5},
    "grass":      {"edible":0, "animacy":0, "size":0.3},
    "sand":       {"edible":0, "animacy":0, "size":0.3},
    "snow":       {"edible":0, "animacy":0, "size":0.5},
    "mountain":   {"edible":0, "animacy":0, "size":1.0},
    "river":      {"edible":0, "animacy":0, "size":1.0},
    "ocean":      {"edible":0, "animacy":0, "size":1.0},
    "sun":        {"edible":0, "animacy":0, "size":1.0},
    "moon":       {"edible":0, "animacy":0, "size":1.0},
}

ATTR_NAMES = ["edible", "animacy", "size"]

CONTEXT_TEMPLATES = [
    "The {word} is here",
    "I see a {word}",
    "The {word} was found",
    "Look at that {word}",
    "A {word} appeared",
    "This {word} looks nice",
    "Every {word} has features",
]

TEST_CONCEPTS = ["apple", "dog", "hammer", "salmon", "water", "tree"]


def find_token_index(tokens, target_word):
    target_lower = target_word.lower().strip()
    for i, t in enumerate(tokens):
        if t.lower().strip() == target_lower:
            return i
    for i, t in enumerate(tokens):
        if t.lower().strip()[:3] == target_lower[:3]:
            return i
    for i, t in enumerate(tokens):
        if t.lower().strip()[:2] == target_lower[:2]:
            return i
    return -1


# ============================================================================
# 核心函数
# ============================================================================

def collect_hs_at_layer(model, tokenizer, device, word, template, layer_idx):
    """收集指定层的hidden state"""
    layers = get_layers(model)
    sent = template.format(word=word)
    toks = tokenizer(sent, return_tensors="pt").to(device)
    tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
    dep_idx = find_token_index(tokens_list, word)
    if dep_idx < 0:
        return None, dep_idx

    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float().cpu().numpy()
        else:
            captured['h'] = output.detach().float().cpu().numpy()
    
    h_handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**toks)
    h_handle.remove()

    if 'h' in captured:
        return captured['h'][0, dep_idx, :], dep_idx
    return None, dep_idx


def collect_all_layer_hs(model, tokenizer, device, word, template, n_layers):
    """收集ALL层的hidden states"""
    layers = get_layers(model)
    sent = template.format(word=word)
    toks = tokenizer(sent, return_tensors="pt").to(device)
    tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
    dep_idx = find_token_index(tokens_list, word)
    if dep_idx < 0:
        return None, dep_idx

    captured = {}
    def make_hook(li):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu().numpy()
            else:
                captured[li] = output.detach().float().cpu().numpy()
        return hook

    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        _ = model(**toks)

    for h in hooks:
        h.remove()

    result = {}
    for li in range(n_layers):
        if li in captured:
            result[li] = captured[li][0, dep_idx, :]
    return result, dep_idx


def inject_and_collect(model, tokenizer, device, word, template,
                      source_layer, direction, epsilon, n_layers, dep_idx_cache=None):
    """在source_layer注入扰动, 收集ALL后续层的hidden states和logits"""
    layers = get_layers(model)
    sent = template.format(word=word)
    toks = tokenizer(sent, return_tensors="pt").to(device)
    tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
    dep_idx = find_token_index(tokens_list, word)
    if dep_idx < 0:
        return None, None, dep_idx

    captured = {}

    def make_hook(li):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu().numpy()
            else:
                captured[li] = output.detach().float().cpu().numpy()
        return hook

    def make_inject_hook(li, dir_np, eps):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0].clone()
            else:
                out = output.clone()
            delta = torch.tensor(eps * dir_np, dtype=out.dtype, device=device)
            out[0, dep_idx, :] += delta
            captured[li] = out.detach().float().cpu().numpy()
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
        return hook

    hooks = []
    for li in range(n_layers):
        if li == source_layer:
            hooks.append(layers[li].register_forward_hook(
                make_inject_hook(li, direction, epsilon)))
        else:
            hooks.append(layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        outputs = model(**toks)
        logits = outputs.logits[0, dep_idx, :].detach().float().cpu().numpy()

    for h in hooks:
        h.remove()

    result = {}
    for li in range(n_layers):
        if li in captured:
            result[li] = captured[li][0, dep_idx, :]
    return result, logits, dep_idx


# ============================================================================
# 35A: Jacobian SVD — 隐式Jacobian的奇异值谱
# ============================================================================

def expA_jacobian_svd(model_name, model, tokenizer, device):
    """
    35A: 通过有限差分隐式计算Jacobian, 然后做SVD
    
    方法: 
    1. 在源层l, 用d_model个正交方向注入微小扰动
    2. 测量在目标层l+1产生的变化
    3. 这些变化组成Jacobian的列 → 可以做SVD
    
    但d_model太大(2560-4096), 不可能逐列测量
    → 替代方案: 用随机投影 + Johnson-Lindenstrauss
    - 注入k个随机方向(k~200), 测量目标变化
    - 组成k×k的"有效Jacobian"子矩阵
    - SVD给出有效秩和谱
    
    ★★★ 更高效的方法: 直接测少量关键方向 ★★★
    - 测probe方向, W_U行方向, random方向
    - 看这些方向在下一层的放大倍数和旋转角度
    - 再加一个"最放大方向"和"最压缩方向"(从权重矩阵推导)
    """
    print(f"\n{'='*70}")
    print(f"35A: Jacobian SVD — 隐式Jacobian的奇异值谱")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # 采样层 (相邻层对)
    if n_layers <= 12:
        sample_layers = list(range(0, n_layers - 1))
    else:
        step = max(1, (n_layers - 1) // 9)
        sample_layers = sorted(set(list(range(0, n_layers - 1, step)) + [n_layers - 2]))
    
    print(f"模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")
    print(f"采样层: {sample_layers}")

    # ===== Step 1: 用随机方向测量有效Jacobian谱 =====
    # 对于每对相邻层(l, l+1), 注入k个随机方向, 测量下一层的变化
    # 这给出J_l的k个列采样 → 可以估算奇异值谱

    n_random_dirs = 150  # 随机方向数
    eps = 0.01  # 小扰动, 确保在线性区域
    
    # 选3个不同概念, 测Jacobian的input-dependence
    concepts_for_jacobian = ["apple", "dog", "hammer"]
    
    results = {
        "model": model_name, "exp": "A",
        "experiment": "jacobian_svd",
        "n_layers": n_layers, "d_model": d_model,
        "sample_layers": sample_layers,
        "n_random_dirs": n_random_dirs,
        "eps": eps,
        "svd_spectra": {},  # {layer: {concept: {sv: [...]}}}
        "effective_ranks": {},  # {layer: {concept: rank}}
        "jacobian_input_dependence": {},  # {layer: {metric: value}}
    }

    for li in sample_layers:
        results["svd_spectra"][str(li)] = {}
        results["effective_ranks"][str(li)] = {}
        
        layer_jacobians = {}  # {concept: jacobian_sample [k, d_model]}
        
        for concept in concepts_for_jacobian:
            print(f"\n  L{li} → L{li+1}, concept={concept}:")
            
            # 收集baseline
            baseline_hs, _ = collect_all_layer_hs(
                model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
            if baseline_hs is None or li not in baseline_hs or (li+1) not in baseline_hs:
                print(f"    baseline收集失败, 跳过")
                continue
            
            h_l_base = baseline_hs[li]
            h_l1_base = baseline_hs[li + 1]
            
            # 注入随机方向, 测量下一层变化
            np.random.seed(42)
            delta_matrix = np.random.randn(n_random_dirs, d_model).astype(np.float32)
            # 归一化每行
            for i in range(n_random_dirs):
                delta_matrix[i] /= np.linalg.norm(delta_matrix[i])
            
            # 逐个注入 (太慢!) → 批量: 用分组平均
            # 策略: 分10组, 每组15个方向, 用平均方向注入
            # → 这给出了Jacobian对随机方向的平均响应
            
            # 更好: 直接注入前50个随机方向 (GPU时间允许)
            n_to_inject = min(50, n_random_dirs)
            delta_at_next = np.zeros((n_to_inject, d_model), dtype=np.float32)
            amp_values = []
            
            for i in range(n_to_inject):
                dir_vec = delta_matrix[i]
                actual_eps = eps * np.linalg.norm(h_l_base)
                
                # 在层l注入, 收集层l+1
                perturbed_hs, _, _ = inject_and_collect(
                    model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    li, dir_vec, actual_eps, n_layers)
                
                if perturbed_hs is not None and (li + 1) in perturbed_hs:
                    delta_next = perturbed_hs[li + 1] - h_l1_base
                    delta_at_next[i] = delta_next
                    amp_values.append(np.linalg.norm(delta_next) / max(actual_eps, 1e-10))
            
            if len(amp_values) < 10:
                print(f"    有效注入数不足 ({len(amp_values)}), 跳过")
                continue
            
            # 用delta_at_next做SVD: [n_to_inject, d_model]
            # 这是一个"Jacobian的行采样" (J·V_i ≈ delta_at_next[i])
            # SVD: delta_at_next = U S Vt
            # → S的奇异值反映了Jacobian的有效放大谱
            try:
                U, S, Vt = np.linalg.svd(delta_at_next, full_matrices=False)
                
                # 有效秩: 奇异值 > max(S) * 0.01 的个数
                effective_rank = int(np.sum(S > max(S) * 0.01))
                # 另一个有效秩: 90%能量
                total_energy = np.sum(S**2)
                cum_energy = np.cumsum(S**2)
                rank_90 = int(np.searchsorted(cum_energy, 0.9 * total_energy)) + 1
                
                results["svd_spectra"][str(li)][concept] = {
                    "top_20_sv": [float(s) for s in S[:20]],
                    "bottom_5_sv": [float(s) for s in S[-5:]] if len(S) >= 5 else [float(s) for s in S],
                    "effective_rank": effective_rank,
                    "rank_90": rank_90,
                    "condition_number": float(S[0] / max(S[-1], 1e-10)),
                    "mean_amplification": float(np.mean(amp_values)),
                    "std_amplification": float(np.std(amp_values)),
                    "cv_amplification": float(np.std(amp_values) / max(np.mean(amp_values), 1e-10)),
                }
                results["effective_ranks"][str(li)][concept] = {
                    "effective_rank": effective_rank,
                    "rank_90": rank_90,
                }
                
                print(f"    effective_rank={effective_rank}, rank_90={rank_90}, "
                      f"cond={S[0]/max(S[-1],1e-10):.1f}, "
                      f"mean_amp={np.mean(amp_values):.3f}, CV={np.std(amp_values)/max(np.mean(amp_values),1e-10):.3f}")
                print(f"    top5 sv: {[f'{s:.2f}' for s in S[:5]]}")
                
            except Exception as e:
                print(f"    SVD失败: {e}")
        
        # Step 1.5: Jacobian的input-dependence
        # 比较不同概念的有效秩和谱
        if len(results["svd_spectra"][str(li)]) >= 2:
            concepts_done = list(results["svd_spectra"][str(li)].keys())
            ranks = [results["svd_spectra"][str(li)][c]["effective_rank"] for c in concepts_done]
            rank_cv = np.std(ranks) / max(np.mean(ranks), 1e-10) if len(ranks) > 1 else 0
            
            # 比较top奇异值
            top_svs = [results["svd_spectra"][str(li)][c]["top_20_sv"][:5] for c in concepts_done]
            top_sv_cv = np.std([sv[0] for sv in top_svs]) / max(np.mean([sv[0] for sv in top_svs]), 1e-10)
            
            results["jacobian_input_dependence"][str(li)] = {
                "rank_cv": float(rank_cv),
                "top_sv_cv": float(top_sv_cv),
                "ranks": {c: results["svd_spectra"][str(li)][c]["effective_rank"] for c in concepts_done},
            }
            print(f"    ★ input-dependence: rank_CV={rank_cv:.3f}, top_sv_CV={top_sv_cv:.3f}")

    # ===== Step 2: 特定方向的放大谱 =====
    # 对probe方向和W_U方向, 测量它们在每层Jacobian中的放大倍数
    print(f"\n--- Step 2: 特定方向的Jacobian放大谱 ---")
    
    # 先获取W_U
    W_U = get_W_U(model)  # [vocab_size, d_model]
    
    attr_token_words = {
        "edible": ["edible", "food", "eat", "delicious"],
        "animacy": ["alive", "living", "animate", "animal"],
    }
    attr_token_ids = {}
    lm_head_dirs = {}
    for attr, words in attr_token_words.items():
        ids = []
        for w in words:
            ids.extend(tokenizer.encode(w, add_special_tokens=False))
        ids = list(set(ids))
        attr_token_ids[attr] = ids
        if len(ids) > 0:
            dir_vec = np.mean(W_U[ids], axis=0)
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-10:
                lm_head_dirs[attr] = dir_vec / norm
    
    # 在所有采样层训练probe
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    probe_weights = {}
    for li in sample_layers[:5]:  # 只在部分层训练, 节省时间
        all_hs = []
        all_words = []
        layers_list = get_layers(model)
        for template in CONTEXT_TEMPLATES[:3]:  # 减少模板数
            for word in concepts[:30]:  # 减少概念数
                sent = template.format(word=word)
                toks = tokenizer(sent, return_tensors="pt").to(device)
                tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
                dep_idx = find_token_index(tokens_list, word)
                if dep_idx < 0:
                    continue
                captured = {}
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        captured['h'] = output[0].detach().float().cpu().numpy()
                    else:
                        captured['h'] = output.detach().float().cpu().numpy()
                h_handle = layers_list[li].register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(**toks)
                h_handle.remove()
                if 'h' in captured:
                    all_hs.append(captured['h'][0, dep_idx, :])
                    all_words.append(word)
        
        if len(all_hs) < 10:
            continue
        
        H = np.array(all_hs)
        valid_V = np.array([[CONCEPT_DATASET[w][attr] for attr in ATTR_NAMES] for w in all_words])
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(valid_V, axis=0, keepdims=True)
        V_centered = valid_V - V_mean
        
        layer_probes = {}
        for ai, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, ai])
            layer_probes[attr] = ridge.coef_.copy()
            norm = np.linalg.norm(layer_probes[attr])
            if norm > 1e-10:
                layer_probes[attr] = layer_probes[attr] / norm
        
        probe_weights[li] = layer_probes
    
    # 测量特定方向在相邻层的放大倍数
    direction_amplification = {}  # {layer: {concept: {dir_name: amp}}}
    
    for li in sample_layers:
        direction_amplification[str(li)] = {}
        
        # 准备方向: probe + lm_head + random
        directions = {}
        if li in probe_weights:
            directions["probe_edible"] = probe_weights[li].get("edible")
            directions["probe_animacy"] = probe_weights[li].get("animacy")
        directions["lm_head_edible"] = lm_head_dirs.get("edible")
        directions["lm_head_animacy"] = lm_head_dirs.get("animacy")
        np.random.seed(42)
        directions["random1"] = np.random.randn(d_model)
        directions["random1"] /= np.linalg.norm(directions["random1"])
        np.random.seed(123)
        directions["random2"] = np.random.randn(d_model)
        directions["random2"] /= np.linalg.norm(directions["random2"])
        
        # 过滤无效方向
        directions = {k: v for k, v in directions.items() if v is not None and np.linalg.norm(v) > 1e-10}
        
        for concept in ["apple", "dog", "hammer"]:
            baseline_hs, _ = collect_all_layer_hs(
                model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
            if baseline_hs is None or li not in baseline_hs or (li+1) not in baseline_hs:
                continue
            
            h_l1_base = baseline_hs[li + 1]
            h_scale = float(np.linalg.norm(baseline_hs[li]))
            actual_eps = 0.01 * h_scale  # 小扰动
            
            direction_amplification[str(li)][concept] = {}
            
            for dir_name, dir_vec in directions.items():
                # 注入并收集
                perturbed_hs, _, _ = inject_and_collect(
                    model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    li, dir_vec, actual_eps, n_layers)
                
                if perturbed_hs is not None and (li + 1) in perturbed_hs:
                    delta_next = perturbed_hs[li + 1] - h_l1_base
                    amp = float(np.linalg.norm(delta_next) / max(actual_eps, 1e-10))
                    cos = float(np.dot(delta_next, dir_vec) / 
                               (np.linalg.norm(delta_next) * max(np.linalg.norm(dir_vec), 1e-10)))
                    direction_amplification[str(li)][concept][dir_name] = {
                        "amplification": amp,
                        "cos_alignment": cos,
                    }
    
    results["direction_amplification"] = direction_amplification
    
    # 打印摘要
    print(f"\n  ★ 特定方向放大谱摘要 ★")
    for li_str in sorted(direction_amplification.keys(), key=int):
        li = int(li_str)
        print(f"  L{li} → L{li+1}:")
        for concept in ["apple", "dog", "hammer"]:
            if concept in direction_amplification[li_str]:
                dirs = direction_amplification[li_str][concept]
                dir_summary = ", ".join(f"{k}={v['amplification']:.3f}" for k, v in dirs.items())
                print(f"    {concept}: {dir_summary}")

    return results


# ============================================================================
# 35B: W_U子空间对齐 — 扰动如何变成"可读信号"
# ============================================================================

def expB_wu_alignment(model_name, model, tokenizer, device):
    """
    35B: 测量扰动delta在W_U行空间中的投影比, 随层变化
    
    核心问题: 
    - 早期层注入的扰动, 经过几十层后, 有多少投影落在W_U的行空间?
    - 如果投影比不降 → Jacobian链在"保护"W_U可读子空间 (信息路由!)
    - 如果投影比降 → 信号在W_U不可读子空间中"泄漏"
    
    方法:
    1. 在源层l注入扰动δ
    2. 在目标层l+k测量δ_k = h_{l+k}(perturbed) - h_{l+k}(baseline)
    3. 计算δ_k在W_U行空间中的投影比
    4. 看投影比如何随k变化
    """
    print(f"\n{'='*70}")
    print(f"35B: W_U子空间对齐 — 扰动如何变成可读信号")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # 预计算W_U的SVD (只取top-k, 节省内存)
    W_U = get_W_U(model)  # [vocab_size, d_model]
    
    # W_U^T的SVD: [d_model, vocab] → U [d_model, k] S [k]
    # U的列 = W_U行空间的基
    from scipy.sparse.linalg import svds
    k_svd = min(200, min(d_model, W_U.shape[0]) - 2)
    k_svd = max(k_svd, 10)
    print(f"  计算W_U^T SVD (k={k_svd})...")
    W_U_T = W_U.T.astype(np.float32)
    U_wut, s_wut, _ = svds(W_U_T, k=k_svd)
    U_wut = np.asarray(U_wut, dtype=np.float64)  # [d_model, k_svd]
    print(f"  W_U SVD done. Top singular values: {s_wut[:5]}")
    
    # W_U行空间的能量: 前200个奇异值的总能量占比
    # 全谱近似: 总能量 ≈ ||W_U||_F^2
    total_energy = float(np.sum(W_U**2))
    top_energy = float(np.sum(s_wut**2))
    print(f"  W_U前{k_svd}个分量占总能量的{top_energy/total_energy*100:.1f}%")

    # 采样源层
    if n_layers <= 10:
        source_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 8)
        source_layers = sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))
    
    print(f"源层: {source_layers}")
    
    results = {
        "model": model_name, "exp": "B",
        "experiment": "wu_alignment",
        "n_layers": n_layers, "d_model": d_model,
        "k_svd": k_svd,
        "wu_energy_fraction": float(top_energy / total_energy),
        "wu_top_sv": [float(s) for s in s_wut[:20]],
        "alignment_data": {},
    }

    # 方向: probe_edible, lm_head_edible, random
    # 先训练probe (只在部分层)
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    probe_weights = {}
    for li in source_layers[:4]:
        all_hs = []
        all_words = []
        layers_list = get_layers(model)
        for template in CONTEXT_TEMPLATES[:3]:
            for word in concepts[:30]:
                sent = template.format(word=word)
                toks = tokenizer(sent, return_tensors="pt").to(device)
                tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
                dep_idx = find_token_index(tokens_list, word)
                if dep_idx < 0:
                    continue
                captured = {}
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        captured['h'] = output[0].detach().float().cpu().numpy()
                    else:
                        captured['h'] = output.detach().float().cpu().numpy()
                h_handle = layers_list[li].register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(**toks)
                h_handle.remove()
                if 'h' in captured:
                    all_hs.append(captured['h'][0, dep_idx, :])
                    all_words.append(word)
        
        if len(all_hs) < 10:
            continue
        H = np.array(all_hs)
        valid_V = np.array([[CONCEPT_DATASET[w][attr] for attr in ATTR_NAMES] for w in all_words])
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(valid_V, axis=0, keepdims=True)
        V_centered = valid_V - V_mean
        layer_probes = {}
        for ai, attr in enumerate(ATTR_NAMES[:2]):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, ai])
            layer_probes[attr] = ridge.coef_.copy()
            norm = np.linalg.norm(layer_probes[attr])
            if norm > 1e-10:
                layer_probes[attr] = layer_probes[attr] / norm
        probe_weights[li] = layer_probes

    # lm_head方向
    attr_token_words = {
        "edible": ["edible", "food", "eat", "delicious"],
        "animacy": ["alive", "living", "animate", "animal"],
    }
    lm_head_dirs = {}
    for attr, words in attr_token_words.items():
        ids = []
        for w in words:
            ids.extend(tokenizer.encode(w, add_special_tokens=False))
        ids = list(set(ids))
        if len(ids) > 0:
            dir_vec = np.mean(W_U[ids], axis=0)
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-10:
                lm_head_dirs[attr] = dir_vec / norm
    
    # 注入测试
    eps = 0.1  # 相对epsilon
    test_concepts = ["apple", "dog", "hammer"]
    
    np.random.seed(42)
    random_dir = np.random.randn(d_model)
    random_dir /= np.linalg.norm(random_dir)
    
    for concept in test_concepts:
        results["alignment_data"][concept] = {}
        
        # 收集baseline
        baseline_hs, _ = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if baseline_hs is None:
            continue
        
        for src_layer in source_layers[:5]:  # 限制源层数, 控制时间
            if src_layer not in baseline_hs:
                continue
            
            h_src = baseline_hs[src_layer]
            h_scale = float(np.linalg.norm(h_src))
            actual_eps = eps * h_scale
            
            results["alignment_data"][concept][str(src_layer)] = {}
            
            # 方向列表
            directions = {
                "random": random_dir,
                "lm_head_edible": lm_head_dirs.get("edible", np.zeros(d_model)),
            }
            if src_layer in probe_weights:
                directions["probe_edible"] = probe_weights[src_layer].get("edible", np.zeros(d_model))
            
            directions = {k: v for k, v in directions.items() 
                         if np.linalg.norm(v) > 1e-10}
            
            for dir_name, dir_vec in directions.items():
                # 注入扰动
                perturbed_hs, perturbed_logits, _ = inject_and_collect(
                    model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    src_layer, dir_vec, actual_eps, n_layers)
                
                if perturbed_hs is None:
                    continue
                
                # 在多个目标层测量W_U投影
                alignment_results = {}
                
                # 采样目标层
                target_layers = list(range(src_layer, n_layers, max(1, (n_layers - src_layer) // 8)))
                if (n_layers - 1) not in target_layers:
                    target_layers.append(n_layers - 1)
                
                for tgt_layer in target_layers:
                    if tgt_layer not in perturbed_hs or tgt_layer not in baseline_hs:
                        continue
                    
                    delta = perturbed_hs[tgt_layer] - baseline_hs[tgt_layer]
                    delta_norm = np.linalg.norm(delta)
                    
                    if delta_norm < 1e-10:
                        continue
                    
                    # 在W_U行空间中的投影
                    proj_coeffs = U_wut.T @ delta  # [k_svd]
                    proj_energy = float(np.sum(proj_coeffs**2))
                    proj_ratio = proj_energy / max(delta_norm**2, 1e-20)
                    
                    # Top-10投影系数的能量占比
                    top10_energy = float(np.sum(np.sort(proj_coeffs**2)[-10:]))
                    top10_ratio = top10_energy / max(proj_energy, 1e-20)
                    
                    # 注入方向本身的W_U投影 (参照)
                    inj_proj_coeffs = U_wut.T @ dir_vec
                    inj_proj_ratio = float(np.sum(inj_proj_coeffs**2)) / max(np.linalg.norm(dir_vec)**2, 1e-20)
                    
                    alignment_results[str(tgt_layer)] = {
                        "delta_norm": float(delta_norm),
                        "wu_proj_ratio": float(min(proj_ratio, 1.0)),
                        "wu_proj_energy": float(proj_energy),
                        "top10_ratio": float(top10_ratio),
                        "injected_wu_proj_ratio": float(min(inj_proj_ratio, 1.0)),
                        "proj_ratio_change": float(min(proj_ratio, 1.0) - min(inj_proj_ratio, 1.0)),
                    }
                
                results["alignment_data"][concept][str(src_layer)][dir_name] = {
                    "epsilon": float(actual_eps),
                    "alignment": alignment_results,
                }
                
                # 打印摘要
                if alignment_results:
                    last_tl = max(alignment_results.keys(), key=int)
                    first_tl = min(alignment_results.keys(), key=int)
                    r_first = alignment_results[first_tl]["wu_proj_ratio"]
                    r_last = alignment_results[last_tl]["wu_proj_ratio"]
                    r_inj = alignment_results[first_tl]["injected_wu_proj_ratio"]
                    print(f"  {concept} L{src_layer} {dir_name}: "
                          f"W_U_proj: {r_inj:.3f}(inject) → {r_first:.3f}(L{first_tl}) → {r_last:.3f}(L{last_tl})")

    return results


# ============================================================================
# 35C: 语义守恒测试 — g(h_l)是否跨层一致?
# ============================================================================

def expC_semantic_conservation(model_name, model, tokenizer, device):
    """
    35C: 测量语义函数g(h)在不同层的一致性
    
    核心问题:
    - 如果方向在旋转, 但函数g在"补偿" → 语义守恒
    - 如果方向在旋转, 函数g也在变 → 语义在逐层变化
    
    方法:
    1. 在每层训练probe: g_l(h) = w_l · h + b_l → 预测属性值
    2. 测试: g_l(h_l) vs g_l(h_{l+k}) — 用l层的probe读l+k层的表示
    3. 如果g_l(h_{l+k}) ≈ g_l(h_l) → 语义是跨层不变量
    4. 如果g_l(h_{l+k}) ≠ g_l(h_l) → 语义在逐层变化
    
    ★ 更深入: 用l层的probe读所有层的表示, 构建"语义一致性矩阵" ★
    """
    print(f"\n{'='*70}")
    print(f"35C: 语义守恒测试 — g(h_l)是否跨层一致?")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # 采样层
    if n_layers <= 12:
        probe_layers = list(range(0, n_layers, 2)) + [n_layers - 1]
    else:
        step = max(1, n_layers // 8)
        probe_layers = sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))
    
    print(f"模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")
    print(f"Probe层: {probe_layers}")

    # ===== Step 1: 在每层训练probe =====
    print(f"\n--- Step 1: 训练各层probe ---")
    
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    # 收集所有层所有概念的hidden states
    all_hs = {}  # {concept: {layer: h}}
    
    # 用更多模板增加鲁棒性
    for concept in concepts:
        hs_dict = {}
        for template in CONTEXT_TEMPLATES:
            hs, _ = collect_all_layer_hs(model, tokenizer, device, concept, template, n_layers)
            if hs is not None:
                for l, h in hs.items():
                    if l not in hs_dict:
                        hs_dict[l] = []
                    hs_dict[l].append(h)
        # 平均
        all_hs[concept] = {}
        for l, h_list in hs_dict.items():
            all_hs[concept][l] = np.mean(h_list, axis=0)
    
    print(f"  收集了 {len(all_hs)} 个概念的hidden states")
    
    # 训练probe
    probe_data = {}  # {layer: {attr: (weight, bias, mean_h, mean_v)}}
    
    for li in probe_layers:
        H_list = []
        valid_concepts = []
        for c in concepts:
            if li in all_hs[c]:
                H_list.append(all_hs[c][li])
                valid_concepts.append(c)
        
        if len(H_list) < 10:
            continue
        
        H = np.array(H_list)
        V_valid = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in valid_concepts])
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(V_valid, axis=0, keepdims=True)
        V_centered = V_valid - V_mean
        
        layer_probes = {}
        for ai, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, ai])
            pred = ridge.predict(H_centered)
            ss_res = np.sum((V_centered[:, ai] - pred)**2)
            ss_tot = np.sum(V_centered[:, ai]**2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            
            layer_probes[attr] = {
                "weight": ridge.coef_.copy(),
                "bias": float(ridge.intercept_),
                "r2": float(r2),
                "mean_h": H_mean.flatten().copy(),
                "mean_v": float(V_mean[0, ai]),
            }
            print(f"  L{li} {attr}: R²={r2:.3f}")
        
        probe_data[li] = layer_probes
    
    # ===== Step 2: 跨层语义一致性 =====
    print(f"\n--- Step 2: 跨层语义一致性矩阵 ---")
    
    # 对每个(probe层, 读出层)对, 计算:
    # R²(probe_l 在 h_k 上的预测)
    # → 如果高 → h_k在l层的probe方向上保持语义
    # → 如果低 → h_k在l层的probe方向上丢失语义
    
    conservation_matrix = {}  # {probe_layer: {read_layer: {attr: r2}}}
    
    for probe_li in probe_layers:
        if probe_li not in probe_data:
            continue
        
        conservation_matrix[str(probe_li)] = {}
        
        for read_li in probe_layers:
            # 用probe_li的权重读取read_li的表示
            H_read = []
            V_read = []
            valid_concepts = []
            
            for c in concepts:
                if read_li in all_hs[c] and probe_li in all_hs[c]:
                    H_read.append(all_hs[c][read_li])
                    V_read.append([CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES])
                    valid_concepts.append(c)
            
            if len(H_read) < 10:
                continue
            
            H_read = np.array(H_read)
            V_read = np.array(V_read)
            
            for ai, attr in enumerate(ATTR_NAMES):
                if attr not in probe_data[probe_li]:
                    continue
                
                probe = probe_data[probe_li][attr]
                w = probe["weight"]
                b = probe["bias"]
                mean_h = probe["mean_h"]
                mean_v = probe["mean_v"]
                
                # 预测: w · (h_read - mean_h_probe) + mean_v_probe
                H_centered = H_read - mean_h
                pred = H_centered @ w + b + mean_v
                
                # 真值
                V_centered = V_read[:, ai] - mean_v
                pred_centered = pred - mean_v
                
                ss_res = np.sum((V_centered - pred_centered)**2)
                ss_tot = np.sum(V_centered**2)
                r2 = 1 - ss_res / max(ss_tot, 1e-10)
                
                if attr not in conservation_matrix[str(probe_li)]:
                    conservation_matrix[str(probe_li)][attr] = {}
                conservation_matrix[str(probe_li)][attr][str(read_li)] = float(r2)
    
    # 打印摘要
    print(f"\n  ★ 语义守恒矩阵 (对角线应为1.0, 远离对角线反映守恒程度) ★")
    for attr in ATTR_NAMES:
        print(f"\n  {attr}:")
        print(f"  {'Probe\\Read':>12}", end="")
        for read_li in probe_layers:
            if str(read_li) in conservation_matrix.get(str(probe_layers[0]), {}).get(attr, {}):
                print(f"  L{read_li:>3}", end="")
        print()
        
        for probe_li in probe_layers:
            if str(probe_li) not in conservation_matrix:
                continue
            if attr not in conservation_matrix[str(probe_li)]:
                continue
            print(f"  L{probe_li:>10}", end="")
            for read_li in probe_layers:
                if str(read_li) in conservation_matrix[str(probe_li)][attr]:
                    r2 = conservation_matrix[str(probe_li)][attr][str(read_li)]
                    marker = "★" if probe_li == read_li else " "
                    print(f" {r2:>5.2f}{marker}", end="")
            print()
    
    results = {
        "model": model_name, "exp": "C",
        "experiment": "semantic_conservation",
        "n_layers": n_layers, "d_model": d_model,
        "probe_layers": probe_layers,
        "probe_r2s": {str(li): {attr: probe_data[li][attr]["r2"] 
                                for attr in ATTR_NAMES if attr in probe_data.get(li, {})}
                      for li in probe_layers if li in probe_data},
        "conservation_matrix": conservation_matrix,
    }
    
    return results


# ============================================================================
# 序列化辅助
# ============================================================================

def _convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 35: 算子谱分析")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                        help="实验编号: 0=全部, 1=Jacobian SVD, 2=W_U对齐, 3=语义守恒")
    args = parser.parse_args()

    model_name = args.model
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    print(f"模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5_temp")
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.exp in [0, 1]:
        resA = expA_jacobian_svd(model_name, model, tokenizer, device)
        with open(os.path.join(output_dir, f"ccml_phase35_expA_{model_name}_results.json"),
                  'w', encoding='utf-8') as f:
            json.dump(_convert_to_serializable(resA), f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[ExpA] Jacobian SVD结果已保存")

    if args.exp in [0, 2]:
        resB = expB_wu_alignment(model_name, model, tokenizer, device)
        with open(os.path.join(output_dir, f"ccml_phase35_expB_{model_name}_results.json"),
                  'w', encoding='utf-8') as f:
            json.dump(_convert_to_serializable(resB), f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[ExpB] W_U对齐结果已保存")

    if args.exp in [0, 3]:
        resC = expC_semantic_conservation(model_name, model, tokenizer, device)
        with open(os.path.join(output_dir, f"ccml_phase35_expC_{model_name}_results.json"),
                  'w', encoding='utf-8') as f:
            json.dump(_convert_to_serializable(resC), f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[ExpC] 语义守恒结果已保存")

    release_model(model)
    print(f"\n模型 {model_name} 已释放")


if __name__ == "__main__":
    main()
