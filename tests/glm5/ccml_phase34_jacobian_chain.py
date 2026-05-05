"""
CCML(Phase 34): 从方向到算子 — 扰动传播与Jacobian链
====================================================

Phase 33批评分析 (已采纳全部5条):

  ❌ 问题1: gradient ≠ 因果方向 ✔正确
     gradient = ∇_h ℓ = 局部最陡上升方向, input-dependent
     apple的gradient ≠ dog的gradient → 不是稳定语义方向
     正确说法: gradient = 局部决策边界方向(local decision boundary geometry)

  ❌ 问题2: probe负效应 ≠ 完全无用 ✔正确
     probe捕捉 E[y|h] (静态相关性)
     gradient捕捉 ∂logit/∂h (局部变化率)
     在非线性系统中, correlation正而derivative负是完全可能的

  ❌ 问题3: 病态矩阵解释过度 ✔正确
     d_model~4000时, 几乎所有learned matrix都高cond
     A^{-T}w≠w的原因不止零空间: 非正交基/多解性/噪声/分布变化

  ❌ 问题4: 残差 ≠ 非线性语义 ✔正确
     残差可能是: 未拟合的线性结构/噪声+bias/局部曲率
     只证明了linear_model不够, 没证明r=f_nonlinear(h)

  ❌ 问题5: 内在维度 ≠ 流形 ✔正确
     局部低维还可能是: clustered cloud / mixture of subspaces / anisotropic Gaussian
     真正流形需要: 光滑性/可微结构/geodesic consistency

三个瓶颈全部准确:
  🧱 瓶颈1: 混淆表示空间/解码空间/因果路径
  🧱 瓶颈2: 只看端点, 没有建模路径
  🧱 瓶颈3: 还在找坐标, 系统是算子

Phase 34核心任务:
  34A: ★★★★★★★★★ 扰动传播图 (最关键!)
    → 在每层注入扰动, 测量对后续所有层和最终输出的影响
    → 构建"因果路径"的完整图景 (Jacobian链的隐式测量)
    → 对比不同概念(apple/dog/hammer)的传播 → 算子一致性
    → 多epsilon值检验线性性

  34B: ★★★★★★★ 表示流 (Representation Flow)
    → 追踪表示如何在空间中移动: 层间位移/累积位移/速度
    → 属性如何沿路径变化
    → 概念间距离如何随层演化

关键数学框架:
  扰动传播 = Jacobian链的隐式计算:
    δ_{l+k} ≈ J_{l+k-1} · J_{l+k-2} · ... · J_l · δ_l
    
  如果效果与ε成正比 → 线性Jacobian有效
  如果效果不成正比 → 非线性显著

  算子一致性 = Jacobian的input-dependence:
    J_l(h_apple) ≈ J_l(h_dog)? → 线性 vs 非线性
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


def compute_cos(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ============================================================================
# 数据定义 (同Phase 33)
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

TEST_CONCEPTS = ["apple", "dog", "hammer"]


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
# 核心函数: 收集baseline和扰动传播
# ============================================================================

def collect_hs_at_layers(model, tokenizer, device, word, template, layer_indices):
    """收集指定层的hidden states (单次forward pass)"""
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
    for li in layer_indices:
        if li < len(layers):
            hooks.append(layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        _ = model(**toks)

    for h in hooks:
        h.remove()

    result = {}
    for li in layer_indices:
        if li in captured:
            result[li] = captured[li][0, dep_idx, :]

    return result, dep_idx


def collect_all_layer_hs(model, tokenizer, device, word, template, n_layers):
    """收集ALL层的hidden states"""
    return collect_hs_at_layers(model, tokenizer, device, word, template,
                                list(range(n_layers)))


def inject_perturbation(model, tokenizer, device, word, template,
                        source_layer, direction, epsilon, n_layers):
    """
    在source_layer注入扰动, 收集ALL后续层的hidden states和logits
    
    返回: (perturbed_hs_dict, logits_array) 或 (None, None)
    """
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
            # 也保存注入后的结果
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
# 34A: 扰动传播图 + 算子一致性
# ============================================================================

def expA_perturbation_propagation(model_name, model, tokenizer, device):
    """
    34A: 在每层注入扰动, 测量对后续所有层和输出的影响
    
    关键问题:
    1. 扰动如何在层间传播? (放大/抑制/旋转)
    2. 不同概念的传播是否一致? (算子一致性 → 线性 vs 非线性)
    3. 不同方向的传播差异? (probe vs random vs lm_head)
    4. 传播是否与epsilon成正比? (线性性检验)
    """
    print(f"\n{'='*70}")
    print(f"34A: 扰动传播图 — 构建因果路径的完整图景")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # 确定源层 (均匀采样 ~9层)
    if n_layers <= 10:
        source_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 9)
        source_layers = sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))

    print(f"模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")
    print(f"源层: {source_layers}")

    results = {
        "model": model_name, "exp": "A",
        "experiment": "perturbation_propagation",
        "n_layers": n_layers, "d_model": d_model,
        "source_layers": source_layers,
        "propagation": {},
        "logit_effects": {},
        "operator_consistency": {},
    }

    # ===== Step 1: 训练probes at source layers =====
    print(f"\n--- Step 1: 训练probes ---")

    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])

    probe_weights = {}  # {layer: {attr: weight_vector}}
    probe_r2s = {}     # {layer: {attr: r2}}

    for li in source_layers:
        all_hs = []
        all_words = []
        layers_list = get_layers(model)
        for template in CONTEXT_TEMPLATES:
            for word in concepts:
                sent = template.format(word=word)
                toks = tokenizer(sent, return_tensors="pt").to(device)
                tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
                dep_idx = find_token_index(tokens_list, word)
                if dep_idx < 0:
                    continue

                captured = {}
                target_layer = layers_list[li]
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        captured['h'] = output[0].detach().float().cpu().numpy()
                    else:
                        captured['h'] = output.detach().float().cpu().numpy()
                h_handle = target_layer.register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(**toks)
                h_handle.remove()

                if 'h' in captured:
                    all_hs.append(captured['h'][0, dep_idx, :])
                    all_words.append(word)

        if len(all_hs) < 20:
            print(f"  L{li}: 样本不足 ({len(all_hs)}), 跳过")
            continue

        H = np.array(all_hs)
        valid_V = np.array([[CONCEPT_DATASET[w][attr] for attr in ATTR_NAMES] for w in all_words])

        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(valid_V, axis=0, keepdims=True)
        V_centered = valid_V - V_mean

        layer_probes = {}
        layer_r2s = {}
        for ai, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, ai])
            layer_probes[attr] = ridge.coef_.copy()
            pred = ridge.predict(H_centered)
            ss_res = np.sum((V_centered[:, ai] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, ai] ** 2)
            layer_r2s[attr] = float(1 - ss_res / max(ss_tot, 1e-10))

        probe_weights[li] = layer_probes
        probe_r2s[li] = layer_r2s
        print(f"  L{li}: probe R² = {', '.join(f'{a}={r:.3f}' for a, r in layer_r2s.items())}")

    # ===== Step 2: 获取lm_head方向 =====
    print(f"\n--- Step 2: 获取lm_head方向 ---")
    W_U = get_W_U(model)  # [vocab_size, d_model]

    attr_token_words = {
        "edible": ["edible", "food", "eat", "delicious"],
        "animacy": ["alive", "living", "animate", "animal"],
        "size": ["large", "small", "big", "tiny"],
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
            # 平均W_U行作为方向
            dir_vec = np.mean(W_U[ids], axis=0)
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-10:
                lm_head_dirs[attr] = dir_vec / norm
            else:
                lm_head_dirs[attr] = np.zeros(d_model)
        print(f"  {attr}: token_ids={ids[:5]}, lm_head_dir norm={np.linalg.norm(lm_head_dirs.get(attr, np.zeros(d_model))):.4f}")

    # ===== Step 3: 收集baseline =====
    print(f"\n--- Step 3: 收集baseline ---")

    baseline_hs = {}  # {concept: {layer: h_vector}}
    baseline_logits = {}  # {concept: {attr: logit_value}}

    for concept in TEST_CONCEPTS:
        hs, dep_idx = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if hs is None:
            print(f"  {concept}: baseline收集失败")
            continue

        baseline_hs[concept] = hs

        # 用lm_head直接计算logits (从最后一层)
        h_last = hs[max(hs.keys())]
        h_tensor = torch.tensor(h_last, dtype=model.lm_head.weight.dtype,
                               device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model.lm_head(h_tensor)[0].detach().float().cpu().numpy()

        concept_logits = {}
        for attr, ids in attr_token_ids.items():
            if len(ids) > 0:
                concept_logits[attr] = float(logits[ids].mean())
            else:
                concept_logits[attr] = 0.0
        baseline_logits[concept] = concept_logits

        print(f"  {concept}: baseline logits = {concept_logits}")

    # ===== Step 4: 扰动传播测试 =====
    print(f"\n--- Step 4: 扰动传播测试 ---")

    epsilons = [0.1, 1.0]  # 相对epsilon (乘以||h - h_mean||)
    np.random.seed(42)
    random_dir = np.random.randn(d_model)
    random_dir = random_dir / np.linalg.norm(random_dir)
    np.random.seed(123)
    random_dir2 = np.random.randn(d_model)
    random_dir2 = random_dir2 / np.linalg.norm(random_dir2)

    # 对每个(概念, 源层, 方向, epsilon), 运行扰动测试
    total_tests = len(TEST_CONCEPTS) * len(source_layers) * 5 * len(epsilons)
    test_count = 0
    t_start = time.time()

    propagation_data = {}  # {concept: {source_layer: {direction: {epsilon: {target_layer: metrics}}}}}
    logit_effects_data = {}  # {concept: {source_layer: {direction: {epsilon: {attr: delta}}}}}

    for concept in TEST_CONCEPTS:
        if concept not in baseline_hs:
            continue

        propagation_data[concept] = {}
        logit_effects_data[concept] = {}

        for src_layer in source_layers:
            if src_layer not in probe_weights:
                continue
            if src_layer not in baseline_hs[concept]:
                continue

            propagation_data[concept][src_layer] = {}
            logit_effects_data[concept][src_layer] = {}

            h_src = baseline_hs[concept][src_layer]
            h_mean = np.mean(list(baseline_hs[concept].values()), axis=0)
            h_scale = float(np.linalg.norm(h_src - h_mean))

            # 方向列表
            directions = {
                "probe_edible": probe_weights[src_layer].get("edible", np.zeros(d_model)),
                "probe_animacy": probe_weights[src_layer].get("animacy", np.zeros(d_model)),
                "lm_head_edible": lm_head_dirs.get("edible", np.zeros(d_model)),
                "random": random_dir,
                "random2": random_dir2,
            }

            # 归一化方向
            for dname, dvec in directions.items():
                norm = np.linalg.norm(dvec)
                if norm > 1e-10:
                    directions[dname] = dvec / norm

            for dir_name, dir_vec in directions.items():
                if np.linalg.norm(dir_vec) < 1e-10:
                    continue

                propagation_data[concept][src_layer][dir_name] = {}
                logit_effects_data[concept][src_layer][dir_name] = {}

                for eps in epsilons:
                    actual_eps = eps * h_scale

                    test_count += 1
                    if test_count % 20 == 0:
                        elapsed = time.time() - t_start
                        eta = elapsed / test_count * (total_tests - test_count)
                        print(f"  进度: {test_count}/{total_tests} ({elapsed:.0f}s, ETA:{eta:.0f}s)")

                    # 运行扰动测试
                    perturbed_hs, perturbed_logits, dep_idx = inject_perturbation(
                        model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                        src_layer, dir_vec, actual_eps, n_layers)

                    if perturbed_hs is None:
                        continue

                    # 计算传播指标
                    prop_metrics = {}
                    prev_delta = None
                    for tgt_layer in range(src_layer + 1, n_layers, max(1, (n_layers - src_layer) // 8)):
                        if tgt_layer not in perturbed_hs:
                            continue
                        if tgt_layer not in baseline_hs[concept]:
                            continue

                        h_pert = perturbed_hs[tgt_layer]
                        h_base = baseline_hs[concept][tgt_layer]
                        delta = h_pert - h_base

                        delta_norm = float(np.linalg.norm(delta))
                        injected_norm = float(np.linalg.norm(dir_vec * actual_eps))
                        delta_cos_orig = compute_cos(delta, dir_vec)

                        if prev_delta is not None:
                            delta_cos_prev = compute_cos(delta, prev_delta)
                        else:
                            delta_cos_prev = None

                        prop_metrics[tgt_layer] = {
                            "delta_norm": delta_norm,
                            "amplification": delta_norm / max(injected_norm, 1e-10),
                            "cos_with_original": delta_cos_orig,
                            "cos_with_prev": delta_cos_prev,
                        }
                        prev_delta = delta

                    # 也测最后一层
                    last_layer = n_layers - 1
                    if last_layer not in prop_metrics and last_layer in perturbed_hs and last_layer in baseline_hs[concept]:
                        h_pert = perturbed_hs[last_layer]
                        h_base = baseline_hs[concept][last_layer]
                        delta = h_pert - h_base
                        delta_norm = float(np.linalg.norm(delta))
                        injected_norm = float(np.linalg.norm(dir_vec * actual_eps))
                        prop_metrics[last_layer] = {
                            "delta_norm": delta_norm,
                            "amplification": delta_norm / max(injected_norm, 1e-10),
                            "cos_with_original": compute_cos(delta, dir_vec),
                            "cos_with_prev": compute_cos(delta, prev_delta) if prev_delta is not None else None,
                        }

                    propagation_data[concept][src_layer][dir_name][str(eps)] = prop_metrics

                    # Logit变化
                    logit_deltas = {}
                    for attr, ids in attr_token_ids.items():
                        if len(ids) > 0:
                            base_logit = baseline_logits[concept].get(attr, 0.0)
                            pert_logit = float(perturbed_logits[ids].mean())
                            logit_deltas[attr] = pert_logit - base_logit

                    logit_effects_data[concept][src_layer][dir_name][str(eps)] = logit_deltas

    elapsed = time.time() - t_start
    print(f"\n  扰动测试完成: {test_count} tests in {elapsed:.0f}s")

    # ===== Step 5: 算子一致性分析 =====
    print(f"\n--- Step 5: 算子一致性分析 ---")

    consistency_results = {}

    for src_layer in source_layers:
        consistency_results[src_layer] = {}
        for dir_name in ["probe_edible", "probe_animacy", "lm_head_edible", "random"]:
            for eps in epsilons:
                # 收集所有概念在(target_layer=n_layers-1)的amplification
                amps = []
                for concept in TEST_CONCEPTS:
                    try:
                        last_layer = n_layers - 1
                        # 找最近的target_layer
                        prop = propagation_data[concept][src_layer][dir_name][str(eps)]
                        closest_tl = min(prop.keys(), key=lambda x: abs(x - last_layer))
                        amps.append(prop[closest_tl]["amplification"])
                    except (KeyError, ValueError):
                        pass

                if len(amps) >= 2:
                    mean_amp = np.mean(amps)
                    std_amp = np.std(amps)
                    cv = std_amp / max(abs(mean_amp), 1e-10)  # 变异系数
                    consistency_results[src_layer][f"{dir_name}_eps{eps}"] = {
                        "mean_amplification": float(mean_amp),
                        "std_amplification": float(std_amp),
                        "cv": float(cv),
                        "values": [float(a) for a in amps],
                    }

    # 打印一致性摘要
    print(f"\n  ★ 算子一致性 (变异系数CV, 越小越一致→线性) ★")
    for src_layer in source_layers:
        if src_layer not in consistency_results:
            continue
        print(f"  L{src_layer}:")
        for key, val in consistency_results[src_layer].items():
            if "probe_edible" in key or "random" in key:
                marker = "← 一致(线性)" if val["cv"] < 0.3 else "← 不一致(非线性)"
                print(f"    {key}: CV={val['cv']:.3f} mean_amp={val['mean_amplification']:.3f} {marker}")

    # ===== Step 6: 线性性检验 (epsilon scaling) =====
    print(f"\n--- Step 6: 线性性检验 ---")

    linearity_results = {}
    for concept in TEST_CONCEPTS:
        for src_layer in source_layers:
            for dir_name in ["probe_edible", "random"]:
                try:
                    prop_01 = propagation_data[concept][src_layer][dir_name]["0.1"]
                    prop_10 = propagation_data[concept][src_layer][dir_name]["1.0"]

                    # 找共同的target layers
                    common_tls = set(prop_01.keys()) & set(prop_10.keys())
                    for tl in common_tls:
                        amp_01 = prop_01[tl]["amplification"]
                        amp_10 = prop_10[tl]["amplification"]
                        # 如果线性: amp_10应该 ≈ amp_01 (amplification是相对值, 不依赖epsilon)
                        # 或者: delta_norm_10 / delta_norm_01 ≈ 10 (epsilon比)
                        ratio = amp_10 / max(amp_01, 1e-10)

                        key = f"{concept}_L{src_layer}_{dir_name}_L{tl}"
                        linearity_results[key] = {
                            "amp_eps01": float(amp_01),
                            "amp_eps10": float(amp_10),
                            "ratio": float(ratio),
                            "linear_if_close_to_1": abs(ratio - 1.0) < 0.5,
                        }
                except (KeyError, TypeError):
                    pass

    # 打印线性性摘要
    linear_count = sum(1 for v in linearity_results.values() if v["linear_if_close_to_1"])
    total_count = len(linearity_results)
    if total_count > 0:
        print(f"  线性性: {linear_count}/{total_count} 的amplification在不同epsilon下近似一致")
        print(f"  (如果amplification不依赖epsilon → Jacobian近似线性)")

    # 保存结果
    results["propagation"] = _convert_to_serializable(propagation_data)
    results["logit_effects"] = _convert_to_serializable(logit_effects_data)
    results["operator_consistency"] = _convert_to_serializable(consistency_results)
    results["linearity"] = _convert_to_serializable(linearity_results)
    results["probe_r2s"] = _convert_to_serializable(probe_r2s)
    results["baseline_logits"] = _convert_to_serializable(baseline_logits)

    return results


# ============================================================================
# 34B: 表示流 (Representation Flow)
# ============================================================================

def expB_representation_flow(model_name, model, tokenizer, device):
    """
    34B: 追踪表示如何在空间中移动

    关键问题:
    1. 表示的"速度"(层间位移)如何随层变化?
    2. 不同概念的轨迹如何分叉/汇合?
    3. 属性何时"涌现"(probe预测何时变强)?
    """
    print(f"\n{'='*70}")
    print(f"34B: 表示流 — 追踪表示如何在空间中移动")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    results = {
        "model": model_name, "exp": "B",
        "experiment": "representation_flow",
        "n_layers": n_layers, "d_model": d_model,
    }

    # 收集所有层的hidden states
    all_concept_hs = {}  # {concept: {layer: h_vector}}

    # 用更多概念来分析
    flow_concepts = ["apple", "dog", "hammer", "orange", "cat", "knife",
                     "banana", "eagle", "chair", "water", "fire", "tree"]

    for concept in flow_concepts:
        if concept not in CONCEPT_DATASET:
            continue
        hs, dep_idx = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if hs is not None:
            all_concept_hs[concept] = hs
            print(f"  {concept}: 收集了 {len(hs)} 层的hidden states")

    if len(all_concept_hs) < 3:
        print("  概念数不足, 跳过")
        return results

    # ===== 分析1: 层间位移 =====
    print(f"\n  ★ 层间位移 (每层表示移动了多少?) ★")

    displacement_norms = {}  # {concept: {layer: norm}}
    for concept, hs in all_concept_hs.items():
        displacement_norms[concept] = {}
        sorted_layers = sorted(hs.keys())
        for i in range(len(sorted_layers) - 1):
            l1 = sorted_layers[i]
            l2 = sorted_layers[i + 1]
            disp = hs[l2] - hs[l1]
            displacement_norms[concept][l2] = float(np.linalg.norm(disp))

    # 打印摘要
    for concept in TEST_CONCEPTS:
        if concept in displacement_norms:
            disps = displacement_norms[concept]
            sorted_layers = sorted(disps.keys())
            early = np.mean([disps[l] for l in sorted_layers[:len(sorted_layers)//3]])
            mid = np.mean([disps[l] for l in sorted_layers[len(sorted_layers)//3:2*len(sorted_layers)//3]])
            late = np.mean([disps[l] for l in sorted_layers[2*len(sorted_layers)//3:]])
            print(f"    {concept}: 早期={early:.2f}, 中期={mid:.2f}, 晚期={late:.2f}")

    # ===== 分析2: 累积位移 =====
    print(f"\n  ★ 累积位移 (从L0走了多远?) ★")

    cumulative_norms = {}
    for concept, hs in all_concept_hs.items():
        cumulative_norms[concept] = {}
        h0 = hs[0]
        for l in sorted(hs.keys()):
            cumulative_norms[concept][l] = float(np.linalg.norm(hs[l] - h0))

    # ===== 分析3: 概念间距离 =====
    print(f"\n  ★ 概念间距离 (何时分离?) ★")

    concept_pairs = [("apple", "dog"), ("apple", "hammer"), ("dog", "hammer"),
                     ("apple", "orange"), ("dog", "cat"), ("hammer", "knife")]
    inter_distances = {}  # {layer: {pair: distance}}

    for l in range(n_layers):
        inter_distances[l] = {}
        for c1, c2 in concept_pairs:
            if c1 in all_concept_hs and c2 in all_concept_hs:
                if l in all_concept_hs[c1] and l in all_concept_hs[c2]:
                    dist = float(np.linalg.norm(all_concept_hs[c1][l] - all_concept_hs[c2][l]))
                    inter_distances[l][f"{c1}-{c2}"] = dist

    # 找到edible/non-edible分离最大的层
    edible_concepts = [c for c in all_concept_hs if CONCEPT_DATASET[c]["edible"] == 1]
    non_edible_concepts = [c for c in all_concept_hs if CONCEPT_DATASET[c]["edible"] == 0]

    edible_separation = {}
    for l in range(n_layers):
        edible_hs = [all_concept_hs[c][l] for c in edible_concepts if l in all_concept_hs[c]]
        non_edible_hs = [all_concept_hs[c][l] for c in non_edible_concepts if l in all_concept_hs[c]]
        if len(edible_hs) >= 2 and len(non_edible_hs) >= 2:
            edible_center = np.mean(edible_hs, axis=0)
            non_edible_center = np.mean(non_edible_hs, axis=0)
            edible_separation[l] = float(np.linalg.norm(edible_center - non_edible_center))

    if edible_separation:
        max_sep_layer = max(edible_separation, key=edible_separation.get)
        print(f"    Edible/Non-edible最大分离层: L{max_sep_layer} (距离={edible_separation[max_sep_layer]:.2f})")

    # 同理: animacy分离
    animate_concepts = [c for c in all_concept_hs if CONCEPT_DATASET[c]["animacy"] == 1]
    inanimate_concepts = [c for c in all_concept_hs if CONCEPT_DATASET[c]["animacy"] == 0]

    animacy_separation = {}
    for l in range(n_layers):
        ani_hs = [all_concept_hs[c][l] for c in animate_concepts if l in all_concept_hs[c]]
        inani_hs = [all_concept_hs[c][l] for c in inanimate_concepts if l in all_concept_hs[c]]
        if len(ani_hs) >= 2 and len(inani_hs) >= 2:
            ani_center = np.mean(ani_hs, axis=0)
            inani_center = np.mean(inani_hs, axis=0)
            animacy_separation[l] = float(np.linalg.norm(ani_center - inani_center))

    if animacy_separation:
        max_ani_layer = max(animacy_separation, key=animacy_separation.get)
        print(f"    Animate/Inanimate最大分离层: L{max_ani_layer} (距离={animacy_separation[max_ani_layer]:.2f})")

    # ===== 分析4: 属性涌现 =====
    print(f"\n  ★ 属性涌现 (edible/animacy何时可被线性读取?) ★")

    # 在每层训练probe, 看R²随层变化
    all_concepts_list = list(all_concept_hs.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in all_concepts_list])

    attr_emergence = {}  # {attr: {layer: r2}}

    for l in range(n_layers):
        H = []
        valid_indices = []
        for i, c in enumerate(all_concepts_list):
            if l in all_concept_hs[c]:
                H.append(all_concept_hs[c][l])
                valid_indices.append(i)

        if len(H) < 10:
            continue

        H = np.array(H)
        V_valid = V[valid_indices]

        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(V_valid, axis=0, keepdims=True)
        V_centered = V_valid - V_mean

        for ai, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, ai])
            pred = ridge.predict(H_centered)
            ss_res = np.sum((V_centered[:, ai] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, ai] ** 2)
            r2 = float(1 - ss_res / max(ss_tot, 1e-10))

            if attr not in attr_emergence:
                attr_emergence[attr] = {}
            attr_emergence[attr][l] = r2

    # 打印涌现摘要
    for attr in ATTR_NAMES:
        if attr in attr_emergence:
            r2s = attr_emergence[attr]
            sorted_layers = sorted(r2s.keys())
            # 找R²首次超过0.5的层
            emergence_layer = None
            for l in sorted_layers:
                if r2s[l] > 0.5:
                    emergence_layer = l
                    break
            max_r2 = max(r2s.values())
            max_r2_layer = max(r2s, key=r2s.get)
            print(f"    {attr}: R²>0.5首次出现在L{emergence_layer}, "
                  f"峰值R²={max_r2:.3f}在L{max_r2_layer}")

    # 保存结果
    results["displacement_norms"] = _convert_to_serializable(displacement_norms)
    results["cumulative_norms"] = _convert_to_serializable(cumulative_norms)
    results["inter_distances_sample"] = _convert_to_serializable(
        {l: v for l, v in inter_distances.items() if l % max(1, n_layers // 10) == 0 or l == n_layers - 1})
    results["edible_separation"] = _convert_to_serializable(
        {l: v for l, v in edible_separation.items() if l % max(1, n_layers // 10) == 0 or l == n_layers - 1})
    results["animacy_separation"] = _convert_to_serializable(
        {l: v for l, v in animacy_separation.items() if l % max(1, n_layers // 10) == 0 or l == n_layers - 1})
    results["attr_emergence"] = _convert_to_serializable(attr_emergence)

    return results


# ============================================================================
# 序列化辅助
# ============================================================================

def _convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型以便JSON序列化"""
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
    parser = argparse.ArgumentParser(description="Phase 34: Jacobian链与扰动传播")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                        help="实验编号: 0=全部, 1=扰动传播, 2=表示流")
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
        resA = expA_perturbation_propagation(model_name, model, tokenizer, device)
        with open(os.path.join(output_dir, f"ccml_phase34_expA_{model_name}_results.json"),
                  'w', encoding='utf-8') as f:
            json.dump(resA, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[ExpA] 扰动传播结果已保存")

    if args.exp in [0, 2]:
        resB = expB_representation_flow(model_name, model, tokenizer, device)
        with open(os.path.join(output_dir, f"ccml_phase34_expB_{model_name}_results.json"),
                  'w', encoding='utf-8') as f:
            json.dump(resB, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[ExpB] 表示流结果已保存")

    release_model(model)
    print(f"\n模型 {model_name} 已释放")


if __name__ == "__main__":
    main()
