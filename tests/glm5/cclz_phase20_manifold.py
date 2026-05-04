"""
CCL-Z(Phase 20): 从描述到推导——语法编码的流形动力学与量化归因
=============================================================================
核心问题(基于Phase 19的5个问题点):
  1. ★★★★★★★★★ 曲率比归因: 6.9是否是采样间距的函数?
     → 用统一层间距(每1层/每2层/每3层)重新计算曲率比
     → 如果曲率比随采样间距变化 → 是人为假象
     → 如果不变 → 是模型的固有属性

  2. ★★★★★★★★★ 范数指数增长的理论推导
     → 推导: h_{l+1} = LN(h_l + Attn(h_l) + MLP(h_l))
     → 在语法子空间上的范数演化方程
     → 验证: 指数增长率 vs 残差增益的理论预测

  3. ★★★★★★★★ 最后层旋转的归因: 内部旋转 vs logit投影
     → 在最后一层hidden state(含LN) vs logits之间测量旋转
     → 区分: Transformer层内旋转 vs lm_head投影旋转
     → 关键: 抽离lm_head的影响

  4. ★★★★★★★ 正交性的数据归因验证
     → 随机方向上的PCA: 是否也正交?
     → 3个随机cluster的PCA → 提取的轴是否正交?
     → 语法cluster vs 随机cluster: 正交性差异?

  5. ★★★★★★★ 因果模式判据的稳定性分析
     → 更多语法角色(nsubj, dobj, amod, poss, nmod)的测试
     → 判据在不同语法角色上是否一致?
     → 蒙特卡洛验证: 随机参数下判据的行为

实验:
  Exp1: ★★★★★★★★★ 曲率比归因——采样间距vs模型属性
    → 用3种采样间距(1层/2层/3层)分别计算曲率比
    → 如果曲率比与采样间距无关 → 模型属性
    → 如果相关 → 人为假象

  Exp2: ★★★★★★★★★ 范数指数增长的理论推导与验证
    → 理论: 语法方向范数 ||v_l|| 的演化方程
    → 推导: 残差连接使得 v_{l+1} ≈ v_l + O(||delta||)
    → 如果 ||delta||/||v_l|| = λ (常数) → ||v_l|| ∝ exp(λl)
    → 验证: 测量每层的 λ = ||delta||/||v_l||, 是否为常数?

  Exp3: ★★★★★★★★ 最后层旋转的归因
    → 在last-1层hidden state上计算语法方向
    → 在logits上计算语法方向(用lm_head投影)
    → 比较旋转角: 层间旋转 vs lm_head旋转
    → 量化: 总旋转中lm_head的贡献比例

  Exp4: ★★★★★★★ 正交性与因果判据的系统性验证
    → 4A: 随机方向PCA对照实验
    → 4B: 更多语法角色的正交性测量
    → 4C: 因果判据的跨角色稳定性
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
from sklearn.decomposition import PCA

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 复用数据 =====
MANIFOLD_ROLES_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom wisely",
            "The doctor treated the patient carefully",
            "The artist painted the portrait beautifully",
            "The soldier defended the castle bravely",
            "The teacher explained the lesson clearly",
            "The chef cooked the meal perfectly",
            "The cat chased the mouse quickly",
            "The dog found the bone happily",
            "The woman drove the car safely",
            "The man fixed the roof carefully",
            "The student read the book quietly",
            "The singer performed the song brilliantly",
            "The baker made the bread daily",
            "The pilot flew the plane smoothly",
            "The farmer grew the crops diligently",
            "The writer wrote the novel slowly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "poss": {
        "sentences": [
            "The king's crown glittered brightly",
            "The doctor's office opened early",
            "The artist's studio looked beautiful",
            "The soldier's uniform was clean",
            "The teacher's book sold quickly",
            "The chef's restaurant opened today",
            "The cat's tail swished gently",
            "The dog's bark echoed loudly",
            "The woman's dress looked elegant",
            "The man's car drove fast",
            "The student's essay read well",
            "The singer's voice rang clearly",
            "The baker's shop smelled wonderful",
            "The pilot's license was renewed",
            "The farmer's land was fertile",
            "The writer's pen wrote smoothly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday",
            "She visited the doctor recently",
            "He admired the artist greatly",
            "We honored the soldier today",
            "You thanked the teacher warmly",
            "The customer tipped the chef generously",
            "The hawk chased the cat swiftly",
            "The boy found the dog outside",
            "The police arrested the woman quickly",
            "The company hired the man recently",
            "I praised the student loudly",
            "They applauded the singer warmly",
            "She visited the baker often",
            "He admired the pilot greatly",
            "We thanked the farmer sincerely",
            "The editor praised the writer highly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "amod": {
        "sentences": [
            "The brave king fought hard",
            "The kind doctor helped many",
            "The creative artist worked well",
            "The strong soldier marched far",
            "The wise teacher explained clearly",
            "The skilled chef cooked perfectly",
            "The quick cat ran fast",
            "The loyal dog stayed close",
            "The old woman walked slowly",
            "The tall man stood quietly",
            "The bright student read carefully",
            "The talented singer performed brilliantly",
            "The patient baker waited calmly",
            "The careful pilot landed smoothly",
            "The hardworking farmer harvested early",
            "The thoughtful writer reflected deeply",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "wise",
            "skilled", "quick", "loyal", "old", "tall",
            "bright", "talented", "patient", "careful", "hardworking", "thoughtful",
        ],
    },
}


def find_token_index(tokens, word):
    word_lower = word.lower()
    word_start = word_lower[:3]
    for i, tok in enumerate(tokens):
        tok_lower = tok.lower().strip()
        if word_lower in tok_lower or tok_lower.startswith(word_start):
            return i
    for i, tok in enumerate(tokens):
        if word_lower[:2] in tok.lower():
            return i
    return None


def collect_hs_at_layer(model, tokenizer, device, sentences, target_words, layer_idx):
    """在指定层收集target token的hidden states"""
    layers = get_layers(model)
    if layer_idx >= len(layers):
        layer_idx = len(layers) - 1
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    target_layer = layers[layer_idx]

    all_h = []
    valid_words = []

    for sent, target_word in zip(sentences, target_words):
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]

        dep_idx = find_token_index(tokens_list, target_word)
        if dep_idx is None:
            continue

        captured = {}
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured['h'] = output[0].detach().float().cpu().numpy()
            else:
                captured['h'] = output.detach().float().cpu().numpy()

        h_handle = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(**toks)
        h_handle.remove()

        if 'h' not in captured:
            continue

        h_vec = captured['h'][0, dep_idx, :]
        all_h.append(h_vec)
        valid_words.append(target_word)

    return np.array(all_h) if all_h else None


def get_syntax_directions_at_layer(model, tokenizer, device, model_info, layer_idx):
    """获取指定层的语法方向(nsubj-dobj, nsubj-amod等)"""
    role_names = ["nsubj", "dobj", "amod"]
    role_h = {}
    for role in role_names:
        data = MANIFOLD_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], layer_idx)
        if H is not None and len(H) > 0:
            role_h[role] = H

    centers = {}
    for role in role_names:
        if role in role_h:
            centers[role] = np.mean(role_h[role], axis=0)

    directions = {}
    if 'nsubj' in centers and 'dobj' in centers:
        d = centers['dobj'] - centers['nsubj']
        norm = np.linalg.norm(d)
        if norm > 0:
            directions['noun_axis'] = d / norm
            directions['noun_axis_norm'] = norm

    if 'nsubj' in centers and 'amod' in centers:
        amod_vec = centers['amod'] - centers['nsubj']
        if 'noun_axis' in directions:
            amod_proj = np.dot(amod_vec, directions['noun_axis']) * directions['noun_axis']
            modifier_axis = amod_vec - amod_proj
            modifier_norm = np.linalg.norm(modifier_axis)
            if modifier_norm > 0:
                directions['modifier_axis'] = modifier_axis / modifier_norm
                directions['modifier_axis_norm'] = modifier_norm
                directions['axis_orthogonality'] = float(np.dot(directions['noun_axis'], modifier_axis / modifier_norm))

    directions['centers'] = centers
    return directions


def compute_grassmannian_distance(p1_noun, p1_mod, p2_noun, p2_mod):
    """计算两个2D平面的Grassmannian距离"""
    P1 = np.column_stack([p1_noun, p1_mod])
    P2 = np.column_stack([p2_noun, p2_mod])
    M = P1.T @ P2
    _, s_m, _ = np.linalg.svd(M)
    d_G = np.sqrt(np.sum(np.arccos(np.clip(s_m, -1, 1)) ** 2))
    return d_G


# ===== Exp1: 曲率比归因 =====
def exp1_curvature_attribution(model, tokenizer, device):
    """曲率比归因: 6.9是采样间距的函数还是模型属性?"""
    print("\n" + "="*70)
    print("Exp1: 曲率比归因 ★★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Step 1: 采集所有层的2D平面
    print("\n  Step 1: 采集所有层的2D语法子空间")

    all_layer_planes = {}
    for li in range(n_layers):
        print(f"  采集L{li}...", end="", flush=True)
        dirs = get_syntax_directions_at_layer(model, tokenizer, device, model_info, li)
        if 'noun_axis' in dirs and 'modifier_axis' in dirs:
            all_layer_planes[li] = {
                'noun_axis': dirs['noun_axis'],
                'modifier_axis': dirs['modifier_axis'],
                'ortho': dirs.get('axis_orthogonality', 0),
            }
            print(f" ok")
        else:
            print(f" 跳过")

    if len(all_layer_planes) < 4:
        print("  采集到的层不足!")
        return results

    # Step 2: 用不同采样间距计算曲率比
    print("\n  Step 2: 不同采样间距的曲率比")

    sorted_layers = sorted(all_layer_planes.keys())
    min_layer = sorted_layers[0]
    max_layer = sorted_layers[-1]

    for step_size in [1, 2, 3, 4]:
        # 选取采样层
        sampled = list(range(min_layer, max_layer + 1, step_size))
        # 确保首尾层都在
        if sampled[-1] != max_layer:
            sampled.append(max_layer)
        # 只保留有数据的层
        sampled = [li for li in sampled if li in all_layer_planes]

        if len(sampled) < 3:
            print(f"  步长={step_size}: 采样点不足({len(sampled)})")
            continue

        # 计算累积Grassmannian距离
        cum_dG = 0
        for i in range(len(sampled) - 1):
            li = sampled[i]
            li1 = sampled[i + 1]
            p = all_layer_planes[li]
            p1 = all_layer_planes[li1]
            dG = compute_grassmannian_distance(
                p['noun_axis'], p['modifier_axis'],
                p1['noun_axis'], p1['modifier_axis']
            )
            cum_dG += dG

        # 首尾直线距离
        p_first = all_layer_planes[sampled[0]]
        p_last = all_layer_planes[sampled[-1]]
        dG_total = compute_grassmannian_distance(
            p_first['noun_axis'], p_first['modifier_axis'],
            p_last['noun_axis'], p_last['modifier_axis']
        )

        curvature_ratio = cum_dG / max(dG_total, 1e-10)

        # 每步的平均旋转角
        avg_step_dG = cum_dG / max(len(sampled) - 1, 1)

        print(f"  步长={step_size}: 采样点={len(sampled)}, "
              f"cum_dG={cum_dG:.4f}, dG_total={dG_total:.4f}, "
              f"曲率比={curvature_ratio:.4f}, 平均步距={avg_step_dG:.4f}")

        results[f'step_{step_size}'] = {
            'n_points': len(sampled),
            'cum_dG': float(cum_dG),
            'dG_total': float(dG_total),
            'curvature_ratio': float(curvature_ratio),
            'avg_step_dG': float(avg_step_dG),
        }

    # Step 3: 理论分析——如果轨迹是直线(测地线), 曲率比=1
    # 如果轨迹是随机游走, 曲率比 ∝ sqrt(步数)
    # 如果轨迹是固定曲率, 曲率比与采样无关
    print("\n  Step 3: 曲率比归因分析")

    ratios = [results[k]['curvature_ratio'] for k in results if k.startswith('step_')]
    steps = [int(k.split('_')[1]) for k in results if k.startswith('step_')]
    n_points = [results[k]['n_points'] for k in results if k.startswith('step_')]

    if len(ratios) >= 2:
        ratio_cv = np.std(ratios) / max(np.mean(ratios), 1e-10)
        print(f"  曲率比变化: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}, CV={ratio_cv:.4f}")

        # 随机游走模型: 曲率比 ≈ sqrt(N), N=采样点数
        rw_predictions = [np.sqrt(n) for n in n_points]
        print(f"  随机游走预测: {[f'{p:.2f}' for p in rw_predictions]}")
        print(f"  实际曲率比: {[f'{r:.2f}' for r in ratios]}")

        if ratio_cv < 0.15:
            print(f"  ★★★ 曲率比近似不变 → 模型的固有属性!")
        else:
            # 检查是否与sqrt(N)成正比
            ratios_over_sqrtn = [r / max(np.sqrt(n), 1e-10) for r, n in zip(ratios, n_points)]
            cv2 = np.std(ratios_over_sqrtn) / max(np.mean(ratios_over_sqrtn), 1e-10)
            if cv2 < 0.15:
                print(f"  ★★★ 曲率比 ∝ sqrt(N) → 随机游走模型! (CV={cv2:.4f})")
                print(f"  → 6.9不是模型属性, 而是采样密度的函数")
            else:
                print(f"  ★ 曲率比与采样间距有复杂关系")

    # Step 4: 每层的单步旋转角分析
    print("\n  Step 4: 单步旋转角的层间分布")

    step_dGs = []
    for i in range(len(sorted_layers) - 1):
        li = sorted_layers[i]
        li1 = sorted_layers[i + 1]
        if li in all_layer_planes and li1 in all_layer_planes:
            p = all_layer_planes[li]
            p1 = all_layer_planes[li1]
            dG = compute_grassmannian_distance(
                p['noun_axis'], p['modifier_axis'],
                p1['noun_axis'], p1['modifier_axis']
            )
            step_dGs.append((li, li1, dG))

    # 旋转角是否随层衰减?
    first_half = [dG for li, _, dG in step_dGs if li < n_layers // 2]
    second_half = [dG for li, _, dG in step_dGs if li >= n_layers // 2]

    if first_half and second_half:
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        decay_ratio = mean_second / max(mean_first, 1e-10)
        print(f"  前半层平均旋转: {mean_first:.4f}")
        print(f"  后半层平均旋转: {mean_second:.4f}")
        print(f"  衰减比(后/前): {decay_ratio:.4f}")
        if decay_ratio < 0.7:
            print(f"  ★★★ 旋转角随层衰减 → 前层旋转大, 后层旋转小")
            print(f"  → 这解释了曲率比>1: 前层大旋转使轨迹偏离直线")

    results['step_dGs'] = [(li, li1, float(dG)) for li, li1, dG in step_dGs]
    if first_half and second_half:
        results['rotation_decay_ratio'] = float(decay_ratio)
        results['mean_rotation_first_half'] = float(mean_first)
        results['mean_rotation_second_half'] = float(mean_second)

    return results


# ===== Exp2: 范数指数增长的理论推导与验证 =====
def exp2_norm_growth_derivation(model, tokenizer, device):
    """范数指数增长的理论推导与验证"""
    print("\n" + "="*70)
    print("Exp2: 范数指数增长的理论推导与验证 ★★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Step 1: 理论推导
    print("\n  Step 1: 理论推导")
    print("""
  Transformer的前向传播:
    h_{l+1} = LN(h_l + Attn(h_l) + MLP(h_l))

  设 v_l 是语法方向在L层的表示(在h_l空间中的投影):
    v_l = P_l^T @ h_l  (投影到语法子空间)

  语法方向的范数: ||v_l||

  残差连接: h_{l+1} = h_l + delta_l  (delta_l = Attn + MLP, 经LN归一化后)
  LN的效果: h_{l+1} = (h_l + delta_l - μ_l) / σ_l

  在语法方向上:
    v_{l+1} ≈ (v_l + delta_v_l) / σ_l  (近似, 忽略均值项对方向的影响)

  其中 delta_v_l = P_{l+1}^T @ delta_l

  如果 ||delta_v_l|| / ||v_l|| = λ (常数), 则:
    ||v_{l+1}|| ≈ ||v_l|| * (1 + λ) / σ_l

  如果 σ_l ≈ σ (层间近似常数), 则:
    ||v_l|| ∝ ((1+λ)/σ)^l  → 指数增长!

  关键: 需要验证两个条件:
    (A) ||delta_v_l|| / ||v_l|| = λ 是否为常数?
    (B) σ_l 是否近似常数?
""")

    # Step 2: 测量每层的语法方向范数
    print("\n  Step 2: 测量每层的语法方向范数")

    layer_norms = {}
    layer_noun_norms = {}
    layer_mod_norms = {}

    for li in range(n_layers):
        print(f"  L{li}...", end="", flush=True)

        # 收集各角色的hidden states
        nsubj_h = collect_hs_at_layer(model, tokenizer, device,
                                       MANIFOLD_ROLES_DATA["nsubj"]["sentences"][:8],
                                       MANIFOLD_ROLES_DATA["nsubj"]["target_words"][:8], li)
        dobj_h = collect_hs_at_layer(model, tokenizer, device,
                                      MANIFOLD_ROLES_DATA["dobj"]["sentences"][:8],
                                      MANIFOLD_ROLES_DATA["dobj"]["target_words"][:8], li)
        amod_h = collect_hs_at_layer(model, tokenizer, device,
                                      MANIFOLD_ROLES_DATA["amod"]["sentences"][:8],
                                      MANIFOLD_ROLES_DATA["amod"]["target_words"][:8], li)

        if nsubj_h is not None and dobj_h is not None and amod_h is not None:
            # 语法方向的范数
            nsubj_center = np.mean(nsubj_h, axis=0)
            dobj_center = np.mean(dobj_h, axis=0)
            amod_center = np.mean(amod_h, axis=0)

            noun_dir = dobj_center - nsubj_center
            noun_norm = np.linalg.norm(noun_dir)

            amod_vec = amod_center - nsubj_center
            # 正交分解
            if noun_norm > 0:
                noun_dir_unit = noun_dir / noun_norm
                amod_proj = np.dot(amod_vec, noun_dir_unit) * noun_dir_unit
                mod_dir = amod_vec - amod_proj
                mod_norm = np.linalg.norm(mod_dir)
            else:
                mod_norm = 0

            # 平均hidden state范数
            avg_hs_norm = np.mean(np.linalg.norm(nsubj_h, axis=1))

            layer_norms[li] = float(avg_hs_norm)
            layer_noun_norms[li] = float(noun_norm)
            layer_mod_norms[li] = float(mod_norm)

            print(f" hs_norm={avg_hs_norm:.2f}, noun_norm={noun_norm:.4f}, mod_norm={mod_norm:.4f}")
        else:
            print(f" 跳过")

    # Step 3: 验证条件A - delta_v/v是否为常数
    print("\n  Step 3: 验证条件A - 层间范数增长率")

    sorted_layers = sorted(layer_noun_norms.keys())
    growth_rates = []

    for i in range(len(sorted_layers) - 1):
        li = sorted_layers[i]
        li1 = sorted_layers[i + 1]

        v_l = layer_noun_norms[li]
        v_l1 = layer_noun_norms[li1]

        if v_l > 1e-10:
            lambda_i = (v_l1 - v_l) / v_l  # 增长率
            growth_rates.append((li, li1, lambda_i))
            print(f"  L{li}→L{li1}: ||v|| {v_l:.4f}→{v_l1:.4f}, λ={lambda_i:.4f}")

    if growth_rates:
        lambdas = [lr[2] for lr in growth_rates]
        mean_lambda = np.mean(lambdas)
        std_lambda = np.std(lambdas)
        cv_lambda = std_lambda / max(abs(mean_lambda), 1e-10)

        print(f"\n  条件A验证: λ = ||delta_v||/||v||")
        print(f"    mean(λ) = {mean_lambda:.4f}")
        print(f"    std(λ) = {std_lambda:.4f}")
        print(f"    CV(λ) = {cv_lambda:.4f}")

        if cv_lambda < 0.3:
            print(f"    ★★★ λ近似常数! 指数增长的条件A成立!")
        else:
            print(f"    ★ λ变化较大, 需要分段分析")

        # 分段分析: 前1/3, 中1/3, 后1/3
        n = len(growth_rates)
        thirds = [growth_rates[:n//3], growth_rates[n//3:2*n//3], growth_rates[2*n//3:]]
        third_names = ["前1/3", "中1/3", "后1/3"]
        for name, third in zip(third_names, thirds):
            if third:
                lambdas_t = [lr[2] for lr in third]
                print(f"    {name}: mean(λ)={np.mean(lambdas_t):.4f}, std={np.std(lambdas_t):.4f}")

        results['lambda_mean'] = float(mean_lambda)
        results['lambda_std'] = float(std_lambda)
        results['lambda_cv'] = float(cv_lambda)

    # Step 4: 验证条件B - LayerNorm的σ是否为常数
    print("\n  Step 4: 验证条件B - LayerNorm的σ是否为常数")

    # LN的σ = sqrt(mean((h - μ)^2)) = RMS(h) (近似)
    # 如果hidden state的范数随层增长, 但LN将其归一化到固定尺度
    # 则σ_l ≈ ||h_l|| / sqrt(d_model) (如果各分量方差相似)

    if layer_norms:
        norms_list = [layer_norms[li] for li in sorted(layer_norms.keys())]
        sigma_estimates = [n / np.sqrt(d_model) for n in norms_list]

        mean_sigma = np.mean(sigma_estimates)
        std_sigma = np.std(sigma_estimates)
        cv_sigma = std_sigma / max(mean_sigma, 1e-10)

        print(f"  条件B验证: σ_l = ||h_l|| / sqrt(d_model)")
        print(f"    mean(σ) = {mean_sigma:.4f}")
        print(f"    std(σ) = {std_sigma:.4f}")
        print(f"    CV(σ) = {cv_sigma:.4f}")

        if cv_sigma < 0.1:
            print(f"    ★★★ σ近似常数! 指数增长的条件B成立!")
            print(f"    → LN将范数归一化到固定尺度, 但语法方向的*相对*范数仍增长")
        else:
            print(f"    ★ σ变化较大")

        results['sigma_cv'] = float(cv_sigma)
        results['sigma_mean'] = float(mean_sigma)

    # Step 5: 关键洞察——LN归一化的是整体范数, 不是语法方向范数
    print("\n  Step 5: 关键洞察")

    print("""
  ★★★ 关键发现:
  LN归一化的是hidden state的整体范数, 而非语法方向的范数!

  h_{l+1} = LN(h_l + delta_l)
    → ||h_{l+1}|| ≈ sqrt(d) (LN后范数固定)
    → 但语法方向在h_l中的*比例*在增长!

  为什么? 因为残差连接使得语法信息逐步累积:
    v_l = v_0 + sum_{k=0}^{l-1} delta_v_k

  如果每层的delta_v_k方向一致(语法方向被强化):
    ||v_l|| ≈ ||v_0|| + l * ||delta_v||  (线性增长)

  但如果delta_v_k也被放大(正反馈):
    ||v_l|| ∝ exp(λl)  (指数增长)

  指数增长的条件: 语法方向被每层的注意力/MLP放大
    → 这是一个正反馈过程: 语法方向越大 → 注意力越聚焦 → 放大越多
""")

    # 验证: 语法方向范数/总范数 的比值是否指数增长
    print("\n  Step 6: 语法方向相对范数(占总范数的比例)的增长")

    if layer_noun_norms and layer_norms:
        relative_norms = {}
        for li in sorted(layer_noun_norms.keys()):
            if li in layer_norms and layer_norms[li] > 0:
                rel = layer_noun_norms[li] / layer_norms[li]
                relative_norms[li] = rel
                print(f"  L{li}: 相对范数 = {rel:.6f}")

        if len(relative_norms) >= 3:
            rel_list = [relative_norms[li] for li in sorted(relative_norms.keys())]
            log_rel = np.log(np.array(rel_list) + 1e-20)
            x = np.arange(len(log_rel))
            slope, intercept = np.polyfit(x, log_rel, 1)
            r_squared = 1 - np.sum((log_rel - (slope * x + intercept))**2) / max(np.sum((log_rel - np.mean(log_rel))**2), 1e-10)

            growth_total = rel_list[-1] / max(rel_list[0], 1e-20)
            print(f"\n  相对范数增长: {rel_list[0]:.6f} → {rel_list[-1]:.6f} ({growth_total:.2f}x)")
            print(f"  指数拟合 R²={r_squared:.4f}")

            results['relative_norm_growth'] = float(growth_total)
            results['relative_norm_r_squared'] = float(r_squared)
            results['relative_norm_slope'] = float(slope)

    results['layer_noun_norms'] = {str(k): v for k, v in layer_noun_norms.items()}
    results['layer_mod_norms'] = {str(k): v for k, v in layer_mod_norms.items()}
    results['layer_norms'] = {str(k): v for k, v in layer_norms.items()}
    results['growth_rates'] = [(li, li1, float(lr)) for li, li1, lr in growth_rates]

    return results


# ===== Exp3: 最后层旋转的归因 =====
def exp3_final_layer_attribution(model, tokenizer, device):
    """最后层旋转的归因: Transformer层内旋转 vs lm_head投影旋转"""
    print("\n" + "="*70)
    print("Exp3: 最后层旋转的归因 ★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Step 1: 在最后2层收集hidden states
    print("\n  Step 1: 在最后3层收集hidden states")

    test_layers = [max(0, n_layers - 3), max(0, n_layers - 2), n_layers - 1]

    layer_planes = {}
    for li in test_layers:
        dirs = get_syntax_directions_at_layer(model, tokenizer, device, model_info, li)
        if 'noun_axis' in dirs and 'modifier_axis' in dirs:
            layer_planes[li] = {
                'noun_axis': dirs['noun_axis'],
                'modifier_axis': dirs['modifier_axis'],
                'ortho': dirs.get('axis_orthogonality', 0),
            }
            print(f"  L{li}: ortho={layer_planes[li]['ortho']:.6f}")

    # Step 2: 在logits空间计算语法方向
    print("\n  Step 2: 在logits空间计算语法方向")

    # 对每个角色, 计算logits
    role_names = ["nsubj", "dobj", "amod"]
    role_logit_centers = {}

    for role in role_names:
        data = MANIFOLD_ROLES_DATA[role]
        logit_vecs = []

        for sent, tw in zip(data["sentences"][:8], data["target_words"][:8]):
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens_list, tw)
            if dep_idx is None:
                continue

            with torch.no_grad():
                outputs = model(**toks)
                logits = outputs.logits[0, dep_idx, :].float().cpu().numpy()

            logit_vecs.append(logits)

        if logit_vecs:
            role_logit_centers[role] = np.mean(logit_vecs, axis=0)
            print(f"  {role}: logit范数={np.linalg.norm(role_logit_centers[role]):.4f}")

    # Step 3: 计算logits空间的语法方向
    print("\n  Step 3: logits空间的语法方向")

    logits_dirs = {}
    if all(r in role_logit_centers for r in role_names):
        nsubj_c = role_logit_centers['nsubj']
        dobj_c = role_logit_centers['dobj']
        amod_c = role_logit_centers['amod']

        noun_dir = dobj_c - nsubj_c
        noun_norm = np.linalg.norm(noun_dir)
        if noun_norm > 0:
            noun_dir = noun_dir / noun_norm

        amod_vec = amod_c - nsubj_c
        amod_proj = np.dot(amod_vec, noun_dir) * noun_dir
        mod_dir = amod_vec - amod_proj
        mod_norm = np.linalg.norm(mod_dir)
        if mod_norm > 0:
            mod_dir = mod_dir / mod_norm

        logits_dirs = {
            'noun_axis': noun_dir,
            'modifier_axis': mod_dir,
            'noun_norm': float(noun_norm),
            'mod_norm': float(mod_norm),
            'ortho': float(np.dot(noun_dir, mod_dir)),
        }
        print(f"  logits正交性: {logits_dirs['ortho']:.6f}")

    # Step 4: 计算旋转角——归因分析
    print("\n  Step 4: 旋转角归因分析")

    last_layer = n_layers - 1
    prev_layer = max(0, n_layers - 2)
    prev2_layer = max(0, n_layers - 3)

    # 层间旋转（同维度，可以用Grassmannian距离）
    dG_ref_step = 0
    dG_last_step = 0

    if prev2_layer in layer_planes and prev_layer in layer_planes:
        dG_ref_step = compute_grassmannian_distance(
            layer_planes[prev2_layer]['noun_axis'], layer_planes[prev2_layer]['modifier_axis'],
            layer_planes[prev_layer]['noun_axis'], layer_planes[prev_layer]['modifier_axis']
        )
    if prev_layer in layer_planes and last_layer in layer_planes:
        dG_last_step = compute_grassmannian_distance(
            layer_planes[prev_layer]['noun_axis'], layer_planes[prev_layer]['modifier_axis'],
            layer_planes[last_layer]['noun_axis'], layer_planes[last_layer]['modifier_axis']
        )

    print(f"  L{prev2_layer}→L{prev_layer} 旋转(参考): {np.degrees(dG_ref_step):.2f}°")
    print(f"  L{prev_layer}→L{last_layer} 旋转(层内): {np.degrees(dG_last_step):.2f}°")

    # lm_head投影的旋转——方法: 用lm_head将最后层的语法方向投影到logits空间
    # 比较投影后的方向与logits空间中直接计算的方向
    print("\n  Step 4B: lm_head投影旋转分析")

    lm_head_rotation_deg = 0
    lm_head_fraction = 0
    layer_fraction = 0

    if last_layer in layer_planes and logits_dirs:
        try:
            lm_head = model.get_output_embeddings()
            if lm_head is not None:
                W = lm_head.weight.detach().float().cpu().numpy()  # [vocab, d_model]

                # 将最后层的语法方向投影到logits空间
                noun_proj = W @ layer_planes[last_layer]['noun_axis']  # [vocab]
                mod_proj = W @ layer_planes[last_layer]['modifier_axis']  # [vocab]

                # 归一化
                noun_proj_norm = np.linalg.norm(noun_proj)
                mod_proj_norm = np.linalg.norm(mod_proj)

                if noun_proj_norm > 0 and mod_proj_norm > 0:
                    noun_proj_unit = noun_proj / noun_proj_norm
                    mod_proj_unit = mod_proj / mod_proj_norm

                    # 投影后的方向与logits空间方向的一致性
                    cos_noun_logit = np.dot(noun_proj_unit, logits_dirs['noun_axis'])
                    cos_mod_logit = np.dot(mod_proj_unit, logits_dirs['modifier_axis'])

                    # 投影后的正交性
                    ortho_projected = float(np.dot(noun_proj_unit, mod_proj_unit))

                    # 投影旋转角 = 1 - cos相似度 (近似)
                    noun_rotation = np.degrees(np.arccos(np.clip(abs(cos_noun_logit), 0, 1)))
                    mod_rotation = np.degrees(np.arccos(np.clip(abs(cos_mod_logit), 0, 1)))

                    print(f"  lm_head投影名词轴 → logits名词轴: cos={cos_noun_logit:.4f}, 旋转≈{noun_rotation:.1f}°")
                    print(f"  lm_head投影修饰语轴 → logits修饰语轴: cos={cos_mod_logit:.4f}, 旋转≈{mod_rotation:.1f}°")
                    print(f"  投影后正交性: {ortho_projected:.6f}")
                    print(f"  logits空间正交性: {logits_dirs['ortho']:.6f}")

                    # 判断: lm_head是否是近似线性变换?
                    # 如果cos > 0.8, 则lm_head近似保持方向 → 旋转主要来自层内
                    # 如果cos < 0.3, 则lm_head大幅旋转方向 → lm_head贡献大
                    avg_cos = (abs(cos_noun_logit) + abs(cos_mod_logit)) / 2
                    lm_head_rotation_deg = (noun_rotation + mod_rotation) / 2

                    print(f"\n  ★★★ 归因分析:")
                    print(f"    层间旋转(L{prev_layer}→L{last_layer}): {np.degrees(dG_last_step):.2f}°")
                    print(f"    lm_head方向保持度(平均cos): {avg_cos:.4f}")
                    print(f"    lm_head旋转: ~{lm_head_rotation_deg:.1f}°")

                    if avg_cos > 0.7:
                        print(f"    → lm_head近似保持方向! 旋转主要来自Transformer层内")
                    elif avg_cos > 0.3:
                        print(f"    → lm_head部分旋转方向, 层内和lm_head都有贡献")
                    else:
                        print(f"    → lm_head大幅旋转方向! 最后层旋转主要由lm_head引起")

                    # 量化归因
                    total_rotation_est = np.degrees(dG_last_step) + lm_head_rotation_deg
                    if total_rotation_est > 1e-10:
                        lm_head_fraction = lm_head_rotation_deg / total_rotation_est
                        layer_fraction = np.degrees(dG_last_step) / total_rotation_est
                    else:
                        lm_head_fraction = 0
                        layer_fraction = 1.0

                    print(f"    层内旋转占比: {layer_fraction:.1%}")
                    print(f"    lm_head投影占比: {lm_head_fraction:.1%}")

                    results['cos_noun_logit'] = float(cos_noun_logit)
                    results['cos_mod_logit'] = float(cos_mod_logit)
                    results['ortho_projected'] = float(ortho_projected)
                    results['avg_cos_lm_head'] = float(avg_cos)

        except Exception as e:
            print(f"  lm_head分析出错: {e}")

    results['dG_last_step'] = float(np.degrees(dG_last_step))
    results['dG_ref_step'] = float(np.degrees(dG_ref_step))
    results['lm_head_rotation_deg'] = float(lm_head_rotation_deg)
    results['lm_head_fraction'] = float(lm_head_fraction)
    results['layer_fraction'] = float(layer_fraction)

    # Step 5: lm_head权重的奇异值分析
    print("\n  Step 5: lm_head权重的奇异值分析")

    # lm_head: [vocab_size, d_model]
    # 语法子空间在d_model中是2维的, 在vocab空间中的投影
    try:
        lm_head = model.get_output_embeddings()
        if lm_head is not None:
            W = lm_head.weight.detach().float().cpu().numpy()  # [vocab, d_model]

            # SVD of lm_head
            # 只计算前50个奇异值(太大无法全算)
            U_w, s_w, Vt_w = np.linalg.svd(W, full_matrices=False)

            print(f"  lm_head: shape={W.shape}")
            print(f"  前10个奇异值: {s_w[:10].tolist()}")
            print(f"  条件数(σ1/σ50): {s_w[0]/max(s_w[min(49,len(s_w)-1)],1e-10):.2f}")

            # lm_head在语法子空间上的投影
            if last_layer in layer_planes:
                noun_axis = layer_planes[last_layer]['noun_axis']
                mod_axis = layer_planes[last_layer]['modifier_axis']

                # 语法方向在V空间中的坐标
                P_syntax = np.column_stack([noun_axis, mod_axis])  # [d_model, 2]
                # lm_head对语法方向的增益: W @ P_syntax
                W_syntax = W @ P_syntax  # [vocab, 2]

                # 语法方向被lm_head放大的程度
                gain_noun = np.linalg.norm(W_syntax[:, 0])
                gain_mod = np.linalg.norm(W_syntax[:, 1])

                # 与W整体范数的比较
                W_norm = np.linalg.norm(W)

                print(f"  lm_head对名词轴增益: {gain_noun:.4f} (相对={gain_noun/W_norm:.6f})")
                print(f"  lm_head对修饰语轴增益: {gain_mod:.4f} (相对={gain_mod/W_norm:.6f})")

                results['lm_head_gain_noun'] = float(gain_noun)
                results['lm_head_gain_mod'] = float(gain_mod)
                results['lm_head_gain_ratio'] = float(gain_noun / max(gain_mod, 1e-10))
    except Exception as e:
        print(f"  lm_head分析出错: {e}")

    results['logits_dirs'] = {
        'ortho': logits_dirs.get('ortho', None),
        'noun_norm': logits_dirs.get('noun_norm', None),
        'mod_norm': logits_dirs.get('mod_norm', None),
    }

    return results


# ===== Exp4: 正交性与因果判据的系统性验证 =====
def exp4_orthogonality_and_criterion(model, tokenizer, device):
    """正交性与因果判据的系统性验证"""
    print("\n" + "="*70)
    print("Exp4: 正交性与因果判据的系统性验证 ★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # ===== 4A: 随机方向PCA对照实验 =====
    print("\n  ===== 4A: 随机方向PCA对照实验 =====")
    print("  问题: PCA是否总是产生正交轴? 语法方向的正交性是否特殊?")

    # 在中间层收集3组随机hidden states
    test_layer = n_layers // 2
    print(f"  测试层: L{test_layer}")

    # 随机组1: 用3组不同的随机句子
    random_sents_group1 = [
        "The weather is nice today",
        "The sun shines brightly",
        "The rain falls gently",
        "The wind blows softly",
        "The snow melts slowly",
        "The clouds drift apart",
        "The storm passed quickly",
        "The fog cleared away",
    ]
    random_sents_group2 = [
        "She runs every morning",
        "He swims in the pool",
        "They dance all night",
        "We sing together now",
        "I read before sleeping",
        "You write very well",
        "It plays the music",
        "She paints the wall",
    ]
    random_sents_group3 = [
        "The table is wooden",
        "The chair looks old",
        "The lamp shines dim",
        "The door closes shut",
        "The window opens wide",
        "The floor feels cold",
        "The ceiling hangs low",
        "The wall stands firm",
    ]

    # 收集每组最后一个token的hidden state
    random_groups = [random_sents_group1, random_sents_group2, random_sents_group3]

    # 随机方向测试: 3组不同句子的中心差是否正交?
    random_centers = []
    for gi, group in enumerate(random_groups):
        hs = []
        for sent in group:
            toks = tokenizer(sent, return_tensors="pt").to(device)
            with torch.no_grad():
                # 收集指定层的hidden state
                captured = {}
                layers_list = get_layers(model)
                target_layer = layers_list[test_layer]
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        captured['h'] = output[0].detach().float().cpu().numpy()
                    else:
                        captured['h'] = output.detach().float().cpu().numpy()
                h_handle = target_layer.register_forward_hook(hook_fn)
                _ = model(**toks)
                h_handle.remove()
                if 'h' in captured:
                    last_idx = captured['h'].shape[1] - 1
                    hs.append(captured['h'][0, last_idx, :])

        if hs:
            center = np.mean(hs, axis=0)
            random_centers.append(center)
            print(f"  随机组{gi+1}: 范数={np.linalg.norm(center):.4f}")

    # 计算随机组之间的方向正交性
    if len(random_centers) >= 3:
        c0, c1, c2 = random_centers[0], random_centers[1], random_centers[2]

        dir_01 = c1 - c0
        dir_02 = c2 - c0

        norm_01 = np.linalg.norm(dir_01)
        norm_02 = np.linalg.norm(dir_02)

        if norm_01 > 0 and norm_02 > 0:
            dir_01_unit = dir_01 / norm_01
            dir_02_unit = dir_02 / norm_02

            # 正交化: Gram-Schmidt
            ortho_component = dir_02_unit - np.dot(dir_02_unit, dir_01_unit) * dir_01_unit
            ortho_norm = np.linalg.norm(ortho_component)

            cos_angle = np.dot(dir_01_unit, dir_02_unit)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            print(f"  随机方向夹角: {angle_deg:.2f}°")
            print(f"  随机方向余弦: {cos_angle:.6f}")

            # 对比: 语法方向的正交性
            dirs = get_syntax_directions_at_layer(model, tokenizer, device, model_info, test_layer)
            if 'axis_orthogonality' in dirs:
                syntax_ortho = dirs['axis_orthogonality']
                syntax_angle = np.degrees(np.arccos(np.clip(abs(syntax_ortho), 0, 1)))
                print(f"  语法方向夹角: {syntax_angle:.2f}°")
                print(f"  语法方向余弦: {syntax_ortho:.6f}")

                if abs(cos_angle) < 0.3 and abs(syntax_ortho) < 0.01:
                    print(f"  ★★★ 随机方向也近似正交(夹角≈{angle_deg:.0f}°), 但语法方向更正交")
                    print(f"  → 语法正交性不是PCA的人为效应, 而是真实的数据结构!")
                elif abs(cos_angle) > 0.3 and abs(syntax_ortho) < 0.01:
                    print(f"  ★★★ 随机方向不正交(夹角={angle_deg:.0f}°), 语法方向正交")
                    print(f"  → 语法正交性是语言特有的结构, PCA只是提取工具")
                else:
                    print(f"  ★ 两者都接近正交, PCA正交化效果显著")

            results['random_cos_angle'] = float(cos_angle)
            results['random_angle_deg'] = float(angle_deg)
            results['syntax_ortho_at_test'] = float(dirs.get('axis_orthogonality', 0))

    # ===== 4B: 更多语法角色的正交性 =====
    print("\n  ===== 4B: 更多语法角色的正交性 =====")

    # 使用5种语法角色的所有组合来测量正交性
    all_role_pairs_ortho = {}
    roles_to_test = ["nsubj", "dobj", "amod", "poss"]

    for test_li in [0, n_layers // 2, n_layers - 1]:
        print(f"\n  L{test_li}:")

        # 收集各角色中心
        role_centers = {}
        for role in roles_to_test:
            data = MANIFOLD_ROLES_DATA[role]
            H = collect_hs_at_layer(model, tokenizer, device,
                                    data["sentences"][:8], data["target_words"][:8], test_li)
            if H is not None:
                role_centers[role] = np.mean(H, axis=0)

        # 计算所有角色对的方向正交性
        role_names_sorted = sorted(role_centers.keys())
        for i, r1 in enumerate(role_names_sorted):
            for j, r2 in enumerate(role_names_sorted):
                if i < j and r1 != r2:
                    dir1 = role_centers[r2] - role_centers[r1]

                    # 找第三个角色做正交分解
                    for r3 in role_names_sorted:
                        if r3 != r1 and r3 != r2:
                            dir2 = role_centers[r3] - role_centers[r1]

                            norm1 = np.linalg.norm(dir1)
                            norm2 = np.linalg.norm(dir2)

                            if norm1 > 0 and norm2 > 0:
                                dir1_unit = dir1 / norm1
                                proj = np.dot(dir2, dir1_unit) * dir1_unit
                                ortho = dir2 - proj
                                ortho_norm = np.linalg.norm(ortho)

                                cos_a = np.dot(dir1_unit, dir2 / norm2)
                                angle = np.degrees(np.arccos(np.clip(abs(cos_a), 0, 1)))

                                pair_key = f"{r1}-{r2}_vs_{r1}-{r3}"
                                if pair_key not in all_role_pairs_ortho:
                                    all_role_pairs_ortho[pair_key] = {}
                                all_role_pairs_ortho[pair_key][test_li] = float(cos_a)

                                print(f"    {r1}→{r2} vs {r1}→{r3}: cos={cos_a:.4f}, angle={angle:.1f}°")
                            break  # 只取第三个角色

    results['role_pairs_ortho'] = all_role_pairs_ortho

    # ===== 4C: 因果判据的跨角色稳定性 =====
    print("\n  ===== 4C: 因果判据的跨角色稳定性 =====")

    # 用不同角色对作为"名词轴"和"修饰语轴"来测量代数量
    # 只在中间层测试
    test_li = n_layers // 2
    print(f"  测试层: L{test_li}")

    role_combination_metrics = {}

    # 定义角色组合: (基准角色, 名词方向角色, 修饰语方向角色)
    role_combos = [
        ("nsubj", "dobj", "amod"),
        ("nsubj", "dobj", "poss"),
        ("nsubj", "amod", "poss"),
    ]

    for base, noun_role, mod_role in role_combos:
        print(f"\n  组合: 基准={base}, 名词方向={noun_role}, 修饰语方向={mod_role}")

        # 收集hidden states
        role_hs = {}
        for role in [base, noun_role, mod_role]:
            data = MANIFOLD_ROLES_DATA[role]
            H = collect_hs_at_layer(model, tokenizer, device,
                                    data["sentences"][:8], data["target_words"][:8], test_li)
            if H is not None:
                role_hs[role] = H

        if len(role_hs) < 3:
            print(f"  数据不足, 跳过")
            continue

        # 计算方向
        base_c = np.mean(role_hs[base], axis=0)
        noun_c = np.mean(role_hs[noun_role], axis=0)
        mod_c = np.mean(role_hs[mod_role], axis=0)

        noun_dir = noun_c - base_c
        noun_norm = np.linalg.norm(noun_dir)
        if noun_norm > 0:
            noun_dir = noun_dir / noun_norm

        mod_vec = mod_c - base_c
        mod_proj = np.dot(mod_vec, noun_dir) * noun_dir
        modifier_dir = mod_vec - mod_proj
        modifier_norm = np.linalg.norm(modifier_dir)
        if modifier_norm > 0:
            modifier_dir = modifier_dir / modifier_norm

        ortho = float(np.dot(noun_dir, modifier_dir))
        print(f"  正交性: {ortho:.6f}")

        # 计算方向导数(简化版, 只用1个测试句)
        test_sent = "The king ruled"
        toks = tokenizer(test_sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        last_idx = input_ids.shape[1] - 1

        embed_layer = model.get_input_embeddings()
        inputs_embeds_base = embed_layer(input_ids).detach().clone().float()

        with torch.no_grad():
            base_logits = model(inputs_embeds=inputs_embeds_base.to(model.dtype)).logits[0, last_idx, :]

        top_k = 30
        base_topk_vals, base_topk_indices = torch.topk(base_logits.float(), top_k)

        epsilon = 0.1
        epsilon2 = 0.5

        noun_tensor = torch.tensor(noun_dir, dtype=torch.float32, device=device)
        mod_tensor = torch.tensor(modifier_dir, dtype=torch.float32, device=device)

        # 方向导数
        inputs_plus_n = inputs_embeds_base.clone()
        inputs_plus_n[0, last_idx, :] += (epsilon * noun_tensor).to(inputs_embeds_base.dtype)
        inputs_minus_n = inputs_embeds_base.clone()
        inputs_minus_n[0, last_idx, :] -= (epsilon * noun_tensor).to(inputs_embeds_base.dtype)

        with torch.no_grad():
            logits_plus_n = model(inputs_embeds=inputs_plus_n.to(model.dtype)).logits[0, last_idx, :]
            logits_minus_n = model(inputs_embeds=inputs_minus_n.to(model.dtype)).logits[0, last_idx, :]

        J_noun = ((logits_plus_n[base_topk_indices] - logits_minus_n[base_topk_indices]) / (2 * epsilon)).float().cpu().numpy()

        inputs_plus_m = inputs_embeds_base.clone()
        inputs_plus_m[0, last_idx, :] += (epsilon * mod_tensor).to(inputs_embeds_base.dtype)
        inputs_minus_m = inputs_embeds_base.clone()
        inputs_minus_m[0, last_idx, :] -= (epsilon * mod_tensor).to(inputs_embeds_base.dtype)

        with torch.no_grad():
            logits_plus_m = model(inputs_embeds=inputs_plus_m.to(model.dtype)).logits[0, last_idx, :]
            logits_minus_m = model(inputs_embeds=inputs_minus_m.to(model.dtype)).logits[0, last_idx, :]

        J_mod = ((logits_plus_m[base_topk_indices] - logits_minus_m[base_topk_indices]) / (2 * epsilon)).float().cpu().numpy()

        # Hessian
        with torch.no_grad():
            logits_center = model(inputs_embeds=inputs_embeds_base.to(model.dtype)).logits[0, last_idx, :]

        inputs_plus2n = inputs_embeds_base.clone()
        inputs_plus2n[0, last_idx, :] += (epsilon2 * noun_tensor).to(inputs_embeds_base.dtype)
        inputs_minus2n = inputs_embeds_base.clone()
        inputs_minus2n[0, last_idx, :] -= (epsilon2 * noun_tensor).to(inputs_embeds_base.dtype)

        with torch.no_grad():
            logits_plus2n = model(inputs_embeds=inputs_plus2n.to(model.dtype)).logits[0, last_idx, :]
            logits_minus2n = model(inputs_embeds=inputs_minus2n.to(model.dtype)).logits[0, last_idx, :]

        H_noun = ((logits_plus2n[base_topk_indices] - 2*logits_center[base_topk_indices] + logits_minus2n[base_topk_indices]) / (epsilon2**2)).float().cpu().numpy()

        inputs_plus2m = inputs_embeds_base.clone()
        inputs_plus2m[0, last_idx, :] += (epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)
        inputs_minus2m = inputs_embeds_base.clone()
        inputs_minus2m[0, last_idx, :] -= (epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)

        with torch.no_grad():
            logits_plus2m = model(inputs_embeds=inputs_plus2m.to(model.dtype)).logits[0, last_idx, :]
            logits_minus2m = model(inputs_embeds=inputs_minus2m.to(model.dtype)).logits[0, last_idx, :]

        H_mod = ((logits_plus2m[base_topk_indices] - 2*logits_center[base_topk_indices] + logits_minus2m[base_topk_indices]) / (epsilon2**2)).float().cpu().numpy()

        # 交叉Hessian
        inputs_pp = inputs_embeds_base.clone()
        inputs_pp[0, last_idx, :] += (epsilon2 * noun_tensor + epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)
        inputs_pm = inputs_embeds_base.clone()
        inputs_pm[0, last_idx, :] += (epsilon2 * noun_tensor - epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)
        inputs_mp = inputs_embeds_base.clone()
        inputs_mp[0, last_idx, :] += (-epsilon2 * noun_tensor + epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)
        inputs_mm = inputs_embeds_base.clone()
        inputs_mm[0, last_idx, :] += (-epsilon2 * noun_tensor - epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)

        with torch.no_grad():
            logits_pp = model(inputs_embeds=inputs_pp.to(model.dtype)).logits[0, last_idx, :]
            logits_pm = model(inputs_embeds=inputs_pm.to(model.dtype)).logits[0, last_idx, :]
            logits_mp = model(inputs_embeds=inputs_mp.to(model.dtype)).logits[0, last_idx, :]
            logits_mm = model(inputs_embeds=inputs_mm.to(model.dtype)).logits[0, last_idx, :]

        H_cross = ((logits_pp[base_topk_indices] - logits_pm[base_topk_indices] - logits_mp[base_topk_indices] + logits_mm[base_topk_indices]) / (4 * epsilon2**2)).float().cpu().numpy()

        # 核心指标
        first_order_noun = np.linalg.norm(J_noun)
        first_order_mod = np.linalg.norm(J_mod)
        noun_curvature = np.linalg.norm(H_noun)
        mod_curvature = np.linalg.norm(H_mod)
        cross_curvature = np.linalg.norm(H_cross)

        nonlinear_ratio_noun = noun_curvature / max(first_order_noun, 1e-10)
        nonlinear_ratio_mod = mod_curvature / max(first_order_mod, 1e-10)
        avg_nonlinearity = (nonlinear_ratio_noun + nonlinear_ratio_mod) / 2

        cos_jacobian = np.dot(J_noun, J_mod) / max(
            np.linalg.norm(J_noun) * np.linalg.norm(J_mod), 1e-10)
        jacobian_overlap = abs(cos_jacobian)

        cross_to_single = cross_curvature / max(
            np.mean([noun_curvature, mod_curvature]), 1e-10)

        combo_key = f"{base}-{noun_role}-{mod_role}"
        role_combination_metrics[combo_key] = {
            'nonlinearity': float(avg_nonlinearity),
            'jacobian_overlap': float(jacobian_overlap),
            'cross_to_single': float(cross_to_single),
            'orthogonality': float(ortho),
        }

        print(f"  非线性: {avg_nonlinearity:.4f}")
        print(f"  Jacobian重叠: {jacobian_overlap:.4f}")
        print(f"  交叉/单轴: {cross_to_single:.4f}")

    # 判据稳定性分析
    if role_combination_metrics:
        print(f"\n  ===== 判据稳定性总结 =====")
        for key, m in role_combination_metrics.items():
            print(f"  {key}: nonlin={m['nonlinearity']:.2f}, overlap={m['jacobian_overlap']:.2f}, "
                  f"cross/single={m['cross_to_single']:.2f}, ortho={m['orthogonality']:.6f}")

        # 判据是否跨角色组合一致?
        nonlinearities = [m['nonlinearity'] for m in role_combination_metrics.values()]
        overlaps = [m['jacobian_overlap'] for m in role_combination_metrics.values()]
        cross_singles = [m['cross_to_single'] for m in role_combination_metrics.values()]

        print(f"\n  判据范围:")
        print(f"    非线性: [{min(nonlinearities):.2f}, {max(nonlinearities):.2f}]")
        print(f"    重叠: [{min(overlaps):.2f}, {max(overlaps):.2f}]")
        print(f"    交叉/单轴: [{min(cross_singles):.2f}, {max(cross_singles):.2f}]")

        nl_cv = np.std(nonlinearities) / max(np.mean(nonlinearities), 1e-10)
        ol_cv = np.std(overlaps) / max(np.mean(overlaps), 1e-10)

        if nl_cv < 0.3 and ol_cv < 0.3:
            print(f"  ★★★ 判据跨角色组合稳定! (CV<0.3)")
        else:
            print(f"  ★ 判据跨角色组合不稳定 (nl_CV={nl_cv:.3f}, ol_CV={ol_cv:.3f})")

    results['role_combination_metrics'] = role_combination_metrics

    return results


# ===== 主函数 =====
def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3, 4])
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CCL-Z Phase20 流形动力学与量化归因 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_curvature_attribution(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_norm_growth_derivation(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_final_layer_attribution(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_orthogonality_and_criterion(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"cclz_exp{args.exp}_{args.model}_results.json")

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return list(convert(v) for v in obj)
            return obj

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {out_path}")

    finally:
        release_model(model)
        print(f"  模型已释放")


if __name__ == "__main__":
    main()
