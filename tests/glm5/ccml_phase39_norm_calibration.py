"""
CCML Phase 39: RMSNorm-Jacobian动力学校准
============================================

核心修正:
1. eps归一化: α = ||δ|| / ||h^l||  (解决硬伤3: eps未归一化)
2. 修正hook: 加法注入, 只修改last token位置 (解决硬伤1: 观测机制问题)
3. 正确分层: 几何(Jacobian) / 归一化(Norm) / 非线性(Activation)

理论模型: δh_out = D_norm · J · δh_in
  Norm = 各向同性投影器 (在h正交子空间中近似各向同性)
  Jacobian = 各向异性放大器

验证三个预测:
P1: 存在α*临界值, α < α* 方向效应被Norm掩盖, α > α* 方向效应显现
P2: 更多RMSNorm层 = 更强方向掩盖 (注入层越深, 方向效应越弱)
P3: 绕过final norm, 方向效应增强

用法:
  python ccml_phase39_norm_calibration.py --model deepseek7b --exp 1
  python ccml_phase39_norm_calibration.py --model deepseek7b --exp 2
  python ccml_phase39_norm_calibration.py --model deepseek7b --exp 3
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
import gc
import time
from scipy import stats

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 配置 =====
ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
N_RANDOM = 10
TEST_SENTENCES = [
    "The cat sat on the mat",
    "She walked to the store yesterday",
    "The scientist discovered a new element",
    "Music fills the quiet room",
    "The river flows through the valley",
]


def get_W_U_np(model):
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.detach().cpu().float().numpy()
    return model.get_output_embeddings().weight.detach().cpu().float().numpy()


def inject_additive(model, input_ids, attention_mask, hook_target, delta,
                    token_pos=-1):
    """
    加法注入: 在hook_target的输出上添加delta (只修改last token位置)
    
    Args:
        hook_target: 注册hook的目标模块
        delta: [d_model] torch tensor (已在正确device上)
        token_pos: 修改的token位置
    
    Returns:
        logits: [vocab_size] numpy array
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            new_h = output[0].clone()
            new_h[0, token_pos, :] += delta.to(new_h.dtype).to(new_h.device)
            return (new_h,) + output[1:]
        new_h = output.clone()
        new_h[0, token_pos, :] += delta.to(new_h.dtype).to(new_h.device)
        return new_h

    handle = hook_target.register_forward_hook(hook_fn)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()
    return outputs.logits[0, -1].cpu().float().numpy()


def get_base_forward(model, tokenizer, device, text):
    """获取base forward的结果"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)

    logits = outputs.logits[0, -1].cpu().float().numpy()
    hs_norms = [outputs.hidden_states[li][0, -1].float().norm().item()
                for li in range(len(outputs.hidden_states))]
    hs_final = outputs.hidden_states[-1][0, -1].detach().cpu().float()

    top_token_id = int(np.argmax(logits))
    second_token_id = int(np.argsort(logits)[-2])
    base_margin = logits[top_token_id] - logits[second_token_id]

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'logits': logits,
        'hs_norms': hs_norms,
        'hs_final': hs_final,
        'top_token_id': top_token_id,
        'second_token_id': second_token_id,
        'base_margin': base_margin,
    }

    del outputs
    torch.cuda.empty_cache()
    return result


def compute_directions(W_U, top_token_id, second_token_id, d_model, n_random=N_RANDOM):
    """计算测试方向 (全部归一化)"""
    # W_U[top]方向
    wu_top = W_U[top_token_id].copy()
    wu_top = wu_top / (np.linalg.norm(wu_top) + 1e-10)

    # W_U[second]方向
    wu_second = W_U[second_token_id].copy()
    wu_second = wu_second / (np.linalg.norm(wu_second) + 1e-10)

    # Margin方向: W_U[top] - W_U[second]
    margin_dir = W_U[top_token_id] - W_U[second_token_id]
    margin_dir = margin_dir / (np.linalg.norm(margin_dir) + 1e-10)

    # -Margin方向
    neg_margin_dir = -margin_dir

    # Random方向
    np.random.seed(42)
    random_dirs = np.random.randn(n_random, d_model)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)

    return {
        'W_U_top': wu_top,
        'W_U_second': wu_second,
        'margin': margin_dir,
        'neg_margin': neg_margin_dir,
        'random': random_dirs,
    }


def measure_direction_effect(base_result, perturbed_logits, top_token_id, second_token_id):
    """测量单个扰动的方向效应"""
    base_margin = base_result['base_margin']
    perturbed_margin = perturbed_logits[top_token_id] - perturbed_logits[second_token_id]
    delta_margin = perturbed_margin - base_margin
    return float(delta_margin)


# ============================================================================
# Sanity Check: 验证加法hook正确性
# ============================================================================

def sanity_check(model, tokenizer, device):
    """验证delta=0的hook不改变logits"""
    print("\n  [Sanity Check] Testing additive hook with delta=0...")

    text = TEST_SENTENCES[0]
    base = get_base_forward(model, tokenizer, device, text)
    input_ids = base['input_ids']
    attention_mask = base['attention_mask']

    # Base logits (no hook)
    base_logits = base['logits']

    # Hook with delta=0
    layers = get_layers(model)
    d_model = base['hs_final'].shape[0]
    delta_zero = torch.zeros(d_model, device=device, dtype=torch.float32)

    hooked_logits = inject_additive(model, input_ids, attention_mask,
                                    layers[-1], delta_zero)

    max_diff = np.max(np.abs(base_logits - hooked_logits))
    mean_diff = np.mean(np.abs(base_logits - hooked_logits))

    print(f"    Max logit diff: {max_diff:.8f}")
    print(f"    Mean logit diff: {mean_diff:.8f}")

    if max_diff < 0.01:
        print("    ✓ PASSED: Additive hook with delta=0 preserves logits")
    else:
        print("    ✗ FAILED: Additive hook changes logits even with delta=0!")
        print(f"    Base top logit: {base_logits.max():.4f}")
        print(f"    Hooked top logit: {hooked_logits.max():.4f}")

    return max_diff < 0.01


# ============================================================================
# 39A: 归一化eps临界值实验 (验证P1)
# ============================================================================

def expA_critical_alpha(model_name, model, tokenizer, device):
    """
    在final hidden state上注入归一化扰动, 寻找α*
    
    α = ||δ|| / ||h_final||
    
    预测: 存在α*, α < α* 方向效应被Norm掩盖, α > α* 方向效应显现
    """
    print(f"\n{'='*70}")
    print(f"39A: 归一化eps临界值实验 — 寻找α* (final layer injection)")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    W_U = get_W_U_np(model)
    d_model = info.d_model
    n_layers = info.n_layers

    all_results = []

    for text in TEST_SENTENCES:
        print(f"\n  Text: '{text[:50]}'")

        base = get_base_forward(model, tokenizer, device, text)
        input_ids = base['input_ids']
        attention_mask = base['attention_mask']
        top_tok = base['top_token_id']
        sec_tok = base['second_token_id']
        hs_norm = base['hs_norms'][-1]  # final layer norm

        print(f"    ||h_final|| = {hs_norm:.2f}, top_token={safe_decode(tokenizer, top_tok)}, "
              f"margin={base['base_margin']:.2f}")

        dirs = compute_directions(W_U, top_tok, sec_tok, d_model)
        layers = get_layers(model)

        for alpha in ALPHAS:
            eps_abs = alpha * hs_norm  # 归一化后的绝对扰动幅度

            # 测量各方向的margin变化
            dm_wu = measure_direction_effect(
                base,
                inject_additive(model, input_ids, attention_mask, layers[-1],
                               torch.tensor(dirs['W_U_top'] * eps_abs,
                                           dtype=torch.float32, device=device)),
                top_tok, sec_tok)

            dm_margin = measure_direction_effect(
                base,
                inject_additive(model, input_ids, attention_mask, layers[-1],
                               torch.tensor(dirs['margin'] * eps_abs,
                                           dtype=torch.float32, device=device)),
                top_tok, sec_tok)

            dm_neg_margin = measure_direction_effect(
                base,
                inject_additive(model, input_ids, attention_mask, layers[-1],
                               torch.tensor(dirs['neg_margin'] * eps_abs,
                                           dtype=torch.float32, device=device)),
                top_tok, sec_tok)

            # Random方向取平均
            dm_random_list = []
            for ri in range(N_RANDOM):
                dm_r = measure_direction_effect(
                    base,
                    inject_additive(model, input_ids, attention_mask, layers[-1],
                                   torch.tensor(dirs['random'][ri] * eps_abs,
                                               dtype=torch.float32, device=device)),
                    top_tok, sec_tok)
                dm_random_list.append(dm_r)
            dm_random_mean = np.mean(np.abs(dm_random_list))
            dm_random_std = np.std(np.abs(dm_random_list))

            # Direction Effect Ratio
            der_wu = np.abs(dm_wu) / (dm_random_mean + 1e-10)
            der_margin = np.abs(dm_margin) / (dm_random_mean + 1e-10)
            der_neg = np.abs(dm_neg_margin) / (dm_random_mean + 1e-10)

            # 综合DER: max(structured) / random
            der_max = max(der_wu, der_margin, der_neg)

            print(f"    α={alpha:.3f} (eps={eps_abs:.2f}): "
                  f"Δmargin: W_U={dm_wu:+.3f}, margin={dm_margin:+.3f}, "
                  f"-margin={dm_neg_margin:+.3f}, rand={dm_random_mean:.3f}±{dm_random_std:.3f} | "
                  f"DER_max={der_max:.2f}x")

            all_results.append({
                'text': text[:40],
                'alpha': alpha,
                'eps_abs': eps_abs,
                'hs_norm': hs_norm,
                'dm_wu_top': dm_wu,
                'dm_margin': dm_margin,
                'dm_neg_margin': dm_neg_margin,
                'dm_random_mean': dm_random_mean,
                'dm_random_std': dm_random_std,
                'der_wu': der_wu,
                'der_margin': der_margin,
                'der_neg': der_neg,
                'der_max': der_max,
            })

            torch.cuda.empty_cache()

    # ===== 汇总 =====
    print(f"\n  ===== 39A SUMMARY =====")

    # 按alpha分组
    by_alpha = {}
    for r in all_results:
        a = r['alpha']
        if a not in by_alpha:
            by_alpha[a] = {'der_max': [], 'dm_margin': [], 'dm_neg_margin': [],
                           'dm_random': [], 'dm_wu': []}
        by_alpha[a]['der_max'].append(r['der_max'])
        by_alpha[a]['dm_margin'].append(r['dm_margin'])
        by_alpha[a]['dm_neg_margin'].append(r['dm_neg_margin'])
        by_alpha[a]['dm_random'].append(r['dm_random_mean'])
        by_alpha[a]['dm_wu'].append(r['dm_wu_top'])

    print(f"\n  {'α':>8} | {'DER_max':>8} | {'Δmargin':>8} | {'Δ-margin':>8} | {'Δrandom':>8} | {'ΔW_U':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    alpha_star = None
    for a in sorted(by_alpha.keys()):
        d = by_alpha[a]
        mean_der = np.mean(d['der_max'])
        mean_dm_margin = np.mean(d['dm_margin'])
        mean_dm_neg = np.mean(d['dm_neg_margin'])
        mean_dm_rand = np.mean(d['dm_random'])
        mean_dm_wu = np.mean(d['dm_wu'])

        marker = ""
        if alpha_star is None and mean_der > 1.5:
            alpha_star = a
            marker = " ← α*?"

        print(f"  {a:>8.3f} | {mean_der:>8.2f}x | {mean_dm_margin:>+8.3f} | "
              f"{mean_dm_neg:>+8.3f} | {mean_dm_rand:>8.3f} | {mean_dm_wu:>+8.3f}{marker}")

    if alpha_star:
        print(f"\n  ★ α* ≈ {alpha_star}: 方向效应开始显现的临界值")
    else:
        print(f"\n  ⚠ 在测试范围内未找到α*")

    return all_results


# ============================================================================
# 39B: 层深度 vs 方向效应 (验证P2)
# ============================================================================

def expB_layer_depth(model_name, model, tokenizer, device):
    """
    在不同层注入相同(归一化)扰动, 测量方向效应
    
    预测: 注入层越深(离输出越近, 经过的RMSNorm越少), 方向效应越强
    注入层越浅(经过的RMSNorm越多), 方向效应越弱
    
    每经过一层DecoderLayer, 扰动要经过2个RMSNorm:
      input_layernorm + post_attention_layernorm
    """
    print(f"\n{'='*70}")
    print(f"39B: 层深度 vs 方向效应 (RMSNorm层数越多, 方向掩盖越强)")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    W_U = get_W_U_np(model)
    d_model = info.d_model
    n_layers = info.n_layers

    # 选择注入层: 从深到浅
    inject_layers = [n_layers - 1, n_layers - 3, n_layers - 6,
                     max(n_layers // 2, 0), max(n_layers // 4, 0)]
    inject_layers = sorted(set([l for l in inject_layers if l >= 0]))

    # 固定alpha (选一个中等值)
    test_alphas = [0.05, 0.1, 0.2, 0.5]

    all_results = []

    for text in TEST_SENTENCES[:3]:
        print(f"\n  Text: '{text[:50]}'")

        base = get_base_forward(model, tokenizer, device, text)
        input_ids = base['input_ids']
        attention_mask = base['attention_mask']
        top_tok = base['top_token_id']
        sec_tok = base['second_token_id']

        dirs = compute_directions(W_U, top_tok, sec_tok, d_model)
        layers = get_layers(model)

        for li in inject_layers:
            hs_norm = base['hs_norms'][li + 1]  # +1因为hidden_states[0]是embedding
            n_norms_remaining = 2 * (n_layers - 1 - li) + 1  # 到输出还有多少个RMSNorm

            for alpha in test_alphas:
                eps_abs = alpha * hs_norm

                # margin方向
                dm_margin = measure_direction_effect(
                    base,
                    inject_additive(model, input_ids, attention_mask, layers[li],
                                   torch.tensor(dirs['margin'] * eps_abs,
                                               dtype=torch.float32, device=device)),
                    top_tok, sec_tok)

                # -margin方向
                dm_neg = measure_direction_effect(
                    base,
                    inject_additive(model, input_ids, attention_mask, layers[li],
                                   torch.tensor(dirs['neg_margin'] * eps_abs,
                                               dtype=torch.float32, device=device)),
                    top_tok, sec_tok)

                # Random方向
                dm_rand_list = []
                for ri in range(N_RANDOM):
                    dm_r = measure_direction_effect(
                        base,
                        inject_additive(model, input_ids, attention_mask, layers[li],
                                       torch.tensor(dirs['random'][ri] * eps_abs,
                                                   dtype=torch.float32, device=device)),
                        top_tok, sec_tok)
                    dm_rand_list.append(dm_r)
                dm_rand_mean = np.mean(np.abs(dm_rand_list))

                # Direction Effect Ratio
                der = max(np.abs(dm_margin), np.abs(dm_neg)) / (dm_rand_mean + 1e-10)

                # 方向不对称性: margin vs -margin
                asymmetry = dm_margin - dm_neg  # 正值=margin方向增大margin更多

                print(f"    L{li} (norms_remaining={n_norms_remaining}, "
                      f"α={alpha:.2f}): DER={der:.2f}x, "
                      f"Δmargin={dm_margin:+.3f}, Δ-margin={dm_neg:+.3f}, "
                      f"asymmetry={asymmetry:+.3f}")

                all_results.append({
                    'text': text[:40],
                    'layer': li,
                    'n_norms_remaining': n_norms_remaining,
                    'alpha': alpha,
                    'hs_norm': hs_norm,
                    'dm_margin': dm_margin,
                    'dm_neg_margin': dm_neg,
                    'dm_random_mean': dm_rand_mean,
                    'der': der,
                    'asymmetry': asymmetry,
                })

                torch.cuda.empty_cache()

    # ===== 汇总 =====
    print(f"\n  ===== 39B SUMMARY =====")

    # 按层分组 (固定alpha=0.1)
    by_layer = {}
    for r in all_results:
        if abs(r['alpha'] - 0.1) < 0.01:
            li = r['layer']
            if li not in by_layer:
                by_layer[li] = {'der': [], 'asymmetry': [], 'n_norms': r['n_norms_remaining']}
            by_layer[li]['der'].append(r['der'])
            by_layer[li]['asymmetry'].append(r['asymmetry'])

    print(f"\n  At α=0.1:")
    print(f"  {'Layer':>6} | {'Norms→out':>10} | {'DER':>8} | {'Asymmetry':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

    for li in sorted(by_layer.keys()):
        d = by_layer[li]
        print(f"  {li:>6} | {d['n_norms']:>10} | {np.mean(d['der']):>8.2f}x | "
              f"{np.mean(d['asymmetry']):>+10.3f}")

    # 按alpha分组 (所有层汇总)
    by_alpha = {}
    for r in all_results:
        a = r['alpha']
        if a not in by_alpha:
            by_alpha[a] = {'der': [], 'asymmetry': []}
        by_alpha[a]['der'].append(r['der'])
        by_alpha[a]['asymmetry'].append(r['asymmetry'])

    print(f"\n  Across all layers:")
    for a in sorted(by_alpha.keys()):
        d = by_alpha[a]
        print(f"    α={a:.2f}: mean DER={np.mean(d['der']):.2f}x, "
              f"mean asymmetry={np.mean(d['asymmetry']):+.3f}")

    # 检查P2: 层越深(norms越少), DER越大?
    layer_der_pairs = [(by_layer[li]['n_norms'], np.mean(by_layer[li]['der']))
                       for li in sorted(by_layer.keys())]
    if len(layer_der_pairs) >= 3:
        norms_list = [p[0] for p in layer_der_pairs]
        ders_list = [p[1] for p in layer_der_pairs]
        r_corr, p_val = stats.pearsonr(norms_list, ders_list)
        print(f"\n  P2检验: r(n_norms, DER) = {r_corr:.3f} (p={p_val:.3f})")
        if r_corr < -0.5:
            print(f"  ★★★ P2确认: 更多RMSNorm = 更弱方向效应 (r={r_corr:.2f})")
        elif r_corr > 0.5:
            print(f"  ⚠ P2被否定: 更多RMSNorm反而增强方向效应?!")
        else:
            print(f"  ★ P2不确定: 无显著相关 (r={r_corr:.2f})")

    return all_results


# ============================================================================
# 39C: 绕过Final Norm (验证P3)
# ============================================================================

def expC_bypass_norm(model_name, model, tokenizer, device):
    """
    对比有/无final RMSNorm的方向效应
    
    P3预测: 绕过norm → 方向效应更强 (在更小α就显现)
    
    实现: 用hook将model.model.norm变为identity
    """
    print(f"\n{'='*70}")
    print(f"39C: 绕过Final Norm — 对比有/无RMSNorm的方向效应")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    W_U = get_W_U_np(model)
    d_model = info.d_model
    n_layers = info.n_layers

    test_alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    all_results = []

    for text in TEST_SENTENCES[:3]:
        print(f"\n  Text: '{text[:50]}'")

        base = get_base_forward(model, tokenizer, device, text)
        input_ids = base['input_ids']
        attention_mask = base['attention_mask']
        top_tok = base['top_token_id']
        sec_tok = base['second_token_id']
        hs_norm = base['hs_norms'][-1]

        dirs = compute_directions(W_U, top_tok, sec_tok, d_model)
        layers = get_layers(model)

        for alpha in test_alphas:
            eps_abs = alpha * hs_norm

            # ===== 有Norm (正常) =====
            dm_margin_norm = measure_direction_effect(
                base,
                inject_additive(model, input_ids, attention_mask, layers[-1],
                               torch.tensor(dirs['margin'] * eps_abs,
                                           dtype=torch.float32, device=device)),
                top_tok, sec_tok)

            dm_neg_norm = measure_direction_effect(
                base,
                inject_additive(model, input_ids, attention_mask, layers[-1],
                               torch.tensor(dirs['neg_margin'] * eps_abs,
                                           dtype=torch.float32, device=device)),
                top_tok, sec_tok)

            dm_rand_norm_list = []
            for ri in range(N_RANDOM):
                dm_r = measure_direction_effect(
                    base,
                    inject_additive(model, input_ids, attention_mask, layers[-1],
                                   torch.tensor(dirs['random'][ri] * eps_abs,
                                               dtype=torch.float32, device=device)),
                    top_tok, sec_tok)
                dm_rand_norm_list.append(dm_r)
            dm_rand_norm_mean = np.mean(np.abs(dm_rand_norm_list))

            der_norm = max(np.abs(dm_margin_norm), np.abs(dm_neg_norm)) / (dm_rand_norm_mean + 1e-10)

            # ===== 无Norm (bypass model.model.norm) =====
            # 用hook将model.model.norm变为identity
            def bypass_hook(module, input, output):
                return input[0]  # 返回输入(跳过norm)

            handle = model.model.norm.register_forward_hook(bypass_hook)

            # 需要重新获取base (因为norm被绕过)
            with torch.no_grad():
                base_no_norm_out = model(input_ids=input_ids, attention_mask=attention_mask)
            base_logits_no_norm = base_no_norm_out.logits[0, -1].cpu().float().numpy()
            base_margin_no_norm = base_logits_no_norm[top_tok] - base_logits_no_norm[sec_tok]

            del base_no_norm_out

            dm_margin_no_norm = measure_direction_effect(
                {'base_margin': base_margin_no_norm},
                inject_additive(model, input_ids, attention_mask, layers[-1],
                               torch.tensor(dirs['margin'] * eps_abs,
                                           dtype=torch.float32, device=device)),
                top_tok, sec_tok)

            dm_neg_no_norm = measure_direction_effect(
                {'base_margin': base_margin_no_norm},
                inject_additive(model, input_ids, attention_mask, layers[-1],
                               torch.tensor(dirs['neg_margin'] * eps_abs,
                                           dtype=torch.float32, device=device)),
                top_tok, sec_tok)

            dm_rand_no_norm_list = []
            for ri in range(N_RANDOM):
                dm_r = measure_direction_effect(
                    {'base_margin': base_margin_no_norm},
                    inject_additive(model, input_ids, attention_mask, layers[-1],
                                   torch.tensor(dirs['random'][ri] * eps_abs,
                                               dtype=torch.float32, device=device)),
                    top_tok, sec_tok)
                dm_rand_no_norm_list.append(dm_r)
            dm_rand_no_norm_mean = np.mean(np.abs(dm_rand_no_norm_list))

            der_no_norm = max(np.abs(dm_margin_no_norm), np.abs(dm_neg_no_norm)) / (dm_rand_no_norm_mean + 1e-10)

            handle.remove()  # 恢复norm

            print(f"    α={alpha:.3f}: DER_norm={der_norm:.2f}x → DER_no_norm={der_no_norm:.2f}x "
                  f"(×{der_no_norm/(der_norm+1e-10):.2f}) | "
                  f"Δmargin: norm={dm_margin_norm:+.3f} no_norm={dm_margin_no_norm:+.3f}")

            all_results.append({
                'text': text[:40],
                'alpha': float(alpha),
                'hs_norm': float(hs_norm),
                'base_margin_norm': float(base['base_margin']),
                'base_margin_no_norm': float(base_margin_no_norm),
                'dm_margin_norm': float(dm_margin_norm),
                'dm_neg_margin_norm': float(dm_neg_norm),
                'dm_random_mean_norm': float(dm_rand_norm_mean),
                'der_norm': float(der_norm),
                'dm_margin_no_norm': float(dm_margin_no_norm),
                'dm_neg_margin_no_norm': float(dm_neg_no_norm),
                'dm_random_mean_no_norm': float(dm_rand_no_norm_mean),
                'der_no_norm': float(der_no_norm),
                'der_ratio': float(der_no_norm / (der_norm + 1e-10)),
            })

            torch.cuda.empty_cache()

    # ===== 汇总 =====
    print(f"\n  ===== 39C SUMMARY =====")

    by_alpha = {}
    for r in all_results:
        a = r['alpha']
        if a not in by_alpha:
            by_alpha[a] = {'der_norm': [], 'der_no_norm': [], 'der_ratio': []}
        by_alpha[a]['der_norm'].append(r['der_norm'])
        by_alpha[a]['der_no_norm'].append(r['der_no_norm'])
        by_alpha[a]['der_ratio'].append(r['der_ratio'])

    print(f"\n  {'α':>8} | {'DER_norm':>10} | {'DER_no_norm':>10} | {'Ratio':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for a in sorted(by_alpha.keys()):
        d = by_alpha[a]
        print(f"  {a:>8.3f} | {np.mean(d['der_norm']):>10.2f}x | "
              f"{np.mean(d['der_no_norm']):>10.2f}x | "
              f"{np.mean(d['der_ratio']):>8.2f}x")

    # P3判定
    mean_ratio = np.mean([r['der_ratio'] for r in all_results])
    mean_der_norm = np.mean([r['der_norm'] for r in all_results])
    mean_der_no_norm = np.mean([r['der_no_norm'] for r in all_results])

    print(f"\n  Overall: DER_norm={mean_der_norm:.2f}x → DER_no_norm={mean_der_no_norm:.2f}x "
          f"(ratio={mean_ratio:.2f}x)")

    if mean_der_no_norm > 1.5 * mean_der_norm:
        print(f"  ★★★ P3确认: 绕过Norm显著增强方向效应!")
    elif mean_der_no_norm > mean_der_norm * 1.1:
        print(f"  ★★ P3部分确认: 绕过Norm轻微增强方向效应")
    else:
        print(f"  ★ P3未确认: 绕过Norm未增强方向效应")

    return all_results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 39: RMSNorm-Jacobian Calibration")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3],
                       help="1=39A critical alpha, 2=39B layer depth, 3=39C bypass norm")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Phase 39: RMSNorm-Jacobian动力学校准")
    print(f"Model: {args.model}, Experiment: {args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)

    try:
        # Sanity check first
        sanity_ok = sanity_check(model, tokenizer, device)
        if not sanity_ok:
            print("  ⚠ Sanity check failed! Results may be unreliable.")

        if args.exp == 1:
            results = expA_critical_alpha(args.model, model, tokenizer, device)
            out_name = f"phase39_expA_{args.model}_results.json"
        elif args.exp == 2:
            results = expB_layer_depth(args.model, model, tokenizer, device)
            out_name = f"phase39_expB_{args.model}_results.json"
        elif args.exp == 3:
            results = expC_bypass_norm(args.model, model, tokenizer, device)
            out_name = f"phase39_expC_{args.model}_results.json"

        out_path = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp', out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")

    finally:
        release_model(model)
