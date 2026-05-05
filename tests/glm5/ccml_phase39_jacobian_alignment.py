"""
CCML Phase 39+ : 归一化eps下重做Jacobian-W_U对齐
===================================================

Phase 37B用绝对eps=1e-3测到Jacobian top direction对齐W_U(3.4x)
Phase 38A用80个方向有限差分测到1.2x(采样不足的null result)

问题: Phase 38的eps=1e-3, ||h||≈200, 所以α=1e-3/200=5e-6, 远在α*以下!
→ 有限差分估计Jacobian时, 扰动太小, 被RMSNorm掩盖

本实验: 用归一化eps(α=0.05-0.1)重新估计Jacobian,
        测Jacobian top output direction vs W_U的对齐

方法: 用大量随机方向(200个)的有限差分, 在正确的α下估计Jacobian谱
      然后测top output direction与W_U[top_tok], margin_dir的对齐

关键修正:
1. eps归一化: α = eps / ||h^l||
2. 修正hook: 加法注入
3. 更多方向(200 vs 80)

用法:
  python ccml_phase39_jacobian_alignment.py --model deepseek7b
  python ccml_phase39_jacobian_alignment.py --model glm4
  python ccml_phase39_jacobian_alignment.py --model qwen3
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


TEST_SENTENCES = [
    "The cat sat on the mat",
    "The scientist discovered a new element",
    "Music fills the quiet room",
]


def get_W_U_np(model):
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.detach().cpu().float().numpy()
    return model.get_output_embeddings().weight.detach().cpu().float().numpy()


def compute_jacobian_cols_normalized(model, tokenizer, device, text, layer_idx,
                                     alpha=0.05, n_dirs=200, token_pos=-1):
    """
    用归一化eps的有限差分估计Jacobian的列
    
    对每个随机方向d:
      Jd ≈ (f(h + α||h||d) - f(h - α||h||d)) / (2α||h||)
    
    这给出Jacobian作用在d上的结果(Jacobian的列空间采样)
    
    Returns:
        J_cols: [n_dirs, d_model] — Jacobian在n_dirs个方向上的输出
        hs_norm: float — 该层hidden state的范数
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Base forward
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    
    hs_base = outputs.hidden_states[layer_idx][0, token_pos].detach().clone()
    hs_norm = hs_base.float().norm().item()
    d_model = hs_base.shape[0]
    eps_abs = alpha * hs_norm
    
    # Base final hidden state
    final_base = outputs.hidden_states[-1][0, token_pos].cpu().float().numpy()
    
    del outputs
    torch.cuda.empty_cache()
    
    # 随机方向
    np.random.seed(42 + layer_idx)
    directions = np.random.randn(n_dirs, d_model)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    J_cols = []
    layers = get_layers(model)
    
    for i in range(n_dirs):
        d_np = directions[i]
        d_t = torch.tensor(d_np * eps_abs, dtype=torch.float32, device=device)
        
        # +扰动
        def make_hook(delta, pos):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    new_h = output[0].clone()
                    new_h[0, pos, :] += delta.to(new_h.dtype).to(new_h.device)
                    return (new_h,) + output[1:]
                new_h = output.clone()
                new_h[0, pos, :] += delta.to(new_h.dtype).to(new_h.device)
                return new_h
            return hook_fn
        
        handle = layers[layer_idx].register_forward_hook(make_hook(d_t, token_pos))
        with torch.no_grad():
            out_p = model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        handle.remove()
        
        final_p = out_p.hidden_states[-1][0, token_pos].cpu().float().numpy()
        del out_p
        
        # -扰动
        handle = layers[layer_idx].register_forward_hook(make_hook(-d_t, token_pos))
        with torch.no_grad():
            out_m = model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        handle.remove()
        
        final_m = out_m.hidden_states[-1][0, token_pos].cpu().float().numpy()
        del out_m
        
        # Jacobian列: (f(h+δ) - f(h-δ)) / (2δ)
        J_col = (final_p - final_m) / (2 * eps_abs)
        J_cols.append(J_col)
        
        if i % 50 == 0:
            print(f"      Direction {i}/{n_dirs}...")
            torch.cuda.empty_cache()
    
    return np.array(J_cols), directions, hs_norm, final_base


def exp_jacobian_wu_alignment(model_name, model, tokenizer, device):
    """
    在归一化eps下重新验证Jacobian-W_U对齐
    
    这是Phase 37B和38A的修正版本:
    - 用α=0.05(在α*以上)代替绝对eps=1e-3(α≈5e-6, 在α*以下)
    - 用200个方向代替80个
    """
    print(f"\n{'='*70}")
    print(f"39+: 归一化eps下Jacobian-W_U对齐验证")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U_np(model)
    print(f"  W_U shape: {W_U.shape}")
    
    # W_U SVD
    print(f"  Computing W_U SVD...")
    _, s_wu, Vt_wu = np.linalg.svd(W_U, full_matrices=False)
    wu_top_dirs = Vt_wu[:20]  # top-20 W_U directions
    
    # 采样层
    sample_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2]
    sample_layers = sorted(set([l for l in sample_layers if l >= 0]))
    print(f"  Sample layers: {sample_layers}")
    
    all_results = []
    
    for text in TEST_SENTENCES:
        print(f"\n  Text: '{text[:50]}'")
        
        # 获取top token
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1].cpu().float().numpy()
        top_tok = int(np.argmax(logits))
        sec_tok = int(np.argsort(logits)[-2])
        del outputs
        torch.cuda.empty_cache()
        
        # W_U方向
        wu_top = W_U[top_tok] / (np.linalg.norm(W_U[top_tok]) + 1e-10)
        wu_sec = W_U[sec_tok] / (np.linalg.norm(W_U[sec_tok]) + 1e-10)
        margin_dir = wu_top - wu_sec
        margin_dir = margin_dir / (np.linalg.norm(margin_dir) + 1e-10)
        
        for li in sample_layers:
            print(f"\n    L{li}: Computing Jacobian (α=0.05, n_dirs=200)...")
            try:
                J_cols, input_dirs, hs_norm, final_base = compute_jacobian_cols_normalized(
                    model, tokenizer, device, text, li,
                    alpha=0.05, n_dirs=200, token_pos=-1
                )
                
                print(f"    ||h|| = {hs_norm:.1f}, eps_abs = {0.05 * hs_norm:.1f}")
                
                # SVD of J_cols → Jacobian output directions
                # J_cols: [200, d_model]
                U_j, sigma_j, Vt_j = np.linalg.svd(J_cols, full_matrices=False)
                # Vt_j: [min(200, d_model), d_model]
                # Vt_j的行是Jacobian output space的主方向
                
                n_top = min(20, len(sigma_j))
                jac_top_dirs = Vt_j[:n_top]
                
                # ===== 核心测量: Jacobian top direction vs W_U方向 =====
                
                # 1. Jacobian top-1 vs W_U[top_tok]
                cos_j_wutop = np.abs(np.dot(jac_top_dirs[0], wu_top))
                
                # 2. Jacobian top-1 vs margin direction
                cos_j_margin = np.abs(np.dot(jac_top_dirs[0], margin_dir))
                
                # 3. Jacobian top-1 vs W_U SVD top-1
                cos_j_wusvd1 = np.abs(np.dot(jac_top_dirs[0], wu_top_dirs[0]))
                
                # 4. Jacobian top-k vs margin (平均)
                cos_j_topk_margin = np.mean([np.abs(np.dot(jac_top_dirs[k], margin_dir))
                                             for k in range(min(5, n_top))])
                
                # 5. Jacobian top-k vs W_U top-20 (最大对齐)
                cos_j_topk_wutop20 = []
                for k in range(min(10, n_top)):
                    max_cos = np.max(np.abs(wu_top_dirs @ jac_top_dirs[k]))
                    cos_j_topk_wutop20.append(max_cos)
                mean_j_wu_align = np.mean(cos_j_topk_wutop20[:5])
                
                # ===== 随机baseline =====
                n_random = 200
                np.random.seed(123 + li)
                rand_vecs = np.random.randn(n_random, d_model)
                rand_vecs = rand_vecs / np.linalg.norm(rand_vecs, axis=1, keepdims=True)
                
                rand_cos_wutop = np.abs(rand_vecs @ wu_top)
                rand_cos_margin = np.abs(rand_vecs @ margin_dir)
                rand_cos_wusvd1 = np.abs(rand_vecs @ wu_top_dirs[0])
                rand_max_wutop20 = np.max(np.abs(rand_vecs @ wu_top_dirs.T), axis=1)
                
                # ===== Jacobian谱 =====
                sigma_ratio = sigma_j[0] / (sigma_j[1] + 1e-10)
                effective_rank = np.sum(sigma_j[:20]) / (np.sum(sigma_j) + 1e-10)
                
                # ===== 增益vs对齐(Phase 38B的修正版) =====
                gains = sigma_j[:20]
                alignments_wu = [np.max(np.abs(wu_top_dirs @ jac_top_dirs[k]))
                                 for k in range(min(20, n_top))]
                alignments_margin = [np.abs(np.dot(jac_top_dirs[k], margin_dir))
                                     for k in range(min(20, n_top))]
                
                if np.std(alignments_wu) > 1e-10:
                    r_gain_wu, p_gw = stats.pearsonr(gains[:len(alignments_wu)], alignments_wu)
                else:
                    r_gain_wu, p_gw = 0, 1
                
                if np.std(alignments_margin) > 1e-10:
                    r_gain_margin, p_gm = stats.pearsonr(gains[:len(alignments_margin)], alignments_margin)
                else:
                    r_gain_margin, p_gm = 0, 1
                
                # ===== 报告 =====
                mult_wutop = cos_j_wutop / (np.mean(rand_cos_wutop) + 1e-10)
                mult_margin = cos_j_margin / (np.mean(rand_cos_margin) + 1e-10)
                mult_wusvd = cos_j_wusvd1 / (np.mean(rand_cos_wusvd1) + 1e-10)
                mult_wutop20 = mean_j_wu_align / (np.mean(rand_max_wutop20) + 1e-10)
                
                print(f"    L{li} Results:")
                print(f"      cos(J_top, W_U[top]) = {cos_j_wutop:.4f} (rand={np.mean(rand_cos_wutop):.4f}, "
                      f"{mult_wutop:.1f}x)")
                print(f"      cos(J_top, margin)   = {cos_j_margin:.4f} (rand={np.mean(rand_cos_margin):.4f}, "
                      f"{mult_margin:.1f}x)")
                print(f"      cos(J_top, W_U_SVD1) = {cos_j_wusvd1:.4f} (rand={np.mean(rand_cos_wusvd1):.4f}, "
                      f"{mult_wusvd:.1f}x)")
                print(f"      mean(J_top5, W_U20)  = {mean_j_wu_align:.4f} (rand={np.mean(rand_max_wutop20):.4f}, "
                      f"{mult_wutop20:.1f}x)")
                print(f"      σ₁/σ₂ = {sigma_ratio:.1f}, eff_rank(20) = {effective_rank:.3f}")
                print(f"      r(gain, W_U_align) = {r_gain_wu:.3f} (p={p_gw:.2e})")
                print(f"      r(gain, margin_align) = {r_gain_margin:.3f} (p={p_gm:.2e})")
                
                all_results.append({
                    'layer': li,
                    'text': text[:40],
                    'hs_norm': float(hs_norm),
                    'cos_j_wutop': float(cos_j_wutop),
                    'cos_j_margin': float(cos_j_margin),
                    'cos_j_wusvd1': float(cos_j_wusvd1),
                    'mean_j_wutop20': float(mean_j_wu_align),
                    'rand_cos_wutop': float(np.mean(rand_cos_wutop)),
                    'rand_cos_margin': float(np.mean(rand_cos_margin)),
                    'rand_cos_wusvd1': float(np.mean(rand_cos_wusvd1)),
                    'rand_max_wutop20': float(np.mean(rand_max_wutop20)),
                    'mult_wutop': float(mult_wutop),
                    'mult_margin': float(mult_margin),
                    'mult_wusvd': float(mult_wusvd),
                    'mult_wutop20': float(mult_wutop20),
                    'sigma_ratio': float(sigma_ratio),
                    'effective_rank': float(effective_rank),
                    'r_gain_wu': float(r_gain_wu),
                    'r_gain_margin': float(r_gain_margin),
                    'sigma_top10': [float(s) for s in sigma_j[:10]],
                })
                
            except Exception as e:
                print(f"    L{li}: ERROR - {e}")
                import traceback
                traceback.print_exc()
            
            torch.cuda.empty_cache()
    
    # ===== 汇总 =====
    print(f"\n  ===== 39+ SUMMARY =====")
    
    # 按层分组
    layer_data = {}
    for r in all_results:
        li = r['layer']
        if li not in layer_data:
            layer_data[li] = {'mult_wutop': [], 'mult_margin': [], 'mult_wusvd': [],
                             'mult_wutop20': [], 'sigma_ratio': [], 'r_gain_margin': []}
        layer_data[li]['mult_wutop'].append(r['mult_wutop'])
        layer_data[li]['mult_margin'].append(r['mult_margin'])
        layer_data[li]['mult_wusvd'].append(r['mult_wusvd'])
        layer_data[li]['mult_wutop20'].append(r['mult_wutop20'])
        layer_data[li]['sigma_ratio'].append(r['sigma_ratio'])
        layer_data[li]['r_gain_margin'].append(r['r_gain_margin'])
    
    print(f"\n  {'Layer':>6} | {'W_U[top]':>8} | {'margin':>8} | {'W_U_SVD':>8} | {'W_U_20':>8} | {'σ₁/σ₂':>6} | {'r(g,margin)':>11}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*11}")
    
    for li in sorted(layer_data.keys()):
        d = layer_data[li]
        print(f"  {li:>6} | {np.mean(d['mult_wutop']):>8.1f}x | {np.mean(d['mult_margin']):>8.1f}x | "
              f"{np.mean(d['mult_wusvd']):>8.1f}x | {np.mean(d['mult_wutop20']):>8.1f}x | "
              f"{np.mean(d['sigma_ratio']):>6.1f} | {np.mean(d['r_gain_margin']):>+11.3f}")
    
    # 总体判定
    all_mult_margin = [r['mult_margin'] for r in all_results]
    all_mult_wutop20 = [r['mult_wutop20'] for r in all_results]
    all_r_gain_margin = [r['r_gain_margin'] for r in all_results]
    
    print(f"\n  Overall:")
    print(f"    Mean mult(margin): {np.mean(all_mult_margin):.1f}x")
    print(f"    Mean mult(W_U_20): {np.mean(all_mult_wutop20):.1f}x")
    print(f"    Mean r(gain, margin_align): {np.mean(all_r_gain_margin):+.3f}")
    
    if np.mean(all_mult_margin) > 5:
        print(f"  ★★★ RASC确认: Jacobian top direction对齐margin方向 ({np.mean(all_mult_margin):.0f}x over random)")
    elif np.mean(all_mult_margin) > 2:
        print(f"  ★★ RASC部分确认: Jacobian对齐margin方向弱于预期 ({np.mean(all_mult_margin):.1f}x)")
    else:
        print(f"  ★ RASC未确认: Jacobian对齐margin方向不显著 ({np.mean(all_mult_margin):.1f}x)")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase 39+: 归一化eps下Jacobian-W_U对齐验证")
    print(f"Model: {args.model}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    
    try:
        results = exp_jacobian_wu_alignment(args.model, model, tokenizer, device)
        
        out_name = f"phase39_jacobian_align_{args.model}_results.json"
        out_path = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp', out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")
    
    finally:
        release_model(model)
