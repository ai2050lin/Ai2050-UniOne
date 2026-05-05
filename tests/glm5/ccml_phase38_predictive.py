"""
CCML Phase 38 (v2): 预测性框架 — 直接预测,不做核回归
======================================================

38A失败原因: 40个方向在3584维空间中不足以重构Jacobian。
核回归外推失败(r≈0),说明模型不是低秩到能用40个方向重构的程度。

但这不否定RASC! RASC的核心主张是:
"Jacobian的dominant output direction对齐W_U"

正确的预测性实验应该直接测试这个主张:

38A-v2: 直接预测 — Jacobian top output direction能否预测最sensitive的logit方向?
  1. 计算Jacobian的top output direction (用n_sample个方向估计)
  2. 计算"最sensitive logit方向" = W_U中使logit变化最大的方向
  3. 如果两者方向一致 → RASC预测正确

38B-v2: 增益vs对齐的因果方向
  用完整Jacobian (不做采样重构) 的谱分解:
  J = U Σ V^T
  绘制 σ_i vs cos(u_i, W_U top direction) 的scatter plot
  判断因果方向

38C-v2: 预测token flip
  RASC预测: 对齐W_U的Jacobian方向 → 最容易flip token
  验证: 在hidden space中沿Jacobian top direction扰动,
  看是否比沿random方向更容易flip top prediction
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

from model_utils import (load_model, get_layers, get_model_info, release_model,
                         safe_decode, MODEL_CONFIGS)


def get_W_U_np(model):
    """获取完整lm_head权重矩阵"""
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.detach().cpu().float().numpy()
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'output_layer'):
        return model.transformer.output_layer.weight.detach().cpu().float().numpy()
    else:
        return model.get_output_embeddings().weight.detach().cpu().float().numpy()


TEST_SENTENCES = [
    "The cat sat on the mat",
    "She walked to the store yesterday",
    "The scientist discovered a new element",
    "Music fills the quiet room",
    "The river flows through the valley",
]


def compute_jacobian_output_dirs(model, tokenizer, device, text, layer_idx,
                                  token_pos=-1, eps=1e-3, n_dirs=100):
    """
    计算Jacobian的output directions (用有限差分采样)
    
    返回J_cols: [n_dirs, d_model] — J作用在n_dirs个随机方向上的结果
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    
    hs_base = outputs.hidden_states[layer_idx][0, token_pos].detach().clone()
    d_model = hs_base.shape[0]
    
    np.random.seed(42 + layer_idx)
    directions = np.random.randn(n_dirs, d_model)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    J_cols = []
    
    for i in range(n_dirs):
        d_t = torch.tensor(directions[i], dtype=torch.float32, device=device)
        hs_plus = hs_base + eps * d_t
        hs_minus = hs_base - eps * d_t
        
        def make_hook(hs):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (hs.unsqueeze(0).unsqueeze(0).expand_as(output[0]),) + output[1:]
                return hs.unsqueeze(0).unsqueeze(0).expand_as(output)
            return hook_fn
        
        handle_p = get_layers(model)[layer_idx].register_forward_hook(
            make_hook(hs_plus.to(device).to(outputs.hidden_states[layer_idx].dtype))
        )
        with torch.no_grad():
            out_p = model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        handle_p.remove()
        
        handle_m = get_layers(model)[layer_idx].register_forward_hook(
            make_hook(hs_minus.to(device).to(outputs.hidden_states[layer_idx].dtype))
        )
        with torch.no_grad():
            out_m = model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        handle_m.remove()
        
        final_p = out_p.hidden_states[-1][0, token_pos].cpu().float().numpy()
        final_m = out_m.hidden_states[-1][0, token_pos].cpu().float().numpy()
        Jd = (final_p - final_m) / (2 * eps)
        J_cols.append(Jd)
        
        del out_p, out_m
        if i % 20 == 0:
            torch.cuda.empty_cache()
    
    return np.array(J_cols), directions  # [n_dirs, d_model], [n_dirs, d_model]


# ============================================================================
# 38A: Jacobian top output direction vs 最sensitive logit方向
# ============================================================================

def expA_jacobian_vs_logit_direction(model_name, model, tokenizer, device):
    """
    预测性实验: Jacobian的top output direction是否指向W_U中最sensitive的方向?
    
    逻辑:
    - "最sensitive logit方向" = W_U中使logit变化最大的行向量
    - 如果Jacobian的top output direction与W_U的top行向量对齐 → RASC预测正确
    - 具体测量: cos(Jacobian top output dir, W_U[top_token_row])
    
    这是最直接的RASC预测:
    如果Jacobian把输入映射到对齐W_U的方向 → 扰动应最影响top token的logit
    """
    print(f"\n{'='*70}")
    print(f"38A: Jacobian top output dir vs W_U[top token] — RASC核心预测")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U_np(model)
    print(f"  W_U shape: {W_U.shape}")
    
    sample_layers = list(range(0, n_layers, max(1, n_layers // 5))) + [n_layers - 2, n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    print(f"  Sample layers: {sample_layers}")
    
    all_results = []
    
    for text in TEST_SENTENCES[:4]:
        print(f"\n  Text: '{text[:40]}...'")
        
        # 获取top token
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1].cpu().float().numpy()
        top_token_id = int(np.argmax(logits))
        second_token_id = int(np.argsort(logits)[-2])
        
        # W_U中top token对应的行向量(归一化)
        wu_top = W_U[top_token_id]
        wu_top_norm = wu_top / (np.linalg.norm(wu_top) + 1e-10)
        
        # W_U中second token对应的行向量
        wu_second = W_U[second_token_id]
        wu_second_norm = wu_second / (np.linalg.norm(wu_second) + 1e-10)
        
        # W_U的SVD (只算top方向)
        U_wu, s_wu, Vt_wu = np.linalg.svd(W_U, full_matrices=False)
        wu_top_dirs = Vt_wu[:10]  # top-10 W_U directions in d_model space
        
        del outputs
        torch.cuda.empty_cache()
        
        for li in sample_layers:
            try:
                J_cols, input_dirs = compute_jacobian_output_dirs(
                    model, tokenizer, device, text, li,
                    token_pos=-1, eps=1e-3, n_dirs=80
                )
                
                # SVD of J_cols → Jacobian的output directions
                U_j, sigma_j, Vt_j = np.linalg.svd(J_cols, full_matrices=False)
                # Vt_j的行是d_model空间中的主方向 (Jacobian output space)
                jac_top_dirs = Vt_j[:10]  # top-10 Jacobian output directions
                
                # 核心预测1: Jacobian top-1 output dir vs W_U[top_token]行向量
                cos_jac_wutop = np.abs(np.dot(jac_top_dirs[0], wu_top_norm))
                
                # 核心预测2: Jacobian top-1 output dir vs W_U SVD top-1方向
                cos_jac_wusvd = np.abs(np.dot(jac_top_dirs[0], wu_top_dirs[0]))
                
                # 对比: random baseline
                n_random = 100
                random_cos_wutop = []
                random_cos_wusvd = []
                for _ in range(n_random):
                    r = np.random.randn(d_model)
                    r = r / np.linalg.norm(r)
                    random_cos_wutop.append(np.abs(np.dot(r, wu_top_norm)))
                    random_cos_wusvd.append(np.abs(np.dot(r, wu_top_dirs[0])))
                
                # Jacobian top-k方向 vs W_U[top_token]的平均对齐
                jac_topk_align_wutop = np.mean([np.abs(np.dot(jac_top_dirs[k], wu_top_norm)) 
                                                for k in range(min(5, len(jac_top_dirs)))])
                
                # 奇异值谱
                sigma_ratio = sigma_j[0] / (sigma_j[1] + 1e-10) if len(sigma_j) > 1 else 0
                
                # margin direction: W_U[top] - W_U[second] (归一化)
                margin_dir = wu_top_norm - wu_second_norm
                margin_dir = margin_dir / (np.linalg.norm(margin_dir) + 1e-10)
                cos_jac_margin = np.abs(np.dot(jac_top_dirs[0], margin_dir))
                random_cos_margin = np.mean([np.abs(np.dot(
                    np.random.randn(d_model) / np.linalg.norm(np.random.randn(d_model)),
                    margin_dir)) for _ in range(100)])
                
                print(f"    L{li}: cos(J_top, W_U[top_tok])={cos_jac_wutop:.4f} "
                      f"(rand={np.mean(random_cos_wutop):.4f}), "
                      f"cos(J_top, W_U_SVD1)={cos_jac_wusvd:.4f} "
                      f"(rand={np.mean(random_cos_wusvd):.4f}), "
                      f"cos(J_top, margin)={cos_jac_margin:.4f} "
                      f"(rand={random_cos_margin:.4f}), "
                      f"σ₁/σ₂={sigma_ratio:.1f}")
                
                all_results.append({
                    'layer': li,
                    'text': text[:30],
                    'cos_jac_wutop': float(cos_jac_wutop),
                    'cos_jac_wusvd': float(cos_jac_wusvd),
                    'cos_jac_margin': float(cos_jac_margin),
                    'rand_cos_wutop': float(np.mean(random_cos_wutop)),
                    'rand_cos_wusvd': float(np.mean(random_cos_wusvd)),
                    'rand_cos_margin': float(random_cos_margin),
                    'jac_topk_align_wutop': float(jac_topk_align_wutop),
                    'sigma_ratio_1_2': float(sigma_ratio),
                    'sigma_top3': sigma_j[:3].tolist(),
                    'top_token_id': top_token_id,
                })
                
            except Exception as e:
                print(f"    L{li}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                continue
            
            torch.cuda.empty_cache()
    
    # 汇总
    cos_jw = [r['cos_jac_wutop'] for r in all_results]
    cos_js = [r['cos_jac_wusvd'] for r in all_results]
    cos_jm = [r['cos_jac_margin'] for r in all_results]
    rand_jw = [r['rand_cos_wutop'] for r in all_results]
    rand_js = [r['rand_cos_wusvd'] for r in all_results]
    rand_jm = [r['rand_cos_margin'] for r in all_results]
    
    print(f"\n  ===== 38A SUMMARY =====")
    print(f"  Mean cos(J_top, W_U[top_tok]): {np.mean(cos_jw):.4f} vs random {np.mean(rand_jw):.4f}")
    print(f"  Mean cos(J_top, W_U_SVD1):     {np.mean(cos_js):.4f} vs random {np.mean(rand_js):.4f}")
    print(f"  Mean cos(J_top, margin_dir):    {np.mean(cos_jm):.4f} vs random {np.mean(rand_jm):.4f}")
    
    # 按层分组
    layer_data = {}
    for r in all_results:
        li = r['layer']
        if li not in layer_data:
            layer_data[li] = {'cos_jw': [], 'cos_js': [], 'cos_jm': []}
        layer_data[li]['cos_jw'].append(r['cos_jac_wutop'])
        layer_data[li]['cos_js'].append(r['cos_jac_wusvd'])
        layer_data[li]['cos_jm'].append(r['cos_jac_margin'])
    
    print(f"\n  By layer:")
    for li in sorted(layer_data.keys()):
        d = layer_data[li]
        print(f"    L{li}: cos(J,W_U_top)={np.mean(d['cos_jw']):.4f}, "
              f"cos(J,W_U_SVD)={np.mean(d['cos_js']):.4f}, "
              f"cos(J,margin)={np.mean(d['cos_jm']):.4f}")
    
    # 判定
    mult_wutop = np.mean(cos_jw) / (np.mean(rand_jw) + 1e-10)
    mult_wusvd = np.mean(cos_js) / (np.mean(rand_js) + 1e-10)
    mult_margin = np.mean(cos_jm) / (np.mean(rand_jm) + 1e-10)
    
    print(f"\n  Multiplication over random:")
    print(f"    W_U[top_tok]: {mult_wutop:.1f}x")
    print(f"    W_U_SVD1:     {mult_wusvd:.1f}x")
    print(f"    Margin dir:   {mult_margin:.1f}x")
    
    if mult_margin > 3:
        print(f"  ★★★ RASC核心预测成立: Jacobian top direction对齐logit margin方向!")
    elif mult_wusvd > 3:
        print(f"  ★★ Jacobian对齐W_U主方向,但不是margin方向")
    else:
        print(f"  ★ Jacobian与W_U方向的对齐弱于预期")
    
    return all_results


# ============================================================================
# 38B: Gain vs Alignment 因果方向
# ============================================================================

def expB_gain_vs_alignment(model_name, model, tokenizer, device):
    """
    分离alignment和gain的因果方向
    
    对Jacobian采样做SVD: J_cols = U Σ V^T
    V的行是Jacobian output space的主方向
    
    对每个V_i (Jacobian output direction):
    - gain = σ_i (奇异值)
    - alignment = |cos(V_i, W_U top direction)|
    
    绘制 gain vs alignment 的scatter
    如果正相关 → alignment drives gain (弱routing)
    如果负相关 → high-gain directions escape W_U alignment
    如果无关 → independent
    """
    print(f"\n{'='*70}")
    print(f"38B: Gain vs Alignment — Jacobian谱结构的因果方向")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U_np(model)
    
    # W_U SVD
    _, s_wu, Vt_wu = np.linalg.svd(W_U, full_matrices=False)
    wu_top_dirs = Vt_wu[:50]  # W_U top-50 directions
    
    sample_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    
    all_results = []
    
    for text in TEST_SENTENCES[:3]:
        print(f"\n  Text: '{text[:40]}...'")
        
        for li in sample_layers:
            try:
                J_cols, _ = compute_jacobian_output_dirs(
                    model, tokenizer, device, text, li,
                    token_pos=-1, eps=1e-3, n_dirs=100
                )
                
                # SVD
                _, sigma_j, Vt_j = np.linalg.svd(J_cols, full_matrices=False)
                # Vt_j: [min(100, d_model), d_model]
                
                # 计算每个output direction与W_U的alignment
                n_dirs_j = min(30, len(sigma_j))  # 只看top-30
                
                gains = sigma_j[:n_dirs_j]
                alignments = []
                for k in range(n_dirs_j):
                    v = Vt_j[k]
                    # 与W_U top-50方向的最大cosine
                    cos_sims = np.abs(wu_top_dirs @ v)
                    max_align = cos_sims.max()
                    alignments.append(max_align)
                
                alignments = np.array(alignments)
                
                # 相关性
                if np.std(alignments) > 1e-10 and np.std(gains) > 1e-10:
                    r, p_val = stats.pearsonr(gains, alignments)
                    rho, _ = stats.spearmanr(gains, alignments)
                else:
                    r, rho, p_val = 0, 0, 1
                
                # 分组分析
                median_gain = np.median(gains)
                high_gain_align = alignments[gains > median_gain].mean()
                low_gain_align = alignments[gains <= median_gain].mean()
                
                median_align = np.median(alignments)
                high_align_gain = gains[alignments > median_align].mean()
                low_align_gain = gains[alignments <= median_align].mean()
                
                print(f"    L{li}: r(σ, align)={r:.3f} (p={p_val:.2e}), "
                      f"high-gain align={high_gain_align:.3f} vs low={low_gain_align:.3f}, "
                      f"high-align gain={high_align_gain:.0f} vs low={low_align_gain:.0f}")
                
                all_results.append({
                    'layer': li,
                    'text': text[:30],
                    'r_gain_align': float(r),
                    'spearman_rho': float(rho),
                    'p_value': float(p_val),
                    'high_gain_align': float(high_gain_align),
                    'low_gain_align': float(low_gain_align),
                    'high_align_gain': float(high_align_gain),
                    'low_align_gain': float(low_align_gain),
                    'gains_top10': gains[:10].tolist(),
                    'alignments_top10': alignments[:10].tolist(),
                })
                
            except Exception as e:
                print(f"    L{li}: ERROR - {e}")
                continue
            
            torch.cuda.empty_cache()
    
    # 汇总
    rs = [r['r_gain_align'] for r in all_results if not np.isnan(r['r_gain_align'])]
    hg = [r['high_gain_align'] for r in all_results]
    lg = [r['low_gain_align'] for r in all_results]
    
    print(f"\n  ===== 38B SUMMARY =====")
    if rs:
        print(f"  Mean r(σ, alignment): {np.mean(rs):.3f} ± {np.std(rs):.3f}")
        print(f"  Mean high-gain alignment: {np.mean(hg):.3f}")
        print(f"  Mean low-gain alignment:  {np.mean(lg):.3f}")
        print(f"  Ratio: {np.mean(hg)/(np.mean(lg)+1e-10):.2f}x")
        
        if np.mean(rs) > 0.3:
            print(f"  ★★★ Gain与alignment正相关 → alignment drives gain → 弱routing!")
        elif np.mean(rs) < -0.3:
            print(f"  ★★ Gain与alignment负相关 → high-gain方向逃离W_U → 反直觉")
        else:
            print(f"  ★ 无相关 → gain和alignment独立 → pure spectral alignment")
    
    return all_results


# ============================================================================
# 38C: 预测token flip
# ============================================================================

def expC_predict_token_flip(model_name, model, tokenizer, device):
    """
    预测性实验: 沿Jacobian top direction扰动是否比random更容易flip token?
    
    RASC预测: Jacobian的top output direction对齐W_U → 
    沿此方向扰动 → logit变化最大 → 最容易flip top prediction
    
    验证:
    1. 计算Jacobian top output direction
    2. 沿此方向注入递增扰动
    3. 沿随机方向注入相同范数扰动
    4. 比较: 哪个更容易flip top prediction?
    
    如果Jacobian top方向更有效 → RASC有预测力
    """
    print(f"\n{'='*70}")
    print(f"38C: 预测token flip — Jacobian top direction vs random")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    sample_layers = [n_layers // 2, 3 * n_layers // 4, n_layers - 2]
    
    all_results = []
    
    for text in TEST_SENTENCES[:3]:
        print(f"\n  Text: '{text[:40]}...'")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_base = outputs.logits[0, -1].cpu().float().numpy()
        top_token_id = int(np.argmax(logits_base))
        top_logit_base = logits_base[top_token_id]
        
        del outputs
        torch.cuda.empty_cache()
        
        for li in sample_layers:
            try:
                # 获取Jacobian top output direction
                J_cols, _ = compute_jacobian_output_dirs(
                    model, tokenizer, device, text, li,
                    token_pos=-1, eps=1e-3, n_dirs=60
                )
                
                _, sigma_j, Vt_j = np.linalg.svd(J_cols, full_matrices=False)
                jac_top_dir = Vt_j[0]  # [d_model]
                jac_top_dir_t = torch.tensor(jac_top_dir, dtype=torch.float32, device=device)
                
                # Random方向 (10个)
                np.random.seed(42 + li)
                n_random_dirs = 5
                random_dirs = np.random.randn(n_random_dirs, d_model)
                random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
                
                # 测试不同扰动幅度
                eps_list = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
                
                jac_flip_rates = []
                rand_flip_rates = []
                jac_logit_drops = []
                rand_logit_drops = []
                
                for eps_val in eps_list:
                    # Jacobian top direction
                    with torch.no_grad():
                        base_out = model(input_ids=input_ids, attention_mask=attention_mask,
                                        output_hidden_states=True)
                    hs_base = base_out.hidden_states[li][0, -1].detach().clone()
                    
                    hs_jac = hs_base + eps_val * jac_top_dir_t.to(hs_base.dtype)
                    
                    def make_hook_j(hs):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                return (hs.unsqueeze(0).unsqueeze(0).expand_as(output[0]),) + output[1:]
                            return hs.unsqueeze(0).unsqueeze(0).expand_as(output)
                        return hook_fn
                    
                    handle = get_layers(model)[li].register_forward_hook(
                        make_hook_j(hs_jac.to(device).to(base_out.hidden_states[li].dtype))
                    )
                    with torch.no_grad():
                        out_jac = model(input_ids=input_ids, attention_mask=attention_mask)
                    handle.remove()
                    
                    logits_jac = out_jac.logits[0, -1].cpu().float().numpy()
                    jac_flipped = int(np.argmax(logits_jac)) != top_token_id
                    jac_logit_drop = top_logit_base - logits_jac[top_token_id]
                    
                    del base_out, out_jac
                    
                    # Random directions (取平均)
                    rand_flipped_count = 0
                    rand_logit_drop_sum = 0
                    
                    for ri in range(n_random_dirs):
                        rd_t = torch.tensor(random_dirs[ri], dtype=torch.float32, device=device)
                        
                        with torch.no_grad():
                            base_out2 = model(input_ids=input_ids, attention_mask=attention_mask,
                                            output_hidden_states=True)
                        hs_base2 = base_out2.hidden_states[li][0, -1].detach().clone()
                        
                        hs_rand = hs_base2 + eps_val * rd_t.to(hs_base2.dtype)
                        
                        def make_hook_r(hs):
                            def hook_fn(module, input, output):
                                if isinstance(output, tuple):
                                    return (hs.unsqueeze(0).unsqueeze(0).expand_as(output[0]),) + output[1:]
                                return hs.unsqueeze(0).unsqueeze(0).expand_as(output)
                            return hook_fn
                        
                        handle2 = get_layers(model)[li].register_forward_hook(
                            make_hook_r(hs_rand.to(device).to(base_out2.hidden_states[li].dtype))
                        )
                        with torch.no_grad():
                            out_rand = model(input_ids=input_ids, attention_mask=attention_mask)
                        handle2.remove()
                        
                        logits_rand = out_rand.logits[0, -1].cpu().float().numpy()
                        if int(np.argmax(logits_rand)) != top_token_id:
                            rand_flipped_count += 1
                        rand_logit_drop_sum += top_logit_base - logits_rand[top_token_id]
                        
                        del base_out2, out_rand
                    
                    jac_flip_rates.append(float(jac_flipped))
                    rand_flip_rates.append(rand_flipped_count / n_random_dirs)
                    jac_logit_drops.append(float(jac_logit_drop))
                    rand_logit_drops.append(rand_logit_drop_sum / n_random_dirs)
                    
                    torch.cuda.empty_cache()
                
                # 找到最小flip eps
                jac_min_flip_eps = eps_list[0]
                for idx, flipped in enumerate(jac_flip_rates):
                    if flipped > 0.5:
                        jac_min_flip_eps = eps_list[idx]
                        break
                
                rand_min_flip_eps = eps_list[-1]
                for idx, rate in enumerate(rand_flip_rates):
                    if rate > 0.5:
                        rand_min_flip_eps = eps_list[idx]
                        break
                
                print(f"    L{li}: jac_logit_drop={jac_logit_drops}, rand_logit_drop={rand_logit_drops}")
                print(f"           jac_flip={jac_flip_rates}, rand_flip={rand_flip_rates}")
                print(f"           min_flip_eps: jac={jac_min_flip_eps}, rand={rand_min_flip_eps}")
                
                all_results.append({
                    'layer': li,
                    'text': text[:30],
                    'eps_list': eps_list,
                    'jac_logit_drops': jac_logit_drops,
                    'rand_logit_drops': rand_logit_drops,
                    'jac_flip_rates': jac_flip_rates,
                    'rand_flip_rates': rand_flip_rates,
                    'jac_min_flip_eps': jac_min_flip_eps,
                    'rand_min_flip_eps': rand_min_flip_eps,
                    'top_token_id': top_token_id,
                })
                
            except Exception as e:
                print(f"    L{li}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                continue
            
            torch.cuda.empty_cache()
    
    # 汇总
    jac_drops = [r['jac_logit_drops'] for r in all_results]
    rand_drops = [r['rand_logit_drops'] for r in all_results]
    
    print(f"\n  ===== 38C SUMMARY =====")
    # 平均logit drop (跨所有eps值)
    jac_mean_drops = [np.mean(d) for d in jac_drops]
    rand_mean_drops = [np.mean(d) for d in rand_drops]
    
    if jac_mean_drops:
        print(f"  Mean logit drop: jac={np.mean(jac_mean_drops):.3f}, rand={np.mean(rand_mean_drops):.3f}")
        print(f"  Ratio (jac/rand): {np.mean(jac_mean_drops)/(np.mean(rand_mean_drops)+1e-10):.2f}x")
        
        if np.mean(jac_mean_drops) > 2 * np.mean(rand_mean_drops):
            print(f"  ★★★ Jacobian top direction显著更有效! RASC有预测力!")
        elif np.mean(jac_mean_drops) > 1.3 * np.mean(rand_mean_drops):
            print(f"  ★★ Jacobian top direction略更有效")
        else:
            print(f"  ★ Jacobian top direction不比random更有效 → RASC预测力弱")
    
    return all_results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3],
                       help="1=38A jacobian vs logit dir, 2=38B gain vs align, 3=38C token flip")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    
    try:
        if args.exp == 1:
            results = expA_jacobian_vs_logit_direction(args.model, model, tokenizer, device)
            out_name = f"ccml_phase38v2_expA_{args.model}_results.json"
        elif args.exp == 2:
            results = expB_gain_vs_alignment(args.model, model, tokenizer, device)
            out_name = f"ccml_phase38v2_expB_{args.model}_results.json"
        elif args.exp == 3:
            results = expC_predict_token_flip(args.model, model, tokenizer, device)
            out_name = f"ccml_phase38v2_expC_{args.model}_results.json"
        
        out_path = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp', out_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")
    
    finally:
        release_model(model)
