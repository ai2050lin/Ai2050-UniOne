"""
Phase CIV-P496/497/498: 输出端因果链 - 从W_down到语言能力的完整因果路径
======================================================================

核心目标: 解决gamma→语言能力因果链断裂问题，从输出端反向建立因果链

Phase CIII核心瓶颈:
1. gamma→语言能力(importance)链断裂(corr<0.05)
2. layer_frac是最强预测因子(但只是位置标记,无因果机制)
3. 缺少从权重→隐状态→logit→PPL的完整因果路径

Phase CIV核心思路:
从输出端反向建立因果链: ΔW_down → Δh_final → Δlogit → ΔPPL
这不是从中间变量(gamma)出发,而是从权重扰动出发,追踪信号如何一步步传播到输出

P496: W_down扰动→h_final变化的因果链
  - 目标: 建立ΔW_down → Δh_final的因果方程
  - 方法:
    a) 对每层W_down施加小扰动(缩放因子α: 0.01-0.5)
    b) 测量h_final的变化量Δh_final
    c) 分析Δh_final = J_l × ΔW_down × h_l (Jacobian链式法则)
    d) 计算Jacobian条件数随层变化 → 解释为什么深层更敏感
    e) 验证: Δh_final的范数是否与W_down的谱特征相关?

P497: h_final变化→logit变化的因果链
  - 目标: 建立Δh_final → Δlogit的因果方程
  - 方法:
    a) 对h_final施加小扰动,测量logit变化
    b) Δlogit = Δh_final @ W_U^T (线性映射!)
    c) 计算logit灵敏度: ||Δlogit|| / ||Δh_final||
    d) 分析logit灵敏度与W_U谱的关系
    e) 找到最敏感的logit维度(哪些词的logit最容易变化?)

P498: 完整因果链 ΔW_down → ΔPPL 的解析公式
  - 目标: 推导importance = f(W_down, h, W_U)的解析公式
  - 方法:
    a) 链式法则: ΔPPL ≈ ∂PPL/∂logit × ∂logit/∂h × ∂h/∂W_down
    b) 第一步: ∂logit/∂h = W_U^T (精确线性)
    c) 第二步: ∂PPL/∂logit = softmax(p) - one_hot (已知)
    d) 第三步: ∂h/∂W_down = 需要Jacobian链(核心难点)
    e) 提出近似: importance ≈ ||W_U^T × J_l × W_down||_F
    f) 验证: 预测的importance与实测ΔPPL的相关性

使用方法:
    python phase_civ_output_causal_chain.py --model qwen3 --experiment p496
    python phase_civ_output_causal_chain.py --model glm4 --experiment p497
    python phase_civ_output_causal_chain.py --model deepseek7b --experiment p498
    python phase_civ_output_causal_chain.py --model qwen3 --experiment all
"""

import sys
import os
import argparse
import numpy as np
import torch
import json
import time
from scipy.stats import spearmanr, pearsonr
from collections import namedtuple

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 直接添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'tests', 'glm5'))

from model_utils import (
    load_model, get_model_info, get_W_U, get_layer_weights,
    get_sample_layers, compute_recoding_ratio, get_layers,
)


# ============================================================
# 工具函数
# ============================================================

def compute_residual_stream(model, input_ids, device):
    """用hook收集各层hidden state"""
    hidden_states = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach())
        else:
            hidden_states.append(output.detach())
    
    hooks = []
    layers_list = get_layers(model)
    for layer in layers_list:
        h = layer.register_forward_hook(hook_fn)
        hooks.append(h)
    
    try:
        with torch.no_grad():
            model(input_ids)
    except Exception as e:
        print(f"  [compute_residual_stream] Forward failed: {e}")
    
    for h in hooks:
        h.remove()
    
    if hidden_states:
        result = []
        for hs in hidden_states:
            result.append(hs[0, -1].cpu().float().numpy())
        return result
    return None


def compute_participation_ratio(s):
    """参与比 = (sum s^2)^2 / (n * sum s^4)"""
    s_sq = s**2
    s_sq_norm = s_sq / (np.sum(s_sq) + 1e-30)
    return 1.0 / (len(s) * np.sum(s_sq_norm**2) + 1e-30)


def compute_effective_dimension(s):
    """有效维度 = (sum s)^2 / (n * sum s^2)"""
    return (np.sum(s)**2) / (len(s) * np.sum(s**2) + 1e-30)


def compute_kl_divergence(logits_baseline, logits_ablated):
    """计算KL散度"""
    p = torch.nn.functional.softmax(logits_baseline, dim=-1)
    q = torch.nn.functional.log_softmax(logits_ablated, dim=-1)
    kl = torch.nn.functional.kl_div(q, p, reduction='batchmean')
    return kl.item()


def perturb_w_down(layers, l_idx, alpha, mlp_type):
    """
    对W_down施加扰动: W_down → W_down * (1 - alpha)
    返回原始权重用于恢复
    """
    if mlp_type == "split_gate_up":
        orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
        layers[l_idx].mlp.down_proj.weight.data = orig_weight * (1 - alpha)
    elif mlp_type == "merged_gate_up":
        orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
        layers[l_idx].mlp.down_proj.weight.data = orig_weight * (1 - alpha)
    else:
        return None
    return orig_weight


def restore_w_down(layers, l_idx, orig_weight, mlp_type):
    """恢复W_down权重"""
    if mlp_type == "split_gate_up":
        layers[l_idx].mlp.down_proj.weight.data = orig_weight
    elif mlp_type == "merged_gate_up":
        layers[l_idx].mlp.down_proj.weight.data = orig_weight


def compute_importance_measures(model, tokenizer, device, text, layers, l_idx, alpha, mlp_type):
    """
    计算消融W_down后的所有重要性指标
    返回: delta_h_final, delta_logits, delta_ppl, kl_div, importance_ppl等
    """
    info = None  # 不需要重复获取
    
    # 基线前向传播
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        baseline_outputs = model(input_ids)
        if hasattr(baseline_outputs, 'logits'):
            baseline_logits = baseline_outputs.logits
        elif isinstance(baseline_outputs, torch.Tensor):
            baseline_logits = baseline_outputs
        else:
            return None
    
    if baseline_logits.dim() == 3:
        baseline_logits = baseline_logits[0]
    
    # 收集基线h_final
    with torch.no_grad():
        h_all_baseline = compute_residual_stream(model, input_ids, device)
    
    if h_all_baseline is None:
        return None
    
    h_final_baseline = h_all_baseline[-1]
    
    # 基线PPL
    baseline_next_tokens = input_ids[0, 1:]
    baseline_pred_logits = baseline_logits[:-1]
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    baseline_loss = loss_fn(baseline_pred_logits, baseline_next_tokens).item()
    baseline_ppl = np.exp(min(baseline_loss, 20))
    
    # 扰动W_down
    orig_weight = perturb_w_down(layers, l_idx, alpha, mlp_type)
    if orig_weight is None:
        return None
    
    # 扰动后前向传播
    with torch.no_grad():
        ablated_outputs = model(input_ids)
        if hasattr(ablated_outputs, 'logits'):
            ablated_logits = ablated_outputs.logits
        elif isinstance(ablated_outputs, torch.Tensor):
            ablated_logits = ablated_outputs
        else:
            restore_w_down(layers, l_idx, orig_weight, mlp_type)
            return None
    
    if ablated_logits.dim() == 3:
        ablated_logits = ablated_logits[0]
    
    # 收集扰动后h_final
    with torch.no_grad():
        h_all_ablated = compute_residual_stream(model, input_ids, device)
    
    h_final_ablated = h_all_ablated[-1] if h_all_ablated is not None else None
    
    # 恢复权重
    restore_w_down(layers, l_idx, orig_weight, mlp_type)
    
    # 扰动后PPL
    ablated_pred_logits = ablated_logits[:-1]
    ablated_loss = loss_fn(ablated_pred_logits, baseline_next_tokens).item()
    ablated_ppl = np.exp(min(ablated_loss, 20))
    
    # 计算变化量
    delta_ppl = ablated_ppl - baseline_ppl
    importance_ppl = delta_ppl / max(baseline_ppl, 1e-10)
    
    # KL散度
    kl_div = compute_kl_divergence(baseline_pred_logits, ablated_pred_logits)
    
    # delta_h_final
    delta_h_final = h_final_ablated - h_final_baseline if h_final_ablated is not None else None
    delta_h_norm = np.linalg.norm(delta_h_final) if delta_h_final is not None else 0
    h_norm = np.linalg.norm(h_final_baseline)
    
    # delta_logits
    delta_logits = ablated_logits - baseline_logits
    delta_logits_norm = torch.norm(delta_logits).item()
    logits_norm = torch.norm(baseline_logits).item()
    
    # next-token准确率变化
    baseline_preds = torch.argmax(baseline_pred_logits, dim=-1)
    ablated_preds = torch.argmax(ablated_pred_logits, dim=-1)
    baseline_acc = (baseline_preds == baseline_next_tokens).float().mean().item()
    ablated_acc = (ablated_preds == baseline_next_tokens).float().mean().item()
    delta_acc = baseline_acc - ablated_acc
    
    return {
        'delta_h_final': delta_h_final,
        'delta_h_norm': delta_h_norm,
        'h_norm': h_norm,
        'delta_h_relative': delta_h_norm / max(h_norm, 1e-10),
        'delta_logits_norm': delta_logits_norm,
        'logits_norm': logits_norm,
        'delta_logits_relative': delta_logits_norm / max(logits_norm, 1e-10),
        'baseline_ppl': baseline_ppl,
        'ablated_ppl': ablated_ppl,
        'delta_ppl': delta_ppl,
        'importance_ppl': importance_ppl,
        'kl_div': kl_div,
        'baseline_acc': baseline_acc,
        'ablated_acc': ablated_acc,
        'delta_acc': delta_acc,
        'alpha': alpha,
        'l_idx': l_idx,
    }


# ============================================================
# P496: W_down扰动→h_final变化的因果链
# ============================================================

def run_p496(model_name, device):
    """
    P496: W_down扰动→h_final变化的因果链
    
    核心问题: 为什么深层W_down消融影响更大?
    假设: 深层扰动通过LayerNorm放大,传播到h_final的变化更大
    
    方法:
    1. 对每层W_down施加扰动alpha
    2. 测量Δh_final的大小
    3. 分析||Δh_final|| / (alpha × ||W_down||) 是否随层变化
    4. 计算每层的"信号传播增益": gain_l = ||Δh_final|| / (alpha × ||W_down_l||)
    5. 验证gain_l是否与层位置(计算深度)相关
    """
    print(f"\n{'='*60}")
    print(f"P496: W_down→h_final因果链 [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mlp_type = info.mlp_type
    
    layers_list = get_layers(model)
    W_U = get_W_U(model)
    
    test_text = "The apple is red and sweet, and it grows on trees in the garden."
    
    # 多个alpha值
    alphas = [0.1, 0.3, 0.5]
    
    # 采样层
    sample_layers = get_sample_layers(n_layers, 10)
    print(f"  采样层: {sample_layers}")
    
    results = []
    
    for l_idx in sample_layers:
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        # 获取W_down谱特征
        lw = get_layer_weights(layers_list[l_idx], d_model, mlp_type)
        W_down = lw.W_down
        W_down_norm = np.linalg.norm(W_down)
        
        # SVD谱特征
        try:
            U_wd, s_wd, Vt_wd = np.linalg.svd(W_down.astype(np.float32), full_matrices=False)
            PR_wd = compute_participation_ratio(s_wd)
            d_eff_wd = compute_effective_dimension(s_wd)
            top10_wd = np.sum(s_wd[:min(10, len(s_wd))]**2) / max(np.sum(s_wd**2), 1e-10)
            kappa_wd = s_wd[0] / max(s_wd[-1], 1e-10)
        except:
            PR_wd = d_eff_wd = top10_wd = kappa_wd = 0
        
        for alpha in alphas:
            print(f"  层L{l_idx} alpha={alpha}...")
            
            meas = compute_importance_measures(
                model, tokenizer, device, test_text,
                layers_list, l_idx, alpha, mlp_type
            )
            
            if meas is None:
                continue
            
            # 信号传播增益
            gain = meas['delta_h_norm'] / max(alpha * W_down_norm, 1e-10)
            
            results.append({
                'layer': l_idx,
                'layer_frac': layer_frac,
                'alpha': alpha,
                'delta_h_norm': meas['delta_h_norm'],
                'delta_h_relative': meas['delta_h_relative'],
                'delta_logits_norm': meas['delta_logits_norm'],
                'delta_logits_relative': meas['delta_logits_relative'],
                'delta_ppl': meas['delta_ppl'],
                'importance_ppl': meas['importance_ppl'],
                'kl_div': meas['kl_div'],
                'delta_acc': meas['delta_acc'],
                'W_down_norm': W_down_norm,
                'gain': gain,  # 信号传播增益
                'PR_wd': PR_wd,
                'd_eff_wd': d_eff_wd,
                'top10_wd': top10_wd,
                'kappa_wd': kappa_wd,
            })
    
    # 统计分析
    if len(results) < 5:
        print("  数据不足,无法进行统计分析")
        return results
    
    print(f"\n--- P496 统计分析 [{model_name}] ---")
    
    # 1. gain与layer_frac的相关性
    gains = [r['gain'] for r in results if r['alpha'] == 0.1]
    lfracs = [r['layer_frac'] for r in results if r['alpha'] == 0.1]
    if len(gains) >= 3:
        r_glf, p_glf = spearmanr(lfracs, gains)
        print(f"  gain vs layer_frac: r={r_glf:.3f}, p={p_glf:.4f}")
    
    # 2. delta_h_relative与delta_ppl的相关性
    dh_rels = [r['delta_h_relative'] for r in results]
    dppls = [r['delta_ppl'] for r in results]
    if len(dh_rels) >= 3:
        r_dh, p_dh = spearmanr(dh_rels, dppls)
        print(f"  delta_h_relative vs delta_ppl: r={r_dh:.3f}, p={p_dh:.4f}")
    
    # 3. gain与谱特征的相关性
    PRs = [r['PR_wd'] for r in results if r['alpha'] == 0.1]
    gains_alpha01 = [r['gain'] for r in results if r['alpha'] == 0.1]
    if len(PRs) >= 3:
        r_pr, p_pr = spearmanr(PRs, gains_alpha01)
        print(f"  PR_wd vs gain: r={r_pr:.3f}, p={p_pr:.4f}")
    
    # 4. delta_h_relative vs kl_div
    kl_divs = [r['kl_div'] for r in results]
    if len(dh_rels) >= 3:
        r_kl, p_kl = spearmanr(dh_rels, kl_divs)
        print(f"  delta_h_relative vs kl_div: r={r_kl:.3f}, p={p_kl:.4f}")
    
    # 5. 多层回归: importance_ppl ~ f(layer_frac, delta_h_relative, PR)
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.preprocessing import StandardScaler
    
    X_features = []
    y_target = []
    for r in results:
        if r['alpha'] == 0.3:  # 用中等扰动
            X_features.append([r['layer_frac'], r['delta_h_relative'], r['PR_wd'], r['kappa_wd']])
            y_target.append(r['importance_ppl'])
    
    if len(X_features) >= 5:
        X = np.array(X_features)
        y = np.array(y_target)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        ols = LinearRegression()
        ols.fit(X_s, y)
        y_pred = ols.predict(X_s)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        
        print(f"  importance_ppl ~ f(lfrac, delta_h_rel, PR, kappa): R2={r2:.3f}")
        feat_names = ['layer_frac', 'delta_h_relative', 'PR_wd', 'kappa_wd']
        for fn, c in zip(feat_names, ols.coef_):
            print(f"    {fn}: {c:.4f}")
    
    # 保存结果
    out_dir = os.path.join(project_root, 'tests', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"p496_{model_name}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  结果已保存: {out_path}")
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    return results


# ============================================================
# P497: h_final→logit变化的因果链
# ============================================================

def run_p497(model_name, device):
    """
    P497: h_final→logit变化的因果链
    
    核心问题: Δh_final如何转化为Δlogit? 
    理论: Δlogit = Δh_final @ W_U^T (精确线性!)
    
    但关键问题是:
    1. logit灵敏度 = ||Δlogit|| / ||Δh_final|| 是多少?
    2. 哪些logit维度最敏感? (对应哪些词?)
    3. logit灵敏度与W_U谱的关系?
    4. 能否从Δh_final预测ΔPPL?
    
    方法:
    1. 对h_final施加方向性扰动(沿W_U的主奇异向量方向)
    2. 测量logit变化
    3. 分析logit灵敏度与扰动方向的关系
    4. 推导ΔPPL ≈ f(Δh_final, W_U)的近似公式
    """
    print(f"\n{'='*60}")
    print(f"P497: h_final→logit因果链 [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    W_U = get_W_U(model)  # [vocab, d_model]
    
    test_text = "The apple is red and sweet, and it grows on trees in the garden."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # 1. 计算基线
    with torch.no_grad():
        baseline_outputs = model(input_ids)
        if hasattr(baseline_outputs, 'logits'):
            baseline_logits = baseline_outputs.logits
        else:
            baseline_logits = baseline_outputs
    
    if baseline_logits.dim() == 3:
        baseline_logits = baseline_logits[0]
    
    # 收集h_final
    h_all = compute_residual_stream(model, input_ids, device)
    if h_all is None:
        print("  无法获取hidden states")
        return []
    
    h_final = h_all[-1]  # 最后一层的last token
    
    # 2. W_U的SVD分析 (使用采样避免内存溢出)
    print(f"  W_U shape: {W_U.shape}")
    print(f"  h_final shape: {h_final.shape}")
    
    # 采样W_U的行(避免大矩阵内存问题)
    vocab_size = W_U.shape[0]
    k_svd = min(300, d_model)
    if vocab_size > 50000:
        # 采样方法: 随机选k_svd*5行,做QR分解得到W_U行空间的近似基
        np.random.seed(42)
        n_sample = min(k_svd * 5, vocab_size)
        indices = np.random.choice(vocab_size, n_sample, replace=False)
        W_U_sub = W_U[indices].astype(np.float32)  # [n_sample, d_model]
        # 对W_U_sub^T做SVD
        from sklearn.utils.extmath import randomized_svd
        U_wut, s_wut, _ = randomized_svd(W_U_sub.T, n_components=k_svd, random_state=42)
        print(f"  W_U 采样SVD: n_sample={n_sample}, k={k_svd}, s_max={s_wut[0]:.2f}, s_min={s_wut[-1]:.2f}")
    else:
        W_UT = W_U.T.astype(np.float32)
        from sklearn.utils.extmath import randomized_svd
        U_wut, s_wut, _ = randomized_svd(W_UT, n_components=k_svd, random_state=42)
        print(f"  W_U^T SVD: k={k_svd}, s_max={s_wut[0]:.2f}, s_min={s_wut[-1]:.2f}")
    
    # 3. 不同方向的扰动
    perturbation_directions = {}
    
    # (a) 沿W_U前10个奇异向量方向
    for i in range(min(10, k_svd)):
        perturbation_directions[f'W_U_sv{i}'] = U_wut[:, i]
    
    # (b) 沿W_U尾部奇异向量方向
    for i in range(max(0, k_svd-5), k_svd):
        perturbation_directions[f'W_U_sv_tail{i}'] = U_wut[:, i]
    
    # (c) 随机方向
    np.random.seed(42)
    for i in range(5):
        rand_dir = np.random.randn(d_model)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        perturbation_directions[f'random_{i}'] = rand_dir
    
    # (d) 沿h_final方向
    h_final_norm = h_final / max(np.linalg.norm(h_final), 1e-10)
    perturbation_directions['h_final_dir'] = h_final_norm
    
    # 4. 施加扰动并测量
    epsilons = [0.01, 0.05, 0.1]  # 扰动强度(相对于h_final范数)
    h_norm = np.linalg.norm(h_final)
    
    results = []
    
    for dir_name, direction in perturbation_directions.items():
        for eps in epsilons:
            # 扰动h_final
            delta_h = eps * h_norm * direction
            h_perturbed = h_final + delta_h
            
            # 转换为tensor
            h_perturbed_tensor = torch.tensor(h_perturbed, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            # 手动计算logit: logit = h @ W_U^T
            # 对大W_U采样避免OOM
            vocab_size_WU = W_U.shape[0]
            if vocab_size_WU > 50000:
                # 采样部分词计算logit变化
                np.random.seed(42)
                n_sample_logits = min(10000, vocab_size_WU)
                sample_idx = np.random.choice(vocab_size_WU, n_sample_logits, replace=False)
                W_U_sample = torch.tensor(W_U[sample_idx], dtype=torch.float32, device=device)
                logits_perturbed = torch.matmul(
                    h_perturbed_tensor.squeeze(), W_U_sample.T
                )
                logits_baseline_manual = torch.matmul(
                    torch.tensor(h_final, dtype=torch.float32, device=device), W_U_sample.T
                )
            else:
                W_U_tensor = torch.tensor(W_U, dtype=torch.float32, device=device)
                logits_perturbed = torch.matmul(
                    h_perturbed_tensor.squeeze(), W_U_tensor.T
                )
                logits_baseline_manual = torch.matmul(
                    torch.tensor(h_final, dtype=torch.float32, device=device), W_U_tensor.T
                )
            
            # Δlogit
            delta_logits = logits_perturbed - logits_baseline_manual
            delta_logits_norm = torch.norm(delta_logits).item()
            logits_norm = torch.norm(logits_baseline_manual).item()
            
            # logit灵敏度
            sensitivity = delta_logits_norm / max(eps * h_norm, 1e-10)
            
            # ΔPPL近似: 用最后一个token位置的logit变化
            # softmax变化量近似
            last_logits_bl = logits_baseline_manual
            last_logits_pt = logits_perturbed
            
            # KL散度(近似)
            p_bl = torch.nn.functional.softmax(last_logits_bl, dim=-1)
            p_pt = torch.nn.functional.softmax(last_logits_pt, dim=-1)
            kl = torch.nn.functional.kl_div(
                torch.log(p_pt + 1e-10), p_bl, reduction='sum'
            ).item()
            
            # top-1预测变化
            top1_bl = torch.argmax(last_logits_bl).item()
            top1_pt = torch.argmax(last_logits_pt).item()
            top1_changed = 1 if top1_bl != top1_pt else 0
            
            # top-5变化
            _, top5_bl = torch.topk(last_logits_bl, 5)
            _, top5_pt = torch.topk(last_logits_pt, 5)
            top5_overlap = len(set(top5_bl.tolist()) & set(top5_pt.tolist()))
            
            results.append({
                'direction': dir_name,
                'epsilon': eps,
                'delta_h_norm': eps * h_norm,
                'delta_logits_norm': delta_logits_norm,
                'logits_norm': logits_norm,
                'delta_logits_relative': delta_logits_norm / max(logits_norm, 1e-10),
                'sensitivity': sensitivity,  # ||Δlogit|| / ||Δh||
                'kl_div': kl,
                'top1_changed': top1_changed,
                'top5_overlap': top5_overlap,
                'logit_change_per_dim': delta_logits_norm / max(eps * h_norm, 1e-10),  # 每维平均变化
            })
    
    # 5. 统计分析
    print(f"\n--- P497 统计分析 [{model_name}] ---")
    
    # (a) 主奇异方向vs尾部方向vs随机方向的灵敏度
    sv_head_sens = [r['sensitivity'] for r in results if r['direction'].startswith('W_U_sv') and not r['direction'].startswith('W_U_sv_tail') and r['epsilon'] == 0.05]
    sv_tail_sens = [r['sensitivity'] for r in results if r['direction'].startswith('W_U_sv_tail') and r['epsilon'] == 0.05]
    rand_sens = [r['sensitivity'] for r in results if r['direction'].startswith('random') and r['epsilon'] == 0.05]
    hdir_sens = [r['sensitivity'] for r in results if r['direction'] == 'h_final_dir' and r['epsilon'] == 0.05]
    
    if sv_head_sens:
        print(f"  W_U主奇异方向灵敏度: mean={np.mean(sv_head_sens):.3f}, std={np.std(sv_head_sens):.3f}")
    if sv_tail_sens:
        print(f"  W_U尾部奇异方向灵敏度: mean={np.mean(sv_tail_sens):.3f}, std={np.std(sv_tail_sens):.3f}")
    if rand_sens:
        print(f"  随机方向灵敏度: mean={np.mean(rand_sens):.3f}, std={np.std(rand_sens):.3f}")
    if hdir_sens:
        print(f"  h_final方向灵敏度: {hdir_sens[0]:.3f}")
    
    # (b) 灵敏度与奇异值大小的关系
    sv_indices = list(range(min(10, k_svd)))
    sv_sensitivities = []
    sv_values = []
    for i in sv_indices:
        for r in results:
            if r['direction'] == f'W_U_sv{i}' and r['epsilon'] == 0.05:
                sv_sensitivities.append(r['sensitivity'])
                sv_values.append(s_wut[i])
                break
    
    if len(sv_sensitivities) >= 3:
        r_sv, p_sv = spearmanr(sv_values, sv_sensitivities)
        print(f"  奇异值大小 vs logit灵敏度: r={r_sv:.3f}, p={p_sv:.4f}")
    
    # (c) KL散度与sensitivity的关系
    kl_divs = [r['kl_div'] for r in results]
    sensitivities = [r['sensitivity'] for r in results]
    if len(kl_divs) >= 3:
        r_kl, p_kl = spearmanr(sensitivities, kl_divs)
        print(f"  sensitivity vs kl_div: r={r_kl:.3f}, p={p_kl:.4f}")
    
    # (d) top-1变化率
    top1_changes = [r['top1_changed'] for r in results if r['epsilon'] == 0.1]
    if top1_changes:
        print(f"  top-1预测变化率(eps=0.1): {np.mean(top1_changes):.3f}")
    
    # 6. 理论验证: ||Δlogit|| ≈ ||Δh|| × ||W_U||_F / sqrt(d) ?
    # 各向同性假设下: E[||Δlogit||^2] = ||Δh||^2 × ||W_U||_F^2 / d
    W_U_frobenius = np.linalg.norm(W_U)
    isotropic_prediction = h_norm * 0.05 * W_U_frobenius / np.sqrt(d_model)
    actual_mean = np.mean([r['delta_logits_norm'] for r in results if r['epsilon'] == 0.05 and r['direction'].startswith('random')])
    
    print(f"\n  各向同性预测 vs 实测:")
    print(f"    预测||Δlogit|| (随机方向, eps=0.05): {isotropic_prediction:.3f}")
    print(f"    实测mean||Δlogit|| (随机方向): {actual_mean:.3f}")
    print(f"    比值: {actual_mean / max(isotropic_prediction, 1e-10):.3f}")
    
    # 保存结果
    out_dir = os.path.join(project_root, 'tests', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"p497_{model_name}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  结果已保存: {out_path}")
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    return results


# ============================================================
# P498: 完整因果链 ΔW_down → ΔPPL 的解析公式
# ============================================================

def run_p498(model_name, device):
    """
    P498: 完整因果链 ΔW_down → ΔPPL
    
    目标: 推导 importance ≈ f(W_down, h, W_U) 的解析公式
    
    理论推导:
    ΔW_down → Δh_l (层内变化)
    Δh_l → Δh_final (传播到最后一层)
    Δh_final → Δlogit (线性映射: W_U^T)
    Δlogit → ΔPPL (softmax非线性)
    
    关键简化:
    1. Δlogit = Δh_final @ W_U^T (精确)
    2. ΔPPL ≈ softmax敏感度 × ||Δlogit|| (近似)
    3. 传播: Δh_final ≈ J_L × ... × J_{l+1} × Δh_l (Jacobian链)
    
    近似公式:
    importance_l ≈ ||W_U^T × J_{L:l} × W_down_l||_F × ||h_l||
    
    其中 J_{L:l} = ∂h_L / ∂h_l 是从层l到层L的Jacobian
    
    验证: 比较预测importance与实测ΔPPL
    """
    print(f"\n{'='*60}")
    print(f"P498: 完整因果链 ΔW_down→ΔPPL [{model_name}]")
    print(f"{'='*60}")
    
    model, tokenizer, model_device = load_model(model_name)
    if model_device != device:
        model = model.to(device)
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mlp_type = info.mlp_type
    
    layers_list = get_layers(model)
    W_U = get_W_U(model)
    
    test_text = "The apple is red and sweet, and it grows on trees in the garden."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # 1. 收集基线数据
    with torch.no_grad():
        h_all = compute_residual_stream(model, input_ids, device)
    
    if h_all is None:
        print("  无法获取hidden states")
        return []
    
    # 2. W_U^T的SVD (用于logit灵敏度分析, 采样避免OOM)
    vocab_size = W_U.shape[0]
    k_svd = min(300, d_model)
    if vocab_size > 50000:
        np.random.seed(42)
        n_sample = min(k_svd * 5, vocab_size)
        indices = np.random.choice(vocab_size, n_sample, replace=False)
        W_U_sub = W_U[indices].astype(np.float32)
        from sklearn.utils.extmath import randomized_svd
        U_wut, s_wut, _ = randomized_svd(W_U_sub.T, n_components=k_svd, random_state=42)
    else:
        W_UT = W_U.T.astype(np.float32)
        from sklearn.utils.extmath import randomized_svd
        U_wut, s_wut, _ = randomized_svd(W_UT, n_components=k_svd, random_state=42)
    
    # W_U投影矩阵: 将d_model维向量投影到W_U行空间
    # P_WU = U_wut @ U_wut^T, 但对大d_model来说太大, 用U_wut直接投影
    # 不需要显式计算P_WU, 用U_wut @ (U_wut^T @ x) 代替 P_WU @ x
    print(f"  W_U SVD: k={k_svd}, s_max={s_wut[0]:.2f}")
    print(f"  W_U投影基U_wut shape: {U_wut.shape}")
    
    # 3. 对每层计算因果链的各个环节
    sample_layers = get_sample_layers(n_layers, 10)
    print(f"  采样层: {sample_layers}")
    
    alpha = 0.3  # 中等扰动
    
    results = []
    
    for l_idx in sample_layers:
        layer_frac = l_idx / max(n_layers - 1, 1)
        print(f"  处理层L{l_idx}...")
        
        # 获取层权重
        lw = get_layer_weights(layers_list[l_idx], d_model, mlp_type)
        W_down = lw.W_down
        
        # (a) 谱特征
        try:
            _, s_wd, _ = np.linalg.svd(W_down.astype(np.float32), full_matrices=False)
            PR_wd = compute_participation_ratio(s_wd)
            d_eff_wd = compute_effective_dimension(s_wd)
            top10_wd = np.sum(s_wd[:min(10, len(s_wd))]**2) / max(np.sum(s_wd**2), 1e-10)
        except Exception as e:
            print(f"  W_down SVD失败(L{l_idx}): {e}, 使用randomized_svd")
            try:
                from sklearn.utils.extmath import randomized_svd
                _, s_wd, _ = randomized_svd(W_down.astype(np.float32), n_components=min(100, min(W_down.shape)-1), random_state=42)
                PR_wd = compute_participation_ratio(s_wd)
                d_eff_wd = compute_effective_dimension(s_wd)
                top10_wd = np.sum(s_wd[:min(10, len(s_wd))]**2) / max(np.sum(s_wd**2), 1e-10)
            except:
                s_wd = np.array([1.0])
                PR_wd = d_eff_wd = top10_wd = 0
        
        W_down_norm = np.linalg.norm(W_down)
        
        # (b) 该层的hidden state
        h_l = h_all[l_idx] if l_idx < len(h_all) else h_all[-1]
        h_l_norm = np.linalg.norm(h_l)
        
        # (c) 最后一层的hidden state
        h_final = h_all[-1]
        h_final_norm = np.linalg.norm(h_final)
        
        # (d) 实测importance (W_down扰动)
        meas = compute_importance_measures(
            model, tokenizer, device, test_text,
            layers_list, l_idx, alpha, mlp_type
        )
        
        if meas is None:
            continue
        
        # (e) 预测importance的几个候选公式
        
        # 公式1: 简单范数公式
        # importance ∝ ||W_down|| × ||h_l||
        pred_formula1 = W_down_norm * h_l_norm
        
        # 公式2: W_U投影公式
        # importance ∝ ||W_down^T × U_wut|| × ||h_l||
        # W_down shape: [d_model, intermediate] (注意: model_utils返回的格式)
        # U_wut: [d_model, k_svd] (W_U^T的左奇异向量)
        # W_down^T: [intermediate, d_model] → W_down^T @ U_wut: [intermediate, k_svd]
        try:
            W_down_coeffs = W_down.T @ U_wut  # [intermediate, k_svd]
            W_down_proj_norm = np.sqrt(np.sum(W_down_coeffs**2))
            pred_formula2 = W_down_proj_norm * h_l_norm
        except Exception as e:
            print(f"  公式2计算失败: {e}, W_down={W_down.shape}, U_wut={U_wut.shape}")
            pred_formula2 = 0
        
        # 公式3: 基于谱的公式
        # importance ∝ sum(s_Wdown_i × s_WU_j) × ||h_l||
        # W_down SVD: W_down = U_wd @ diag(s_wd) @ Vt_wd
        # 取前k个奇异值的乘积和
        k_cross = min(50, len(s_wd), k_svd)
        # 归一化后的乘积和
        s_wd_norm = s_wd[:k_cross] / max(np.sum(s_wd[:k_cross]), 1e-10)
        s_wut_norm = s_wut[:k_cross] / max(np.sum(s_wut[:k_cross]), 1e-10)
        pred_formula3 = np.sum(s_wd_norm * s_wut_norm) * h_l_norm * W_down_norm
        
        # 公式4: 层位置修正公式
        # importance ∝ ||W_down|| × ||h_l|| × (1 + beta × layer_frac)
        # 深层有更大的影响(因为后续层少,LayerNorm放大累积)
        pred_formula4 = W_down_norm * h_l_norm * (1 + 3 * layer_frac)
        
        # (f) 各公式与实测importance的比较
        results.append({
            'layer': l_idx,
            'layer_frac': layer_frac,
            'W_down_norm': W_down_norm,
            'h_l_norm': h_l_norm,
            'h_final_norm': h_final_norm,
            'PR_wd': PR_wd,
            'd_eff_wd': d_eff_wd,
            'top10_wd': top10_wd,
            # 实测
            'delta_h_norm': meas['delta_h_norm'],
            'delta_h_relative': meas['delta_h_relative'],
            'delta_logits_norm': meas['delta_logits_norm'],
            'delta_ppl': meas['delta_ppl'],
            'importance_ppl': meas['importance_ppl'],
            'kl_div': meas['kl_div'],
            'delta_acc': meas['delta_acc'],
            # 预测
            'pred_formula1': pred_formula1,  # ||W_down|| × ||h_l||
            'pred_formula2': pred_formula2,  # ||P_WU × W_down^T|| × ||h_l||
            'pred_formula3': pred_formula3,  # cross-spectral × ||h_l||
            'pred_formula4': pred_formula4,  # ||W_down|| × ||h_l|| × depth
            # 增益
            'gain': meas['delta_h_norm'] / max(alpha * W_down_norm, 1e-10),
        })
    
    # 4. 验证各公式
    print(f"\n--- P498 公式验证 [{model_name}] ---")
    
    importance_vals = [r['importance_ppl'] for r in results]
    kl_div_vals = [r['kl_div'] for r in results]
    
    for formula_name, formula_key in [
        ("公式1: ||W_down||×||h_l||", 'pred_formula1'),
        ("公式2: ||P_WU×W_down^T||×||h_l||", 'pred_formula2'),
        ("公式3: cross-spectral×||h_l||", 'pred_formula3'),
        ("公式4: ||W_down||×||h_l||×depth", 'pred_formula4'),
    ]:
        preds = [r[formula_key] for r in results]
        if len(preds) >= 3:
            r_ppl, p_ppl = spearmanr(preds, importance_vals)
            r_kl, p_kl = spearmanr(preds, kl_div_vals)
            print(f"  {formula_name}:")
            print(f"    vs importance_ppl: r={r_ppl:.3f}, p={p_ppl:.4f}")
            print(f"    vs kl_div: r={r_kl:.3f}, p={p_kl:.4f}")
    
    # 5. 最优回归: importance ~ f(哪些特征组合?)
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.preprocessing import StandardScaler
    
    feature_names = ['layer_frac', 'W_down_norm', 'h_l_norm', 'PR_wd', 'd_eff_wd', 'top10_wd', 'delta_h_relative']
    X_list = []
    y_list = []
    for r in results:
        X_list.append([r[fn] for fn in feature_names])
        y_list.append(r['kl_div'])  # 用kl_div作为target(更稳定)
    
    if len(X_list) >= 5:
        X = np.array(X_list)
        y = np.array(y_list)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # Lasso
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_s, y)
        y_pred = lasso.predict(X_s)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        
        print(f"\n  最优回归 kl_div ~ f(features):")
        print(f"    R2={r2:.3f}")
        for fn, c in zip(feature_names, lasso.coef_):
            print(f"    {fn}: {c:.4f}")
    
    # 6. 逐环节因果验证
    print(f"\n--- 逐环节因果验证 ---")
    
    # ΔW_down → Δh_final (实测)
    delta_h_rels = [r['delta_h_relative'] for r in results]
    lfracs = [r['layer_frac'] for r in results]
    if len(delta_h_rels) >= 3:
        r1, p1 = spearmanr(lfracs, delta_h_rels)
        print(f"  layer_frac → delta_h_relative: r={r1:.3f}, p={p1:.4f}")
    
    # Δh_final → ΔPPL (实测)
    delta_ppls = [r['delta_ppl'] for r in results]
    if len(delta_h_rels) >= 3:
        r2, p2 = spearmanr(delta_h_rels, delta_ppls)
        print(f"  delta_h_relative → delta_ppl: r={r2:.3f}, p={p2:.4f}")
    
    # Δh_final → kl_div (实测)
    if len(delta_h_rels) >= 3:
        r3, p3 = spearmanr(delta_h_rels, kl_div_vals)
        print(f"  delta_h_relative → kl_div: r={r3:.3f}, p={p3:.4f}")
    
    # 完整链: layer_frac → delta_h → delta_ppl
    print(f"\n  因果链强度: layer_frac({r1:.2f}) → delta_h({r2:.2f}) → delta_ppl")
    print(f"  因果链强度: layer_frac({r1:.2f}) → delta_h({r3:.2f}) → kl_div")
    
    # 7. W_U投影效率分析
    print(f"\n--- W_U投影效率 ---")
    gains = [r['gain'] for r in results]
    if len(gains) >= 3:
        print(f"  信号传播增益: mean={np.mean(gains):.4f}, range=[{np.min(gains):.4f}, {np.max(gains):.4f}]")
        r_gain_lf, _ = spearmanr(lfracs, gains)
        print(f"  gain vs layer_frac: r={r_gain_lf:.3f}")
    
    # 保存结果
    out_dir = os.path.join(project_root, 'tests', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"p498_{model_name}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  结果已保存: {out_path}")
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase CIV: 输出端因果链")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p496", "p497", "p498", "all"],
                       help="实验编号")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if args.experiment == "p496" or args.experiment == "all":
        run_p496(args.model, device)
    
    if args.experiment == "p497" or args.experiment == "all":
        run_p497(args.model, device)
    
    if args.experiment == "p498" or args.experiment == "all":
        run_p498(args.model, device)
    
    print("\nPhase CIV 完成!")


if __name__ == "__main__":
    main()
