"""
Phase CIX-P511/P512/P513: 信号传播稳定性预测与W_gate因果验证
==========================================================

Phase CVIII核心发现:
- DS7B 93%层不稳定(vs Qwen3 15%, GLM4 50%)
- Qwen3/GLM4遵循指数传播模型(R2>0.92), DS7B不遵循(R2=0.30)
- W_gate是谱尖锐度主要来源(PR=0.58-0.66)
- DS7B W_gate MP偏离=962(2x Qwen3)

Phase CIX核心思路:
1. 从权重特征预测传播稳定性 → 不需要实际前向传播即可判断
2. W_gate因果干预 → 验证W_gate谱结构是否直接导致不稳定
3. 跨模型传播理论 → 推导geo_mean_ratio的权重预测公式

P511: 从权重预测传播稳定性
  - 收集每层的W_gate谱特征(PR, kappa, top10%, MP偏离)和W_down/W_up特征
  - 加上post_ln_norm, layer_frac等Phase CVI-CVII已验证的重要特征
  - 目标: 训练分类器预测层是否稳定(max_ratio>2), AUC>0.8
  - 验证: 权重特征能否替代实际前向传播来评估稳定性

P512: W_gate因果干预
  - 对W_gate的奇异值进行干预:
    a. 压缩top-k奇异值 → 降低谱尖锐度
    b. 放大bottom-k奇异值 → 增加谱均匀性
    c. 用MP分布替换奇异值 → 完全随机化
  - 测量干预后传播稳定性的变化
  - 验证: W_gate谱尖锐度是否直接导致不稳定(因果而非相关)

P513: 跨模型传播理论
  - 推导: geo_mean_ratio ≈ f(W_gate_kappa, W_down_PR, post_ln_norm)
  - 验证: 公式能否跨模型泛化(Qwen3/GLM4/DS7B)
  - 分析: 每层放大率的权重预测公式

使用方法:
    python phase_cix_stability_prediction.py --model qwen3 --experiment p511
    python phase_cix_stability_prediction.py --model glm4 --experiment p512
    python phase_cix_stability_prediction.py --model deepseek7b --experiment p513
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'tests', 'glm5'))

from model_utils import (
    load_model, get_model_info, get_W_U, get_layer_weights,
    get_sample_layers, get_layers,
)


def compute_kl_divergence(logits_baseline, logits_ablated):
    p = torch.nn.functional.softmax(logits_baseline, dim=-1)
    q = torch.nn.functional.log_softmax(logits_ablated, dim=-1)
    kl = torch.nn.functional.kl_div(q, p, reduction='batchmean')
    return kl.item()


def perturb_w_down(layers, l_idx, alpha, mlp_type):
    orig_weight = layers[l_idx].mlp.down_proj.weight.data.clone()
    layers[l_idx].mlp.down_proj.weight.data = orig_weight * (1 - alpha)
    return orig_weight


def restore_w_down(layers, l_idx, orig_weight, mlp_type):
    layers[l_idx].mlp.down_proj.weight.data = orig_weight


def get_post_ln_norm(layer):
    """获取post-attention layernorm权重范数"""
    if hasattr(layer, 'post_attention_layernorm'):
        w = layer.post_attention_layernorm.weight.data
        return torch.norm(w.float()).item()
    elif hasattr(layer, 'input_layernorm'):
        # 对于没有post_attention_layernorm的模型
        return 0.0
    return 0.0


def get_ln_norm(layer):
    """获取input layernorm权重范数"""
    if hasattr(layer, 'input_layernorm'):
        w = layer.input_layernorm.weight.data
        return torch.norm(w.float()).item()
    return 0.0


def compute_spectral_features(W, n_components=200):
    """计算权重矩阵的谱特征"""
    if isinstance(W, torch.Tensor):
        W_np = W.float().cpu().numpy()
    else:
        W_np = np.asarray(W, dtype=np.float32)
    m, n = W_np.shape
    
    # 使用randomized SVD加速
    from sklearn.utils.extmath import randomized_svd
    n_comp = min(n_components, min(m, n) - 1)
    try:
        U, s, Vt = randomized_svd(W_np, n_components=n_comp, random_state=42)
    except:
        U, s, Vt = np.linalg.svd(W_np, full_matrices=False)
        s = s[:n_comp]
    
    total_var = np.sum(s**2)
    if total_var < 1e-10:
        return {'pr': 0, 'kappa': 1, 'top10_pct': 1, 'mp_deviation': 0,
                'spectral_entropy': 0, 'mean_sigma': 0, 'std_sigma': 0}
    
    # 参与率 PR = (sum sigma_i^2)^2 / sum sigma_i^4
    pr = (np.sum(s**2))**2 / np.sum(s**4)
    pr_normalized = pr / len(s)  # 归一化到[0,1]
    
    # 条件数
    kappa = s[0] / max(s[-1], 1e-10)
    
    # top10%能量占比
    n_top = max(1, len(s) // 10)
    top10_pct = np.sum(s[:n_top]**2) / total_var
    
    # MP偏离: sigma^2的均值/方差 vs 理论MP分布
    s2 = s**2
    mean_s2 = np.mean(s2)
    var_s2 = np.var(s2)
    # MP理论: 对于随机矩阵, mean≈var (Marchenko-Pastur)
    mp_deviation = var_s2 / max(mean_s2**2, 1e-10)
    
    # 谱熵
    p = s2 / total_var
    p = p[p > 0]
    spectral_entropy = -np.sum(p * np.log(p))
    
    return {
        'pr': pr_normalized,
        'pr_raw': pr,
        'kappa': kappa,
        'top10_pct': top10_pct,
        'mp_deviation': mp_deviation,
        'spectral_entropy': spectral_entropy,
        'mean_sigma': np.mean(s),
        'std_sigma': np.std(s),
        'n_singular': len(s),
    }


def measure_stability(model, tokenizer, device, layers_list, l_idx, alpha, mlp_type, texts):
    """测量层的传播稳定性"""
    n_layers = len(layers_list)
    max_ratios = []
    mean_ratios = []
    kl_divs = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        # 基线
        hidden_baseline = []
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_baseline.append(output[0][0, -1].detach().cpu().float().numpy())
                else:
                    hidden_baseline.append(output[0, -1].detach().cpu().float().numpy())
            return hook_fn
        
        hooks = []
        for i, layer in enumerate(layers_list):
            hooks.append(layer.register_forward_hook(make_hook(i)))
        with torch.no_grad():
            model(input_ids)
        for h in hooks:
            h.remove()
        
        # 扰动
        orig_weight = perturb_w_down(layers_list, l_idx, alpha, mlp_type)
        hidden_ablated = []
        def make_hook2(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_ablated.append(output[0][0, -1].detach().cpu().float().numpy())
                else:
                    hidden_ablated.append(output[0, -1].detach().cpu().float().numpy())
            return hook_fn
        
        hooks2 = []
        for i, layer in enumerate(layers_list):
            hooks2.append(layer.register_forward_hook(make_hook2(i)))
        with torch.no_grad():
            model(input_ids)
        for h in hooks2:
            h.remove()
        
        restore_w_down(layers_list, l_idx, orig_weight, mlp_type)
        
        # 计算传播比
        if len(hidden_baseline) != len(hidden_ablated) or len(hidden_baseline) == 0:
            continue
        
        delta_h_norms = []
        for k in range(len(hidden_baseline)):
            dh = hidden_ablated[k] - hidden_baseline[k]
            delta_h_norms.append(np.linalg.norm(dh))
        
        ratios = []
        for k in range(len(delta_h_norms) - 1):
            if delta_h_norms[k] > 1e-10:
                ratios.append(delta_h_norms[k+1] / delta_h_norms[k])
        
        if len(ratios) > 0:
            max_ratios.append(max(ratios))
            mean_ratios.append(np.mean(ratios))
        
        # KL散度
        with torch.no_grad():
            logits_b = model(input_ids).logits[0, -1]
        orig_weight2 = perturb_w_down(layers_list, l_idx, alpha, mlp_type)
        with torch.no_grad():
            logits_a = model(input_ids).logits[0, -1]
        restore_w_down(layers_list, l_idx, orig_weight2, mlp_type)
        kl_divs.append(compute_kl_divergence(logits_b, logits_a))
    
    is_stable = np.mean(max_ratios) < 2.0 if len(max_ratios) > 0 else True
    return {
        'max_ratio': np.mean(max_ratios) if max_ratios else 0,
        'mean_ratio': np.mean(mean_ratios) if mean_ratios else 0,
        'kl_div': np.mean(kl_divs) if kl_divs else 0,
        'is_stable': is_stable,
    }


# ============================================================
# P511: 从权重预测传播稳定性
# ============================================================
def run_p511(model_name):
    """从权重特征预测传播稳定性"""
    print(f"\n{'='*60}")
    print(f"P511: 从权重预测传播稳定性 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    
    test_texts = [
        "The quantum computer solved the complex mathematical problem.",
        "In the depths of the ocean, bioluminescent creatures glowed.",
        "The architect designed a sustainable building with solar panels.",
        "Music theory explains how different notes create harmony.",
        "The detective carefully examined the evidence at the scene.",
    ]
    
    # 1. 收集所有层的权重特征
    print("\n[1] 收集权重特征...")
    all_features = []
    
    for l in range(n_layers):
        layer = layers_list[l]
        weights = get_layer_weights(layer, info.d_model, info.mlp_type)
        layer_frac = l / max(n_layers - 1, 1)
        
        features = {'layer': l, 'layer_frac': layer_frac}
        
        # 谱特征
        for wname in ['W_down', 'W_gate', 'W_up', 'W_o']:
            W = getattr(weights, wname, None)
            if W is not None:
                spec = compute_spectral_features(W)
                for k, v in spec.items():
                    features[f'{wname}_{k}'] = v
            # 也计算范数
            if W is not None:
                features[f'{wname}_norm'] = np.linalg.norm(W)
                features[f'{wname}_frobenius'] = np.linalg.norm(W, 'fro')
        
        # LN权重
        features['post_ln_norm'] = get_post_ln_norm(layer)
        features['ln_norm'] = get_ln_norm(layer)
        
        all_features.append(features)
        print(f"  L{l}: W_gate_PR={features.get('W_gate_pr', 0):.3f}, "
              f"W_gate_kappa={features.get('W_gate_kappa', 0):.1f}, "
              f"post_ln_norm={features['post_ln_norm']:.3f}")
    
    # 2. 测量每层的传播稳定性
    print("\n[2] 测量传播稳定性...")
    stability_data = []
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    
    for l in sample_layers:
        print(f"  测量L{l}...")
        stab = measure_stability(model, tokenizer, device, layers_list, l, 0.1, info.mlp_type, test_texts[:3])
        stability_data.append({'layer': l, **stab})
        print(f"    max_ratio={stab['max_ratio']:.2f}, "
              f"mean_ratio={stab['mean_ratio']:.2f}, "
              f"kl_div={stab['kl_div']:.4f}, stable={stab['is_stable']}")
    
    # 3. 分析权重特征与稳定性的关系
    print("\n[3] 权重特征与稳定性的相关性...")
    
    # 将稳定性数据匹配到特征
    for sd in stability_data:
        l = sd['layer']
        if l < len(all_features):
            all_features[l]['max_ratio'] = sd['max_ratio']
            all_features[l]['mean_ratio'] = sd['mean_ratio']
            all_features[l]['kl_div'] = sd['kl_div']
            all_features[l]['is_stable'] = 1 if sd['is_stable'] else 0
    
    # 只分析有稳定性数据的层
    analyzed = [f for f in all_features if 'max_ratio' in f]
    
    if len(analyzed) < 3:
        print("  数据不足, 跳过分析")
        return
    
    # 连续特征相关性
    feature_keys = [k for k in analyzed[0].keys() 
                    if k not in ['layer', 'is_stable'] and isinstance(analyzed[0][k], (int, float))]
    
    print("\n  --- 与max_ratio的相关性 ---")
    correlations = []
    for key in feature_keys:
        vals = [f[key] for f in analyzed]
        ratios = [f['max_ratio'] for f in analyzed]
        if np.std(vals) > 1e-10 and np.std(ratios) > 1e-10:
            r, p = pearsonr(vals, ratios)
            correlations.append((key, r, p))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for key, r, p in correlations[:15]:
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {key}: r={r:.3f}{sig}")
    
    # 4. 逻辑回归分类(稳定vs不稳定)
    print("\n[4] 逻辑回归分类...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        
        # 选择数值特征
        numeric_keys = [k for k in feature_keys if k != 'max_ratio' and k != 'mean_ratio' and k != 'kl_div']
        X = np.array([[f[k] for k in numeric_keys] for f in analyzed])
        y = np.array([f['is_stable'] for f in analyzed])
        
        # 替换NaN/Inf
        X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
        
        if len(set(y)) >= 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_scaled, y)
            
            y_pred_proba = clf.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
            print(f"  AUC = {auc:.3f}")
            
            # 特征重要性
            importance = list(zip(numeric_keys, abs(clf.coef_[0])))
            importance.sort(key=lambda x: x[1], reverse=True)
            print("  Top-10权重特征:")
            for name, imp in importance[:10]:
                print(f"    {name}: {imp:.4f}")
        else:
            print(f"  类别不足({len(set(y))}类), 无法分类")
    except ImportError:
        print("  sklearn不可用, 跳过分类")
    
    # 5. 线性回归预测max_ratio
    print("\n[5] 线性回归预测max_ratio...")
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        reg_keys = [k for k in feature_keys if k != 'max_ratio' and k != 'mean_ratio' and k != 'kl_div']
        X = np.array([[f[k] for k in reg_keys] for f in analyzed])
        y_ratio = np.array([f['max_ratio'] for f in analyzed])
        X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 对数变换max_ratio(因为可能很大)
        y_log = np.log1p(y_ratio)
        
        reg = LinearRegression()
        reg.fit(X_scaled, y_log)
        R2 = reg.score(X_scaled, y_log)
        print(f"  R2(log(max_ratio+1)) = {R2:.3f}")
        
        importance = list(zip(reg_keys, abs(reg.coef_)))
        importance.sort(key=lambda x: x[1], reverse=True)
        print("  Top-10预测特征:")
        for name, imp in importance[:10]:
            print(f"    {name}: {imp:.4f}")
        
        # 简化公式: 只用top-5特征
        top5_keys = [k for k, _ in importance[:5]]
        X_top5 = np.array([[f[k] for k in top5_keys] for f in analyzed])
        X_top5 = np.nan_to_num(X_top5, nan=0, posinf=1e10, neginf=-1e10)
        X_top5_scaled = StandardScaler().fit_transform(X_top5)
        
        reg5 = LinearRegression()
        reg5.fit(X_top5_scaled, y_log)
        R2_5 = reg5.score(X_top5_scaled, y_log)
        print(f"  R2(top-5) = {R2_5:.3f}")
        print(f"  简化公式: log(max_ratio+1) ≈ ", end="")
        terms = []
        for k, c in zip(top5_keys, reg5.coef_):
            if abs(c) > 0.01:
                terms.append(f"{c:.2f}*{k}")
        print(" + ".join(terms) + f" + {reg5.intercept_:.2f}")
        
    except ImportError:
        print("  sklearn不可用, 跳过回归")
    
    # 6. 关键指标总结
    print("\n[6] 关键发现总结")
    n_stable = sum(1 for f in analyzed if f['is_stable'])
    n_unstable = len(analyzed) - n_stable
    print(f"  稳定层: {n_stable}/{len(analyzed)} ({100*n_stable/len(analyzed):.0f}%)")
    print(f"  不稳定层: {n_unstable}/{len(analyzed)} ({100*n_unstable/len(analyzed):.0f}%)")
    
    # top-5相关特征
    if correlations:
        print(f"  最相关特征: {correlations[0][0]} (r={correlations[0][1]:.3f})")
        print(f"  最相关(top5): " + ", ".join(f"{k}(r={r:.2f})" for k, r, _ in correlations[:5]))
    
    del model
    torch.cuda.empty_cache()
    print("P511完成")


# ============================================================
# P512: W_gate因果干预
# ============================================================
def run_p512(model_name):
    """W_gate因果干预验证"""
    print(f"\n{'='*60}")
    print(f"P512: W_gate因果干预 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    
    test_texts = [
        "The quantum computer solved the complex mathematical problem.",
        "In the depths of the ocean, bioluminescent creatures glowed.",
    ]
    
    # 选择3-4个代表性层(浅/中/深)
    sample_layers = [n_layers // 5, n_layers // 2, 3 * n_layers // 5, 4 * n_layers // 5]
    sample_layers = [l for l in sample_layers if l < n_layers]
    sample_layers = list(set(sample_layers))[:4]
    sample_layers.sort()
    
    print(f"测试层: {sample_layers}")
    
    results = []
    
    for l_idx in sample_layers:
        layer = layers_list[l_idx]
        layer_frac = l_idx / max(n_layers - 1, 1)
        print(f"\n--- L{l_idx} (frac={layer_frac:.2f}) ---")
        
        # 获取W_gate (兼容不同架构)
        weights = get_layer_weights(layer, info.d_model, info.mlp_type)
        W_gate = weights.W_gate
        if W_gate is None:
            print(f"  L{l_idx}: 无W_gate, 跳过")
            continue
        
        # 根据架构获取原始权重张量
        if info.mlp_type == "merged_gate_up":
            # GLM4: gate_up_proj前半是gate
            full_weight = layer.mlp.gate_up_proj.weight.data.clone()
            half = full_weight.shape[0] // 2
            W_gate_torch = full_weight[:half].clone()
            W_up_torch = full_weight[half:].clone()
            is_merged = True
        else:
            # Qwen3/DS7B: 独立gate_proj
            W_gate_torch = layer.mlp.gate_proj.weight.data.clone()
            is_merged = False
        
        m, n = W_gate_torch.shape
        
        # SVD分解
        U, S, Vt = torch.linalg.svd(W_gate_torch.float(), full_matrices=False)
        print(f"  W_gate: shape={m}x{n}, top-5 S={S[:5].tolist()}, merged={is_merged}")
        
        # === 干预1: 压缩top-k奇异值 ===
        print("\n  [干预1] 压缩top-k奇异值")
        for k_ratio in [0.1, 0.3, 0.5]:
            k = max(1, int(k_ratio * len(S)))
            S_modified = S.clone()
            S_modified[:k] = S[:k] * 0.5  # 压缩top-k到50%
            
            # 重构W_gate并设置
            W_new = U @ torch.diag(S_modified) @ Vt
            if is_merged:
                full_w = torch.cat([W_new.to(full_weight.dtype), W_up_torch], dim=0)
                layer.mlp.gate_up_proj.weight.data = full_w
            else:
                layer.mlp.gate_proj.weight.data = W_new.to(layer.mlp.gate_proj.weight.dtype)
            
            # 测量稳定性
            stab = measure_stability(model, tokenizer, device, layers_list, l_idx, 0.1, 
                                    info.mlp_type, test_texts[:2])
            
            # 测量kl_div
            kl_divs = []
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits_b = model(inputs["input_ids"]).logits[0, -1]
                orig_w = perturb_w_down(layers_list, l_idx, 0.1, info.mlp_type)
                with torch.no_grad():
                    logits_a = model(inputs["input_ids"]).logits[0, -1]
                restore_w_down(layers_list, l_idx, orig_w, info.mlp_type)
                kl_divs.append(compute_kl_divergence(logits_b, logits_a))
            
            mean_kl = np.mean(kl_divs)
            print(f"    compress top-{k_ratio:.0%}(k={k}): "
                  f"max_ratio={stab['max_ratio']:.2f}, kl_div={mean_kl:.4f}")
            
            results.append({
                'layer': l_idx, 'intervention': 'compress_top',
                'k_ratio': k_ratio, 'k': k,
                'max_ratio': stab['max_ratio'],
                'kl_div': mean_kl,
            })
            
            # 恢复
            if is_merged:
                layer.mlp.gate_up_proj.weight.data = torch.cat([W_gate_torch.to(full_weight.dtype), W_up_torch], dim=0)
            else:
                layer.mlp.gate_proj.weight.data = W_gate_torch.clone()
        
        # === 干预2: 放大bottom-k奇异值 ===
        print("\n  [干预2] 放大bottom-k奇异值")
        for k_ratio in [0.1, 0.3, 0.5]:
            k = max(1, int(k_ratio * len(S)))
            S_modified = S.clone()
            S_modified[-k:] = S[-k:] * 2.0  # 放大bottom-k 2倍
            
            W_new = U @ torch.diag(S_modified) @ Vt
            if is_merged:
                full_w = torch.cat([W_new.to(full_weight.dtype), W_up_torch], dim=0)
                layer.mlp.gate_up_proj.weight.data = full_w
            else:
                layer.mlp.gate_proj.weight.data = W_new.to(layer.mlp.gate_proj.weight.dtype)
            
            stab = measure_stability(model, tokenizer, device, layers_list, l_idx, 0.1,
                                    info.mlp_type, test_texts[:2])
            
            kl_divs = []
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits_b = model(inputs["input_ids"]).logits[0, -1]
                orig_w = perturb_w_down(layers_list, l_idx, 0.1, info.mlp_type)
                with torch.no_grad():
                    logits_a = model(inputs["input_ids"]).logits[0, -1]
                restore_w_down(layers_list, l_idx, orig_w, info.mlp_type)
                kl_divs.append(compute_kl_divergence(logits_b, logits_a))
            
            mean_kl = np.mean(kl_divs)
            print(f"    amplify bottom-{k_ratio:.0%}(k={k}): "
                  f"max_ratio={stab['max_ratio']:.2f}, kl_div={mean_kl:.4f}")
            
            results.append({
                'layer': l_idx, 'intervention': 'amplify_bottom',
                'k_ratio': k_ratio, 'k': k,
                'max_ratio': stab['max_ratio'],
                'kl_div': mean_kl,
            })
            
            if is_merged:
                layer.mlp.gate_up_proj.weight.data = torch.cat([W_gate_torch.to(full_weight.dtype), W_up_torch], dim=0)
            else:
                layer.mlp.gate_proj.weight.data = W_gate_torch.clone()
        
        # === 干预3: 用MP分布替换奇异值 ===
        print("\n  [干预3] MP分布替换奇异值")
        # Marchenko-Pastur分布: sigma_i ~ sqrt(mean(sigma^2)) * random
        # 简化: 用等间距替换
        S_mean = torch.mean(S**2).sqrt()
        S_mp = torch.linspace(S[0] * 0.8, S[-1] * 1.2, len(S), device=S.device)  # 线性分布
        # 或: 完全均匀化
        S_uniform = torch.ones_like(S) * S_mean
        
        for s_type, S_new in [('linear', S_mp), ('uniform', S_uniform)]:
            W_new = U @ torch.diag(S_new.to(S.dtype)) @ Vt
            if is_merged:
                full_w = torch.cat([W_new.to(full_weight.dtype), W_up_torch], dim=0)
                layer.mlp.gate_up_proj.weight.data = full_w
            else:
                layer.mlp.gate_proj.weight.data = W_new.to(layer.mlp.gate_proj.weight.dtype)
            
            stab = measure_stability(model, tokenizer, device, layers_list, l_idx, 0.1,
                                    info.mlp_type, test_texts[:2])
            
            kl_divs = []
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits_b = model(inputs["input_ids"]).logits[0, -1]
                orig_w = perturb_w_down(layers_list, l_idx, 0.1, info.mlp_type)
                with torch.no_grad():
                    logits_a = model(inputs["input_ids"]).logits[0, -1]
                restore_w_down(layers_list, l_idx, orig_w, info.mlp_type)
                kl_divs.append(compute_kl_divergence(logits_b, logits_a))
            
            mean_kl = np.mean(kl_divs)
            print(f"    replace_{s_type}: max_ratio={stab['max_ratio']:.2f}, kl_div={mean_kl:.4f}")
            
            results.append({
                'layer': l_idx, 'intervention': f'replace_{s_type}',
                'max_ratio': stab['max_ratio'],
                'kl_div': mean_kl,
            })
            
            if is_merged:
                layer.mlp.gate_up_proj.weight.data = torch.cat([W_gate_torch.to(full_weight.dtype), W_up_torch], dim=0)
            else:
                layer.mlp.gate_proj.weight.data = W_gate_torch.clone()
        
        # 基线
        stab_base = measure_stability(model, tokenizer, device, layers_list, l_idx, 0.1,
                                      info.mlp_type, test_texts[:2])
        print(f"\n  [基线] max_ratio={stab_base['max_ratio']:.2f}")
        results.append({
            'layer': l_idx, 'intervention': 'baseline',
            'max_ratio': stab_base['max_ratio'],
            'kl_div': 0,  # 基线kl_div为0(无扰动)
        })
    
    # 总结
    print(f"\n{'='*40}")
    print("P512 因果干预总结:")
    print(f"{'='*40}")
    
    for l_idx in sample_layers:
        layer_results = [r for r in results if r['layer'] == l_idx]
        if not layer_results:
            continue
        
        baseline = [r for r in layer_results if r['intervention'] == 'baseline']
        base_ratio = baseline[0]['max_ratio'] if baseline else 0
        
        print(f"\n  L{l_idx}: 基线max_ratio={base_ratio:.2f}")
        for r in layer_results:
            if r['intervention'] != 'baseline':
                change = r['max_ratio'] - base_ratio
                direction = "↑" if change > 0.1 else "↓" if change < -0.1 else "→"
                print(f"    {r['intervention']}: ratio={r['max_ratio']:.2f} {direction}(Δ={change:+.2f})")
    
    del model
    torch.cuda.empty_cache()
    print("P512完成")


# ============================================================
# P513: 跨模型传播理论
# ============================================================
def run_p513(model_name):
    """跨模型传播理论: 从权重预测geo_mean_ratio"""
    print(f"\n{'='*60}")
    print(f"P513: 跨模型传播理论 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    
    test_texts = [
        "The quantum computer solved the complex mathematical problem.",
        "In the depths of the ocean, bioluminescent creatures glowed.",
    ]
    
    # 1. 对所有层测量geo_mean_ratio和权重特征
    print("\n[1] 测量逐层geo_mean_ratio...")
    layer_data = []
    
    # 采样8-10层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if len(sample_layers) > 10:
        sample_layers = sample_layers[:10]
    
    for l_idx in sample_layers:
        layer = layers_list[l_idx]
        layer_frac = l_idx / max(n_layers - 1, 1)
        
        # 获取权重特征
        weights = get_layer_weights(layer, info.d_model, info.mlp_type)
        W_gate_spec = compute_spectral_features(weights.W_gate) if weights.W_gate is not None else {}
        W_down_spec = compute_spectral_features(weights.W_down) if weights.W_down is not None else {}
        
        post_ln = get_post_ln_norm(layer)
        ln_norm = get_ln_norm(layer)
        
        # 测量传播比
        inputs = tokenizer(test_texts[0], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        # 基线
        hidden_baseline = []
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_baseline.append(output[0][0, -1].detach().cpu().float().numpy())
                else:
                    hidden_baseline.append(output[0, -1].detach().cpu().float().numpy())
            return hook_fn
        
        hooks = []
        for i, lyr in enumerate(layers_list):
            hooks.append(lyr.register_forward_hook(make_hook(i)))
        with torch.no_grad():
            model(input_ids)
        for h in hooks:
            h.remove()
        
        # 扰动
        orig_weight = perturb_w_down(layers_list, l_idx, 0.1, info.mlp_type)
        hidden_ablated = []
        def make_hook2(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_ablated.append(output[0][0, -1].detach().cpu().float().numpy())
                else:
                    hidden_ablated.append(output[0, -1].detach().cpu().float().numpy())
            return hook_fn
        
        hooks2 = []
        for i, lyr in enumerate(layers_list):
            hooks2.append(lyr.register_forward_hook(make_hook2(i)))
        with torch.no_grad():
            model(input_ids)
        for h in hooks2:
            h.remove()
        
        restore_w_down(layers_list, l_idx, orig_weight, info.mlp_type)
        
        # 计算逐层传播比
        delta_h_norms = []
        for k in range(len(hidden_baseline)):
            dh = hidden_ablated[k] - hidden_baseline[k]
            delta_h_norms.append(np.linalg.norm(dh))
        
        ratios = []
        for k in range(len(delta_h_norms) - 1):
            if delta_h_norms[k] > 1e-10:
                ratios.append(delta_h_norms[k+1] / delta_h_norms[k])
        
        # 只取l_idx之后的传播比(后续传播)
        post_l_ratios = ratios[l_idx:] if l_idx < len(ratios) else ratios
        if len(post_l_ratios) == 0:
            post_l_ratios = ratios
        
        geo_mean = np.exp(np.mean(np.log(np.maximum(post_l_ratios, 1e-10))))
        
        data = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'geo_mean_ratio': geo_mean,
            'mean_ratio': np.mean(post_l_ratios) if post_l_ratios else 0,
            'std_ratio': np.std(post_l_ratios) if len(post_l_ratios) > 1 else 0,
            'max_ratio': max(post_l_ratios) if post_l_ratios else 0,
            'post_ln_norm': post_ln,
            'ln_norm': ln_norm,
            'W_gate_kappa': W_gate_spec.get('kappa', 0),
            'W_gate_pr': W_gate_spec.get('pr', 0),
            'W_gate_mp_dev': W_gate_spec.get('mp_deviation', 0),
            'W_gate_top10': W_gate_spec.get('top10_pct', 0),
            'W_down_kappa': W_down_spec.get('kappa', 0),
            'W_down_pr': W_down_spec.get('pr', 0),
            'W_down_mp_dev': W_down_spec.get('mp_deviation', 0),
        }
        
        layer_data.append(data)
        print(f"  L{l_idx}: geo_mean_ratio={geo_mean:.3f}, "
              f"W_gate_kappa={data['W_gate_kappa']:.1f}, "
              f"post_ln_norm={post_ln:.3f}")
    
    # 2. 分析geo_mean_ratio与权重特征的关系
    print("\n[2] geo_mean_ratio与权重特征的相关性...")
    
    feature_keys = [k for k in layer_data[0].keys() 
                    if k not in ['layer', 'geo_mean_ratio', 'mean_ratio', 'std_ratio', 'max_ratio']
                    and isinstance(layer_data[0][k], (int, float))]
    
    for key in feature_keys:
        vals = [d[key] for d in layer_data]
        ratios = [d['geo_mean_ratio'] for d in layer_data]
        if np.std(vals) > 1e-10 and np.std(ratios) > 1e-10:
            r, p = pearsonr(vals, ratios)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {key}: r={r:.3f}{sig}")
    
    # 3. 多元回归: geo_mean_ratio ≈ f(W_gate_kappa, W_down_PR, post_ln_norm, ...)
    print("\n[3] 多元回归预测geo_mean_ratio...")
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        X = np.array([[d[k] for k in feature_keys] for d in layer_data])
        y = np.array([d['geo_mean_ratio'] for d in layer_data])
        X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
        y = np.nan_to_num(y, nan=1.0, posinf=10.0, neginf=0.0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 全特征
        reg = LinearRegression()
        reg.fit(X_scaled, y)
        R2_full = reg.score(X_scaled, y)
        print(f"  R2(全特征) = {R2_full:.3f}")
        
        importance = list(zip(feature_keys, reg.coef_))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)
        print("  特征权重:")
        for name, c in importance:
            print(f"    {name}: {c:.4f}")
        
        # 核心公式: 只用W_gate_kappa + W_down_PR + post_ln_norm
        core_keys = ['W_gate_kappa', 'W_down_pr', 'post_ln_norm', 'layer_frac']
        available_core = [k for k in core_keys if k in feature_keys]
        if available_core:
            X_core = np.array([[d[k] for k in available_core] for d in layer_data])
            X_core = np.nan_to_num(X_core, nan=0, posinf=1e10, neginf=-1e10)
            X_core_scaled = StandardScaler().fit_transform(X_core)
            
            reg_core = LinearRegression()
            reg_core.fit(X_core_scaled, y)
            R2_core = reg_core.score(X_core_scaled, y)
            print(f"\n  R2(核心公式: {', '.join(available_core)}) = {R2_core:.3f}")
            for k, c in zip(available_core, reg_core.coef_):
                print(f"    {k}: {c:.4f}")
            print(f"    intercept: {reg_core.intercept_:.4f}")
        
        # 对数回归: log(geo_mean_ratio) ≈ a * log(W_gate_kappa) + b * log(W_down_PR) + ...
        print("\n[4] 对数回归...")
        y_log = np.log(np.maximum(y, 1e-10))
        reg_log = LinearRegression()
        reg_log.fit(X_scaled, y_log)
        R2_log = reg_log.score(X_scaled, y_log)
        print(f"  R2(log回归) = {R2_log:.3f}")
        
        importance_log = list(zip(feature_keys, reg_log.coef_))
        importance_log.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, c in importance_log[:5]:
            print(f"    {name}: {c:.4f}")
        
    except ImportError:
        print("  sklearn不可用, 跳过回归")
    
    # 4. 传播稳定性与权重的关系总结
    print(f"\n[5] 关键发现总结")
    print(f"  模型: {model_name}")
    print(f"  采样层数: {len(layer_data)}")
    
    stable = [d for d in layer_data if d['max_ratio'] < 2.0]
    unstable = [d for d in layer_data if d['max_ratio'] >= 2.0]
    print(f"  稳定层: {len(stable)}/{len(layer_data)}")
    print(f"  不稳定层: {len(unstable)}/{len(layer_data)}")
    
    if stable and unstable:
        # 对比稳定/不稳定层的权重特征
        print("\n  稳定vs不稳定层特征对比:")
        for key in ['W_gate_kappa', 'W_gate_pr', 'W_gate_mp_dev', 'W_down_kappa', 
                     'post_ln_norm', 'layer_frac']:
            if key in feature_keys:
                s_val = np.mean([d[key] for d in stable])
                u_val = np.mean([d[key] for d in unstable])
                ratio = u_val / max(s_val, 1e-10)
                print(f"    {key}: 稳定={s_val:.3f}, 不稳定={u_val:.3f}, ratio={ratio:.2f}")
    
    del model
    torch.cuda.empty_cache()
    print("P513完成")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase CIX: 信号传播稳定性预测")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p511", "p512", "p513"])
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.experiment == "p511":
        run_p511(args.model)
    elif args.experiment == "p512":
        run_p512(args.model)
    elif args.experiment == "p513":
        run_p513(args.model)
    
    elapsed = time.time() - start_time
    print(f"\n耗时: {elapsed:.1f}秒")
