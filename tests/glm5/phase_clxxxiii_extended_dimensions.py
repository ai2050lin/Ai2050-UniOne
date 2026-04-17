"""
Phase CLXXXIII: 扩大功能维度 — 从5维到50-200维
核心问题: 5维太薄(5/4096=0.12%)，所有统计指标≈随机
如果扩展到50-200维，是否能在W_U中找到统计上显著的功能子空间?

测试:
P831: 残差流PCA — 35个功能对的前N个主成分是否特殊?
P832: W_U投影检验 — PCA子空间在W_U中的能量是否>随机?
P833: 逐维度因果干预 — 删除PCA的每个主方向，效果是否>随机?
P834: 训练方法与功能厚度 — 不同模型的"有效功能维度"是多少?

Usage:
    python phase_clxxxiii_extended_dimensions.py --model glm4 --n-random 50
    python phase_clxxxiii_extended_dimensions.py --model qwen3 --n-random 50
    python phase_clxxxiii_extended_dimensions.py --model deepseek7b --n-random 50
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

# 确保可以导入model_utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import load_model, release_model
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def get_function_pairs():
    """定义35个功能对"""
    return {
        # Syntax (7 pairs)
        'syntax': [
            ("The cat sits", "Cats sit"),         # singular/plural
            ("She runs", "She run"),              # agreement
            ("The big cat", "The cat big"),       # word order
            ("I have eaten", "I eat have"),       # tense word order
            ("He is running", "Him running is"),  # case+order
            ("The cat that runs", "Cat the that runs"),  # determiner order
            ("She will go", "Will she go"),       # statement vs question
        ],
        # Semantic (7 pairs)
        'semantic': [
            ("The cat sits", "The dog sits"),     # animal swap
            ("She is happy", "She is sad"),       # emotion swap
            ("The house is big", "The house is small"),  # size
            ("He loves music", "He hates music"), # attitude
            ("The king ruled", "The queen ruled"), # gender
            ("I bought food", "I stole food"),    # action intent
            ("The sun rises", "The sun sets"),    # direction
        ],
        # Style (7 pairs)
        'style': [
            ("I am happy", "I'm happy"),          # contraction
            ("Do not go", "Don't go"),            # contraction
            ("It is fine", "It's fine"),           # contraction
            ("She will come", "She'll come"),      # contraction
            ("I have done", "I've done"),          # contraction
            ("Cannot do", "Can't do"),             # contraction
            ("We are here", "We're here"),         # contraction
        ],
        # Tense (7 pairs)
        'tense': [
            ("She walks", "She walked"),          # present/past
            ("He runs", "He ran"),                # present/past
            ("I go", "I went"),                   # present/past
            ("They see", "They saw"),             # present/past
            ("We eat", "We ate"),                 # present/past
            ("She writes", "She wrote"),          # present/past
            ("He thinks", "He thought"),          # present/past
        ],
        # Polarity (7 pairs)
        'polarity': [
            ("She is happy", "She is not happy"), # affirmation/negation
            ("I like it", "I don't like it"),     # affirmation/negation
            ("He can go", "He cannot go"),        # can/cannot
            ("This is good", "This is not good"), # is/is not
            ("She will come", "She will not come"),# will/will not
            ("They have food", "They have no food"), # have/have no
            ("We know", "We do not know"),        # know/not know
        ],
    }


def extract_directions_from_pairs(model, tokenizer, device, pairs_dict, layer_idx=0):
    """从功能对中提取残差流方向"""
    directions = []
    labels = []
    
    for func_type, pairs in pairs_dict.items():
        for i, (text_a, text_b) in enumerate(pairs):
            # 获取两个文本在layer_idx的残差流
            for text, sign in [(text_a, 1.0), (text_b, -1.0)]:
                tokens = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**tokens, output_hidden_states=True)
                hs = to_numpy(outputs.hidden_states[layer_idx][0, -1])  # last token
                directions.append(sign * hs)
            
            labels.append(f"{func_type}_{i}")
    
    # directions: [2*n_pairs, d_model], 我们需要每对的差
    n_pairs = len(labels)
    diffs = np.zeros((n_pairs, len(directions[0])))
    for i in range(n_pairs):
        diffs[i] = directions[2*i] + directions[2*i+1]  # sign already encoded
    
    return diffs, labels


def run_experiment(model_name, n_random=50):
    """运行CLXXXIII实验"""
    print(f"\n{'='*60}", flush=True)
    print(f"Phase CLXXXIII: Extended Functional Dimensions - {model_name}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    d_model = model.config.hidden_size
    
    # 获取功能对
    pairs_dict = get_function_pairs()
    n_total_pairs = sum(len(v) for v in pairs_dict.values())
    print(f"d_model={d_model}, total_pairs={n_total_pairs}", flush=True)
    
    # === P831: 残差流PCA ===
    print(f"\n--- P831: Residual Stream PCA ---", flush=True)
    
    # 提取所有功能对的方向
    diffs, labels = extract_directions_from_pairs(model, tokenizer, device, pairs_dict, layer_idx=0)
    print(f"Extracted {len(diffs)} difference vectors, shape: {diffs.shape}", flush=True)
    
    # PCA
    from sklearn.decomposition import PCA
    n_pca = min(35, diffs.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    pca.fit(diffs)
    
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    print(f"PCA top-5 explained variance: {explained_var[:5]}", flush=True)
    print(f"Cumulative: 5={cum_var[4]:.3f}, 10={cum_var[min(9,n_pca-1)]:.3f}, 20={cum_var[min(19,n_pca-1)]:.3f}", flush=True)
    
    # PCA主方向
    pca_dirs = pca.components_  # [n_pca, d_model]
    
    # === P832: W_U投影检验 ===
    print(f"\n--- P832: W_U Projection Test ---", flush=True)
    
    # 获取W_U
    W_U = to_numpy(model.lm_head.weight)  # [vocab, d_model]
    print(f"W_U shape: {W_U.shape}", flush=True)
    
    # 计算PCA子空间在W_U中的能量
    results_p832 = {}
    
    for k in [5, 10, 20, 30, 35]:
        # 功能子空间 = PCA前k个主方向
        V_func = pca_dirs[:k]  # [k, d_model]
        
        # 功能子空间在W_U中的能量占比
        proj_func = W_U @ V_func.T  # [vocab, k]
        energy_func = np.sum(proj_func ** 2) / np.sum(W_U ** 2)
        
        # 随机基线
        rng = np.random.default_rng(42)
        random_energies = []
        for _ in range(n_random):
            V_rand = rng.standard_normal((k, d_model))
            Q_rand, _ = np.linalg.qr(V_rand.T)
            Q_rand = Q_rand.T[:k]  # [k, d_model]
            proj_rand = W_U @ Q_rand.T
            energy_rand = np.sum(proj_rand ** 2) / np.sum(W_U ** 2)
            random_energies.append(energy_rand)
        
        rand_mean = np.mean(random_energies)
        rand_std = np.std(random_energies)
        z = (energy_func - rand_mean) / max(rand_std, 1e-10)
        import math as _math
        p = 0.5 * (1 + _math.erf(-abs(z) / _math.sqrt(2)))
        
        results_p832[f'k={k}'] = {
            'func_energy': float(energy_func),
            'rand_mean': float(rand_mean),
            'rand_std': float(rand_std),
            'z': float(z),
            'p': float(p),
            'ratio': float(energy_func / rand_mean) if rand_mean > 0 else 0,
            'significant': bool(p < 0.05)
        }
        print(f"  k={k}: func={energy_func:.6f}, rand={rand_mean:.6f}±{rand_std:.6f}, z={z:.1f}, p={p:.4f}, ratio={energy_func/rand_mean:.2f}x", flush=True)
    
    # === P833: 逐维度因果干预 ===
    print(f"\n--- P833: Per-Direction Causal Intervention ---", flush=True)
    
    test_text = "The cat sits on the mat"
    tokens = tokenizer(test_text, return_tensors="pt").to(device)
    
    # 基线输出
    with torch.no_grad():
        base_outputs = model(**tokens)
    base_logits = base_outputs.logits[0, -1].float()
    base_probs = torch.softmax(base_logits, dim=-1).cpu().numpy()
    
    # 干预中间层
    layer_idx = model.config.num_hidden_layers // 2  # 中间层
    
    # 对PCA的每个主方向做干预
    pca_intervention_results = []
    for i in range(min(10, n_pca)):  # 前10个主方向
        dir_np = pca_dirs[i]
        
        def make_hook(direction, scale=1.0):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                dir_tensor = torch.tensor(direction, dtype=hs.dtype, device=hs.device) * scale
                proj = torch.matmul(hs, dir_tensor)
                hs_new = hs - proj.unsqueeze(-1) * dir_tensor.unsqueeze(0).unsqueeze(0)
                if isinstance(output, tuple):
                    return (hs_new,) + output[1:]
                return hs_new
            return hook_fn
        
        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_hook(make_hook(dir_np))
        
        try:
            with torch.no_grad():
                outputs = model(**tokens)
            logits = outputs.logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            kl = np.sum(base_probs * np.log(base_probs / (probs + 1e-10) + 1e-10))
        finally:
            hook.remove()
        
        pca_intervention_results.append({
            'direction': f'PCA_{i}',
            'var_explained': float(explained_var[i]),
            'kl_divergence': float(kl)
        })
        print(f"  PCA[{i}]: var={explained_var[i]:.4f}, KL={kl:.6f}", flush=True)
    
    # 对随机方向做同样的干预
    rng_rand = np.random.default_rng(123)
    random_intervention_results = []
    for i in range(10):
        rand_dir = rng_rand.standard_normal(d_model)
        rand_dir /= np.linalg.norm(rand_dir)
        
        def make_hook2(direction):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                dir_tensor = torch.tensor(direction, dtype=hs.dtype, device=hs.device)
                proj = torch.matmul(hs, dir_tensor)
                hs_new = hs - proj.unsqueeze(-1) * dir_tensor.unsqueeze(0).unsqueeze(0)
                if isinstance(output, tuple):
                    return (hs_new,) + output[1:]
                return hs_new
            return hook_fn
        
        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_hook(make_hook2(rand_dir))
        
        try:
            with torch.no_grad():
                outputs = model(**tokens)
            logits = outputs.logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            kl = np.sum(base_probs * np.log(base_probs / (probs + 1e-10) + 1e-10))
        finally:
            hook.remove()
        
        random_intervention_results.append({
            'direction': f'random_{i}',
            'kl_divergence': float(kl)
        })
        print(f"  Random[{i}]: KL={kl:.6f}", flush=True)
    
    # 比较
    pca_kls = [r['kl_divergence'] for r in pca_intervention_results]
    rand_kls = [r['kl_divergence'] for r in random_intervention_results]
    print(f"\n  PCA KL: mean={np.mean(pca_kls):.6f}, std={np.std(pca_kls):.6f}", flush=True)
    print(f"  Random KL: mean={np.mean(rand_kls):.6f}, std={np.std(rand_kls):.6f}", flush=True)
    
    # 置换检验
    all_kls = pca_kls + rand_kls
    n_pca_vals = len(pca_kls)
    observed_diff = np.mean(pca_kls) - np.mean(rand_kls)
    n_perm = 1000
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(len(all_kls))
        perm_diff = np.mean([all_kls[i] for i in perm[:n_pca_vals]]) - np.mean([all_kls[i] for i in perm[n_pca_vals:]])
        if perm_diff >= observed_diff:
            count += 1
    p_perm = count / n_perm
    print(f"  Permutation test: observed_diff={observed_diff:.6f}, p={p_perm:.4f}", flush=True)
    
    # === P834: 有效功能维度 ===
    print(f"\n--- P834: Effective Functional Dimensionality ---", flush=True)
    
    # 用PCA的累积方差确定"有效功能维度"
    for threshold in [0.5, 0.8, 0.9, 0.95]:
        eff_dim = int(np.searchsorted(cum_var, threshold)) + 1
        if eff_dim > n_pca:
            eff_dim = n_pca
        print(f"  Threshold {threshold:.0%}: {eff_dim} dimensions needed", flush=True)
    
    # 也对每层做PCA看功能维度的层间变化
    print(f"\n--- Layer-wise functional dimensionality ---", flush=True)
    layer_eff_dims = []
    for layer_idx in [0, model.config.num_hidden_layers//4, model.config.num_hidden_layers//2, 
                      3*model.config.num_hidden_layers//4, model.config.num_hidden_layers-1]:
        diffs_layer, _ = extract_directions_from_pairs(model, tokenizer, device, pairs_dict, layer_idx=layer_idx)
        pca_layer = PCA(n_components=min(35, diffs_layer.shape[0]-1), random_state=42)
        pca_layer.fit(diffs_layer)
        cum_var_layer = np.cumsum(pca_layer.explained_variance_ratio_)
        
        eff90 = int(np.searchsorted(cum_var_layer, 0.9)) + 1
        eff95 = int(np.searchsorted(cum_var_layer, 0.95)) + 1
        layer_eff_dims.append({
            'layer': layer_idx,
            'eff_dim_90': min(eff90, n_pca),
            'eff_dim_95': min(eff95, n_pca),
            'top5_var': pca_layer.explained_variance_ratio_[:5].tolist()
        })
        print(f"  Layer {layer_idx}: eff_dim_90={layer_eff_dims[-1]['eff_dim_90']}, eff_dim_95={layer_eff_dims[-1]['eff_dim_95']}", flush=True)
    
    # 释放模型
    release_model(model)
    
    # 保存结果
    results = {
        'model': model_name,
        'd_model': d_model,
        'n_total_pairs': n_total_pairs,
        'timestamp': datetime.now().isoformat(),
        'config': {'n_random': n_random, 'seed': 42},
        
        'p831_pca': {
            'n_components': n_pca,
            'explained_variance_ratio': explained_var.tolist(),
            'cumulative_variance': cum_var.tolist(),
        },
        
        'p832_wu_projection': results_p832,
        
        'p833_intervention': {
            'pca_results': pca_intervention_results,
            'random_results': random_intervention_results,
            'pca_kl_mean': float(np.mean(pca_kls)),
            'random_kl_mean': float(np.mean(rand_kls)),
            'permutation_p': float(p_perm),
            'observed_diff': float(observed_diff),
        },
        
        'p834_effective_dim': {
            'layer_eff_dims': layer_eff_dims,
        },
    }
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'phase_clxxxiii')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{model_name}_results.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}", flush=True)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'qwen3', 'deepseek7b'])
    parser.add_argument('--n-random', type=int, default=50)
    args = parser.parse_args()
    
    run_experiment(args.model, args.n_random)
